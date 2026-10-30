#include "gemma4/Gemma4ReferenceGraph.hpp"

#include "gemma4/Gemma4Config.hpp"
#include "package/TensorStore.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace gemma_runtime::gemma4 {
namespace {

constexpr size_t kSequence = 6210;
constexpr size_t kPerLayerStride = Gemma4Config::kLayers * Gemma4Config::kPerLayerWidth;

MPSShape *shape(std::initializer_list<size_t> dimensions) {
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:dimensions.size()];
    for (size_t dimension : dimensions) [result addObject:@(dimension)];
    return result;
}

MPSShape *shape(const std::vector<size_t> &dimensions) {
    NSMutableArray<NSNumber *> *result = [NSMutableArray arrayWithCapacity:dimensions.size()];
    for (size_t dimension : dimensions) [result addObject:@(dimension)];
    return result;
}

uint16_t toBfloat16(float value) noexcept {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t leastSignificantBit = (bits >> 16U) & 1U;
    bits += 0x7FFFU + leastSignificantBit;
    return static_cast<uint16_t>(bits >> 16U);
}

float fromBfloat16(uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

double bfloat16Ulp(uint16_t bits) noexcept {
    const uint16_t magnitude = bits & 0x7FFFU;
    if (magnitude == 0 || magnitude >= 0x7F80U) return std::numeric_limits<float>::denorm_min();
    const double value = std::abs(static_cast<double>(fromBfloat16(magnitude)));
    const double above = std::abs(static_cast<double>(fromBfloat16(magnitude + 1U)) - value);
    const double below = std::abs(value - static_cast<double>(fromBfloat16(magnitude - 1U)));
    return std::max(above, below);
}

void requireDescriptor(
    const package::TensorView &view,
    const std::string &name,
    const char *dtype,
    std::initializer_list<uint64_t> expectedShape) {
    if (view.descriptor->dtype != dtype) throw std::invalid_argument(name + " has the wrong dtype");
    if (view.descriptor->shape != std::vector<uint64_t>(expectedShape)) {
        throw std::invalid_argument(name + " has the wrong shape");
    }
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, std::span<const std::byte> bytes) {
    if (bytes.empty()) throw std::invalid_argument("cannot feed an empty tensor");
    id<MTLBuffer> result = [device newBufferWithBytes:bytes.data()
                                             length:bytes.size()
                                            options:MTLResourceStorageModeShared];
    if (!result) throw std::runtime_error("unable to allocate MPSGraph tensor buffer");
    return result;
}

MPSGraphTensor *feedBfloat16(
    MPSGraph *graph,
    id<MTLDevice> device,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    std::span<const std::byte> bytes,
    MPSShape *tensorShape,
    NSString *name) {
    MPSGraphTensor *tensor = [graph placeholderWithShape:tensorShape dataType:MPSDataTypeBFloat16 name:name];
    id<MTLBuffer> buffer = makeBuffer(device, bytes);
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                      shape:tensorShape
                                                                   dataType:MPSDataTypeBFloat16];
    if (!data) throw std::runtime_error("unable to wrap MPSGraph BF16 input");
    feeds[tensor] = data;
    return tensor;
}

MPSGraphTensor *feedBfloat16(
    MPSGraph *graph,
    id<MTLDevice> device,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    std::span<const uint16_t> values,
    MPSShape *tensorShape,
    NSString *name) {
    return feedBfloat16(graph, device, feeds, std::as_bytes(values), tensorShape, name);
}

MPSGraphTensor *feedModelTensor(
    MPSGraph *graph,
    id<MTLDevice> device,
    NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds,
    const package::TensorStore &model,
    const std::string &name,
    std::initializer_list<uint64_t> expectedShape) {
    const package::TensorView view = model.tensor(name);
    requireDescriptor(view, name, "BF16", expectedShape);
    std::vector<size_t> dimensions;
    dimensions.reserve(view.descriptor->shape.size());
    for (uint64_t dimension : view.descriptor->shape) dimensions.push_back(static_cast<size_t>(dimension));
    return feedBfloat16(graph, device, feeds, view.bytes, shape(dimensions), [NSString stringWithUTF8String:name.c_str()]);
}

MPSGraphTensor *scalar(MPSGraph *graph, double value, MPSDataType dataType) {
    return [graph constantWithScalar:value dataType:dataType];
}

MPSGraphTensor *cast(MPSGraph *graph, MPSGraphTensor *tensor, MPSDataType dataType, NSString *name) {
    return [graph castTensor:tensor toType:dataType name:name];
}

MPSGraphTensor *rmsNorm(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    const std::vector<size_t> &reducedShape,
    size_t lastAxis,
    NSString *name) {
    MPSGraphTensor *input32 = cast(graph, input, MPSDataTypeFloat32, [name stringByAppendingString:@".input32"]);
    MPSGraphTensor *squared = [graph squareWithTensor:input32 name:[name stringByAppendingString:@".square"]];
    MPSGraphTensor *mean = [graph meanOfTensor:squared axes:@[@(lastAxis)] name:[name stringByAppendingString:@".mean"]];
    mean = [graph reshapeTensor:mean withShape:shape(reducedShape) name:[name stringByAppendingString:@".mean_reshape"]];
    MPSGraphTensor *epsilon = scalar(graph, Gemma4Config::kRmsEpsilon, MPSDataTypeFloat32);
    mean = [graph additionWithPrimaryTensor:mean secondaryTensor:epsilon name:[name stringByAppendingString:@".epsilon"]];
    MPSGraphTensor *exponent = scalar(graph, -0.5, MPSDataTypeFloat32);
    MPSGraphTensor *inverse = [graph powerWithPrimaryTensor:mean
                                           secondaryTensor:exponent
                                                      name:[name stringByAppendingString:@".inverse"]];
    MPSGraphTensor *normalized = [graph multiplicationWithPrimaryTensor:input32
                                                        secondaryTensor:inverse
                                                                   name:[name stringByAppendingString:@".normalized"]];
    if (weight) {
        MPSGraphTensor *weight32 = cast(graph, weight, MPSDataTypeFloat32, [name stringByAppendingString:@".weight32"]);
        normalized = [graph multiplicationWithPrimaryTensor:normalized
                                            secondaryTensor:weight32
                                                       name:[name stringByAppendingString:@".scaled"]];
    }
    return cast(graph, normalized, MPSDataTypeBFloat16, name);
}

MPSGraphTensor *dense(
    MPSGraph *graph,
    MPSGraphTensor *input,
    MPSGraphTensor *weight,
    NSString *name) {
    MPSGraphTensor *transposed = [graph transposeTensor:weight
                                              dimension:0
                                          withDimension:1
                                                   name:[name stringByAppendingString:@".weight_t"]];
    return [graph matrixMultiplicationWithPrimaryTensor:input secondaryTensor:transposed name:name];
}

MPSGraphTensor *geluPytorchTanh(MPSGraph *graph, MPSGraphTensor *input, NSString *name) {
    MPSGraphTensor *x = cast(graph, input, MPSDataTypeFloat32, [name stringByAppendingString:@".input32"]);
    MPSGraphTensor *x2 = [graph squareWithTensor:x name:[name stringByAppendingString:@".x2"]];
    MPSGraphTensor *x3 = [graph multiplicationWithPrimaryTensor:x
                                               secondaryTensor:x2
                                                          name:[name stringByAppendingString:@".x3"]];
    MPSGraphTensor *cubicScale = scalar(graph, 0.044715, MPSDataTypeFloat32);
    MPSGraphTensor *cubic = [graph multiplicationWithPrimaryTensor:x3
                                                   secondaryTensor:cubicScale
                                                              name:[name stringByAppendingString:@".cubic"]];
    MPSGraphTensor *sum = [graph additionWithPrimaryTensor:x
                                          secondaryTensor:cubic
                                                     name:[name stringByAppendingString:@".sum"]];
    MPSGraphTensor *coefficient = scalar(graph, 0.7978845608028654, MPSDataTypeFloat32);
    MPSGraphTensor *argument = [graph multiplicationWithPrimaryTensor:sum
                                                      secondaryTensor:coefficient
                                                                 name:[name stringByAppendingString:@".argument"]];
    MPSGraphTensor *activated = [graph tanhWithTensor:argument name:[name stringByAppendingString:@".tanh"]];
    MPSGraphTensor *one = scalar(graph, 1.0, MPSDataTypeFloat32);
    activated = [graph additionWithPrimaryTensor:activated
                                secondaryTensor:one
                                           name:[name stringByAppendingString:@".plus_one"]];
    MPSGraphTensor *half = scalar(graph, 0.5, MPSDataTypeFloat32);
    activated = [graph multiplicationWithPrimaryTensor:activated
                                      secondaryTensor:half
                                                 name:[name stringByAppendingString:@".half"]];
    activated = [graph multiplicationWithPrimaryTensor:x
                                       secondaryTensor:activated
                                                  name:[name stringByAppendingString:@".output32"]];
    return cast(graph, activated, MPSDataTypeBFloat16, name);
}

std::vector<uint16_t> makeRope(
    size_t sequence,
    size_t dimension,
    AttentionKind kind,
    bool sine,
    size_t positionOffset = 0) {
    std::vector<uint16_t> result(sequence * dimension);
    const size_t half = dimension / 2;
    const size_t activeAngles = kind == AttentionKind::Full
                                    ? static_cast<size_t>(Gemma4Config::kFullRotaryFraction * dimension / 2)
                                    : half;
    const float theta = kind == AttentionKind::Full
                            ? Gemma4Config::kFullRopeTheta
                            : Gemma4Config::kSlidingRopeTheta;
    for (size_t position = 0; position < sequence; ++position) {
        for (size_t index = 0; index < half; ++index) {
            const float exponent = static_cast<float>(2 * index) / static_cast<float>(dimension);
            const float inverseFrequency = index < activeAngles ? 1.0F / std::pow(theta, exponent) : 0.0F;
            const float angle = static_cast<float>(position + positionOffset) * inverseFrequency;
            const float value = sine ? std::sin(angle) : std::cos(angle);
            const uint16_t bits = toBfloat16(value);
            result[position * dimension + index] = bits;
            result[position * dimension + index + half] = bits;
        }
    }
    return result;
}

std::vector<uint16_t> makeAttentionMask(size_t sequence, AttentionKind kind) {
    std::vector<uint16_t> result(sequence * sequence, toBfloat16(-std::numeric_limits<float>::infinity()));
    for (size_t query = 0; query < sequence; ++query) {
        const size_t start = kind == AttentionKind::Sliding && query + 1 > Gemma4Config::kSlidingWindow
                                 ? query + 1 - Gemma4Config::kSlidingWindow
                                 : 0;
        std::fill(
            result.begin() + static_cast<std::ptrdiff_t>(query * sequence + start),
            result.begin() + static_cast<std::ptrdiff_t>(query * sequence + query + 1),
            uint16_t{0});
    }
    return result;
}

std::span<const int64_t> int64Values(const package::TensorView &view) {
    if (view.descriptor->dtype != "I64" || view.bytes.size() % sizeof(int64_t) != 0) {
        throw std::invalid_argument("expected an aligned I64 oracle tensor");
    }
    return {reinterpret_cast<const int64_t *>(view.bytes.data()), view.bytes.size() / sizeof(int64_t)};
}

std::vector<uint16_t> makeLayerTokenEmbedding(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    size_t layer,
    bool transition) {
    const std::string idsName = transition ? "transition.input_ids" : "processor.input_ids";
    const size_t sequence = transition ? 1 : kSequence;
    const package::TensorView idsView = oracle.tensor(idsName);
    requireDescriptor(idsView, idsName, "I64", {1, sequence});
    const auto ids = int64Values(idsView);
    std::span<const int64_t> types;
    if (!transition) {
        const package::TensorView typesView = oracle.tensor("processor.mm_token_type_ids");
        requireDescriptor(typesView, "processor.mm_token_type_ids", "I64", {1, sequence});
        types = int64Values(typesView);
    }

    const std::string name = "model.language_model.embed_tokens_per_layer.weight";
    const package::TensorView embedding = model.tensor(name);
    requireDescriptor(
        embedding,
        name,
        "BF16",
        {Gemma4Config::kVocabulary, kPerLayerStride});
    const auto *weights = reinterpret_cast<const uint16_t *>(embedding.bytes.data());
    std::vector<uint16_t> result(sequence * Gemma4Config::kPerLayerWidth);
    const uint16_t scale = toBfloat16(std::sqrt(static_cast<float>(Gemma4Config::kPerLayerWidth)));
    const float scaleValue = fromBfloat16(scale);
    for (size_t token = 0; token < sequence; ++token) {
        const int64_t id = transition || types[token] == 0 ? ids[token] : 0;
        if (id < 0 || id >= static_cast<int64_t>(Gemma4Config::kVocabulary)) {
            throw std::invalid_argument("oracle input id is outside the per-layer vocabulary");
        }
        const size_t source = static_cast<size_t>(id) * kPerLayerStride + layer * Gemma4Config::kPerLayerWidth;
        const size_t destination = token * Gemma4Config::kPerLayerWidth;
        for (size_t column = 0; column < Gemma4Config::kPerLayerWidth; ++column) {
            result[destination + column] = toBfloat16(fromBfloat16(weights[source + column]) * scaleValue);
        }
    }
    return result;
}

LayerParityMetrics compareBfloat16(
    std::span<const uint16_t> actual,
    std::span<const std::byte> expectedBytes) {
    if (expectedBytes.size() != actual.size() * sizeof(uint16_t)) {
        throw std::invalid_argument("layer output and oracle have different byte counts");
    }
    const auto *expected = reinterpret_cast<const uint16_t *>(expectedBytes.data());
    LayerParityMetrics result{.elements = actual.size()};
    double absoluteTotal = 0.0;
    double squaredTotal = 0.0;
    double dot = 0.0;
    double actualSquared = 0.0;
    double expectedSquared = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double actualValue = fromBfloat16(actual[index]);
        const double expectedValue = fromBfloat16(expected[index]);
        if (!std::isfinite(actualValue)) throw std::runtime_error("layer output contains a non-finite value");
        const double difference = actualValue - expectedValue;
        const double absolute = std::abs(difference);
        const double allowedBfloat16Error = std::max(0.75, 3.0 * bfloat16Ulp(expected[index]));
        result.maximumBfloat16ToleranceRatio = std::max(
            result.maximumBfloat16ToleranceRatio,
            absolute / allowedBfloat16Error);
        absoluteTotal += absolute;
        squaredTotal += difference * difference;
        dot += actualValue * expectedValue;
        actualSquared += actualValue * actualValue;
        expectedSquared += expectedValue * expectedValue;
        if (absolute > result.maximumAbsoluteError) {
            result.maximumAbsoluteError = absolute;
            result.maximumErrorIndex = index;
            result.actualAtMaximum = actualValue;
            result.expectedAtMaximum = expectedValue;
        }
    }
    result.meanAbsoluteError = absoluteTotal / static_cast<double>(actual.size());
    result.rootMeanSquaredError = std::sqrt(squaredTotal / static_cast<double>(actual.size()));
    result.cosineSimilarity = dot / std::sqrt(actualSquared * expectedSquared);
    return result;
}

} // namespace

LayerParityReceipt runLayerImpl(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    size_t layer,
    std::span<const uint16_t> hiddenOverride,
    std::span<const uint16_t> sharedKeyOverride,
    std::span<const uint16_t> sharedValueOverride,
    bool transition) {
    @autoreleasepool {
        if (layer >= Gemma4Config::kLayers) throw std::out_of_range("Gemma 4 layer index must be below 42");
        const LayerSpec spec = Gemma4Config::layer(static_cast<uint32_t>(layer));
        const size_t sequence = transition ? 1 : kSequence;
        const size_t positionOffset = transition ? kSequence : 0;
        const size_t headDimension = spec.headDimension;
        const size_t queryWidth = static_cast<size_t>(spec.queryHeads) * headDimension;
        const size_t keyValueWidth = static_cast<size_t>(spec.keyValueHeads) * headDimension;
        const std::string hiddenPrefix = transition ? "decoder.transition_hidden_" : "decoder.hidden_";
        const std::string hiddenInputName = hiddenPrefix + std::to_string(layer);
        const std::string hiddenOutputName = hiddenPrefix + std::to_string(layer + 1);
        const package::TensorView hiddenInput = oracle.tensor(hiddenInputName);
        const package::TensorView hiddenOutput = oracle.tensor(hiddenOutputName);
        requireDescriptor(hiddenInput, hiddenInputName, "BF16", {1, sequence, Gemma4Config::kHidden});
        requireDescriptor(hiddenOutput, hiddenOutputName, "BF16", {1, sequence, Gemma4Config::kHidden});
        const size_t hiddenElements = sequence * Gemma4Config::kHidden;
        if (!hiddenOverride.empty() && hiddenOverride.size() != hiddenElements) {
            throw std::invalid_argument("layer hidden override has the wrong shape");
        }
        if (sharedKeyOverride.empty() != sharedValueOverride.empty()) {
            throw std::invalid_argument("shared K/V overrides must be supplied together");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal is unavailable for the layer reference graph");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("unable to create the layer reference command queue");

        MPSGraph *graph = [[MPSGraph alloc] init];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [NSMutableDictionary dictionary];
        MPSGraphTensor *hidden = hiddenOverride.empty()
                                     ? feedBfloat16(
                                           graph,
                                           device,
                                           feeds,
                                           hiddenInput.bytes,
                                           shape({sequence, Gemma4Config::kHidden}),
                                           @"hidden_input")
                                     : feedBfloat16(
                                           graph,
                                           device,
                                           feeds,
                                           hiddenOverride,
                                           shape({sequence, Gemma4Config::kHidden}),
                                           @"hidden_override");
        MPSGraphTensor *initialHidden = hidden;
        MPSGraphTensor *perLayerModelInput = initialHidden;
        if (layer != 0) {
            const std::string modelInputName = hiddenPrefix + "0";
            const package::TensorView modelInput = oracle.tensor(modelInputName);
            requireDescriptor(
                modelInput,
                modelInputName,
                "BF16",
                {1, sequence, Gemma4Config::kHidden});
            perLayerModelInput = feedBfloat16(
                graph,
                device,
                feeds,
                modelInput.bytes,
                shape({sequence, Gemma4Config::kHidden}),
                @"per_layer_model_input");
        }

        const std::string base = "model.language_model.layers." + std::to_string(layer) + ".";
        MPSGraphTensor *inputNormWeight = feedModelTensor(
            graph, device, feeds, model, base + "input_layernorm.weight", {Gemma4Config::kHidden});
        MPSGraphTensor *normalized = rmsNorm(
            graph,
            hidden,
            inputNormWeight,
            {sequence, 1},
            1,
            @"input_norm");
        MPSGraphTensor *inputNormOutput = normalized;

        MPSGraphTensor *qWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.q_proj.weight",
            {queryWidth, Gemma4Config::kHidden});
        MPSGraphTensor *kWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.k_proj.weight",
            {keyValueWidth, Gemma4Config::kHidden});
        MPSGraphTensor *vWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.v_proj.weight",
            {keyValueWidth, Gemma4Config::kHidden});
        MPSGraphTensor *query = dense(graph, normalized, qWeight, @"q_proj");
        MPSGraphTensor *key = dense(graph, normalized, kWeight, @"k_proj");
        MPSGraphTensor *value = dense(graph, normalized, vWeight, @"v_proj");
        MPSGraphTensor *queryProjection = query;
        MPSGraphTensor *keyProjection = key;
        MPSGraphTensor *valueProjection = value;
        query = [graph reshapeTensor:query
                           withShape:shape({sequence, Gemma4Config::kQueryHeads, headDimension})
                                name:@"q_heads"];
        key = [graph reshapeTensor:key
                         withShape:shape({sequence, Gemma4Config::kKeyValueHeads, headDimension})
                              name:@"k_heads"];
        value = [graph reshapeTensor:value
                           withShape:shape({sequence, Gemma4Config::kKeyValueHeads, headDimension})
                                name:@"v_heads"];

        MPSGraphTensor *qNormWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.q_norm.weight",
            {headDimension});
        MPSGraphTensor *kNormWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.k_norm.weight",
            {headDimension});
        query = rmsNorm(
            graph,
            query,
            qNormWeight,
            {sequence, Gemma4Config::kQueryHeads, 1},
            2,
            @"q_norm");
        key = rmsNorm(
            graph,
            key,
            kNormWeight,
            {sequence, Gemma4Config::kKeyValueHeads, 1},
            2,
            @"k_norm");
        value = rmsNorm(
            graph,
            value,
            nil,
            {sequence, Gemma4Config::kKeyValueHeads, 1},
            2,
            @"v_norm");
        MPSGraphTensor *queryNormOutput = query;
        MPSGraphTensor *keyNormOutput = key;
        MPSGraphTensor *valueNormOutput = value;

        const std::vector<uint16_t> ropeCos = makeRope(
            sequence, headDimension, spec.attention, false, positionOffset);
        const std::vector<uint16_t> ropeSin = makeRope(
            sequence, headDimension, spec.attention, true, positionOffset);
        MPSGraphTensor *cosine = feedBfloat16(
            graph, device, feeds, ropeCos, shape({sequence, 1, headDimension}), @"rope_cos");
        MPSGraphTensor *sine = feedBfloat16(
            graph, device, feeds, ropeSin, shape({sequence, 1, headDimension}), @"rope_sin");
        auto applyRope = [&](MPSGraphTensor *input, NSString *name) {
            MPSGraphTensor *first = [graph sliceTensor:input
                                            dimension:2
                                                start:0
                                               length:headDimension / 2
                                                 name:[name stringByAppendingString:@".first"]];
            MPSGraphTensor *second = [graph sliceTensor:input
                                             dimension:2
                                                 start:headDimension / 2
                                                length:headDimension / 2
                                                  name:[name stringByAppendingString:@".second"]];
            second = [graph negativeWithTensor:second name:[name stringByAppendingString:@".negative_second"]];
            MPSGraphTensor *rotated = [graph concatTensor:second
                                              withTensor:first
                                               dimension:2
                                                    name:[name stringByAppendingString:@".rotated"]];
            MPSGraphTensor *primary = [graph multiplicationWithPrimaryTensor:input
                                                             secondaryTensor:cosine
                                                                        name:[name stringByAppendingString:@".cos"]];
            MPSGraphTensor *secondary = [graph multiplicationWithPrimaryTensor:rotated
                                                               secondaryTensor:sine
                                                                          name:[name stringByAppendingString:@".sin"]];
            return [graph additionWithPrimaryTensor:primary secondaryTensor:secondary name:name];
        };
        query = applyRope(query, @"q_rope");
        key = applyRope(key, @"k_rope");

        query = [graph transposeTensor:query permutation:@[@1, @0, @2] name:@"q_transpose"];
        key = [graph transposeTensor:key permutation:@[@1, @0, @2] name:@"k_transpose"];
        value = [graph transposeTensor:value permutation:@[@1, @0, @2] name:@"v_transpose"];
        query = [graph reshapeTensor:query
                           withShape:shape({1, Gemma4Config::kQueryHeads, sequence, headDimension})
                                name:@"q_batched"];
        key = [graph reshapeTensor:key
                         withShape:shape({1, Gemma4Config::kKeyValueHeads, sequence, headDimension})
                              name:@"k_batched"];
        value = [graph reshapeTensor:value
                           withShape:shape({1, Gemma4Config::kKeyValueHeads, sequence, headDimension})
                                name:@"v_batched"];
        size_t attentionLength = sequence;
        if (spec.sharesKeyValue) {
            if (!sharedKeyOverride.empty()) {
                const size_t rowElements = Gemma4Config::kKeyValueHeads * headDimension;
                if (sharedKeyOverride.size() != sharedValueOverride.size() ||
                    sharedKeyOverride.size() % rowElements != 0) {
                    throw std::invalid_argument("shared K/V override has the wrong shape");
                }
                attentionLength = sharedKeyOverride.size() / rowElements;
                key = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedKeyOverride,
                    shape({1, Gemma4Config::kKeyValueHeads, attentionLength, headDimension}),
                    @"shared_key_override");
                value = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedValueOverride,
                    shape({1, Gemma4Config::kKeyValueHeads, attentionLength, headDimension}),
                    @"shared_value_override");
            } else if (!transition) {
                const std::string owner = std::to_string(*spec.sharedOwner);
                const std::string keyName = "cache.prefill.shared_" + owner + ".key";
                const std::string valueName = "cache.prefill.shared_" + owner + ".value";
                const package::TensorView sharedKey = oracle.tensor(keyName);
                const package::TensorView sharedValue = oracle.tensor(valueName);
                requireDescriptor(
                    sharedKey,
                    keyName,
                    "BF16",
                    {1, Gemma4Config::kKeyValueHeads, sequence, headDimension});
                requireDescriptor(
                    sharedValue,
                    valueName,
                    "BF16",
                    {1, Gemma4Config::kKeyValueHeads, sequence, headDimension});
                key = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedKey.bytes,
                    shape({1, Gemma4Config::kKeyValueHeads, sequence, headDimension}),
                    @"shared_key");
                value = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedValue.bytes,
                    shape({1, Gemma4Config::kKeyValueHeads, sequence, headDimension}),
                    @"shared_value");
            } else {
                const std::string owner = std::to_string(*spec.sharedOwner);
                const std::string keyName = "cache.transition.layer_" + owner + ".key";
                const std::string valueName = "cache.transition.layer_" + owner + ".value";
                const package::TensorView sharedKey = oracle.tensor(keyName);
                const package::TensorView sharedValue = oracle.tensor(valueName);
                if (sharedKey.descriptor->dtype != "BF16" || sharedValue.descriptor->dtype != "BF16" ||
                    sharedKey.descriptor->shape.size() != 4 ||
                    sharedKey.descriptor->shape != sharedValue.descriptor->shape ||
                    sharedKey.descriptor->shape[0] != 1 ||
                    sharedKey.descriptor->shape[1] != Gemma4Config::kKeyValueHeads ||
                    sharedKey.descriptor->shape[3] != headDimension) {
                    throw std::invalid_argument("transition shared K/V has the wrong shape");
                }
                attentionLength = static_cast<size_t>(sharedKey.descriptor->shape[2]);
                key = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedKey.bytes,
                    shape({1, Gemma4Config::kKeyValueHeads, attentionLength, headDimension}),
                    @"shared_transition_key");
                value = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    sharedValue.bytes,
                    shape({1, Gemma4Config::kKeyValueHeads, attentionLength, headDimension}),
                    @"shared_transition_value");
            }
        } else if (transition) {
            const std::string cachePrefix = "cache.prefill.layer_" + std::to_string(layer);
            const package::TensorView pastKey = oracle.tensor(cachePrefix + ".key");
            const package::TensorView pastValue = oracle.tensor(cachePrefix + ".value");
            if (pastKey.descriptor->dtype != "BF16" || pastValue.descriptor->dtype != "BF16" ||
                pastKey.descriptor->shape.size() != 4 || pastKey.descriptor->shape != pastValue.descriptor->shape ||
                pastKey.descriptor->shape[0] != 1 ||
                pastKey.descriptor->shape[1] != Gemma4Config::kKeyValueHeads ||
                pastKey.descriptor->shape[3] != headDimension) {
                throw std::invalid_argument("prefill cache tensor has the wrong transition shape");
            }
            const size_t pastLength = static_cast<size_t>(pastKey.descriptor->shape[2]);
            MPSGraphTensor *pastKeyTensor = feedBfloat16(
                graph,
                device,
                feeds,
                pastKey.bytes,
                shape({1, Gemma4Config::kKeyValueHeads, pastLength, headDimension}),
                @"past_key");
            MPSGraphTensor *pastValueTensor = feedBfloat16(
                graph,
                device,
                feeds,
                pastValue.bytes,
                shape({1, Gemma4Config::kKeyValueHeads, pastLength, headDimension}),
                @"past_value");
            key = [graph concatTensor:pastKeyTensor withTensor:key dimension:2 name:@"updated_key"];
            value = [graph concatTensor:pastValueTensor withTensor:value dimension:2 name:@"updated_value"];
            attentionLength = pastLength + sequence;
        }
        MPSGraphTensor *ownedKey = layer == 22 || layer == 23 ? key : nil;
        MPSGraphTensor *ownedValue = layer == 22 || layer == 23 ? value : nil;
        key = [graph reshapeTensor:key
                         withShape:shape({1, Gemma4Config::kKeyValueHeads, 1, attentionLength, headDimension})
                              name:@"k_grouped"];
        value = [graph reshapeTensor:value
                           withShape:shape({1, Gemma4Config::kKeyValueHeads, 1, attentionLength, headDimension})
                                name:@"v_grouped"];
        constexpr size_t groups = Gemma4Config::kQueryHeads / Gemma4Config::kKeyValueHeads;
        key = [graph tileTensor:key withMultiplier:shape({1, 1, groups, 1, 1}) name:@"k_tiled"];
        value = [graph tileTensor:value withMultiplier:shape({1, 1, groups, 1, 1}) name:@"v_tiled"];
        key = [graph reshapeTensor:key
                         withShape:shape({1, Gemma4Config::kQueryHeads, attentionLength, headDimension})
                              name:@"k_repeated"];
        value = [graph reshapeTensor:value
                           withShape:shape({1, Gemma4Config::kQueryHeads, attentionLength, headDimension})
                                name:@"v_repeated"];

        const std::vector<uint16_t> maskValues = transition
                                                     ? std::vector<uint16_t>(
                                                           sequence * attentionLength,
                                                           uint16_t{0})
                                                     : makeAttentionMask(sequence, spec.attention);
        MPSGraphTensor *mask = feedBfloat16(
            graph, device, feeds, maskValues, shape({1, 1, sequence, attentionLength}), @"attention_mask");
        MPSGraphTensor *attention = [graph scaledDotProductAttentionWithQueryTensor:query
                                                                          keyTensor:key
                                                                        valueTensor:value
                                                                         maskTensor:mask
                                                                              scale:1.0F
                                                                               name:@"attention"];
        attention = [graph transposeTensor:attention permutation:@[@0, @2, @1, @3] name:@"attention_transpose"];
        attention = [graph reshapeTensor:attention
                               withShape:shape({sequence, queryWidth})
                                    name:@"attention_flat"];
        MPSGraphTensor *oWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "self_attn.o_proj.weight",
            {Gemma4Config::kHidden, queryWidth});
        MPSGraphTensor *attentionOutput = dense(graph, attention, oWeight, @"o_proj");
        MPSGraphTensor *oProjectionOutput = attentionOutput;
        MPSGraphTensor *postAttentionWeight = feedModelTensor(
            graph, device, feeds, model, base + "post_attention_layernorm.weight", {Gemma4Config::kHidden});
        attentionOutput = rmsNorm(
            graph,
            attentionOutput,
            postAttentionWeight,
            {sequence, 1},
            1,
            @"post_attention_norm");
        MPSGraphTensor *postAttentionNormOutput = attentionOutput;
        hidden = [graph additionWithPrimaryTensor:hidden secondaryTensor:attentionOutput name:@"attention_residual"];

        MPSGraphTensor *preFeedforwardWeight = feedModelTensor(
            graph, device, feeds, model, base + "pre_feedforward_layernorm.weight", {Gemma4Config::kHidden});
        normalized = rmsNorm(
            graph,
            hidden,
            preFeedforwardWeight,
            {sequence, 1},
            1,
            @"pre_feedforward_norm");
        MPSGraphTensor *preFeedforwardNormOutput = normalized;
        MPSGraphTensor *gateWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "mlp.gate_proj.weight",
            {Gemma4Config::kIntermediate, Gemma4Config::kHidden});
        MPSGraphTensor *upWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "mlp.up_proj.weight",
            {Gemma4Config::kIntermediate, Gemma4Config::kHidden});
        MPSGraphTensor *downWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "mlp.down_proj.weight",
            {Gemma4Config::kHidden, Gemma4Config::kIntermediate});
        MPSGraphTensor *gate = dense(graph, normalized, gateWeight, @"mlp_gate");
        gate = geluPytorchTanh(graph, gate, @"mlp_gelu");
        MPSGraphTensor *up = dense(graph, normalized, upWeight, @"mlp_up");
        MPSGraphTensor *mlp = [graph multiplicationWithPrimaryTensor:gate secondaryTensor:up name:@"mlp_product"];
        mlp = dense(graph, mlp, downWeight, @"mlp_down");
        MPSGraphTensor *mlpOutput = mlp;
        MPSGraphTensor *postFeedforwardWeight = feedModelTensor(
            graph, device, feeds, model, base + "post_feedforward_layernorm.weight", {Gemma4Config::kHidden});
        mlp = rmsNorm(
            graph,
            mlp,
            postFeedforwardWeight,
            {sequence, 1},
            1,
            @"post_feedforward_norm");
        MPSGraphTensor *postFeedforwardNormOutput = mlp;
        hidden = [graph additionWithPrimaryTensor:hidden secondaryTensor:mlp name:@"feedforward_residual"];

        const std::vector<uint16_t> tokenEmbedding = makeLayerTokenEmbedding(model, oracle, layer, transition);
        MPSGraphTensor *perLayerToken = feedBfloat16(
            graph,
            device,
            feeds,
            tokenEmbedding,
            shape({sequence, Gemma4Config::kPerLayerWidth}),
            @"per_layer_token");
        const std::string projectionName = "model.language_model.per_layer_model_projection.weight";
        const package::TensorView projectionView = model.tensor(projectionName);
        requireDescriptor(
            projectionView,
            projectionName,
            "BF16",
            {kPerLayerStride, Gemma4Config::kHidden});
        const size_t layerProjectionBytes = Gemma4Config::kPerLayerWidth * Gemma4Config::kHidden * sizeof(uint16_t);
        const size_t layerProjectionOffset = layer * layerProjectionBytes;
        MPSGraphTensor *projectionWeight = feedBfloat16(
            graph,
            device,
            feeds,
            projectionView.bytes.subspan(layerProjectionOffset, layerProjectionBytes),
            shape({Gemma4Config::kPerLayerWidth, Gemma4Config::kHidden}),
            @"per_layer_model_projection.weight.slice");
        // Transformers projects the original multimodal model input once for
        // all 42 layers; it does not re-project each layer's evolving hidden.
        MPSGraphTensor *perLayerInput = dense(
            graph, perLayerModelInput, projectionWeight, @"per_layer_model_projection");
        MPSGraphTensor *projectionScale = scalar(
            graph,
            1.0 / std::sqrt(static_cast<double>(Gemma4Config::kHidden)),
            MPSDataTypeBFloat16);
        perLayerInput = [graph multiplicationWithPrimaryTensor:perLayerInput
                                               secondaryTensor:projectionScale
                                                          name:@"per_layer_model_projection_scaled"];
        MPSGraphTensor *projectionNormWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            "model.language_model.per_layer_projection_norm.weight",
            {Gemma4Config::kPerLayerWidth});
        perLayerInput = rmsNorm(
            graph,
            perLayerInput,
            projectionNormWeight,
            {sequence, 1},
            1,
            @"per_layer_projection_norm");
        MPSGraphTensor *perLayerProjectedOutput = perLayerInput;
        perLayerInput = [graph additionWithPrimaryTensor:perLayerInput
                                         secondaryTensor:perLayerToken
                                                    name:@"per_layer_input_sum"];
        MPSGraphTensor *inputScale = scalar(graph, std::sqrt(0.5), MPSDataTypeBFloat16);
        perLayerInput = [graph multiplicationWithPrimaryTensor:perLayerInput
                                               secondaryTensor:inputScale
                                                          name:@"per_layer_input"];
        MPSGraphTensor *perLayerInputOutput = perLayerInput;

        MPSGraphTensor *perLayerGateWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "per_layer_input_gate.weight",
            {Gemma4Config::kPerLayerWidth, Gemma4Config::kHidden});
        MPSGraphTensor *perLayerGate = dense(graph, hidden, perLayerGateWeight, @"per_layer_gate");
        MPSGraphTensor *perLayerGateOutput = perLayerGate;
        perLayerGate = geluPytorchTanh(graph, perLayerGate, @"per_layer_gelu");
        perLayerGate = [graph multiplicationWithPrimaryTensor:perLayerGate
                                              secondaryTensor:perLayerInput
                                                         name:@"per_layer_product"];
        MPSGraphTensor *perLayerProjectionWeight = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            base + "per_layer_projection.weight",
            {Gemma4Config::kHidden, Gemma4Config::kPerLayerWidth});
        MPSGraphTensor *perLayerOutput = dense(
            graph, perLayerGate, perLayerProjectionWeight, @"per_layer_projection");
        MPSGraphTensor *perLayerProjectionOutput = perLayerOutput;
        MPSGraphTensor *postPerLayerWeight = feedModelTensor(
            graph, device, feeds, model, base + "post_per_layer_input_norm.weight", {Gemma4Config::kHidden});
        perLayerOutput = rmsNorm(
            graph,
            perLayerOutput,
            postPerLayerWeight,
            {sequence, 1},
            1,
            @"post_per_layer_norm");
        MPSGraphTensor *postPerLayerNormOutput = perLayerOutput;
        hidden = [graph additionWithPrimaryTensor:hidden secondaryTensor:perLayerOutput name:@"per_layer_residual"];
        MPSGraphTensor *layerScalar = feedModelTensor(
            graph, device, feeds, model, base + "layer_scalar", {1});
        hidden = [graph multiplicationWithPrimaryTensor:hidden secondaryTensor:layerScalar name:@"layer_output"];
        MPSGraphTensor *rawLayerOutput = hidden;
        MPSGraphTensor *oracleRawFinalNorm = nil;
        if (layer + 1 == Gemma4Config::kLayers) {
            MPSGraphTensor *finalNormWeight = feedModelTensor(
                graph,
                device,
                feeds,
                model,
                "model.language_model.norm.weight",
                {Gemma4Config::kHidden});
            hidden = rmsNorm(
                graph,
                hidden,
                finalNormWeight,
                {sequence, 1},
                1,
                @"final_norm");
            const std::string rawOutputName = "layer" + std::to_string(layer) + ".raw_output";
            if (oracle.index().entries().contains(rawOutputName)) {
                const package::TensorView oracleRawOutput = oracle.tensor(rawOutputName);
                requireDescriptor(
                    oracleRawOutput,
                    rawOutputName,
                    "BF16",
                    {1, sequence, Gemma4Config::kHidden});
                MPSGraphTensor *oracleRawInput = feedBfloat16(
                    graph,
                    device,
                    feeds,
                    oracleRawOutput.bytes,
                    shape({sequence, Gemma4Config::kHidden}),
                    @"oracle_raw_final_norm_input");
                oracleRawFinalNorm = rmsNorm(
                    graph,
                    oracleRawInput,
                    finalNormWeight,
                    {sequence, 1},
                    1,
                    @"oracle_raw_final_norm");
            }
        }

        struct Target {
            std::string label;
            std::string expected;
            MPSGraphTensor *tensor;
        };
        std::vector<Target> targets;
        const std::string stagePrefix = "layer" + std::to_string(layer) + ".";
        if (oracle.index().entries().contains(stagePrefix + "input_norm")) {
            const std::vector<std::pair<std::string, MPSGraphTensor *>> candidates{
                {stagePrefix + "input_norm", inputNormOutput},
                {stagePrefix + "q_proj", queryProjection},
                {stagePrefix + "q_norm", queryNormOutput},
                {stagePrefix + "k_proj", keyProjection},
                {stagePrefix + "k_norm", keyNormOutput},
                {stagePrefix + "v_proj", valueProjection},
                {stagePrefix + "v_norm", valueNormOutput},
                {stagePrefix + "o_proj", oProjectionOutput},
                {stagePrefix + "post_attention_norm", postAttentionNormOutput},
                {stagePrefix + "pre_feedforward_norm", preFeedforwardNormOutput},
                {stagePrefix + "mlp", mlpOutput},
                {stagePrefix + "post_feedforward_norm", postFeedforwardNormOutput},
                {stagePrefix + "per_layer_token", perLayerToken},
                {stagePrefix + "per_layer_projected", perLayerProjectedOutput},
                {stagePrefix + "per_layer_input", perLayerInputOutput},
                {stagePrefix + "per_layer_gate", perLayerGateOutput},
                {stagePrefix + "per_layer_projection", perLayerProjectionOutput},
                {stagePrefix + "post_per_layer_norm", postPerLayerNormOutput},
            };
            for (const auto &[name, tensor] : candidates) {
                if (oracle.index().entries().contains(name)) targets.push_back({name, name, tensor});
            }
            if (oracle.index().entries().contains(stagePrefix + "raw_output")) {
                targets.push_back({stagePrefix + "raw_output", stagePrefix + "raw_output", rawLayerOutput});
            }
            targets.push_back({stagePrefix + "output", stagePrefix + "output", hidden});
            if (oracleRawFinalNorm) {
                targets.push_back({
                    stagePrefix + "oracle_raw_final_norm",
                    stagePrefix + "output",
                    oracleRawFinalNorm,
                });
            }
        } else {
            targets.push_back({hiddenOutputName, hiddenOutputName, hidden});
        }
        NSMutableArray<MPSGraphTensor *> *targetTensors = [NSMutableArray arrayWithCapacity:targets.size()];
        for (const auto &target : targets) {
            [targetTensors addObject:target.tensor];
        }
        if (ownedKey) {
            [targetTensors addObject:ownedKey];
            [targetTensors addObject:ownedValue];
        }
        MPSGraphTensorDataDictionary *results = [graph runWithMTLCommandQueue:queue
                                                                  feeds:feeds
                                                          targetTensors:targetTensors
                                                       targetOperations:nil];
        LayerParityReceipt receipt;
        receipt.stages.reserve(targets.size());
        for (const auto &target : targets) {
            MPSGraphTensorData *output = results[target.tensor];
            if (!output || output.dataType != MPSDataTypeBFloat16) {
                throw std::runtime_error("layer reference graph returned an invalid stage: " + target.label);
            }
            const package::TensorView expected = oracle.tensor(target.expected);
            if (expected.descriptor->dtype != "BF16") {
                throw std::invalid_argument("layer stage oracle is not BF16: " + target.expected);
            }
            std::vector<uint16_t> bits(expected.bytes.size() / sizeof(uint16_t));
            [[output mpsndarray] readBytes:bits.data() strideBytes:nil];
            if (target.label == hiddenOutputName || target.label == stagePrefix + "output") {
                receipt.output = bits;
            }
            receipt.stages.push_back(LayerStageParity{
                .name = target.label,
                .metrics = compareBfloat16(bits, expected.bytes),
            });
        }
        if (ownedKey) {
            const size_t elements = Gemma4Config::kKeyValueHeads * attentionLength * headDimension;
            receipt.ownedKey.resize(elements);
            receipt.ownedValue.resize(elements);
            [[results[ownedKey] mpsndarray] readBytes:receipt.ownedKey.data() strideBytes:nil];
            [[results[ownedValue] mpsndarray] readBytes:receipt.ownedValue.data() strideBytes:nil];
        }
        if (receipt.output.empty()) throw std::runtime_error("layer reference graph did not return its output");
        return receipt;
    }
}

LayerParityReceipt Gemma4ReferenceGraph::runLayer(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    size_t layer,
    std::span<const uint16_t> hiddenOverride,
    std::span<const uint16_t> sharedKeyOverride,
    std::span<const uint16_t> sharedValueOverride) {
    return runLayerImpl(
        model,
        oracle,
        layer,
        hiddenOverride,
        sharedKeyOverride,
        sharedValueOverride,
        false);
}

LayerParityReceipt Gemma4ReferenceGraph::runTransitionLayer(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    size_t layer,
    std::span<const uint16_t> hiddenOverride,
    std::span<const uint16_t> sharedKeyOverride,
    std::span<const uint16_t> sharedValueOverride) {
    return runLayerImpl(
        model,
        oracle,
        layer,
        hiddenOverride,
        sharedKeyOverride,
        sharedValueOverride,
        true);
}

LogitParityReceipt runLogitsImpl(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    std::span<const uint16_t> finalHidden,
    bool transition) {
    @autoreleasepool {
        const size_t sequence = transition ? 1 : kSequence;
        const size_t hiddenElements = sequence * Gemma4Config::kHidden;
        if (finalHidden.size() != hiddenElements) {
            throw std::invalid_argument("final hidden state has the wrong shape");
        }
        const std::string logitsName = transition ? "decoder.transition_logits" : "decoder.last_logits";
        const std::string tokenName = transition
                                          ? "decoder.transition_greedy_next_token"
                                          : "decoder.greedy_next_token";
        const package::TensorView expected = oracle.tensor(logitsName);
        if (expected.descriptor->dtype != "BF16" ||
            expected.bytes.size() != Gemma4Config::kVocabulary * sizeof(uint16_t)) {
            throw std::invalid_argument(logitsName + " has the wrong descriptor");
        }
        const package::TensorView expectedToken = oracle.tensor(tokenName);
        if (expectedToken.descriptor->dtype != "I64" || expectedToken.bytes.size() != sizeof(int64_t)) {
            throw std::invalid_argument(tokenName + " has the wrong descriptor");
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal is unavailable for the logit reference graph");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("unable to create the logit reference command queue");
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSMutableDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = [NSMutableDictionary dictionary];
        const std::span<const uint16_t> lastToken = finalHidden.subspan(
            (sequence - 1) * Gemma4Config::kHidden,
            Gemma4Config::kHidden);
        MPSGraphTensor *hidden = feedBfloat16(
            graph,
            device,
            feeds,
            lastToken,
            shape({1, Gemma4Config::kHidden}),
            @"last_hidden");
        MPSGraphTensor *embedding = feedModelTensor(
            graph,
            device,
            feeds,
            model,
            "model.language_model.embed_tokens.weight",
            {Gemma4Config::kVocabulary, Gemma4Config::kHidden});
        MPSGraphTensor *logits = dense(graph, hidden, embedding, @"lm_head");
        MPSGraphTensor *cap = scalar(graph, Gemma4Config::kFinalLogitCap, MPSDataTypeBFloat16);
        logits = [graph divisionWithPrimaryTensor:logits secondaryTensor:cap name:@"logit_cap_divide"];
        logits = [graph tanhWithTensor:logits name:@"logit_cap_tanh"];
        logits = [graph multiplicationWithPrimaryTensor:logits secondaryTensor:cap name:@"logit_cap_scale"];

        MPSGraphTensorDataDictionary *results = [graph runWithMTLCommandQueue:queue
                                                                  feeds:feeds
                                                          targetTensors:@[logits]
                                                       targetOperations:nil];
        MPSGraphTensorData *output = results[logits];
        if (!output || output.dataType != MPSDataTypeBFloat16) {
            throw std::runtime_error("logit reference graph returned an invalid output");
        }
        std::vector<uint16_t> bits(Gemma4Config::kVocabulary);
        [[output mpsndarray] readBytes:bits.data() strideBytes:nil];
        const auto maximum = std::max_element(bits.begin(), bits.end(), [](uint16_t left, uint16_t right) {
            return fromBfloat16(left) < fromBfloat16(right);
        });
        const auto expectedTokens = int64Values(expectedToken);
        return LogitParityReceipt{
            .metrics = compareBfloat16(bits, expected.bytes),
            .actualGreedyToken = static_cast<int64_t>(std::distance(bits.begin(), maximum)),
            .expectedGreedyToken = expectedTokens.front(),
        };
    }
}

LogitParityReceipt Gemma4ReferenceGraph::runLastTokenLogits(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    std::span<const uint16_t> finalHidden) {
    return runLogitsImpl(model, oracle, finalHidden, false);
}

LogitParityReceipt Gemma4ReferenceGraph::runTransitionLogits(
    const package::TensorStore &model,
    const package::TensorStore &oracle,
    std::span<const uint16_t> finalHidden) {
    return runLogitsImpl(model, oracle, finalHidden, true);
}

} // namespace gemma_runtime::gemma4
