#include "ops/MpsReference.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <bit>
#include <cstdint>
#include <stdexcept>

namespace gemma_runtime::ops {
namespace {

uint16_t toBfloat16(float value) noexcept {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t leastSignificantBit = (bits >> 16U) & 1U;
    bits += 0x7FFFU + leastSignificantBit;
    return static_cast<uint16_t>(bits >> 16U);
}

float fromBfloat16(uint16_t value) noexcept {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

id<MTLBuffer> makeBuffer(id<MTLDevice> device, std::span<const float> values) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:values.size() * sizeof(uint16_t) options:MTLResourceStorageModeShared];
    if (!buffer) throw std::runtime_error("unable to allocate BF16 graph input");
    auto *destination = static_cast<uint16_t *>(buffer.contents);
    for (size_t index = 0; index < values.size(); ++index) destination[index] = toBfloat16(values[index]);
    return buffer;
}

MPSShape *shape(size_t rows, size_t columns) {
    return @[@(rows), @(columns)];
}

} // namespace

std::vector<float> MpsReference::bf16Matmul(
    std::span<const float> left,
    std::span<const float> right,
    size_t rows,
    size_t inner,
    size_t columns) {
    if (!rows || !inner || !columns || left.size() != rows * inner || right.size() != inner * columns) {
        throw std::invalid_argument("invalid BF16 reference matrix dimensions");
    }
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("Metal is unavailable for BF16 reference graph");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("unable to create BF16 reference command queue");

        MPSGraph *graph = [[MPSGraph alloc] init];
        MPSGraphTensor *leftTensor = [graph placeholderWithShape:shape(rows, inner)
                                                        dataType:MPSDataTypeBFloat16
                                                            name:@"left"];
        MPSGraphTensor *rightTensor = [graph placeholderWithShape:shape(inner, columns)
                                                         dataType:MPSDataTypeBFloat16
                                                             name:@"right"];
        MPSGraphTensor *outputTensor = [graph matrixMultiplicationWithPrimaryTensor:leftTensor
                                                                    secondaryTensor:rightTensor
                                                                               name:@"product"];
        id<MTLBuffer> leftBuffer = makeBuffer(device, left);
        id<MTLBuffer> rightBuffer = makeBuffer(device, right);
        MPSGraphTensorData *leftData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:leftBuffer
                        shape:shape(rows, inner)
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorData *rightData = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:rightBuffer
                        shape:shape(inner, columns)
                     dataType:MPSDataTypeBFloat16];
        MPSGraphTensorDataDictionary *results = [graph
            runWithMTLCommandQueue:queue
                             feeds:@{leftTensor: leftData, rightTensor: rightData}
                     targetTensors:@[outputTensor]
                  targetOperations:nil];
        MPSGraphTensorData *outputData = results[outputTensor];
        if (!outputData || outputData.dataType != MPSDataTypeBFloat16) {
            throw std::runtime_error("BF16 reference graph returned an invalid result");
        }
        std::vector<uint16_t> bits(rows * columns);
        [[outputData mpsndarray] readBytes:bits.data() strideBytes:nil];
        std::vector<float> result(bits.size());
        for (size_t index = 0; index < bits.size(); ++index) result[index] = fromBfloat16(bits[index]);
        return result;
    }
}

} // namespace gemma_runtime::ops
