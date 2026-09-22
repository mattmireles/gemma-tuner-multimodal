#include "gemma4/PerLayerEmbedding.hpp"

#include "gemma4/Gemma4Config.hpp"
#include "gemma4/Gemma4Decoder.hpp"

#include <cmath>
#include <stdexcept>

namespace gemma_runtime::gemma4 {

std::vector<float> PerLayerEmbedding::combine(
    std::span<const float> projected,
    std::span<const float> tokenEmbedding,
    std::span<const float> normScale,
    size_t layers,
    size_t width) {
    if (!layers || !width || projected.size() != layers * width || tokenEmbedding.size() != projected.size() ||
        normScale.size() != width) {
        throw std::invalid_argument("invalid per-layer embedding dimensions");
    }
    constexpr float inputScale = 0.7071067811865475F;
    const float projectionScale = 1.0F / std::sqrt(static_cast<float>(Gemma4Config::kHidden));
    const float tokenEmbeddingScale = std::sqrt(static_cast<float>(Gemma4Config::kPerLayerWidth));
    std::vector<float> output(projected.size());
    for (size_t layer = 0; layer < layers; ++layer) {
        const auto slice = projected.subspan(layer * width, width);
        std::vector<float> scaledProjection(slice.size());
        for (size_t index = 0; index < slice.size(); ++index) {
            scaledProjection[index] = slice[index] * projectionScale;
        }
        const std::vector<float> normalized = Gemma4Decoder::rmsNorm(scaledProjection, normScale);
        for (size_t index = 0; index < width; ++index) {
            output[layer * width + index] =
                (normalized[index] + tokenEmbedding[layer * width + index] * tokenEmbeddingScale) * inputScale;
        }
    }
    return output;
}

std::vector<float> PerLayerEmbedding::inject(
    std::span<const float> hidden,
    std::span<const float> perLayerInput,
    std::span<const float> gateWeights,
    std::span<const float> projectionWeights,
    std::span<const float> postNormScale) {
    if (hidden.empty() || perLayerInput.empty() || gateWeights.size() != hidden.size() * perLayerInput.size() ||
        projectionWeights.size() != hidden.size() * perLayerInput.size() || postNormScale.size() != hidden.size()) {
        throw std::invalid_argument("invalid per-layer injection dimensions");
    }
    std::vector<float> gated = Gemma4Decoder::dense(hidden, gateWeights, perLayerInput.size());
    for (size_t index = 0; index < gated.size(); ++index) {
        gated[index] = Gemma4Decoder::geluPytorchTanh(gated[index]) * perLayerInput[index];
    }
    const std::vector<float> projected = Gemma4Decoder::dense(gated, projectionWeights, hidden.size());
    const std::vector<float> normalized = Gemma4Decoder::rmsNorm(projected, postNormScale);
    std::vector<float> output(hidden.size());
    for (size_t index = 0; index < output.size(); ++index) output[index] = hidden[index] + normalized[index];
    return output;
}

} // namespace gemma_runtime::gemma4
