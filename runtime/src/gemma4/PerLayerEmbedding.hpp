#pragma once

#include <span>
#include <vector>

namespace gemma_runtime::gemma4 {

class PerLayerEmbedding {
  public:
    [[nodiscard]] static std::vector<float> combine(
        std::span<const float> projectedMainEmbedding,
        std::span<const float> tokenPerLayerEmbedding,
        std::span<const float> projectionNormScale,
        size_t layers,
        size_t perLayerWidth);

    [[nodiscard]] static std::vector<float> inject(
        std::span<const float> hidden,
        std::span<const float> perLayerInput,
        std::span<const float> gateWeights,
        std::span<const float> projectionWeights,
        std::span<const float> postNormScale);
};

} // namespace gemma_runtime::gemma4
