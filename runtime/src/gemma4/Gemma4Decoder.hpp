#pragma once

#include "gemma4/Gemma4Config.hpp"

#include <optional>
#include <span>
#include <vector>

namespace gemma_runtime::gemma4 {

class Gemma4Decoder {
  public:
    [[nodiscard]] static std::vector<float> dense(
        std::span<const float> input,
        std::span<const float> rowMajorWeights,
        size_t outputFeatures);

    [[nodiscard]] static std::vector<float> rmsNorm(
        std::span<const float> input,
        std::span<const float> scale = {},
        float epsilon = Gemma4Config::kRmsEpsilon);

    [[nodiscard]] static float geluPytorchTanh(float value) noexcept;
    [[nodiscard]] static std::vector<float> geluPytorchTanh(std::span<const float> input);

    [[nodiscard]] static std::vector<float> applyRope(
        std::span<const float> head,
        uint64_t position,
        AttentionKind kind);

    [[nodiscard]] static std::vector<float> causalAttention(
        std::span<const float> query,
        std::span<const float> keys,
        std::span<const float> values,
        size_t tokens,
        std::optional<float> attentionLogitCap = std::nullopt);

    static void softcapLogits(
        std::span<float> logits,
        float cap = Gemma4Config::kFinalLogitCap) noexcept;
};

} // namespace gemma_runtime::gemma4
