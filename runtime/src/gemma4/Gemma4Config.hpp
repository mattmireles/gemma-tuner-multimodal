#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace gemma_runtime::gemma4 {

enum class AttentionKind : uint8_t { Sliding, Full };

struct LayerSpec {
    AttentionKind attention = AttentionKind::Sliding;
    uint32_t headDimension = 0;
    uint32_t queryHeads = 0;
    uint32_t keyValueHeads = 0;
    bool sharesKeyValue = false;
    std::optional<uint32_t> sharedOwner;
};

struct Gemma4Config {
    static constexpr uint32_t kLayers = 42;
    static constexpr uint32_t kHidden = 2560;
    static constexpr uint32_t kIntermediate = 10240;
    static constexpr uint32_t kVocabulary = 262144;
    static constexpr uint32_t kQueryHeads = 8;
    static constexpr uint32_t kKeyValueHeads = 2;
    static constexpr uint32_t kLocalHeadDimension = 256;
    static constexpr uint32_t kGlobalHeadDimension = 512;
    static constexpr uint32_t kSlidingWindow = 512;
    static constexpr uint32_t kSharedKeyValueLayers = 18;
    static constexpr uint32_t kFirstSharedLayer = kLayers - kSharedKeyValueLayers;
    static constexpr uint32_t kPerLayerWidth = 256;
    static constexpr uint32_t kVisionSoftTokens = 280;
    static constexpr float kRmsEpsilon = 1.0e-6F;
    static constexpr float kFinalLogitCap = 30.0F;
    static constexpr float kSlidingRopeTheta = 10000.0F;
    static constexpr float kFullRopeTheta = 1000000.0F;
    static constexpr float kFullRotaryFraction = 0.25F;

    [[nodiscard]] static constexpr AttentionKind attentionKind(uint32_t layer) {
        return (layer + 1) % 6 == 0 ? AttentionKind::Full : AttentionKind::Sliding;
    }

    [[nodiscard]] static LayerSpec layer(uint32_t layer);
    [[nodiscard]] static std::string_view attentionName(AttentionKind kind) noexcept;
};

} // namespace gemma_runtime::gemma4
