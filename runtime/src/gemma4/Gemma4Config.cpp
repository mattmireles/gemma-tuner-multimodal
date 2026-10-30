#include "gemma4/Gemma4Config.hpp"

#include <stdexcept>

namespace gemma_runtime::gemma4 {

LayerSpec Gemma4Config::layer(uint32_t index) {
    if (index >= kLayers) throw std::out_of_range("Gemma 4 layer index must be below 42");
    const AttentionKind kind = attentionKind(index);
    const bool shared = index >= kFirstSharedLayer;
    std::optional<uint32_t> owner;
    if (shared) {
        // The last non-shared layer of each type owns the K/V reused by the
        // final 18 layers: sliding layer 22 and full-attention layer 23.
        owner = kind == AttentionKind::Full ? 23U : 22U;
    }
    return LayerSpec{
        .attention = kind,
        .headDimension = kind == AttentionKind::Full ? kGlobalHeadDimension : kLocalHeadDimension,
        .queryHeads = kQueryHeads,
        .keyValueHeads = kKeyValueHeads,
        .sharesKeyValue = shared,
        .sharedOwner = owner,
    };
}

std::string_view Gemma4Config::attentionName(AttentionKind kind) noexcept {
    return kind == AttentionKind::Full ? "full_attention" : "sliding_attention";
}

} // namespace gemma_runtime::gemma4
