#include "gemma4/Gemma4Weights.hpp"

#include "gemma4/Gemma4Config.hpp"
#include "package/TensorIndex.hpp"

#include <stdexcept>

namespace gemma_runtime::gemma4 {
namespace {

void add(Gemma4Weights::Inventory &result, const std::string &name, Gemma4Weights::Shape shape) {
    if (!result.emplace(name, std::move(shape)).second) {
        throw std::logic_error("duplicate Gemma 4 reference tensor contract: " + name);
    }
}

std::string prefix(uint32_t layer) {
    return "model.language_model.layers." + std::to_string(layer) + '.';
}

} // namespace

Gemma4Weights::Inventory Gemma4Weights::expectedReferenceInventory() {
    Inventory result;
    add(result, "model.language_model.embed_tokens.weight", {Gemma4Config::kVocabulary, Gemma4Config::kHidden});
    add(result,
        "model.language_model.embed_tokens_per_layer.weight",
        {Gemma4Config::kVocabulary, Gemma4Config::kLayers * Gemma4Config::kPerLayerWidth});
    add(result, "model.language_model.norm.weight", {Gemma4Config::kHidden});
    add(result,
        "model.language_model.per_layer_model_projection.weight",
        {Gemma4Config::kLayers * Gemma4Config::kPerLayerWidth, Gemma4Config::kHidden});
    add(result, "model.language_model.per_layer_projection_norm.weight", {Gemma4Config::kPerLayerWidth});

    for (uint32_t layer = 0; layer < Gemma4Config::kLayers; ++layer) {
        const LayerSpec spec = Gemma4Config::layer(layer);
        const uint64_t queryWidth = static_cast<uint64_t>(spec.queryHeads) * spec.headDimension;
        const uint64_t keyValueWidth = static_cast<uint64_t>(spec.keyValueHeads) * spec.headDimension;
        const std::string base = prefix(layer);
        add(result, base + "input_layernorm.weight", {Gemma4Config::kHidden});
        add(result, base + "post_attention_layernorm.weight", {Gemma4Config::kHidden});
        add(result, base + "pre_feedforward_layernorm.weight", {Gemma4Config::kHidden});
        add(result, base + "post_feedforward_layernorm.weight", {Gemma4Config::kHidden});
        add(result, base + "post_per_layer_input_norm.weight", {Gemma4Config::kHidden});
        add(result, base + "layer_scalar", {1});
        add(result, base + "self_attn.q_proj.weight", {queryWidth, Gemma4Config::kHidden});
        add(result, base + "self_attn.o_proj.weight", {Gemma4Config::kHidden, queryWidth});
        add(result, base + "self_attn.q_norm.weight", {spec.headDimension});
        // The official BF16 checkpoint retains K/V tensors even for the 18
        // layers whose forward pass reuses owners 22/23. A later W6 conversion
        // may omit these dead tensors, but the reference inventory must bind
        // the exact upstream checkpoint rather than infer an optimized layout.
        add(result, base + "self_attn.k_proj.weight", {keyValueWidth, Gemma4Config::kHidden});
        add(result, base + "self_attn.v_proj.weight", {keyValueWidth, Gemma4Config::kHidden});
        add(result, base + "self_attn.k_norm.weight", {spec.headDimension});
        add(result,
            base + "mlp.gate_proj.weight",
            {Gemma4Config::kIntermediate, Gemma4Config::kHidden});
        add(result,
            base + "mlp.up_proj.weight",
            {Gemma4Config::kIntermediate, Gemma4Config::kHidden});
        add(result,
            base + "mlp.down_proj.weight",
            {Gemma4Config::kHidden, Gemma4Config::kIntermediate});
        add(result,
            base + "per_layer_input_gate.weight",
            {Gemma4Config::kPerLayerWidth, Gemma4Config::kHidden});
        add(result,
            base + "per_layer_projection.weight",
            {Gemma4Config::kHidden, Gemma4Config::kPerLayerWidth});
    }
    return result;
}

void Gemma4Weights::validateReferenceInventory(const Inventory &inventory) {
    const Inventory expected = expectedReferenceInventory();
    for (const auto &[name, shape] : expected) {
        const auto found = inventory.find(name);
        if (found == inventory.end()) throw std::invalid_argument("missing Gemma 4 tensor: " + name);
        if (found->second != shape) throw std::invalid_argument("Gemma 4 tensor shape mismatch: " + name);
    }
}

void Gemma4Weights::validateReferenceIndex(const package::TensorIndex &index) {
    Inventory inventory;
    for (const auto &[name, descriptor] : index.entries()) {
        if (!name.starts_with("model.language_model.")) continue;
        if (descriptor.dtype != "BF16") {
            throw std::invalid_argument("reference Gemma 4 tensor is not BF16: " + name);
        }
        inventory.emplace(name, descriptor.shape);
    }
    validateReferenceInventory(inventory);
}

} // namespace gemma_runtime::gemma4
