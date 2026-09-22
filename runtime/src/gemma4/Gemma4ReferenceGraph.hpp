#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gemma_runtime::package {
class TensorStore;
}

namespace gemma_runtime::gemma4 {

struct LayerParityMetrics {
    size_t elements = 0;
    size_t maximumErrorIndex = 0;
    double meanAbsoluteError = 0.0;
    double rootMeanSquaredError = 0.0;
    double maximumAbsoluteError = 0.0;
    double cosineSimilarity = 0.0;
    double actualAtMaximum = 0.0;
    double expectedAtMaximum = 0.0;
    double maximumBfloat16ToleranceRatio = 0.0;
};

struct LayerStageParity {
    std::string name;
    LayerParityMetrics metrics;
};

struct LayerParityReceipt {
    std::vector<LayerStageParity> stages;
    std::vector<uint16_t> output;
    std::vector<uint16_t> ownedKey;
    std::vector<uint16_t> ownedValue;
};

struct LogitParityReceipt {
    LayerParityMetrics metrics;
    int64_t actualGreedyToken = -1;
    int64_t expectedGreedyToken = -1;
};

class Gemma4ReferenceGraph {
  public:
    // Runs one complete E4B decoder layer in BF16 MPSGraph and compares it
    // with the corresponding capture_reference.py receipt. Optional overrides
    // let the slow reference driver propagate candidate state across layers.
    [[nodiscard]] static LayerParityReceipt runLayer(
        const package::TensorStore &model,
        const package::TensorStore &oracle,
        size_t layer,
        std::span<const uint16_t> hiddenOverride = {},
        std::span<const uint16_t> sharedKeyOverride = {},
        std::span<const uint16_t> sharedValueOverride = {});

    [[nodiscard]] static LayerParityReceipt runTransitionLayer(
        const package::TensorStore &model,
        const package::TensorStore &oracle,
        size_t layer,
        std::span<const uint16_t> hiddenOverride = {},
        std::span<const uint16_t> sharedKeyOverride = {},
        std::span<const uint16_t> sharedValueOverride = {});

    [[nodiscard]] static LogitParityReceipt runLastTokenLogits(
        const package::TensorStore &model,
        const package::TensorStore &oracle,
        std::span<const uint16_t> finalHidden);

    [[nodiscard]] static LogitParityReceipt runTransitionLogits(
        const package::TensorStore &model,
        const package::TensorStore &oracle,
        std::span<const uint16_t> finalHidden);
};

} // namespace gemma_runtime::gemma4
