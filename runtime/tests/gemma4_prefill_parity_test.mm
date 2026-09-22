#include "gemma4/Gemma4Config.hpp"
#include "gemma4/Gemma4ReferenceGraph.hpp"
#include "package/TensorStore.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

void printMetrics(const char *label, const gemma_runtime::gemma4::LayerParityMetrics &metrics) {
    std::cout << std::setprecision(10)
              << label
              << " elements=" << metrics.elements
              << " mae=" << metrics.meanAbsoluteError
              << " rmse=" << metrics.rootMeanSquaredError
              << " max_abs=" << metrics.maximumAbsoluteError
              << " bf16_ratio=" << metrics.maximumBfloat16ToleranceRatio
              << " cosine=" << metrics.cosineSimilarity << '\n';
}

} // namespace

int main(int argc, char **argv) {
    using namespace gemma_runtime::gemma4;
    if (argc != 5) {
        std::cerr << "usage: gemma4_prefill_parity_test MODEL_INDEX MODEL_ROOT ORACLE_INDEX ORACLE_ROOT\n";
        return EXIT_FAILURE;
    }
    const auto model = gemma_runtime::package::TensorStore::openIndexed(argv[1], argv[2]);
    const auto oracle = gemma_runtime::package::TensorStore::openIndexed(argv[3], argv[4]);

    std::vector<uint16_t> hidden;
    std::vector<uint16_t> owner22Key;
    std::vector<uint16_t> owner22Value;
    std::vector<uint16_t> owner23Key;
    std::vector<uint16_t> owner23Value;
    LayerParityMetrics finalMetrics;
    for (size_t layer = 0; layer < Gemma4Config::kLayers; ++layer) {
        const LayerSpec spec = Gemma4Config::layer(static_cast<uint32_t>(layer));
        std::span<const uint16_t> sharedKey;
        std::span<const uint16_t> sharedValue;
        if (spec.sharedOwner == 22) {
            sharedKey = owner22Key;
            sharedValue = owner22Value;
        } else if (spec.sharedOwner == 23) {
            sharedKey = owner23Key;
            sharedValue = owner23Value;
        }
        auto receipt = Gemma4ReferenceGraph::runLayer(
            model,
            oracle,
            layer,
            hidden,
            sharedKey,
            sharedValue);
        const auto &metrics = receipt.stages.back().metrics;
        const std::string label = "decoder.accumulated_hidden_" + std::to_string(layer + 1);
        printMetrics(label.c_str(), metrics);
        if (layer == 22) {
            owner22Key = std::move(receipt.ownedKey);
            owner22Value = std::move(receipt.ownedValue);
        } else if (layer == 23) {
            owner23Key = std::move(receipt.ownedKey);
            owner23Value = std::move(receipt.ownedValue);
        }
        hidden = std::move(receipt.output);
        finalMetrics = metrics;
    }

    const auto logits = Gemma4ReferenceGraph::runLastTokenLogits(model, oracle, hidden);
    printMetrics("decoder.accumulated_last_logits", logits.metrics);
    std::cout << "greedy actual=" << logits.actualGreedyToken
              << " expected=" << logits.expectedGreedyToken << '\n';

    // Declared before the first accumulated candidate run. The final hidden
    // keeps the existing aggregate layer gate; logits are capped to [-30, 30]
    // and must preserve their greedy decision as well as close global shape.
    const bool finalHiddenPasses = finalMetrics.meanAbsoluteError <= 0.03 &&
                                   finalMetrics.rootMeanSquaredError <= 0.08 &&
                                   finalMetrics.cosineSimilarity >= 0.9999;
    const bool logitsPass = logits.metrics.meanAbsoluteError <= 0.05 &&
                            logits.metrics.rootMeanSquaredError <= 0.15 &&
                            logits.metrics.cosineSimilarity >= 0.999 &&
                            logits.actualGreedyToken == logits.expectedGreedyToken;
    if (!finalHiddenPasses || !logitsPass) {
        std::cerr << "FAIL: accumulated prefill parity thresholds were not met\n";
        return EXIT_FAILURE;
    }
    std::cout << "gemma4 accumulated prefill parity: PASS\n";
    return EXIT_SUCCESS;
}
