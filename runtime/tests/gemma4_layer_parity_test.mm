#include "gemma4/Gemma4ReferenceGraph.hpp"
#include "package/TensorStore.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: gemma4_layer_parity_test MODEL_INDEX MODEL_ROOT ORACLE_INDEX ORACLE_ROOT LAYER [transition]\n";
        return EXIT_FAILURE;
    }
    const size_t layer = std::stoul(argv[5]);
    const bool transition = argc == 7 && std::string(argv[6]) == "transition";
    if (argc == 7 && !transition) {
        std::cerr << "FAIL: optional mode must be transition\n";
        return EXIT_FAILURE;
    }
    const auto model = gemma_runtime::package::TensorStore::openIndexed(argv[1], argv[2]);
    const auto oracle = gemma_runtime::package::TensorStore::openIndexed(argv[3], argv[4]);
    const auto receipt = transition
                             ? gemma_runtime::gemma4::Gemma4ReferenceGraph::runTransitionLayer(model, oracle, layer)
                             : gemma_runtime::gemma4::Gemma4ReferenceGraph::runLayer(model, oracle, layer);
    for (const auto &stage : receipt.stages) {
        const auto &metrics = stage.metrics;
        std::cout << std::setprecision(10)
                  << stage.name
                  << " elements=" << metrics.elements
                  << " mae=" << metrics.meanAbsoluteError
                  << " rmse=" << metrics.rootMeanSquaredError
                  << " max_abs=" << metrics.maximumAbsoluteError
                  << " max_index=" << metrics.maximumErrorIndex
                  << " actual=" << metrics.actualAtMaximum
                  << " expected=" << metrics.expectedAtMaximum
                  << " bf16_ratio=" << metrics.maximumBfloat16ToleranceRatio
                  << " cosine=" << metrics.cosineSimilarity << '\n';
    }
    const std::string outputName = "layer" + std::to_string(layer) + ".output";
    const std::string hiddenName = transition
                                       ? "decoder.transition_hidden_" + std::to_string(layer + 1)
                                       : "decoder.hidden_" + std::to_string(layer + 1);
    const auto output = std::find_if(receipt.stages.begin(), receipt.stages.end(), [&](const auto &stage) {
        return stage.name == outputName || stage.name == hiddenName;
    });
    if (output == receipt.stages.end()) {
        std::cerr << "FAIL: complete layer output was not returned\n";
        return EXIT_FAILURE;
    }
    const auto &metrics = output->metrics;

    // Aggregate bounds were declared before the first candidate run. The
    // elementwise bound is three BF16 ULPs with the original 0.75 floor: layer 5 exposed
    // why a fixed 0.75 absolute limit is not precision-aware once activations
    // exceed 64. Layer 23 is the held-out full-attention validation layer.
    const bool passes = metrics.meanAbsoluteError <= 0.03 &&
                        metrics.rootMeanSquaredError <= 0.08 &&
                        metrics.maximumBfloat16ToleranceRatio <= 1.0 &&
                        metrics.cosineSimilarity >= 0.9999;
    if (!passes) {
        std::cerr << "FAIL: complete layer-" << layer << " BF16 parity thresholds were not met\n";
        return EXIT_FAILURE;
    }
    std::cout << "gemma4 complete layer-" << layer << " BF16 parity: PASS\n";
    return EXIT_SUCCESS;
}
