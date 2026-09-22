#include "gemma4/Gemma4Decoder.hpp"
#include "gemma4/Gemma4Weights.hpp"
#include "gemma4/PerLayerEmbedding.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void close(std::span<const float> actual, std::span<const float> expected, float tolerance, const char *message) {
    require(actual.size() == expected.size(), "comparison size mismatch");
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance) {
            std::cerr << "FAIL: " << message << " at " << index << ": " << actual[index]
                      << " != " << expected[index] << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

} // namespace

int main() {
    using namespace gemma_runtime::gemma4;

    const std::vector<float> input{1, -2, 3, -4};
    const std::vector<float> scale{0.5F, 1.0F, 1.5F, 2.0F};
    const std::vector<float> rmsExpected{
        0.1825741827F, -0.7302967310F, 1.6431677341F, -2.9211869240F};
    close(Gemma4Decoder::rmsNorm(input, scale), rmsExpected, 2.0e-6F, "RMSNorm golden");

    const std::vector<float> geluInput{-2, -1, 0, 1, 2};
    const std::vector<float> geluExpected{
        -0.0454022884F, -0.158807993F, 0.0F, 0.841192007F, 1.954597712F};
    close(Gemma4Decoder::geluPytorchTanh(geluInput), geluExpected, 2.0e-6F, "GELU golden");

    const std::vector<float> ropeInput{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> slidingExpected{
        -1.695592523F, 0.137551665F, 2.788681507F, 3.975982189F,
        -4.808842659F, 6.323059559F, 7.086836815F, 8.011963844F};
    const std::vector<float> fullExpected{
        -1.695592523F, 2.0F, 3.0F, 4.0F, -4.808842659F, 6.0F, 7.0F, 8.0F};
    close(Gemma4Decoder::applyRope(ropeInput, 3, AttentionKind::Sliding),
          slidingExpected, 2.0e-6F, "sliding RoPE golden");
    close(Gemma4Decoder::applyRope(ropeInput, 3, AttentionKind::Full),
          fullExpected, 2.0e-6F, "full proportional RoPE golden");

    std::vector<float> logits{-100, -30, 0, 30, 100};
    const std::vector<float> logitsExpected{
        -29.923740387F, -22.847826004F, 0.0F, 22.847826004F, 29.923740387F};
    Gemma4Decoder::softcapLogits(logits);
    close(logits, logitsExpected, 3.0e-6F, "logit softcap golden");

    const std::vector<float> query{1, 0};
    const std::vector<float> keys{1, 0, 0, 1};
    const std::vector<float> values{10, 20, 30, 40};
    const float p0 = std::exp(1.0F) / (std::exp(1.0F) + 1.0F);
    const std::vector<float> attentionExpected{
        p0 * 10.0F + (1.0F - p0) * 30.0F,
        p0 * 20.0F + (1.0F - p0) * 40.0F};
    close(Gemma4Decoder::causalAttention(query, keys, values, 2),
          attentionExpected, 2.0e-6F, "attention scaling golden");

    const auto inventory = Gemma4Weights::expectedReferenceInventory();
    require(inventory.size() == 719, "reference language tensor count changed");
    Gemma4Weights::validateReferenceInventory(inventory);
    auto broken = inventory;
    broken["model.language_model.layers.5.self_attn.q_proj.weight"] = {1, 1};
    try {
        Gemma4Weights::validateReferenceInventory(broken);
        require(false, "shape mismatch must fail");
    } catch (const std::invalid_argument &) {
    }

    const std::vector<float> projected{1, 2, 3, 4};
    const std::vector<float> tokenEmbedding{0.5F, -0.5F, 1.0F, -1.0F};
    const std::vector<float> norm{1, 1};
    const auto combined = PerLayerEmbedding::combine(projected, tokenEmbedding, norm, 2, 2);
    const std::vector<float> combinedExpected{
        6.103838444F, -4.762884617F, 11.913647652F, -10.513790131F};
    close(combined, combinedExpected, 2.0e-6F, "per-layer projection and embedding scales");
    const std::vector<float> hidden{1, 2};
    const std::vector<float> perLayer{0.25F};
    const std::vector<float> gate{1, -1};
    const std::vector<float> projection{2, 3};
    const auto injected = PerLayerEmbedding::inject(hidden, perLayer, gate, projection, norm);
    require(injected.size() == hidden.size(), "per-layer injection shape");
    require(std::isfinite(injected[0]) && std::isfinite(injected[1]), "per-layer injection finite");

    std::cout << "gemma4 reference primitives: PASS\n";
    return EXIT_SUCCESS;
}
