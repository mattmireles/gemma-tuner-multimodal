#include "gemma4/Gemma4Decoder.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace gemma_runtime::gemma4 {

std::vector<float> Gemma4Decoder::dense(
    std::span<const float> input,
    std::span<const float> weights,
    size_t outputFeatures) {
    if (input.empty() || !outputFeatures || weights.size() != input.size() * outputFeatures) {
        throw std::invalid_argument("dense weight shape does not match input and output dimensions");
    }
    std::vector<float> output(outputFeatures, 0.0F);
    for (size_t row = 0; row < outputFeatures; ++row) {
        output[row] = std::inner_product(
            input.begin(), input.end(), weights.begin() + static_cast<std::ptrdiff_t>(row * input.size()), 0.0F);
    }
    return output;
}

std::vector<float> Gemma4Decoder::rmsNorm(
    std::span<const float> input,
    std::span<const float> scale,
    float epsilon) {
    if (input.empty() || (!scale.empty() && scale.size() != input.size()) || !(epsilon > 0.0F)) {
        throw std::invalid_argument("invalid RMSNorm dimensions or epsilon");
    }
    double squareSum = 0.0;
    for (const float value : input) squareSum += static_cast<double>(value) * value;
    const float inverse = std::pow(static_cast<float>(squareSum / input.size()) + epsilon, -0.5F);
    std::vector<float> output(input.size());
    for (size_t index = 0; index < input.size(); ++index) {
        output[index] = input[index] * inverse * (scale.empty() ? 1.0F : scale[index]);
    }
    return output;
}

float Gemma4Decoder::geluPytorchTanh(float value) noexcept {
    constexpr float coefficient = 0.7978845608028654F; // sqrt(2/pi)
    return 0.5F * value *
           (1.0F + std::tanh(coefficient * (value + 0.044715F * value * value * value)));
}

std::vector<float> Gemma4Decoder::geluPytorchTanh(std::span<const float> input) {
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](float value) {
        return geluPytorchTanh(value);
    });
    return output;
}

std::vector<float> Gemma4Decoder::applyRope(
    std::span<const float> head,
    uint64_t position,
    AttentionKind kind) {
    if (head.empty() || head.size() % 2 != 0) throw std::invalid_argument("RoPE head width must be positive and even");
    const size_t half = head.size() / 2;
    const float theta = kind == AttentionKind::Full ? Gemma4Config::kFullRopeTheta
                                                     : Gemma4Config::kSlidingRopeTheta;
    const size_t rotatedAngles = kind == AttentionKind::Full
                                     ? static_cast<size_t>(Gemma4Config::kFullRotaryFraction * head.size() / 2)
                                     : half;
    std::vector<float> output(head.size());
    for (size_t index = 0; index < half; ++index) {
        const float inverseFrequency = index < rotatedAngles
                                           ? 1.0F / std::pow(theta, static_cast<float>(2 * index) / head.size())
                                           : 0.0F;
        const float angle = static_cast<float>(position) * inverseFrequency;
        const float cosine = std::cos(angle);
        const float sine = std::sin(angle);
        output[index] = head[index] * cosine - head[index + half] * sine;
        output[index + half] = head[index + half] * cosine + head[index] * sine;
    }
    return output;
}

std::vector<float> Gemma4Decoder::causalAttention(
    std::span<const float> query,
    std::span<const float> keys,
    std::span<const float> values,
    size_t tokens,
    std::optional<float> attentionLogitCap) {
    if (query.empty() || !tokens || keys.size() != tokens * query.size() || values.size() != keys.size()) {
        throw std::invalid_argument("invalid attention dimensions");
    }
    std::vector<float> scores(tokens);
    for (size_t token = 0; token < tokens; ++token) {
        scores[token] = std::inner_product(
            query.begin(), query.end(), keys.begin() + static_cast<std::ptrdiff_t>(token * query.size()), 0.0F);
        if (attentionLogitCap) {
            scores[token] = std::tanh(scores[token] / *attentionLogitCap) * *attentionLogitCap;
        }
    }
    const float maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    for (float &score : scores) {
        score = std::exp(score - maximum);
        denominator += score;
    }
    std::vector<float> output(query.size(), 0.0F);
    for (size_t token = 0; token < tokens; ++token) {
        const float probability = static_cast<float>(scores[token] / denominator);
        for (size_t dimension = 0; dimension < output.size(); ++dimension) {
            output[dimension] += probability * values[token * output.size() + dimension];
        }
    }
    return output;
}

void Gemma4Decoder::softcapLogits(std::span<float> logits, float cap) noexcept {
    for (float &logit : logits) logit = std::tanh(logit / cap) * cap;
}

} // namespace gemma_runtime::gemma4
