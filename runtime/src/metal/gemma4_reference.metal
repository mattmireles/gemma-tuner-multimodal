#include <metal_stdlib>
using namespace metal;

template <typename T>
inline void rms_norm(
    device const T *input,
    device const T *scale,
    device T *output,
    uint width,
    float epsilon) {
    float square_sum = 0.0f;
    for (uint index = 0; index < width; ++index) {
        const float value = float(input[index]);
        square_sum += value * value;
    }
    const float inverse = pow(square_sum / float(width) + epsilon, -0.5f);
    for (uint index = 0; index < width; ++index) {
        output[index] = T(float(input[index]) * inverse * float(scale[index]));
    }
}

template <typename T>
inline void gelu_tanh(device const T *input, device T *output, uint index) {
    const float value = float(input[index]);
    constexpr float coefficient = 0.7978845608028654f;
    output[index] = T(0.5f * value *
                      (1.0f + tanh(coefficient * (value + 0.044715f * value * value * value))));
}

template <typename T>
inline void softcap(device const T *input, device T *output, float cap, uint index) {
    output[index] = T(tanh(float(input[index]) / cap) * cap);
}

kernel void gemma4_rms_norm_f16(
    device const half *input [[buffer(0)]],
    device const half *scale [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0) rms_norm(input, scale, output, width, epsilon);
}

kernel void gemma4_rms_norm_bf16(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *scale [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &width [[buffer(3)]],
    constant float &epsilon [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index == 0) rms_norm(input, scale, output, width, epsilon);
}

kernel void gemma4_gelu_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < width) gelu_tanh(input, output, index);
}

kernel void gemma4_gelu_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index < width) gelu_tanh(input, output, index);
}

kernel void gemma4_softcap_f16(
    device const half *input [[buffer(0)]],
    device half *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant float &cap [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < width) softcap(input, output, cap, index);
}

kernel void gemma4_softcap_bf16(
    device const bfloat *input [[buffer(0)]],
    device bfloat *output [[buffer(1)]],
    constant uint &width [[buffer(2)]],
    constant float &cap [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < width) softcap(input, output, cap, index);
}
