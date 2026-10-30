#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

#define GEMMA_MATRIX_ELEMENTS(matrix) reinterpret_cast<thread float2 &>((matrix).thread_elements())

inline float load_unaligned_bfloat16(device const uchar *bytes, ulong byte_offset) {
    const uint bits = uint(bytes[byte_offset]) | (uint(bytes[byte_offset + 1]) << 8);
    return as_type<float>(bits << 16);
}

inline float load_affine_w6(
    device const uchar *packed_weights,
    device const uchar *scales,
    device const uchar *biases,
    uint row,
    uint column,
    uint columns,
    ulong weight_offset,
    ulong scale_offset,
    ulong bias_offset) {
    const ulong row_bytes = ulong(columns) * 6 / 8;
    const uint bit = column * 6;
    const uint byte = bit >> 3;
    const uint shift = bit & 7;
    device const uchar *packed = packed_weights + weight_offset + ulong(row) * row_bytes + byte;
    uint window = uint(packed[0]);
    if (shift > 2) window |= uint(packed[1]) << 8;
    const uint quantized = (window >> shift) & 0x3f;
    const ulong group = ulong(row) * (columns / 64) + column / 64;
    const float scale = load_unaligned_bfloat16(scales, scale_offset + group * 2);
    const float bias = load_unaligned_bfloat16(biases, bias_offset + group * 2);
    return float(quantized) * scale + bias;
}

inline ushort2 simd_matrix_coordinate(uint lane) {
    const ushort qid = ushort(lane / 4);
    const ushort row = (qid & 4) + ushort((lane / 2) % 4);
    const ushort column = (qid & 2) * 2 + ushort(lane % 2) * 2;
    return ushort2(column, row);
}

// Metal-3-compatible specialization of the proven affine-QMV layout: two SIMD
// groups per threadgroup, four output rows per SIMD, and eight contiguous K
// values per lane. Six-bit weights decode from exactly six bytes in registers.
kernel void gemma4_w6a16_matvec_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uchar *packed_weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant ulong &weight_offset [[buffer(7)]],
    constant ulong &scale_offset [[buffer(8)]],
    constant ulong &bias_offset [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_simd = 4;
    constexpr uint values_per_lane = 8;
    constexpr uint block = 256;
    const uint first_row = tile * 8 + simd_group * rows_per_simd;
    if (first_row >= rows || columns % block != 0) return;
    const uint row_bytes = columns * 6 / 8;
    const uint groups = columns / 64;
    float sums[rows_per_simd] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < columns; k += block) {
        const uint column = k + lane * values_per_lane;
        const float4 activations0 = float4(
            float(input[column]),
            float(input[column + 1]),
            float(input[column + 2]),
            float(input[column + 3]));
        const float4 activations1 = float4(
            float(input[column + 4]),
            float(input[column + 5]),
            float(input[column + 6]),
            float(input[column + 7]));
        const uint group = k / 64 + lane / 8;
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            const uint row = first_row + row_offset;
            if (row >= rows) continue;
            device const uchar *packed =
                packed_weights + weight_offset + row * row_bytes + k * 6 / 8 + lane * 6;
            const ulong auxiliary = ulong(row * groups + group) * 2;
            const float scale = load_unaligned_bfloat16(scales, scale_offset + auxiliary);
            const float bias = load_unaligned_bfloat16(biases, bias_offset + auxiliary);
            const float4 weights0 = float4(
                float(packed[0] & 0x3f),
                float(((packed[0] >> 6) & 0x03) | ((packed[1] & 0x0f) << 2)),
                float(((packed[1] >> 4) & 0x0f) | ((packed[2] & 0x03) << 4)),
                float((packed[2] >> 2) & 0x3f)) * scale + bias;
            const float4 weights1 = float4(
                float(packed[3] & 0x3f),
                float(((packed[3] >> 6) & 0x03) | ((packed[4] & 0x0f) << 2)),
                float(((packed[4] >> 4) & 0x0f) | ((packed[5] & 0x03) << 4)),
                float((packed[5] >> 2) & 0x3f)) * scale + bias;
            sums[row_offset] += dot(activations0, weights0) + dot(activations1, weights1);
        }
    }
    for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
        const float sum = simd_sum(sums[row_offset]);
        const uint row = first_row + row_offset;
        if (lane == 0 && row < rows) output[row] = bfloat(sum);
    }
}

kernel void gemma4_w6a16_matvec_narrow_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uchar *packed_weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &rows [[buffer(5)]],
    constant uint &columns [[buffer(6)]],
    constant ulong &weight_offset [[buffer(7)]],
    constant ulong &scale_offset [[buffer(8)]],
    constant ulong &bias_offset [[buffer(9)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_simd = 4;
    constexpr uint values_per_lane = 4;
    constexpr uint block = 128;
    const uint first_row = tile * 8 + simd_group * rows_per_simd;
    if (first_row >= rows || columns % block != 0) return;
    const uint row_bytes = columns * 6 / 8;
    const uint groups = columns / 64;
    float sums[rows_per_simd] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < columns; k += block) {
        const uint column = k + lane * values_per_lane;
        const float4 activations = float4(
            float(input[column]),
            float(input[column + 1]),
            float(input[column + 2]),
            float(input[column + 3]));
        const uint group = k / 64 + lane / 16;
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            const uint row = first_row + row_offset;
            if (row >= rows) continue;
            device const uchar *packed =
                packed_weights + weight_offset + row * row_bytes + k * 6 / 8 + lane * 3;
            const ulong auxiliary = ulong(row * groups + group) * 2;
            const float scale = load_unaligned_bfloat16(scales, scale_offset + auxiliary);
            const float bias = load_unaligned_bfloat16(biases, bias_offset + auxiliary);
            const float4 weights = float4(
                float(packed[0] & 0x3f),
                float(((packed[0] >> 6) & 0x03) | ((packed[1] & 0x0f) << 2)),
                float(((packed[1] >> 4) & 0x0f) | ((packed[2] & 0x03) << 4)),
                float((packed[2] >> 2) & 0x3f)) * scale + bias;
            sums[row_offset] += dot(activations, weights);
        }
    }
    for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
        const float sum = simd_sum(sums[row_offset]);
        const uint row = first_row + row_offset;
        if (lane == 0 && row < rows) output[row] = bfloat(sum);
    }
}

kernel void gemma4_bf16_matvec(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &rows [[buffer(3)]],
    constant uint &columns [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint rows_per_simd = 4;
    constexpr uint values_per_lane = 4;
    constexpr uint block = 128;
    const uint first_row = tile * 8 + simd_group * rows_per_simd;
    if (first_row >= rows || columns % block != 0) return;
    float sums[rows_per_simd] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = 0; k < columns; k += block) {
        const uint column = k + lane * values_per_lane;
        const float4 activations = float4(
            float(input[column]),
            float(input[column + 1]),
            float(input[column + 2]),
            float(input[column + 3]));
        for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
            const uint row = first_row + row_offset;
            if (row >= rows) continue;
            device const bfloat *weight = weights + row * columns + column;
            sums[row_offset] += dot(
                activations,
                float4(float(weight[0]), float(weight[1]), float(weight[2]), float(weight[3])));
        }
    }
    for (uint row_offset = 0; row_offset < rows_per_simd; ++row_offset) {
        const float sum = simd_sum(sums[row_offset]);
        const uint row = first_row + row_offset;
        if (lane == 0 && row < rows) output[row] = bfloat(sum);
    }
}

// Quantized token/per-layer embedding lookup. The host selects either the full
// 2,560-wide token row or a 256-wide layer slice through column_offset; no
// dense vocabulary table or intermediate dequantized row is materialized.
kernel void gemma4_w6_embedding_gather_bf16(
    device const uint *token_ids [[buffer(0)]],
    device const uchar *packed_weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &token_count [[buffer(5)]],
    constant uint &vocabulary [[buffer(6)]],
    constant uint &source_columns [[buffer(7)]],
    constant uint &column_offset [[buffer(8)]],
    constant uint &output_columns [[buffer(9)]],
    constant float &output_scale [[buffer(10)]],
    constant ulong &weight_offset [[buffer(11)]],
    constant ulong &scale_offset [[buffer(12)]],
    constant ulong &bias_offset [[buffer(13)]],
    uint position [[thread_position_in_grid]]) {
    const uint output_elements = token_count * output_columns;
    if (position >= output_elements) return;
    const uint token = position / output_columns;
    const uint output_column = position - token * output_columns;
    const uint row = token_ids[token];
    const uint source_column = column_offset + output_column;
    if (row >= vocabulary || source_column >= source_columns) {
        output[position] = bfloat(0.0f);
        return;
    }

    const uint row_bytes = source_columns * 6 / 8;
    const uint bit = source_column * 6;
    const uint byte = bit >> 3;
    const uint shift = bit & 7;
    device const uchar *packed = packed_weights + weight_offset + row * row_bytes + byte;
    uint window = uint(packed[0]);
    if (shift > 2) window |= uint(packed[1]) << 8;
    const uint quantized = (window >> shift) & 0x3f;
    const uint groups = source_columns / 64;
    const uint group = row * groups + source_column / 64;
    const float scale = load_unaligned_bfloat16(scales, scale_offset + ulong(group) * 2);
    const float bias = load_unaligned_bfloat16(biases, bias_offset + ulong(group) * 2);
    const float value = float(quantized) * scale + bias;
    output[position] = bfloat(value * output_scale);
}

// Reference tiled QMM for prefill: each SIMD group computes a 16x16 output
// tile with four 8x8 simdgroup matrix accumulators. Four SIMD groups cover a
// 16x64 threadgroup tile. The first version favors a small, inspectable
// Metal-3 path; family tuning can add shared-memory staging without changing
// the package or operator ABI.
kernel void gemma4_w6a16_matmul_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uchar *packed_weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &m_size [[buffer(5)]],
    constant uint &n_size [[buffer(6)]],
    constant uint &k_size [[buffer(7)]],
    constant ulong &weight_offset [[buffer(8)]],
    constant ulong &scale_offset [[buffer(9)]],
    constant ulong &bias_offset [[buffer(10)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint base_m = tile.y * 16;
    const uint base_n = (tile.x * 4 + simd_group) * 16;
    if (base_m >= m_size || base_n >= n_size) return;
    const ushort2 coordinate = simd_matrix_coordinate(lane);

    simdgroup_matrix<float, 8, 8> c00;
    simdgroup_matrix<float, 8, 8> c01;
    simdgroup_matrix<float, 8, 8> c10;
    simdgroup_matrix<float, 8, 8> c11;
    reinterpret_cast<thread float2 &>(c00.thread_elements()) = float2(0.0f);
    reinterpret_cast<thread float2 &>(c01.thread_elements()) = float2(0.0f);
    reinterpret_cast<thread float2 &>(c10.thread_elements()) = float2(0.0f);
    reinterpret_cast<thread float2 &>(c11.thread_elements()) = float2(0.0f);

    for (uint k = 0; k < k_size; k += 8) {
        simdgroup_matrix<float, 8, 8> a0;
        simdgroup_matrix<float, 8, 8> a1;
        simdgroup_matrix<float, 8, 8> b0;
        simdgroup_matrix<float, 8, 8> b1;
        for (uint element = 0; element < 2; ++element) {
            const uint a_column = k + coordinate.x + element;
            const uint a_row0 = base_m + coordinate.y;
            const uint a_row1 = a_row0 + 8;
            GEMMA_MATRIX_ELEMENTS(a0)[element] =
                a_row0 < m_size && a_column < k_size ? float(input[a_row0 * k_size + a_column]) : 0.0f;
            GEMMA_MATRIX_ELEMENTS(a1)[element] =
                a_row1 < m_size && a_column < k_size ? float(input[a_row1 * k_size + a_column]) : 0.0f;

            const uint weight_column = k + coordinate.y;
            const uint weight_row0 = base_n + coordinate.x + element;
            const uint weight_row1 = weight_row0 + 8;
            GEMMA_MATRIX_ELEMENTS(b0)[element] = weight_row0 < n_size
                ? load_affine_w6(
                      packed_weights,
                      scales,
                      biases,
                      weight_row0,
                      weight_column,
                      k_size,
                      weight_offset,
                      scale_offset,
                      bias_offset)
                : 0.0f;
            GEMMA_MATRIX_ELEMENTS(b1)[element] = weight_row1 < n_size
                ? load_affine_w6(
                      packed_weights,
                      scales,
                      biases,
                      weight_row1,
                      weight_column,
                      k_size,
                      weight_offset,
                      scale_offset,
                      bias_offset)
                : 0.0f;
        }
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }

    for (uint element = 0; element < 2; ++element) {
        const uint row0 = base_m + coordinate.y;
        const uint row1 = row0 + 8;
        const uint column0 = base_n + coordinate.x + element;
        const uint column1 = column0 + 8;
        if (row0 < m_size && column0 < n_size) output[row0 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c00)[element]);
        if (row0 < m_size && column1 < n_size) output[row0 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c01)[element]);
        if (row1 < m_size && column0 < n_size) output[row1 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c10)[element]);
        if (row1 < m_size && column1 < n_size) output[row1 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c11)[element]);
    }
}

kernel void gemma4_bf16_matmul(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &m_size [[buffer(3)]],
    constant uint &n_size [[buffer(4)]],
    constant uint &k_size [[buffer(5)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    const uint base_m = tile.y * 16;
    const uint base_n = (tile.x * 4 + simd_group) * 16;
    if (base_m >= m_size || base_n >= n_size) return;
    const ushort2 coordinate = simd_matrix_coordinate(lane);
    simdgroup_matrix<float, 8, 8> c00;
    simdgroup_matrix<float, 8, 8> c01;
    simdgroup_matrix<float, 8, 8> c10;
    simdgroup_matrix<float, 8, 8> c11;
    GEMMA_MATRIX_ELEMENTS(c00) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c01) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c10) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c11) = float2(0.0f);
    for (uint k = 0; k < k_size; k += 8) {
        simdgroup_matrix<float, 8, 8> a0;
        simdgroup_matrix<float, 8, 8> a1;
        simdgroup_matrix<float, 8, 8> b0;
        simdgroup_matrix<float, 8, 8> b1;
        for (uint element = 0; element < 2; ++element) {
            const uint a_column = k + coordinate.x + element;
            const uint a_row0 = base_m + coordinate.y;
            const uint a_row1 = a_row0 + 8;
            GEMMA_MATRIX_ELEMENTS(a0)[element] =
                a_row0 < m_size && a_column < k_size ? float(input[a_row0 * k_size + a_column]) : 0.0f;
            GEMMA_MATRIX_ELEMENTS(a1)[element] =
                a_row1 < m_size && a_column < k_size ? float(input[a_row1 * k_size + a_column]) : 0.0f;
            const uint weight_column = k + coordinate.y;
            const uint weight_row0 = base_n + coordinate.x + element;
            const uint weight_row1 = weight_row0 + 8;
            GEMMA_MATRIX_ELEMENTS(b0)[element] = weight_row0 < n_size
                ? float(weights[weight_row0 * k_size + weight_column])
                : 0.0f;
            GEMMA_MATRIX_ELEMENTS(b1)[element] = weight_row1 < n_size
                ? float(weights[weight_row1 * k_size + weight_column])
                : 0.0f;
        }
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    for (uint element = 0; element < 2; ++element) {
        const uint row0 = base_m + coordinate.y;
        const uint row1 = row0 + 8;
        const uint column0 = base_n + coordinate.x + element;
        const uint column1 = column0 + 8;
        if (row0 < m_size && column0 < n_size) output[row0 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c00)[element]);
        if (row0 < m_size && column1 < n_size) output[row0 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c01)[element]);
        if (row1 < m_size && column0 < n_size) output[row1 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c10)[element]);
        if (row1 < m_size && column1 < n_size) output[row1 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c11)[element]);
    }
}

kernel void gemma4_w6a16_matmul_tiled_bf16(
    device const bfloat *input [[buffer(0)]],
    device const uchar *packed_weights [[buffer(1)]],
    device const uchar *scales [[buffer(2)]],
    device const uchar *biases [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant uint &m_size [[buffer(5)]],
    constant uint &n_size [[buffer(6)]],
    constant uint &k_size [[buffer(7)]],
    constant ulong &weight_offset [[buffer(8)]],
    constant ulong &scale_offset [[buffer(9)]],
    constant ulong &bias_offset [[buffer(10)]],
    threadgroup bfloat *tiles [[threadgroup(0)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint block = 32;
    constexpr uint stride = 34;
    constexpr uint tile_elements = block * stride;
    threadgroup bfloat *input_tile = tiles;
    threadgroup bfloat *weight_tile = tiles + tile_elements;
    const uint base_m = tile.y * block;
    const uint base_n = tile.x * block;
    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;
    const ushort2 coordinate = simd_matrix_coordinate(lane);

    simdgroup_matrix<float, 8, 8> c00;
    simdgroup_matrix<float, 8, 8> c01;
    simdgroup_matrix<float, 8, 8> c10;
    simdgroup_matrix<float, 8, 8> c11;
    GEMMA_MATRIX_ELEMENTS(c00) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c01) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c10) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c11) = float2(0.0f);

    for (uint k = 0; k < k_size; k += block) {
        const uint local_row = thread_index / 4;
        const uint local_k = (thread_index % 4) * 8;
        const uint global_m = base_m + local_row;
        const uint global_n = base_n + local_row;
        const uint global_k = k + local_k;
        for (uint element = 0; element < 8; ++element) {
            input_tile[local_row * stride + local_k + element] =
                global_m < m_size && global_k + element < k_size
                ? input[global_m * k_size + global_k + element]
                : bfloat(0.0f);
        }
        if (global_n < n_size && global_k + 7 < k_size) {
            const ulong row_bytes = ulong(k_size) * 6 / 8;
            device const uchar *packed =
                packed_weights + weight_offset + ulong(global_n) * row_bytes + ulong(global_k) * 6 / 8;
            const ulong group = ulong(global_n) * (k_size / 64) + global_k / 64;
            const float scale = load_unaligned_bfloat16(scales, scale_offset + group * 2);
            const float bias = load_unaligned_bfloat16(biases, bias_offset + group * 2);
            const uint q0 = uint(packed[0] & 0x3f);
            const uint q1 = uint(((packed[0] >> 6) & 0x03) | ((packed[1] & 0x0f) << 2));
            const uint q2 = uint(((packed[1] >> 4) & 0x0f) | ((packed[2] & 0x03) << 4));
            const uint q3 = uint((packed[2] >> 2) & 0x3f);
            const uint q4 = uint(packed[3] & 0x3f);
            const uint q5 = uint(((packed[3] >> 6) & 0x03) | ((packed[4] & 0x0f) << 2));
            const uint q6 = uint(((packed[4] >> 4) & 0x0f) | ((packed[5] & 0x03) << 4));
            const uint q7 = uint((packed[5] >> 2) & 0x3f);
            weight_tile[(local_k + 0) * stride + local_row] = bfloat(float(q0) * scale + bias);
            weight_tile[(local_k + 1) * stride + local_row] = bfloat(float(q1) * scale + bias);
            weight_tile[(local_k + 2) * stride + local_row] = bfloat(float(q2) * scale + bias);
            weight_tile[(local_k + 3) * stride + local_row] = bfloat(float(q3) * scale + bias);
            weight_tile[(local_k + 4) * stride + local_row] = bfloat(float(q4) * scale + bias);
            weight_tile[(local_k + 5) * stride + local_row] = bfloat(float(q5) * scale + bias);
            weight_tile[(local_k + 6) * stride + local_row] = bfloat(float(q6) * scale + bias);
            weight_tile[(local_k + 7) * stride + local_row] = bfloat(float(q7) * scale + bias);
        } else {
            for (uint element = 0; element < 8; ++element) {
                weight_tile[(local_k + element) * stride + local_row] = bfloat(0.0f);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < block; inner += 8) {
            simdgroup_matrix<float, 8, 8> a0;
            simdgroup_matrix<float, 8, 8> a1;
            simdgroup_matrix<float, 8, 8> b0;
            simdgroup_matrix<float, 8, 8> b1;
            for (uint element = 0; element < 2; ++element) {
                GEMMA_MATRIX_ELEMENTS(a0)[element] = float(
                    input_tile[(simd_m + coordinate.y) * stride + inner + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(a1)[element] = float(
                    input_tile[(simd_m + 8 + coordinate.y) * stride + inner + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(b0)[element] = float(
                    weight_tile[(inner + coordinate.y) * stride + simd_n + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(b1)[element] = float(
                    weight_tile[(inner + coordinate.y) * stride + simd_n + 8 + coordinate.x + element]);
            }
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint element = 0; element < 2; ++element) {
        const uint row0 = base_m + simd_m + coordinate.y;
        const uint row1 = row0 + 8;
        const uint column0 = base_n + simd_n + coordinate.x + element;
        const uint column1 = column0 + 8;
        if (row0 < m_size && column0 < n_size) output[row0 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c00)[element]);
        if (row0 < m_size && column1 < n_size) output[row0 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c01)[element]);
        if (row1 < m_size && column0 < n_size) output[row1 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c10)[element]);
        if (row1 < m_size && column1 < n_size) output[row1 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c11)[element]);
    }
}

kernel void gemma4_bf16_matmul_tiled(
    device const bfloat *input [[buffer(0)]],
    device const bfloat *weights [[buffer(1)]],
    device bfloat *output [[buffer(2)]],
    constant uint &m_size [[buffer(3)]],
    constant uint &n_size [[buffer(4)]],
    constant uint &k_size [[buffer(5)]],
    threadgroup bfloat *tiles [[threadgroup(0)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint block = 32;
    constexpr uint stride = 34;
    constexpr uint tile_elements = block * stride;
    threadgroup bfloat *input_tile = tiles;
    threadgroup bfloat *weight_tile = tiles + tile_elements;
    const uint base_m = tile.y * block;
    const uint base_n = tile.x * block;
    const uint simd_m = (simd_group / 2) * 16;
    const uint simd_n = (simd_group % 2) * 16;
    const ushort2 coordinate = simd_matrix_coordinate(lane);
    simdgroup_matrix<float, 8, 8> c00;
    simdgroup_matrix<float, 8, 8> c01;
    simdgroup_matrix<float, 8, 8> c10;
    simdgroup_matrix<float, 8, 8> c11;
    GEMMA_MATRIX_ELEMENTS(c00) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c01) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c10) = float2(0.0f);
    GEMMA_MATRIX_ELEMENTS(c11) = float2(0.0f);
    for (uint k = 0; k < k_size; k += block) {
        const uint local_row = thread_index / 4;
        const uint local_k = (thread_index % 4) * 8;
        const uint global_m = base_m + local_row;
        const uint global_n = base_n + local_row;
        const uint global_k = k + local_k;
        for (uint element = 0; element < 8; ++element) {
            input_tile[local_row * stride + local_k + element] =
                global_m < m_size && global_k + element < k_size
                ? input[global_m * k_size + global_k + element]
                : bfloat(0.0f);
            weight_tile[(local_k + element) * stride + local_row] =
                global_n < n_size && global_k + element < k_size
                ? weights[global_n * k_size + global_k + element]
                : bfloat(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint inner = 0; inner < block; inner += 8) {
            simdgroup_matrix<float, 8, 8> a0;
            simdgroup_matrix<float, 8, 8> a1;
            simdgroup_matrix<float, 8, 8> b0;
            simdgroup_matrix<float, 8, 8> b1;
            for (uint element = 0; element < 2; ++element) {
                GEMMA_MATRIX_ELEMENTS(a0)[element] = float(
                    input_tile[(simd_m + coordinate.y) * stride + inner + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(a1)[element] = float(
                    input_tile[(simd_m + 8 + coordinate.y) * stride + inner + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(b0)[element] = float(
                    weight_tile[(inner + coordinate.y) * stride + simd_n + coordinate.x + element]);
                GEMMA_MATRIX_ELEMENTS(b1)[element] = float(
                    weight_tile[(inner + coordinate.y) * stride + simd_n + 8 + coordinate.x + element]);
            }
            simdgroup_multiply_accumulate(c00, a0, b0, c00);
            simdgroup_multiply_accumulate(c01, a0, b1, c01);
            simdgroup_multiply_accumulate(c10, a1, b0, c10);
            simdgroup_multiply_accumulate(c11, a1, b1, c11);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint element = 0; element < 2; ++element) {
        const uint row0 = base_m + simd_m + coordinate.y;
        const uint row1 = row0 + 8;
        const uint column0 = base_n + simd_n + coordinate.x + element;
        const uint column1 = column0 + 8;
        if (row0 < m_size && column0 < n_size) output[row0 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c00)[element]);
        if (row0 < m_size && column1 < n_size) output[row0 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c01)[element]);
        if (row1 < m_size && column0 < n_size) output[row1 * n_size + column0] = bfloat(GEMMA_MATRIX_ELEMENTS(c10)[element]);
        if (row1 < m_size && column1 < n_size) output[row1 * n_size + column1] = bfloat(GEMMA_MATRIX_ELEMENTS(c11)[element]);
    }
}
