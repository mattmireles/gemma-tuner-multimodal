#include "package/W6Tensor.hpp"

#include <bit>
#include <limits>
#include <stdexcept>

namespace gemma_runtime::package {
namespace {

void requireRankTwo(const TensorView &view, const std::string &name, const char *dtype) {
    if (view.descriptor->dtype != dtype || view.descriptor->shape.size() != 2) {
        throw std::invalid_argument("W6 tensor has the wrong dtype or rank: " + name);
    }
}

} // namespace

W6Tensor::W6Tensor(
    TensorView weight,
    TensorView scales,
    TensorView biases,
    size_t rows,
    size_t columns)
    : weight_(weight), scales_(scales), biases_(biases), rows_(rows), columns_(columns) {}

W6Tensor W6Tensor::open(const TensorStore &store, const std::string &baseName) {
    const std::string weightName = baseName + ".weight";
    const std::string scalesName = baseName + ".scales";
    const std::string biasesName = baseName + ".biases";
    const TensorView weight = store.tensor(weightName);
    const TensorView scales = store.tensor(scalesName);
    const TensorView biases = store.tensor(biasesName);
    requireRankTwo(weight, weightName, "U32");
    requireRankTwo(scales, scalesName, "BF16");
    requireRankTwo(biases, biasesName, "BF16");
    const uint64_t rows = weight.descriptor->shape[0];
    const uint64_t packedColumns = weight.descriptor->shape[1];
    if (packedColumns > std::numeric_limits<uint64_t>::max() / 32 || (packedColumns * 32) % kBits != 0) {
        throw std::invalid_argument("W6 packed width is invalid: " + weightName);
    }
    const uint64_t columns = packedColumns * 32 / kBits;
    if (columns == 0 || columns % kGroupSize != 0) {
        throw std::invalid_argument("W6 logical width is not group-aligned: " + weightName);
    }
    const std::vector<uint64_t> auxiliaryShape{rows, columns / kGroupSize};
    if (scales.descriptor->shape != auxiliaryShape || biases.descriptor->shape != auxiliaryShape) {
        throw std::invalid_argument("W6 scale/bias shape mismatch: " + baseName);
    }
    return W6Tensor(weight, scales, biases, static_cast<size_t>(rows), static_cast<size_t>(columns));
}

uint8_t W6Tensor::unpack(std::span<const std::byte> packed, size_t index) {
    const size_t bit = index * kBits;
    const size_t byte = bit / 8;
    const size_t shift = bit % 8;
    if (byte >= packed.size() || (shift > 2 && byte + 1 >= packed.size())) {
        throw std::out_of_range("W6 packed index is outside the payload");
    }
    uint16_t window = std::to_integer<uint8_t>(packed[byte]);
    if (byte + 1 < packed.size()) {
        window |= static_cast<uint16_t>(std::to_integer<uint8_t>(packed[byte + 1])) << 8U;
    }
    return static_cast<uint8_t>((window >> shift) & 0x3FU);
}

size_t W6Tensor::rows() const noexcept { return rows_; }
size_t W6Tensor::columns() const noexcept { return columns_; }
size_t W6Tensor::groupsPerRow() const noexcept { return columns_ / kGroupSize; }

bool W6Tensor::hasNaturallyAlignedPayloads() const noexcept {
    return weight_.descriptor->fileOffset % alignof(uint32_t) == 0 &&
           scales_.descriptor->fileOffset % alignof(uint16_t) == 0 &&
           biases_.descriptor->fileOffset % alignof(uint16_t) == 0;
}

uint8_t W6Tensor::quantized(size_t row, size_t column) const {
    if (row >= rows_ || column >= columns_) throw std::out_of_range("W6 tensor coordinate is out of range");
    const size_t rowBytes = columns_ * kBits / 8;
    return unpack(weight_.bytes.subspan(row * rowBytes, rowBytes), column);
}

float W6Tensor::bfloat16At(std::span<const std::byte> bytes, size_t index) {
    const size_t offset = index * sizeof(uint16_t);
    if (offset + sizeof(uint16_t) > bytes.size()) throw std::out_of_range("BF16 index is out of range");
    const uint16_t bits = static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[offset])) |
                          static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[offset + 1])) << 8U;
    return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16U);
}

float W6Tensor::value(size_t row, size_t column) const {
    if (row >= rows_ || column >= columns_) throw std::out_of_range("W6 tensor coordinate is out of range");
    const size_t auxiliaryIndex = row * groupsPerRow() + column / kGroupSize;
    const float scale = bfloat16At(scales_.bytes, auxiliaryIndex);
    const float bias = bfloat16At(biases_.bytes, auxiliaryIndex);
    return static_cast<float>(quantized(row, column)) * scale + bias;
}

std::vector<float> W6Tensor::dequantizeRow(size_t row) const {
    if (row >= rows_) throw std::out_of_range("W6 row is out of range");
    std::vector<float> result(columns_);
    for (size_t column = 0; column < columns_; ++column) result[column] = value(row, column);
    return result;
}

const TensorView &W6Tensor::packedWeight() const noexcept { return weight_; }
const TensorView &W6Tensor::scales() const noexcept { return scales_; }
const TensorView &W6Tensor::biases() const noexcept { return biases_; }

} // namespace gemma_runtime::package
