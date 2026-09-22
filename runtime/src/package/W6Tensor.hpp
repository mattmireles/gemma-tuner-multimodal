#pragma once

#include "package/TensorStore.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace gemma_runtime::package {

class W6Tensor {
  public:
    static constexpr size_t kBits = 6;
    static constexpr size_t kGroupSize = 64;

    [[nodiscard]] static W6Tensor open(const TensorStore &store, const std::string &baseName);
    [[nodiscard]] static uint8_t unpack(std::span<const std::byte> packed, size_t index);

    [[nodiscard]] size_t rows() const noexcept;
    [[nodiscard]] size_t columns() const noexcept;
    [[nodiscard]] size_t groupsPerRow() const noexcept;
    [[nodiscard]] bool hasNaturallyAlignedPayloads() const noexcept;
    [[nodiscard]] uint8_t quantized(size_t row, size_t column) const;
    [[nodiscard]] float value(size_t row, size_t column) const;
    [[nodiscard]] std::vector<float> dequantizeRow(size_t row) const;

    [[nodiscard]] const TensorView &packedWeight() const noexcept;
    [[nodiscard]] const TensorView &scales() const noexcept;
    [[nodiscard]] const TensorView &biases() const noexcept;

  private:
    W6Tensor(TensorView weight, TensorView scales, TensorView biases, size_t rows, size_t columns);

    [[nodiscard]] static float bfloat16At(std::span<const std::byte> bytes, size_t index);

    TensorView weight_;
    TensorView scales_;
    TensorView biases_;
    size_t rows_;
    size_t columns_;
};

} // namespace gemma_runtime::package
