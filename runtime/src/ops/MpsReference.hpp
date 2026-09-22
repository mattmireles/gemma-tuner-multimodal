#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace gemma_runtime::ops {

class MpsReference {
  public:
    // Compute [rows, inner] x [inner, columns] in BF16 on Metal and return
    // float32 values copied back for reference comparison.
    [[nodiscard]] static std::vector<float> bf16Matmul(
        std::span<const float> left,
        std::span<const float> right,
        size_t rows,
        size_t inner,
        size_t columns);
};

} // namespace gemma_runtime::ops
