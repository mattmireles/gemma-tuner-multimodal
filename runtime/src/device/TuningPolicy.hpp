#pragma once

#include <cstdint>

namespace gemma_runtime {

enum class W6MatvecVariant {
    Wide8,
    Narrow4,
};

struct TuningPolicy {
    [[nodiscard]] static W6MatvecVariant selectW6Matvec(
        uint32_t appleGpuFamily,
        uint32_t gpuCoreCount,
        uint32_t rows,
        uint32_t columns) noexcept;
};

} // namespace gemma_runtime
