#include "device/TuningPolicy.hpp"

namespace gemma_runtime {

W6MatvecVariant TuningPolicy::selectW6Matvec(
    uint32_t appleGpuFamily,
    uint32_t gpuCoreCount,
    uint32_t rows,
    uint32_t columns) noexcept {
    // Measured on a 10-core M2 Air: the four-value schedule wins for the
    // 2,560-wide Q and MLP gate projections. The same schedule loses on the
    // 8-core M1 and 60-core M2 Ultra, and for the 10,240-wide down projection.
    // Unknown core counts therefore remain on the conservative wide schedule.
    if (appleGpuFamily == 8 && gpuCoreCount > 0 && gpuCoreCount <= 10 &&
        columns == 2560 && rows <= 10240) {
        return W6MatvecVariant::Narrow4;
    }
    return W6MatvecVariant::Wide8;
}

} // namespace gemma_runtime
