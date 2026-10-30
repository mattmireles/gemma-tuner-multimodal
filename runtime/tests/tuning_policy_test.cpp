#include "device/TuningPolicy.hpp"

#include <cstdlib>
#include <iostream>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    using gemma_runtime::TuningPolicy;
    using gemma_runtime::W6MatvecVariant;

    require(
        TuningPolicy::selectW6Matvec(8, 10, 2048, 2560) == W6MatvecVariant::Narrow4,
        "10-core Apple8 Q projection should use narrow4");
    require(
        TuningPolicy::selectW6Matvec(8, 10, 10240, 2560) == W6MatvecVariant::Narrow4,
        "10-core Apple8 gate projection should use narrow4");
    require(
        TuningPolicy::selectW6Matvec(8, 10, 2560, 10240) == W6MatvecVariant::Wide8,
        "10-core Apple8 down projection should use wide8");
    require(
        TuningPolicy::selectW6Matvec(8, 60, 10240, 2560) == W6MatvecVariant::Wide8,
        "60-core Apple8 should use wide8");
    require(
        TuningPolicy::selectW6Matvec(7, 8, 10240, 2560) == W6MatvecVariant::Wide8,
        "Apple7 should use wide8");
    require(
        TuningPolicy::selectW6Matvec(8, 0, 10240, 2560) == W6MatvecVariant::Wide8,
        "unknown core count should fail conservatively to wide8");

    std::cout << "W6 tuning policy: PASS\n";
    return EXIT_SUCCESS;
}
