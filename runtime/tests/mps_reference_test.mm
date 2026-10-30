#include "ops/MpsReference.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <span>
#include <vector>

namespace {

void close(std::span<const float> actual, std::span<const float> expected, float tolerance) {
    if (actual.size() != expected.size()) std::exit(EXIT_FAILURE);
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance) {
            std::cerr << "BF16 MPS mismatch at " << index << ": " << actual[index]
                      << " != " << expected[index] << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
}

} // namespace

int main() {
    const std::vector<float> left{1, 2, 3, 4, 5, 6};
    const std::vector<float> right{7, 8, 9, 10, 11, 12};
    const std::vector<float> expected{58, 64, 139, 154};
    close(gemma_runtime::ops::MpsReference::bf16Matmul(left, right, 2, 3, 2), expected, 0.01F);
    std::cout << "gemma4 MPS BF16 reference: PASS\n";
    return EXIT_SUCCESS;
}
