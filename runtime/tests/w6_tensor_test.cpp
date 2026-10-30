#include "package/W6Tensor.hpp"

#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    const std::array<uint8_t, 8> values{0, 1, 2, 3, 15, 31, 47, 63};
    std::vector<std::byte> packed(6, std::byte{0});
    for (size_t index = 0; index < values.size(); ++index) {
        const size_t bit = index * 6;
        const size_t byte = bit / 8;
        const size_t shift = bit % 8;
        uint16_t window = static_cast<uint16_t>(values[index]) << shift;
        packed[byte] |= static_cast<std::byte>(window & 0xFFU);
        if (byte + 1 < packed.size()) packed[byte + 1] |= static_cast<std::byte>(window >> 8U);
    }
    for (size_t index = 0; index < values.size(); ++index) {
        require(
            gemma_runtime::package::W6Tensor::unpack(packed, index) == values[index],
            "six-bit unpack mismatch");
    }
    try {
        static_cast<void>(gemma_runtime::package::W6Tensor::unpack(packed, values.size()));
        require(false, "out-of-range six-bit unpack must fail");
    } catch (const std::out_of_range &) {
    }
    std::cout << "W6 tensor bit packing: PASS\n";
    return EXIT_SUCCESS;
}
