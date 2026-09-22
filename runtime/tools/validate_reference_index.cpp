#include "gemma4/Gemma4Weights.hpp"
#include "package/TensorIndex.hpp"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "usage: validate-reference-index TENSOR_INDEX\n";
        return EXIT_FAILURE;
    }
    try {
        const auto index = gemma_runtime::package::TensorIndex::load(argv[1]);
        gemma_runtime::gemma4::Gemma4Weights::validateReferenceIndex(index);
        std::cout << "gemma4 BF16 reference index: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
