#include "package/TensorStore.hpp"
#include "package/W6Tensor.hpp"

#include <cstdlib>
#include <iostream>
#include <set>
#include <string>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: validate-w6-index INDEX PAYLOAD_ROOT\n";
        return EXIT_FAILURE;
    }
    const auto store = gemma_runtime::package::TensorStore::openIndexed(argv[1], argv[2]);
    std::set<std::string> bases;
    for (const auto &[name, descriptor] : store.index().entries()) {
        if (descriptor.dtype == "U32" && name.ends_with(".weight")) {
            bases.insert(name.substr(0, name.size() - std::string(".weight").size()));
        }
    }
    size_t unaligned = 0;
    for (const std::string &base : bases) {
        const auto tensor = gemma_runtime::package::W6Tensor::open(store, base);
        if (!tensor.hasNaturallyAlignedPayloads()) ++unaligned;
        static_cast<void>(tensor.value(0, 0));
        static_cast<void>(tensor.value(tensor.rows() - 1, tensor.columns() - 1));
    }
    std::cout << "validated " << bases.size() << " affine W6 tensors; "
              << unaligned << " require byte-addressed binding or aligned W6 staging\n";
    return EXIT_SUCCESS;
}
