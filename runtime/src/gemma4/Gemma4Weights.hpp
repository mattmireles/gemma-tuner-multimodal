#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <vector>

namespace gemma_runtime::package {
class TensorIndex;
}

namespace gemma_runtime::gemma4 {

class Gemma4Weights {
  public:
    using Shape = std::vector<uint64_t>;
    using Inventory = std::map<std::string, Shape>;

    [[nodiscard]] static Inventory expectedReferenceInventory();
    static void validateReferenceInventory(const Inventory &inventory);
    static void validateReferenceIndex(const package::TensorIndex &index);
};

} // namespace gemma_runtime::gemma4
