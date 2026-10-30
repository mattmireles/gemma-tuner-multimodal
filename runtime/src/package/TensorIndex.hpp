#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace gemma_runtime::package {

struct TensorDescriptor {
    std::string name;
    std::string dtype;
    std::filesystem::path relativePath;
    uint64_t fileOffset;
    uint64_t byteLength;
    std::vector<uint64_t> shape;
};

class TensorIndex {
  public:
    [[nodiscard]] static TensorIndex load(const std::filesystem::path &path);

    [[nodiscard]] const TensorDescriptor &at(const std::string &name) const;
    [[nodiscard]] const std::map<std::string, TensorDescriptor> &entries() const noexcept;
    void validateFiles(const std::filesystem::path &packageRoot) const;

  private:
    std::map<std::string, TensorDescriptor> entries_;
};

} // namespace gemma_runtime::package
