#pragma once

#include "package/TensorIndex.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

namespace gemma_runtime::package {

struct TensorView {
    const TensorDescriptor *descriptor;
    std::span<const std::byte> bytes;
};

struct MappedFileView {
    const std::byte *data;
    size_t byteLength;
    size_t allocationLength;
};

class TensorStore {
  public:
    [[nodiscard]] static TensorStore open(const std::filesystem::path &packageRoot);
    [[nodiscard]] static TensorStore openIndexed(
        const std::filesystem::path &indexPath,
        const std::filesystem::path &payloadRoot);

    TensorStore(TensorStore &&) noexcept;
    TensorStore &operator=(TensorStore &&) noexcept;
    TensorStore(const TensorStore &) = delete;
    TensorStore &operator=(const TensorStore &) = delete;
    ~TensorStore();

    [[nodiscard]] const TensorIndex &index() const noexcept;
    [[nodiscard]] TensorView tensor(const std::string &name) const;
    [[nodiscard]] MappedFileView mappedFile(const std::filesystem::path &relativePath) const;

  private:
    struct Impl;
    explicit TensorStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace gemma_runtime::package
