#include "package/TensorStore.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace gemma_runtime::package {
namespace {

class Mapping {
  public:
    explicit Mapping(const std::filesystem::path &path) {
        descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (descriptor_ < 0) fail("open", path);
        struct stat status {};
        if (::fstat(descriptor_, &status) != 0) fail("stat", path);
        if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
            throw std::runtime_error("tensor shard is not a non-empty regular file: " + path.string());
        }
        size_ = static_cast<size_t>(status.st_size);
        if (static_cast<off_t>(size_) != status.st_size) {
            throw std::runtime_error("tensor shard exceeds addressable size: " + path.string());
        }
        const long pageSize = ::sysconf(_SC_PAGESIZE);
        if (pageSize <= 0 || size_ > std::numeric_limits<size_t>::max() - static_cast<size_t>(pageSize - 1)) {
            throw std::runtime_error("unable to determine a safe tensor mapping size: " + path.string());
        }
        allocationSize_ =
            (size_ + static_cast<size_t>(pageSize - 1)) / static_cast<size_t>(pageSize) *
            static_cast<size_t>(pageSize);
        data_ = ::mmap(nullptr, allocationSize_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            fail("map", path);
        }
    }

    Mapping(Mapping &&other) noexcept
        : descriptor_(other.descriptor_), data_(other.data_), size_(other.size_), allocationSize_(other.allocationSize_) {
        other.descriptor_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
        other.allocationSize_ = 0;
    }

    Mapping &operator=(Mapping &&other) noexcept {
        if (this == &other) return *this;
        release();
        descriptor_ = other.descriptor_;
        data_ = other.data_;
        size_ = other.size_;
        allocationSize_ = other.allocationSize_;
        other.descriptor_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
        other.allocationSize_ = 0;
        return *this;
    }

    Mapping(const Mapping &) = delete;
    Mapping &operator=(const Mapping &) = delete;
    ~Mapping() { release(); }

    [[nodiscard]] const std::byte *data() const noexcept { return static_cast<const std::byte *>(data_); }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] size_t allocationSize() const noexcept { return allocationSize_; }

  private:
    [[noreturn]] void fail(const char *operation, const std::filesystem::path &path) {
        const int error = errno;
        release();
        throw std::runtime_error(std::string("unable to ") + operation + " tensor shard " + path.string() +
                                 ": " + std::strerror(error));
    }

    void release() noexcept {
        if (data_ != nullptr) ::munmap(data_, allocationSize_);
        if (descriptor_ >= 0) ::close(descriptor_);
        descriptor_ = -1;
        data_ = nullptr;
        size_ = 0;
        allocationSize_ = 0;
    }

    int descriptor_ = -1;
    void *data_ = nullptr;
    size_t size_ = 0;
    size_t allocationSize_ = 0;
};

} // namespace

struct TensorStore::Impl {
    TensorIndex index;
    std::map<std::filesystem::path, Mapping> mappings;
};

TensorStore::TensorStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TensorStore::TensorStore(TensorStore &&) noexcept = default;
TensorStore &TensorStore::operator=(TensorStore &&) noexcept = default;
TensorStore::~TensorStore() = default;

TensorStore TensorStore::open(const std::filesystem::path &packageRoot) {
    return openIndexed(packageRoot / "metadata/tensor-index.tsv", packageRoot);
}

TensorStore TensorStore::openIndexed(
    const std::filesystem::path &indexPath,
    const std::filesystem::path &payloadRoot) {
    auto impl = std::make_unique<Impl>(Impl{
        .index = TensorIndex::load(indexPath),
        .mappings = {},
    });
    impl->index.validateFiles(payloadRoot);
    std::set<std::filesystem::path> paths;
    for (const auto &[name, descriptor] : impl->index.entries()) {
        static_cast<void>(name);
        paths.insert(descriptor.relativePath);
    }
    for (const auto &relativePath : paths) {
        impl->mappings.emplace(relativePath, Mapping(payloadRoot / relativePath));
    }
    return TensorStore(std::move(impl));
}

const TensorIndex &TensorStore::index() const noexcept { return impl_->index; }

TensorView TensorStore::tensor(const std::string &name) const {
    const TensorDescriptor &descriptor = impl_->index.at(name);
    const auto found = impl_->mappings.find(descriptor.relativePath);
    if (found == impl_->mappings.end()) throw std::logic_error("tensor shard was not mapped: " + name);
    const Mapping &mapping = found->second;
    if (descriptor.fileOffset > mapping.size() || descriptor.byteLength > mapping.size() - descriptor.fileOffset) {
        throw std::logic_error("validated tensor slice is out of bounds: " + name);
    }
    return TensorView{
        .descriptor = &descriptor,
        .bytes = std::span(mapping.data() + descriptor.fileOffset, static_cast<size_t>(descriptor.byteLength)),
    };
}

MappedFileView TensorStore::mappedFile(const std::filesystem::path &relativePath) const {
    const auto found = impl_->mappings.find(relativePath);
    if (found == impl_->mappings.end()) {
        throw std::out_of_range("tensor shard was not mapped: " + relativePath.string());
    }
    const Mapping &mapping = found->second;
    return MappedFileView{
        .data = mapping.data(),
        .byteLength = mapping.size(),
        .allocationLength = mapping.allocationSize(),
    };
}

} // namespace gemma_runtime::package
