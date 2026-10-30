#include "package/TensorIndex.hpp"

#include <charconv>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace gemma_runtime::package {
namespace {

constexpr std::string_view kVersion = "gemma4-tensor-index-v1";

std::vector<std::string> splitPreservingEmpty(const std::string &value, char delimiter) {
    std::vector<std::string> result;
    size_t start = 0;
    while (true) {
        const size_t end = value.find(delimiter, start);
        if (end == std::string::npos) {
            result.emplace_back(value.substr(start));
            return result;
        }
        result.emplace_back(value.substr(start, end - start));
        start = end + 1;
    }
}

uint64_t parseUnsigned(const std::string &value, std::string_view field, size_t lineNumber) {
    uint64_t result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (value.empty() || parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
        throw std::runtime_error("invalid " + std::string(field) + " on tensor-index line " +
                                 std::to_string(lineNumber));
    }
    return result;
}

bool isSafeRelativePath(const std::filesystem::path &path) {
    if (path.empty() || path.is_absolute() || path.has_root_path()) return false;
    for (const auto &component : path) {
        if (component.empty() || component == "." || component == "..") return false;
    }
    return true;
}

std::vector<uint64_t> parseShape(const std::string &value, uint64_t rank, size_t lineNumber) {
    if (rank == 0) {
        if (!value.empty()) throw std::runtime_error("scalar tensor has dimensions on line " + std::to_string(lineNumber));
        return {};
    }
    const std::vector<std::string> fields = splitPreservingEmpty(value, ',');
    if (fields.size() != rank) throw std::runtime_error("tensor rank mismatch on line " + std::to_string(lineNumber));
    std::vector<uint64_t> result;
    result.reserve(fields.size());
    for (const std::string &field : fields) result.push_back(parseUnsigned(field, "dimension", lineNumber));
    return result;
}

} // namespace

TensorIndex TensorIndex::load(const std::filesystem::path &path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("unable to open tensor index: " + path.string());
    std::string line;
    if (!std::getline(stream, line) || line != kVersion) {
        throw std::runtime_error("unsupported tensor-index version");
    }

    TensorIndex result;
    size_t lineNumber = 1;
    while (std::getline(stream, line)) {
        ++lineNumber;
        if (line.empty()) throw std::runtime_error("empty tensor-index line " + std::to_string(lineNumber));
        const std::vector<std::string> fields = splitPreservingEmpty(line, '\t');
        if (fields.size() != 7) throw std::runtime_error("invalid tensor-index line " + std::to_string(lineNumber));
        if (fields[0].empty() || fields[0].find_first_of("\r\n") != std::string::npos) {
            throw std::runtime_error("invalid tensor name on line " + std::to_string(lineNumber));
        }
        const std::filesystem::path relativePath(fields[2]);
        if (!isSafeRelativePath(relativePath)) {
            throw std::runtime_error("unsafe tensor path on line " + std::to_string(lineNumber));
        }
        const uint64_t rank = parseUnsigned(fields[5], "rank", lineNumber);
        TensorDescriptor descriptor{
            .name = fields[0],
            .dtype = fields[1],
            .relativePath = relativePath,
            .fileOffset = parseUnsigned(fields[3], "file offset", lineNumber),
            .byteLength = parseUnsigned(fields[4], "byte length", lineNumber),
            .shape = parseShape(fields[6], rank, lineNumber),
        };
        if (descriptor.dtype.empty()) throw std::runtime_error("empty tensor dtype on line " + std::to_string(lineNumber));
        if (!result.entries_.emplace(descriptor.name, std::move(descriptor)).second) {
            throw std::runtime_error("duplicate tensor name on line " + std::to_string(lineNumber));
        }
    }
    if (result.entries_.empty()) throw std::runtime_error("tensor index contains no entries");
    return result;
}

const TensorDescriptor &TensorIndex::at(const std::string &name) const {
    const auto found = entries_.find(name);
    if (found == entries_.end()) throw std::out_of_range("tensor is absent from package index: " + name);
    return found->second;
}

const std::map<std::string, TensorDescriptor> &TensorIndex::entries() const noexcept { return entries_; }

void TensorIndex::validateFiles(const std::filesystem::path &packageRoot) const {
    for (const auto &[name, descriptor] : entries_) {
        const std::filesystem::path path = packageRoot / descriptor.relativePath;
        std::error_code error;
        const uint64_t fileSize = std::filesystem::file_size(path, error);
        if (error) throw std::runtime_error("unable to inspect tensor shard for " + name + ": " + error.message());
        if (descriptor.fileOffset > fileSize || descriptor.byteLength > fileSize - descriptor.fileOffset) {
            throw std::runtime_error("tensor extends beyond shard: " + name);
        }
    }
}

} // namespace gemma_runtime::package
