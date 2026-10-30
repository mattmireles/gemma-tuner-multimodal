#include "package/TensorIndex.hpp"
#include "package/TensorStore.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Function> void requireThrows(Function function, const std::string &message) {
    try {
        function();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error(message);
}

} // namespace

int main() {
    const std::filesystem::path root = std::filesystem::temp_directory_path() / "gemma-runtime-tensor-index-test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root / "model");
    std::filesystem::create_directories(root / "metadata");
    {
        std::ofstream shard(root / "model/model.safetensors", std::ios::binary);
        shard << std::string(128, '\0');
    }
    const std::filesystem::path indexPath = root / "metadata/tensor-index.tsv";
    {
        std::ofstream index(indexPath);
        index << "gemma4-tensor-index-v1\n";
        index << "weight\tBF16\tmodel/model.safetensors\t64\t8\t1\t4\n";
        index << "scalar\tF32\tmodel/model.safetensors\t72\t4\t0\t\n";
    }

    const auto index = gemma_runtime::package::TensorIndex::load(indexPath);
    require(index.entries().size() == 2, "wrong tensor-index entry count");
    require(index.at("weight").shape == std::vector<uint64_t>{4}, "wrong tensor shape");
    require(index.at("scalar").shape.empty(), "scalar tensor should have rank zero");
    index.validateFiles(root);
    requireThrows([&] { static_cast<void>(index.at("missing")); }, "missing tensor did not fail");
    const auto store = gemma_runtime::package::TensorStore::open(root);
    const auto weight = store.tensor("weight");
    require(weight.bytes.size() == 8, "wrong mapped tensor length");
    require(weight.descriptor->dtype == "BF16", "wrong mapped tensor dtype");
    const auto mapped = store.mappedFile("model/model.safetensors");
    require(mapped.byteLength == 128, "wrong mapped shard length");
    require(mapped.allocationLength >= mapped.byteLength, "mapped shard allocation is too short");
    require(mapped.allocationLength % static_cast<size_t>(::sysconf(_SC_PAGESIZE)) == 0,
            "mapped shard allocation is not page-aligned");

    const std::filesystem::path directIndex = root / "direct-index.tsv";
    {
        std::ofstream index(directIndex);
        index << "gemma4-tensor-index-v1\n";
        index << "direct\tBF16\tmodel/model.safetensors\t80\t4\t1\t2\n";
    }
    const auto directStore = gemma_runtime::package::TensorStore::openIndexed(directIndex, root);
    require(directStore.tensor("direct").bytes.size() == 4, "direct indexed store mapped the wrong slice");

    {
        std::ofstream bad(indexPath);
        bad << "gemma4-tensor-index-v1\n";
        bad << "bad\tBF16\t../outside\t0\t2\t1\t1\n";
    }
    requireThrows(
        [&] { static_cast<void>(gemma_runtime::package::TensorIndex::load(indexPath)); },
        "unsafe tensor path did not fail");

    std::filesystem::remove_all(root);
    std::cout << "gemma4 tensor index: ok\n";
    return 0;
}
