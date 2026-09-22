#include "package/TensorStore.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    @autoreleasepool {
        try {
            const std::filesystem::path root =
                std::filesystem::temp_directory_path() / "gemma-runtime-metal-tensor-store-test";
            std::filesystem::remove_all(root);
            std::filesystem::create_directories(root / "model");
            std::filesystem::create_directories(root / "metadata");
            {
                std::ofstream shard(root / "model/weights.bin", std::ios::binary);
                shard << std::string(16385, '\0');
            }
            {
                std::ofstream index(root / "metadata/tensor-index.tsv");
                index << "gemma4-tensor-index-v1\n";
                index << "weight\tU8\tmodel/weights.bin\t1\t16384\t1\t16384\n";
            }

            const auto store = gemma_runtime::package::TensorStore::open(root);
            const auto mapped = store.mappedFile("model/weights.bin");
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil, "Metal device is unavailable");
            id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:const_cast<std::byte *>(mapped.data)
                                                             length:mapped.allocationLength
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
            require(buffer != nil, "Metal rejected the page-aligned mmap without copying");
            require(buffer.contents == mapped.data, "Metal no-copy buffer changed the mapped address");
            require(buffer.length == mapped.allocationLength, "Metal no-copy buffer changed the mapped length");
            buffer = nil;
            std::filesystem::remove_all(root);
            std::cout << "Metal tensor-store no-copy mapping: PASS\n";
            return 0;
        } catch (const std::exception &error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return 1;
        }
    }
}
