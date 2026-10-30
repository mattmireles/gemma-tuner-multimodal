#include "ops/W6MetalOps.hpp"
#include "package/TensorStore.hpp"
#include "package/W6Tensor.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

uint16_t toBfloat16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::floor(fraction * static_cast<double>(values.size() - 1)));
    return values[index];
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 11) {
                fail(
                    "usage: benchmark-w6-gather METALLIB INDEX PAYLOAD_ROOT BASE_NAME "
                    "TOKENS COLUMN_OFFSET OUTPUT_COLUMNS OUTPUT_SCALE WARMUPS RUNS");
            }
            auto store = gemma_runtime::package::TensorStore::openIndexed(argv[2], argv[3]);
            const auto tensor = gemma_runtime::package::W6Tensor::open(store, argv[4]);
            const std::string baseName = argv[4];
            const uint32_t tokenCount = static_cast<uint32_t>(std::stoul(argv[5]));
            const uint32_t columnOffset = static_cast<uint32_t>(std::stoul(argv[6]));
            const uint32_t outputColumns = static_cast<uint32_t>(std::stoul(argv[7]));
            const float outputScale = std::stof(argv[8]);
            const int warmups = std::stoi(argv[9]);
            const int runs = std::stoi(argv[10]);
            if (tokenCount == 0 || outputColumns == 0 ||
                static_cast<uint64_t>(columnOffset) + outputColumns > tensor.columns()) {
                fail("invalid gather geometry");
            }
            if (!std::isfinite(outputScale) || warmups < 0 || runs <= 0) {
                fail("scale must be finite; warmups/runs are invalid");
            }
            if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX) {
                fail("tensor geometry exceeds Metal ABI");
            }
            const uint32_t vocabulary = static_cast<uint32_t>(tensor.rows());
            const uint32_t sourceColumns = static_cast<uint32_t>(tensor.columns());
            const size_t outputElements = static_cast<size_t>(tokenCount) * outputColumns;

            std::vector<uint32_t> tokenIds(tokenCount);
            for (uint32_t token = 0; token < tokenCount; ++token) {
                tokenIds[token] = static_cast<uint32_t>(
                    (static_cast<uint64_t>(token) * 104729U + (token % 7U) * 199729U) % vocabulary);
            }

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!device || !library || !queue) {
                fail(error.localizedDescription.UTF8String ?: "unable to initialize W6 gather benchmark");
            }
            gemma_runtime::ops::W6MetalOps ops(device, library, std::move(store));

            id<MTLBuffer> tokenBuffer = [device newBufferWithBytes:tokenIds.data()
                                                         length:tokenIds.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:outputElements * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
            if (!tokenBuffer || !outputBuffer) {
                fail("unable to allocate W6 gather buffers");
            }

            auto execute = [&]() {
                id<MTLCommandBuffer> commands = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                ops.encodeGather(
                    encoder,
                    tokenBuffer,
                    0,
                    outputBuffer,
                    0,
                    baseName,
                    tokenCount,
                    columnOffset,
                    outputColumns,
                    outputScale);
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status != MTLCommandBufferStatusCompleted) fail("Metal gather command failed");
                return (commands.GPUEndTime - commands.GPUStartTime) * 1000.0;
            };

            for (int iteration = 0; iteration < warmups; ++iteration) execute();
            execute();
            const auto *actual = static_cast<const uint16_t *>(outputBuffer.contents);
            const uint32_t sampleTokens[] = {0, tokenCount / 2, tokenCount - 1};
            const uint32_t sampleColumns[] = {0, outputColumns / 2, outputColumns - 1};
            for (uint32_t token : sampleTokens) {
                for (uint32_t column : sampleColumns) {
                    const float value = tensor.value(tokenIds[token], columnOffset + column) * outputScale;
                    const uint16_t expected = toBfloat16(value);
                    if (actual[static_cast<size_t>(token) * outputColumns + column] != expected) {
                        fail("real W6 gather differs from CPU affine decode");
                    }
                }
            }

            std::vector<double> wallTimes;
            std::vector<double> gpuTimes;
            wallTimes.reserve(runs);
            gpuTimes.reserve(runs);
            for (int iteration = 0; iteration < runs; ++iteration) {
                const auto start = std::chrono::steady_clock::now();
                const double gpu = execute();
                const auto end = std::chrono::steady_clock::now();
                wallTimes.push_back(std::chrono::duration<double, std::milli>(end - start).count());
                gpuTimes.push_back(gpu);
            }
            const double wallP50 = percentile(wallTimes, 0.5);
            const double gpuP50 = percentile(gpuTimes, 0.5);
            std::cout << std::setprecision(8)
                      << "{\"rows\":" << tensor.rows()
                      << ",\"source_columns\":" << sourceColumns
                      << ",\"tokens\":" << tokenCount
                      << ",\"column_offset\":" << columnOffset
                      << ",\"output_columns\":" << outputColumns
                      << ",\"weight_bytes\":" << tensor.packedWeight().bytes.size()
                      << ",\"aux_bytes\":" << tensor.scales().bytes.size() + tensor.biases().bytes.size()
                      << ",\"mapped_shards\":" << ops.mappedShardCount()
                      << ",\"mapped_file_bytes\":" << ops.mappedFileBytes()
                      << ",\"copied_weight_bytes\":" << ops.copiedWeightBytes()
                      << ",\"wall_p50_ms\":" << wallP50
                      << ",\"gpu_p50_ms\":" << gpuP50
                      << ",\"output_gib_per_second\":"
                      << (outputElements * sizeof(uint16_t) / 1073741824.0) / (gpuP50 / 1000.0)
                      << "}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception &error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
