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

float fromBfloat16(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::floor(fraction * static_cast<double>(values.size() - 1)));
    return values[index];
}

struct Timings {
    std::vector<double> wallMilliseconds;
    std::vector<double> gpuMilliseconds;
};

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 8) {
                fail("usage: benchmark-w6-matmul METALLIB INDEX PAYLOAD_ROOT BASE_NAME ROWS WARMUPS RUNS");
            }
            auto store = gemma_runtime::package::TensorStore::openIndexed(argv[2], argv[3]);
            const auto tensor = gemma_runtime::package::W6Tensor::open(store, argv[4]);
            const std::string baseName = argv[4];
            const uint32_t mSize = static_cast<uint32_t>(std::stoul(argv[5]));
            const int warmups = std::stoi(argv[6]);
            const int runs = std::stoi(argv[7]);
            if (mSize == 0 || warmups < 0 || runs <= 0) fail("invalid rows, warmups, or runs");
            if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX) fail("tensor geometry exceeds Metal ABI");
            const uint32_t nSize = static_cast<uint32_t>(tensor.rows());
            const uint32_t kSize = static_cast<uint32_t>(tensor.columns());

            std::vector<uint16_t> input(static_cast<size_t>(mSize) * kSize);
            for (uint32_t row = 0; row < mSize; ++row) {
                for (uint32_t column = 0; column < kSize; ++column) {
                    input[static_cast<size_t>(row) * kSize + column] =
                        toBfloat16(std::sin((row * 17.0F + column) * 0.013F) * 0.125F);
                }
            }
            std::vector<uint16_t> denseWeights(static_cast<size_t>(nSize) * kSize);
            for (uint32_t row = 0; row < nSize; ++row) {
                for (uint32_t column = 0; column < kSize; ++column) {
                    denseWeights[static_cast<size_t>(row) * kSize + column] =
                        toBfloat16(tensor.value(row, column));
                }
            }

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            id<MTLFunction> denseFunction = [library newFunctionWithName:@"gemma4_bf16_matmul_tiled"];
            id<MTLComputePipelineState> densePipeline =
                [device newComputePipelineStateWithFunction:denseFunction error:&error];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!device || !library || !denseFunction || !densePipeline || !queue) {
                fail(error.localizedDescription.UTF8String ?: "unable to initialize W6 matmul benchmark");
            }
            gemma_runtime::ops::W6MetalOps ops(device, library, std::move(store));

            id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                            length:input.size() * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> denseBuffer = [device newBufferWithBytes:denseWeights.data()
                                                            length:denseWeights.size() * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:static_cast<size_t>(mSize) * nSize * sizeof(uint16_t)
                                                             options:MTLResourceStorageModeShared];
            if (!inputBuffer || !denseBuffer || !outputBuffer) fail("unable to allocate matmul buffers");

            auto execute = [&](bool w6) {
                id<MTLCommandBuffer> commands = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
                if (w6) {
                    ops.encodeMatmul(encoder, inputBuffer, 0, outputBuffer, 0, baseName, mSize);
                } else {
                    [encoder setComputePipelineState:densePipeline];
                    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
                    [encoder setBuffer:denseBuffer offset:0 atIndex:1];
                    [encoder setBuffer:outputBuffer offset:0 atIndex:2];
                    [encoder setBytes:&mSize length:sizeof(mSize) atIndex:3];
                    [encoder setBytes:&nSize length:sizeof(nSize) atIndex:4];
                    [encoder setBytes:&kSize length:sizeof(kSize) atIndex:5];
                    constexpr NSUInteger threadgroupBytes = 2 * 32 * 34 * sizeof(uint16_t);
                    [encoder setThreadgroupMemoryLength:threadgroupBytes atIndex:0];
                    [encoder dispatchThreadgroups:MTLSizeMake((nSize + 31) / 32, (mSize + 31) / 32, 1)
                                      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                }
                [encoder endEncoding];
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status != MTLCommandBufferStatusCompleted) fail("Metal matmul command failed");
                return (commands.GPUEndTime - commands.GPUStartTime) * 1000.0;
            };

            for (int iteration = 0; iteration < warmups; ++iteration) execute(true);
            execute(true);
            const auto *actual = static_cast<const uint16_t *>(outputBuffer.contents);
            const uint32_t sampleRows[] = {0, mSize / 2, mSize - 1};
            const uint32_t sampleColumns[] = {0, nSize / 2, nSize - 1};
            double maximum = 0.0;
            for (uint32_t row : sampleRows) {
                for (uint32_t outputColumn : sampleColumns) {
                    float expected = 0.0F;
                    for (uint32_t k = 0; k < kSize; ++k) {
                        expected += fromBfloat16(input[static_cast<size_t>(row) * kSize + k]) *
                                    tensor.value(outputColumn, k);
                    }
                    const double difference = std::abs(
                        static_cast<double>(fromBfloat16(actual[static_cast<size_t>(row) * nSize + outputColumn])) -
                        fromBfloat16(toBfloat16(expected)));
                    maximum = std::max(maximum, difference);
                }
            }
            if (maximum > 0.5) fail("real W6 matmul exceeds the BF16 sample tolerance");

            auto measure = [&](bool w6) {
                Timings timings;
                timings.wallMilliseconds.reserve(runs);
                timings.gpuMilliseconds.reserve(runs);
                for (int iteration = 0; iteration < runs; ++iteration) {
                    const auto start = std::chrono::steady_clock::now();
                    const double gpu = execute(w6);
                    const auto end = std::chrono::steady_clock::now();
                    timings.wallMilliseconds.push_back(
                        std::chrono::duration<double, std::milli>(end - start).count());
                    timings.gpuMilliseconds.push_back(gpu);
                }
                return timings;
            };
            const Timings w6Times = measure(true);
            for (int iteration = 0; iteration < warmups; ++iteration) execute(false);
            const Timings denseTimes = measure(false);
            const double w6Wall = percentile(w6Times.wallMilliseconds, 0.5);
            const double denseWall = percentile(denseTimes.wallMilliseconds, 0.5);
            const double w6Gpu = percentile(w6Times.gpuMilliseconds, 0.5);
            const double denseGpu = percentile(denseTimes.gpuMilliseconds, 0.5);
            std::cout << std::setprecision(8)
                      << "{\"m\":" << mSize
                      << ",\"n\":" << nSize
                      << ",\"k\":" << kSize
                      << ",\"mapped_shards\":" << ops.mappedShardCount()
                      << ",\"mapped_file_bytes\":" << ops.mappedFileBytes()
                      << ",\"copied_weight_bytes\":" << ops.copiedWeightBytes()
                      << ",\"sample_max_abs\":" << maximum
                      << ",\"w6_wall_p50_ms\":" << w6Wall
                      << ",\"dense_wall_p50_ms\":" << denseWall
                      << ",\"wall_speedup_vs_dense\":" << denseWall / w6Wall
                      << ",\"w6_gpu_p50_ms\":" << w6Gpu
                      << ",\"dense_gpu_p50_ms\":" << denseGpu
                      << ",\"gpu_speedup_vs_dense\":" << denseGpu / w6Gpu
                      << "}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception &error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
