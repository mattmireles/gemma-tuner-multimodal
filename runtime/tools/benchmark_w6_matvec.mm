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

void encodeW6(
    id<MTLCommandBuffer> commands,
    gemma_runtime::ops::W6MetalOps &ops,
    id<MTLBuffer> input,
    id<MTLBuffer> output,
    const std::string &baseName,
    gemma_runtime::W6MatvecVariant variant) {
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    ops.encodeMatvec(encoder, input, 0, output, 0, baseName, variant);
    [encoder endEncoding];
}

void encodeDense(
    id<MTLCommandBuffer> commands,
    id<MTLComputePipelineState> pipeline,
    id<MTLBuffer> input,
    id<MTLBuffer> weight,
    id<MTLBuffer> output,
    uint32_t rows,
    uint32_t columns) {
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:weight offset:0 atIndex:1];
    [encoder setBuffer:output offset:0 atIndex:2];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:3];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:4];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [encoder endEncoding];
}

} // namespace

int main(int argc, char **argv) {
    @autoreleasepool {
        try {
            if (argc != 7 && argc != 8) {
                fail("usage: benchmark-w6-matvec METALLIB INDEX PAYLOAD_ROOT BASE_NAME WARMUPS RUNS [wide8|narrow4]");
            }
            const int warmups = std::stoi(argv[5]);
            const int runs = std::stoi(argv[6]);
            if (warmups < 0 || runs <= 0) fail("warmups must be non-negative and runs must be positive");
            auto store = gemma_runtime::package::TensorStore::openIndexed(argv[2], argv[3]);
            const auto tensor = gemma_runtime::package::W6Tensor::open(store, argv[4]);
            const std::string baseName = argv[4];
            const std::string variantName = argc == 8 ? argv[7] : "wide8";
            const auto variant = variantName == "wide8"
                                     ? gemma_runtime::W6MatvecVariant::Wide8
                                     : variantName == "narrow4"
                                         ? gemma_runtime::W6MatvecVariant::Narrow4
                                         : throw std::invalid_argument("unknown W6 matvec variant");
            if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX) fail("tensor geometry exceeds Metal ABI");
            const uint32_t rows = static_cast<uint32_t>(tensor.rows());
            const uint32_t columns = static_cast<uint32_t>(tensor.columns());

            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            NSError *error = nil;
            NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
            id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
            id<MTLFunction> denseFunction = [library newFunctionWithName:@"gemma4_bf16_matvec"];
            id<MTLComputePipelineState> densePipeline = [device newComputePipelineStateWithFunction:denseFunction error:&error];
            id<MTLCommandQueue> queue = [device newCommandQueue];
            if (!device || !library || !densePipeline || !queue) {
                fail(error.localizedDescription.UTF8String ?: "unable to initialize W6 benchmark");
            }
            gemma_runtime::ops::W6MetalOps ops(device, library, std::move(store));

            std::vector<uint16_t> input(columns);
            for (uint32_t column = 0; column < columns; ++column) {
                input[column] = toBfloat16(std::sin(static_cast<float>(column) * 0.013F) * 0.25F);
            }
            std::vector<uint16_t> denseWeights(static_cast<size_t>(rows) * columns);
            std::vector<uint16_t> expected(rows);
            for (uint32_t row = 0; row < rows; ++row) {
                float sum = 0.0F;
                for (uint32_t column = 0; column < columns; ++column) {
                    const float weight = tensor.value(row, column);
                    denseWeights[static_cast<size_t>(row) * columns + column] = toBfloat16(weight);
                    sum += fromBfloat16(input[column]) * weight;
                }
                expected[row] = toBfloat16(sum);
            }

            id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                            length:input.size() * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> denseBuffer = [device newBufferWithBytes:denseWeights.data()
                                                            length:denseWeights.size() * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
            id<MTLBuffer> outputBuffer = [device newBufferWithLength:rows * sizeof(uint16_t)
                                                             options:MTLResourceStorageModeShared];

            auto execute = [&](bool w6) {
                id<MTLCommandBuffer> commands = [queue commandBuffer];
                if (w6) {
                    encodeW6(
                        commands,
                        ops,
                        inputBuffer,
                        outputBuffer,
                        baseName,
                        variant);
                } else {
                    encodeDense(
                        commands,
                        densePipeline,
                        inputBuffer,
                        denseBuffer,
                        outputBuffer,
                        rows,
                        columns);
                }
                [commands commit];
                [commands waitUntilCompleted];
                if (commands.status != MTLCommandBufferStatusCompleted) fail("Metal benchmark command failed");
                return (commands.GPUEndTime - commands.GPUStartTime) * 1000.0;
            };
            for (int iteration = 0; iteration < warmups; ++iteration) execute(true);
            execute(true);
            const auto *actual = static_cast<const uint16_t *>(outputBuffer.contents);
            double absoluteTotal = 0.0;
            double maximum = 0.0;
            for (uint32_t row = 0; row < rows; ++row) {
                const double difference = std::abs(
                    static_cast<double>(fromBfloat16(actual[row])) - fromBfloat16(expected[row]));
                absoluteTotal += difference;
                maximum = std::max(maximum, difference);
            }
            const double mae = absoluteTotal / rows;
            if (mae > 0.03 || maximum > 0.5) fail("real W6 matvec exceeds declared BF16 reference tolerance");

            auto measure = [&](bool w6) {
                Timings timings;
                timings.wallMilliseconds.reserve(runs);
                timings.gpuMilliseconds.reserve(runs);
                for (int iteration = 0; iteration < runs; ++iteration) {
                    const auto start = std::chrono::steady_clock::now();
                    const double gpuMilliseconds = execute(w6);
                    const auto end = std::chrono::steady_clock::now();
                    timings.wallMilliseconds.push_back(
                        std::chrono::duration<double, std::milli>(end - start).count());
                    timings.gpuMilliseconds.push_back(gpuMilliseconds);
                }
                return timings;
            };
            const auto w6Times = measure(true);
            for (int iteration = 0; iteration < warmups; ++iteration) execute(false);
            const auto denseTimes = measure(false);
            const double w6P50 = percentile(w6Times.wallMilliseconds, 0.5);
            const double denseP50 = percentile(denseTimes.wallMilliseconds, 0.5);
            const double w6GpuP50 = percentile(w6Times.gpuMilliseconds, 0.5);
            const double denseGpuP50 = percentile(denseTimes.gpuMilliseconds, 0.5);
            std::cout << std::setprecision(8)
                      << "{\"variant\":\"" << variantName << "\""
                      << ",\"rows\":" << rows
                      << ",\"columns\":" << columns
                      << ",\"w6_weight_bytes\":" << tensor.packedWeight().bytes.size()
                      << ",\"w6_aux_bytes\":" << tensor.scales().bytes.size() + tensor.biases().bytes.size()
                      << ",\"dense_comparator_bytes\":" << denseWeights.size() * sizeof(uint16_t)
                      << ",\"mapped_shards\":" << ops.mappedShardCount()
                      << ",\"mapped_file_bytes\":" << ops.mappedFileBytes()
                      << ",\"copied_weight_bytes\":" << ops.copiedWeightBytes()
                      << ",\"w6_mae\":" << mae
                      << ",\"w6_max_abs\":" << maximum
                      << ",\"w6_p50_ms\":" << w6P50
                      << ",\"dense_p50_ms\":" << denseP50
                      << ",\"speedup_vs_dense\":" << denseP50 / w6P50
                      << ",\"w6_gpu_p50_ms\":" << w6GpuP50
                      << ",\"dense_gpu_p50_ms\":" << denseGpuP50
                      << ",\"gpu_speedup_vs_dense\":" << denseGpuP50 / w6GpuP50
                      << "}\n";
            return EXIT_SUCCESS;
        } catch (const std::exception &error) {
            std::cerr << "FAIL: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
}
