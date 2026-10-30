#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

void pack6(std::vector<uint8_t> &bytes, size_t index, uint8_t value) {
    const size_t bit = index * 6;
    const size_t byte = bit / 8;
    const size_t shift = bit % 8;
    const uint16_t window = static_cast<uint16_t>(value & 0x3FU) << shift;
    bytes[byte] |= static_cast<uint8_t>(window & 0xFFU);
    if (byte + 1 < bytes.size()) bytes[byte + 1] |= static_cast<uint8_t>(window >> 8U);
}

} // namespace

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) fail("usage: w6_metal_test <metallib>");
        constexpr uint32_t rows = 16;
        constexpr uint32_t columns = 256;
        constexpr uint32_t groups = columns / 64;
        constexpr uint32_t rowBytes = columns * 6 / 8;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"gemma4_w6a16_matvec_bf16"];
        id<MTLFunction> gatherFunction = [library newFunctionWithName:@"gemma4_w6_embedding_gather_bf16"];
        id<MTLFunction> matmulFunction = [library newFunctionWithName:@"gemma4_w6a16_matmul_bf16"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        id<MTLComputePipelineState> gatherPipeline =
            [device newComputePipelineStateWithFunction:gatherFunction error:&error];
        id<MTLComputePipelineState> matmulPipeline =
            [device newComputePipelineStateWithFunction:matmulFunction error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !library || !function || !gatherFunction || !matmulFunction || !pipeline ||
            !gatherPipeline || !matmulPipeline || !queue) {
            fail(error.localizedDescription.UTF8String ?: "W6 Metal setup failed");
        }

        std::vector<uint16_t> input(columns);
        for (uint32_t column = 0; column < columns; ++column) {
            input[column] = toBfloat16((static_cast<int>(column % 9) - 4) * 0.125F);
        }
        std::vector<uint8_t> packed(rows * rowBytes, 0);
        std::vector<uint16_t> scales(rows * groups);
        std::vector<uint16_t> biases(rows * groups);
        std::vector<uint16_t> expected(rows);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const size_t index = static_cast<size_t>(row) * groups + group;
                scales[index] = toBfloat16(
                    0.015625F * static_cast<float>(row + 1) + 0.00390625F * static_cast<float>(group));
                biases[index] = toBfloat16(
                    -0.25F + 0.0625F * static_cast<float>(row) + 0.03125F * static_cast<float>(group));
            }
            float sum = 0.0F;
            for (uint32_t column = 0; column < columns; ++column) {
                const uint8_t quantized = static_cast<uint8_t>((column * 7 + row * 11) & 0x3F);
                pack6(packed, row * columns + column, quantized);
                const size_t group = static_cast<size_t>(row) * groups + column / 64;
                const float weight =
                    quantized * fromBfloat16(scales[group]) + fromBfloat16(biases[group]);
                sum += fromBfloat16(input[column]) * weight;
            }
            expected[row] = toBfloat16(sum);
        }

        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                        length:input.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> weightBuffer = [device newBufferWithBytes:packed.data()
                                                         length:packed.size()
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> scaleBuffer = [device newBufferWithBytes:scales.data()
                                                        length:scales.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> biasBuffer = [device newBufferWithBytes:biases.data()
                                                       length:biases.size() * sizeof(uint16_t)
                                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:rows * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:inputBuffer offset:0 atIndex:0];
        [encoder setBuffer:weightBuffer offset:0 atIndex:1];
        [encoder setBuffer:scaleBuffer offset:0 atIndex:2];
        [encoder setBuffer:biasBuffer offset:0 atIndex:3];
        [encoder setBuffer:outputBuffer offset:0 atIndex:4];
        [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
        [encoder setBytes:&columns length:sizeof(columns) atIndex:6];
        constexpr uint64_t zeroOffset = 0;
        [encoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:7];
        [encoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:8];
        [encoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) fail("W6 Metal command failed");

        const auto *actual = static_cast<const uint16_t *>(outputBuffer.contents);
        for (uint32_t row = 0; row < rows; ++row) {
            if (actual[row] != expected[row]) {
                fail("W6 fused matvec mismatch at row " + std::to_string(row));
            }
        }

        constexpr uint32_t tokenCount = 2;
        constexpr uint32_t vocabulary = rows;
        constexpr uint32_t columnOffset = 64;
        constexpr uint32_t outputColumns = 64;
        constexpr float outputScale = 1.5F;
        const std::array<uint32_t, tokenCount> tokenIds{3, 1};
        std::vector<uint16_t> gatherExpected(tokenCount * outputColumns);
        for (uint32_t token = 0; token < tokenCount; ++token) {
            const uint32_t row = tokenIds[token];
            for (uint32_t column = 0; column < outputColumns; ++column) {
                const uint32_t sourceColumn = columnOffset + column;
                const uint8_t quantized = static_cast<uint8_t>((sourceColumn * 7 + row * 11) & 0x3F);
                const size_t group = static_cast<size_t>(row) * groups + sourceColumn / 64;
                const float value =
                    quantized * fromBfloat16(scales[group]) + fromBfloat16(biases[group]);
                gatherExpected[token * outputColumns + column] = toBfloat16(value * outputScale);
            }
        }
        id<MTLBuffer> tokenBuffer = [device newBufferWithBytes:tokenIds.data()
                                                         length:tokenIds.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> gatherOutput = [device newBufferWithLength:gatherExpected.size() * sizeof(uint16_t)
                                                           options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> gatherCommands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> gatherEncoder = [gatherCommands computeCommandEncoder];
        [gatherEncoder setComputePipelineState:gatherPipeline];
        [gatherEncoder setBuffer:tokenBuffer offset:0 atIndex:0];
        [gatherEncoder setBuffer:weightBuffer offset:0 atIndex:1];
        [gatherEncoder setBuffer:scaleBuffer offset:0 atIndex:2];
        [gatherEncoder setBuffer:biasBuffer offset:0 atIndex:3];
        [gatherEncoder setBuffer:gatherOutput offset:0 atIndex:4];
        [gatherEncoder setBytes:&tokenCount length:sizeof(tokenCount) atIndex:5];
        [gatherEncoder setBytes:&vocabulary length:sizeof(vocabulary) atIndex:6];
        [gatherEncoder setBytes:&columns length:sizeof(columns) atIndex:7];
        [gatherEncoder setBytes:&columnOffset length:sizeof(columnOffset) atIndex:8];
        [gatherEncoder setBytes:&outputColumns length:sizeof(outputColumns) atIndex:9];
        [gatherEncoder setBytes:&outputScale length:sizeof(outputScale) atIndex:10];
        [gatherEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:11];
        [gatherEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:12];
        [gatherEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:13];
        const NSUInteger gatherWidth = std::min<NSUInteger>(gatherPipeline.maxTotalThreadsPerThreadgroup, 256);
        [gatherEncoder dispatchThreads:MTLSizeMake(gatherExpected.size(), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(gatherWidth, 1, 1)];
        [gatherEncoder endEncoding];
        [gatherCommands commit];
        [gatherCommands waitUntilCompleted];
        if (gatherCommands.status != MTLCommandBufferStatusCompleted) fail("W6 gather command failed");
        const auto *gatherActual = static_cast<const uint16_t *>(gatherOutput.contents);
        for (size_t index = 0; index < gatherExpected.size(); ++index) {
            if (gatherActual[index] != gatherExpected[index]) {
                fail("W6 fused embedding gather mismatch at element " + std::to_string(index));
            }
        }

        constexpr uint32_t matrixRows = 17;
        std::vector<uint16_t> matrixInput(matrixRows * columns);
        for (uint32_t row = 0; row < matrixRows; ++row) {
            for (uint32_t column = 0; column < columns; ++column) {
                matrixInput[static_cast<size_t>(row) * columns + column] =
                    toBfloat16((static_cast<int>((row * 5 + column) % 13) - 6) * 0.03125F);
            }
        }
        id<MTLBuffer> matrixInputBuffer = [device newBufferWithBytes:matrixInput.data()
                                                               length:matrixInput.size() * sizeof(uint16_t)
                                                              options:MTLResourceStorageModeShared];
        id<MTLBuffer> matrixOutputBuffer = [device newBufferWithLength:matrixRows * rows * sizeof(uint16_t)
                                                                  options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> matrixCommands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> matrixEncoder = [matrixCommands computeCommandEncoder];
        [matrixEncoder setComputePipelineState:matmulPipeline];
        [matrixEncoder setBuffer:matrixInputBuffer offset:0 atIndex:0];
        [matrixEncoder setBuffer:weightBuffer offset:0 atIndex:1];
        [matrixEncoder setBuffer:scaleBuffer offset:0 atIndex:2];
        [matrixEncoder setBuffer:biasBuffer offset:0 atIndex:3];
        [matrixEncoder setBuffer:matrixOutputBuffer offset:0 atIndex:4];
        [matrixEncoder setBytes:&matrixRows length:sizeof(matrixRows) atIndex:5];
        [matrixEncoder setBytes:&rows length:sizeof(rows) atIndex:6];
        [matrixEncoder setBytes:&columns length:sizeof(columns) atIndex:7];
        [matrixEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:8];
        [matrixEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:9];
        [matrixEncoder setBytes:&zeroOffset length:sizeof(zeroOffset) atIndex:10];
        [matrixEncoder dispatchThreadgroups:MTLSizeMake((rows + 63) / 64, (matrixRows + 15) / 16, 1)
                                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [matrixEncoder endEncoding];
        [matrixCommands commit];
        [matrixCommands waitUntilCompleted];
        if (matrixCommands.status != MTLCommandBufferStatusCompleted) fail("W6 matmul command failed");
        const auto *matrixActual = static_cast<const uint16_t *>(matrixOutputBuffer.contents);
        for (uint32_t matrixRow = 0; matrixRow < matrixRows; ++matrixRow) {
            for (uint32_t outputRow = 0; outputRow < rows; ++outputRow) {
                float sum = 0.0F;
                for (uint32_t column = 0; column < columns; ++column) {
                    const uint8_t quantized = static_cast<uint8_t>((column * 7 + outputRow * 11) & 0x3F);
                    const size_t group = static_cast<size_t>(outputRow) * groups + column / 64;
                    const float weight = quantized * fromBfloat16(scales[group]) + fromBfloat16(biases[group]);
                    sum += fromBfloat16(matrixInput[static_cast<size_t>(matrixRow) * columns + column]) * weight;
                }
                const float actualValue =
                    fromBfloat16(matrixActual[static_cast<size_t>(matrixRow) * rows + outputRow]);
                const float expectedValue = fromBfloat16(toBfloat16(sum));
                if (std::abs(actualValue - expectedValue) > 0.125F) {
                    fail("W6 fused matmul mismatch at output element");
                }
            }
        }
        std::cout << "gemma4 fused W6A16 Metal matvec, matmul, and embedding gather: PASS\n";
        return EXIT_SUCCESS;
    }
}
