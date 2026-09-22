#include "ops/W6MetalOps.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

void writeBfloat16(std::vector<uint8_t> &bytes, size_t offset, float value) {
    const uint16_t bits = toBfloat16(value);
    bytes[offset] = static_cast<uint8_t>(bits & 0xFFU);
    bytes[offset + 1] = static_cast<uint8_t>(bits >> 8U);
}

void pack6(std::vector<uint8_t> &bytes, size_t base, size_t index, uint8_t value) {
    const size_t bit = index * 6;
    const size_t byte = base + bit / 8;
    const size_t shift = bit % 8;
    const uint16_t window = static_cast<uint16_t>(value & 0x3FU) << shift;
    bytes[byte] |= static_cast<uint8_t>(window & 0xFFU);
    bytes[byte + 1] |= static_cast<uint8_t>(window >> 8U);
}

} // namespace

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) fail("usage: w6_mapped_metal_test <metallib>");
        constexpr uint32_t rows = 4;
        constexpr uint32_t columns = 256;
        constexpr uint32_t groups = columns / 64;
        constexpr size_t weightOffset = 1;
        constexpr size_t weightBytes = rows * columns * 6 / 8;
        constexpr size_t scaleOffset = weightOffset + weightBytes + 2;
        constexpr size_t auxiliaryBytes = rows * groups * sizeof(uint16_t);
        constexpr size_t biasOffset = scaleOffset + auxiliaryBytes + 2;
        constexpr size_t fileBytes = biasOffset + auxiliaryBytes;

        const std::filesystem::path root =
            std::filesystem::temp_directory_path() / "gemma-runtime-w6-mapped-metal-test";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root / "model");
        std::filesystem::create_directories(root / "metadata");
        std::vector<uint8_t> bytes(fileBytes, 0);
        std::array<float, rows * groups> scales{};
        std::array<float, rows * groups> biases{};
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t group = 0; group < groups; ++group) {
                const size_t index = static_cast<size_t>(row) * groups + group;
                scales[index] = fromBfloat16(toBfloat16(
                    0.015625F * static_cast<float>(row + 1) + 0.00390625F * static_cast<float>(group)));
                biases[index] = fromBfloat16(toBfloat16(
                    -0.25F + 0.0625F * static_cast<float>(row) + 0.03125F * static_cast<float>(group)));
                writeBfloat16(bytes, scaleOffset + index * 2, scales[index]);
                writeBfloat16(bytes, biasOffset + index * 2, biases[index]);
            }
            for (uint32_t column = 0; column < columns; ++column) {
                pack6(
                    bytes,
                    weightOffset,
                    static_cast<size_t>(row) * columns + column,
                    static_cast<uint8_t>((column * 7 + row * 11) & 0x3F));
            }
        }
        {
            std::ofstream shard(root / "model/weights.bin", std::ios::binary);
            shard.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        }
        {
            std::ofstream index(root / "metadata/tensor-index.tsv");
            index << "gemma4-tensor-index-v1\n";
            index << "x.weight\tU32\tmodel/weights.bin\t" << weightOffset << '\t' << weightBytes
                  << "\t2\t" << rows << ",48\n";
            index << "x.scales\tBF16\tmodel/weights.bin\t" << scaleOffset << '\t' << auxiliaryBytes
                  << "\t2\t" << rows << ',' << groups << "\n";
            index << "x.biases\tBF16\tmodel/weights.bin\t" << biasOffset << '\t' << auxiliaryBytes
                  << "\t2\t" << rows << ',' << groups << "\n";
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!device || !library || !queue) fail(error.localizedDescription.UTF8String ?: "Metal setup failed");
        gemma_runtime::ops::W6MetalOps ops(
            device,
            library,
            gemma_runtime::package::TensorStore::open(root));
        if (ops.mappedShardCount() != 1 || ops.mappedFileBytes() != fileBytes || ops.copiedWeightBytes() != 0) {
            fail("W6 operator did not retain the expected no-copy shard mapping");
        }

        std::vector<uint16_t> input(columns);
        for (uint32_t column = 0; column < columns; ++column) {
            input[column] = toBfloat16((static_cast<int>(column % 9) - 4) * 0.125F);
        }
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input.data()
                                                        length:input.size() * sizeof(uint16_t)
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:rows * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        ops.encodeMatvec(encoder, inputBuffer, 0, outputBuffer, 0, "x");
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) fail("mapped W6 matvec command failed");
        const auto *matvec = static_cast<const uint16_t *>(outputBuffer.contents);
        for (uint32_t row = 0; row < rows; ++row) {
            float expected = 0.0F;
            for (uint32_t column = 0; column < columns; ++column) {
                const uint8_t quantized = static_cast<uint8_t>((column * 7 + row * 11) & 0x3F);
                const size_t group = static_cast<size_t>(row) * groups + column / 64;
                expected += fromBfloat16(input[column]) * (quantized * scales[group] + biases[group]);
            }
            if (matvec[row] != toBfloat16(expected)) fail("mapped W6 matvec mismatch");
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
        ops.encodeMatmul(matrixEncoder, matrixInputBuffer, 0, matrixOutputBuffer, 0, "x", matrixRows);
        [matrixEncoder endEncoding];
        [matrixCommands commit];
        [matrixCommands waitUntilCompleted];
        if (matrixCommands.status != MTLCommandBufferStatusCompleted) fail("mapped W6 matmul command failed");
        const auto *matrix = static_cast<const uint16_t *>(matrixOutputBuffer.contents);
        for (uint32_t matrixRow = 0; matrixRow < matrixRows; ++matrixRow) {
            for (uint32_t outputRow = 0; outputRow < rows; ++outputRow) {
                float expected = 0.0F;
                for (uint32_t column = 0; column < columns; ++column) {
                    const uint8_t quantized = static_cast<uint8_t>((column * 7 + outputRow * 11) & 0x3F);
                    const size_t group = static_cast<size_t>(outputRow) * groups + column / 64;
                    expected += fromBfloat16(matrixInput[static_cast<size_t>(matrixRow) * columns + column]) *
                                (quantized * scales[group] + biases[group]);
                }
                const float actual = fromBfloat16(matrix[static_cast<size_t>(matrixRow) * rows + outputRow]);
                if (std::abs(actual - fromBfloat16(toBfloat16(expected))) > 0.125F) {
                    fail("mapped W6 matmul mismatch");
                }
            }
        }

        constexpr uint32_t tokenCount = 2;
        constexpr uint32_t gatherOffset = 64;
        constexpr uint32_t gatherColumns = 64;
        constexpr float gatherScale = 1.5F;
        const std::array<uint32_t, tokenCount> tokenIds{3, 1};
        id<MTLBuffer> tokenBuffer = [device newBufferWithBytes:tokenIds.data()
                                                         length:tokenIds.size() * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> gatherBuffer = [device newBufferWithLength:tokenCount * gatherColumns * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> gatherCommands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> gatherEncoder = [gatherCommands computeCommandEncoder];
        ops.encodeGather(
            gatherEncoder,
            tokenBuffer,
            0,
            gatherBuffer,
            0,
            "x",
            tokenCount,
            gatherOffset,
            gatherColumns,
            gatherScale);
        [gatherEncoder endEncoding];
        [gatherCommands commit];
        [gatherCommands waitUntilCompleted];
        if (gatherCommands.status != MTLCommandBufferStatusCompleted) fail("mapped W6 gather command failed");
        const auto *gather = static_cast<const uint16_t *>(gatherBuffer.contents);
        for (uint32_t token = 0; token < tokenCount; ++token) {
            const uint32_t row = tokenIds[token];
            for (uint32_t column = 0; column < gatherColumns; ++column) {
                const uint32_t sourceColumn = gatherOffset + column;
                const uint8_t quantized = static_cast<uint8_t>((sourceColumn * 7 + row * 11) & 0x3F);
                const size_t group = static_cast<size_t>(row) * groups + sourceColumn / 64;
                const uint16_t expected =
                    toBfloat16((quantized * scales[group] + biases[group]) * gatherScale);
                if (gather[token * gatherColumns + column] != expected) fail("mapped W6 gather mismatch");
            }
        }
        std::filesystem::remove_all(root);
        std::cout << "gemma4 mmap-backed W6 Metal operators: PASS\n";
        return EXIT_SUCCESS;
    }
}
