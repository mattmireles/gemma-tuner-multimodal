#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

enum class Precision { F16, BF16 };

uint16_t floatToF16(float value) {
    const _Float16 converted = static_cast<_Float16>(value);
    return std::bit_cast<uint16_t>(converted);
}

float f16ToFloat(uint16_t value) {
    return static_cast<float>(std::bit_cast<_Float16>(value));
}

uint16_t floatToBF16(float value) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float bf16ToFloat(uint16_t value) {
    return std::bit_cast<float>(static_cast<uint32_t>(value) << 16U);
}

void fail(const std::string &message) {
    std::cerr << "FAIL: " << message << '\n';
    std::exit(EXIT_FAILURE);
}

std::vector<float> runRms(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    Precision precision,
    std::span<const float> input,
    std::span<const float> scale) {
    if (input.size() != scale.size()) fail("RMS fixture shape mismatch");
    std::vector<uint16_t> inputBits(input.size());
    std::vector<uint16_t> scaleBits(scale.size());
    for (size_t index = 0; index < input.size(); ++index) {
        inputBits[index] = precision == Precision::F16 ? floatToF16(input[index]) : floatToBF16(input[index]);
        scaleBits[index] = precision == Precision::F16 ? floatToF16(scale[index]) : floatToBF16(scale[index]);
    }
    id<MTLBuffer> inputBuffer = [device newBufferWithBytes:inputBits.data()
                                                     length:inputBits.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> scaleBuffer = [device newBufferWithBytes:scaleBits.data()
                                                     length:scaleBits.size() * sizeof(uint16_t)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:inputBits.size() * sizeof(uint16_t)
                                                     options:MTLResourceStorageModeShared];
    NSString *name = precision == Precision::F16 ? @"gemma4_rms_norm_f16" : @"gemma4_rms_norm_bf16";
    id<MTLFunction> function = [library newFunctionWithName:name];
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!inputBuffer || !scaleBuffer || !outputBuffer || !function || !pipeline || !queue) {
        fail(error.localizedDescription.UTF8String ?: "Metal RMS setup failed");
    }
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    const uint32_t width = static_cast<uint32_t>(input.size());
    const float epsilon = 1.0e-6F;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:scaleBuffer offset:0 atIndex:1];
    [encoder setBuffer:outputBuffer offset:0 atIndex:2];
    [encoder setBytes:&width length:sizeof(width) atIndex:3];
    [encoder setBytes:&epsilon length:sizeof(epsilon) atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) fail("Metal RMS command failed");

    const auto *resultBits = static_cast<const uint16_t *>(outputBuffer.contents);
    std::vector<float> result(input.size());
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = precision == Precision::F16 ? f16ToFloat(resultBits[index]) : bf16ToFloat(resultBits[index]);
    }
    return result;
}

void requireClose(
    std::span<const float> actual,
    std::span<const float> expected,
    float tolerance,
    const char *label) {
    if (actual.size() != expected.size()) fail(std::string(label) + " size mismatch");
    for (size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance) {
            fail(std::string(label) + " mismatch at " + std::to_string(index));
        }
    }
}

} // namespace

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) fail("usage: gemma4_reference_metal_test <metallib>");
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        if (!device || !library) fail(error.localizedDescription.UTF8String ?: "unable to load metallib");

        const std::vector<float> input{1, -2, 3, -4};
        const std::vector<float> scale{0.5F, 1.0F, 1.5F, 2.0F};
        const std::vector<float> expected{
            0.1825741827F, -0.7302967310F, 1.6431677341F, -2.9211869240F};
        requireClose(runRms(device, library, Precision::F16, input, scale), expected, 0.003F, "FP16 RMSNorm");
        requireClose(runRms(device, library, Precision::BF16, input, scale), expected, 0.025F, "BF16 RMSNorm");
        std::cout << "gemma4 FP16/BF16 Metal reference: PASS\n";
        return EXIT_SUCCESS;
    }
}
