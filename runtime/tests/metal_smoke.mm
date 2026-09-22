#include "gemma_runtime/DeviceCapabilities.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <iostream>

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 2) {
            std::cerr << "usage: metal_smoke <metallib>\n";
            return EXIT_FAILURE;
        }
        const auto capabilities = gemma_runtime::DeviceCapabilities::querySystemDefault();
        if (const auto failure = capabilities.validationError()) {
            std::cerr << capabilities.json() << '\n';
            return EXIT_FAILURE;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        id<MTLLibrary> library = [device newLibraryWithURL:url error:&error];
        id<MTLFunction> function = [library newFunctionWithName:@"add_one"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function error:&error];
        id<MTLCommandQueue> queue = [device newCommandQueue];
        const uint32_t original[] = {1, 2, 3, 4};
        id<MTLBuffer> buffer = [device newBufferWithBytes:original
                                                   length:sizeof(original)
                                                  options:MTLResourceStorageModeShared];
        if (!library || !function || !pipeline || !queue || !buffer) {
            std::cerr << "Metal setup failed: "
                      << (error.localizedDescription.UTF8String ?: "unknown error") << '\n';
            return EXIT_FAILURE;
        }
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer offset:0 atIndex:0];
        [encoder dispatchThreads:MTLSizeMake(4, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
        [encoder endEncoding];
        [commands commit];
        [commands waitUntilCompleted];
        if (commands.status != MTLCommandBufferStatusCompleted) {
            std::cerr << "Metal command failed\n";
            return EXIT_FAILURE;
        }
        const auto *values = static_cast<const uint32_t *>(buffer.contents);
        for (uint32_t index = 0; index < 4; ++index) {
            if (values[index] != original[index] + 1) {
                std::cerr << "Metal result mismatch\n";
                return EXIT_FAILURE;
            }
        }
        std::cout << capabilities.json() << '\n';
        std::cout << "native metal smoke: PASS\n";
        return EXIT_SUCCESS;
    }
}
