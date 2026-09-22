#include "gemma_runtime/DeviceCapabilities.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    gemma_runtime::DeviceCapabilities device;
    device.deviceName = "fixture";
    device.macosMajor = 26;
    device.macosMinor = 4;
    device.appleGpuFamily = 8;
    device.gpuCoreCount = 60;
    device.physicalMemoryBytes = 64ULL << 30;
    device.recommendedMaxWorkingSetBytes = 48ULL << 30;
    device.maxBufferLengthBytes = 16ULL << 30;
    device.maxThreadgroupMemoryBytes = 32ULL << 10;
    device.maxThreadgroupWidth = 1024;
    device.hasUnifiedMemory = true;
    if (device.validationError() || device.selectedPolicy() != "apple8-conservative") {
        std::cerr << device.json() << '\n';
        return EXIT_FAILURE;
    }
    device.appleGpuFamily = 11;
    if (device.validationError() != "apple_gpu_family_7_through_10_required") {
        std::cerr << "unknown GPU family did not fail closed\n";
        return EXIT_FAILURE;
    }
    std::cout << "native cpu smoke: PASS\n";
    return EXIT_SUCCESS;
}
