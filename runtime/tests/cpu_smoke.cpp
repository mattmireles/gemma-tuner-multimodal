#include "gemma_runtime/DeviceCapabilities.hpp"

#include <cstdlib>
#include <iostream>

int main() {
    gemma_runtime::DeviceCapabilities device;
    device.deviceName = "fixture";
    device.macosMajor = 15;
    device.macosMinor = 0;
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
    if (device.json().find("\"metal_language\":\"3.1\"") == std::string::npos) {
        std::cerr << "Metal compatibility lane is missing from diagnostics\n";
        return EXIT_FAILURE;
    }
    device.macosMajor = 14;
    device.macosMinor = 6;
    if (device.validationError() != "macos_15_0_required") {
        std::cerr << "unsupported macOS version did not fail closed\n";
        return EXIT_FAILURE;
    }
    device.macosMajor = 15;
    device.macosMinor = 0;
    device.appleGpuFamily = 11;
    if (device.validationError() != "apple_gpu_family_7_through_10_required") {
        std::cerr << "unknown GPU family did not fail closed\n";
        return EXIT_FAILURE;
    }
    std::cout << "native cpu smoke: PASS\n";
    return EXIT_SUCCESS;
}
