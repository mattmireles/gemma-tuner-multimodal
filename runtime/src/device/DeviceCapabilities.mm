// Derived from incoai/splash at 298852ec603a5ef3848e8fb6d58d6d81746713ac.
// Modified to isolate public Metal/Foundation/IOKit device queries for Gemma.
// Licensed under Apache-2.0; see runtime/third_party/splash/LICENSE.
#include "gemma_runtime/DeviceCapabilities.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace gemma_runtime {
namespace {

uint32_t gpuCoreCountForDevice(uint64_t registryId) noexcept {
    uint32_t count = 0;
    const auto read = [&](io_registry_entry_t entry) {
        if (!entry) return false;
        CFTypeRef value = IORegistryEntryCreateCFProperty(
            entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        if (value) {
            int64_t number = 0;
            if (CFGetTypeID(value) == CFNumberGetTypeID() &&
                CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberSInt64Type, &number) &&
                number > 0 && number <= 4096) {
                count = static_cast<uint32_t>(number);
            }
            CFRelease(value);
        }
        return count != 0;
    };
    io_registry_entry_t entry = IOServiceGetMatchingService(
        kIOMainPortDefault, IORegistryEntryIDMatching(registryId));
    for (int depth = 0; entry && depth < 4 && !read(entry); ++depth) {
        io_registry_entry_t parent = MACH_PORT_NULL;
        if (IORegistryEntryGetParentEntry(entry, kIOServicePlane, &parent) != KERN_SUCCESS) {
            parent = MACH_PORT_NULL;
        }
        IOObjectRelease(entry);
        entry = parent;
    }
    if (entry) IOObjectRelease(entry);
    if (!count) {
        io_registry_entry_t accelerator = IOServiceGetMatchingService(
            kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
        if (accelerator) {
            read(accelerator);
            IOObjectRelease(accelerator);
        }
    }
    return count;
}

std::string stringFromNSString(NSString *value) {
    if (!value) return {};
    const char *utf8 = value.UTF8String;
    return utf8 ? utf8 : "";
}

} // namespace

DeviceCapabilities DeviceCapabilities::querySystemDefault() {
    @autoreleasepool {
        DeviceCapabilities result;
        NSOperatingSystemVersion version = NSProcessInfo.processInfo.operatingSystemVersion;
        result.macosMajor = static_cast<uint32_t>(version.majorVersion);
        result.macosMinor = static_cast<uint32_t>(version.minorVersion);
        result.macosPatch = static_cast<uint32_t>(version.patchVersion);
        result.physicalMemoryBytes = NSProcessInfo.processInfo.physicalMemory;

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return result;
        result.deviceName = stringFromNSString(device.name);
        result.gpuCoreCount = gpuCoreCountForDevice(device.registryID);
        for (uint32_t family = kMaximumAppleGpuFamily; family >= kMinimumAppleGpuFamily; --family) {
            if ([device supportsFamily:static_cast<MTLGPUFamily>(1000 + family)]) {
                result.appleGpuFamily = family;
                break;
            }
        }
        result.recommendedMaxWorkingSetBytes = device.recommendedMaxWorkingSetSize;
        result.maxBufferLengthBytes = device.maxBufferLength;
        result.maxThreadgroupMemoryBytes = device.maxThreadgroupMemoryLength;
        result.maxThreadgroupWidth = device.maxThreadsPerThreadgroup.width;
        result.hasUnifiedMemory = device.hasUnifiedMemory;
        return result;
    }
}

} // namespace gemma_runtime
