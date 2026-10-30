// Derived from incoai/splash at 298852ec603a5ef3848e8fb6d58d6d81746713ac.
// Modified for Gemma 4 E4B, Apple7-Apple10, and fail-closed family policies.
// Licensed under Apache-2.0; see runtime/third_party/splash/LICENSE.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace gemma_runtime {

struct DeviceCapabilities {
    static constexpr uint32_t kMinimumMacosMajor = 15;
    static constexpr uint32_t kMinimumMacosMinor = 0;
    static constexpr uint32_t kMinimumAppleGpuFamily = 7;
    static constexpr uint32_t kMaximumAppleGpuFamily = 10;
    static constexpr const char *kMetalLanguageVersion = "3.1";

    std::string deviceName = "unknown";
    uint32_t macosMajor = 0;
    uint32_t macosMinor = 0;
    uint32_t macosPatch = 0;
    uint32_t appleGpuFamily = 0;
    uint32_t gpuCoreCount = 0;
    uint64_t physicalMemoryBytes = 0;
    uint64_t recommendedMaxWorkingSetBytes = 0;
    uint64_t maxBufferLengthBytes = 0;
    uint64_t maxThreadgroupMemoryBytes = 0;
    uint64_t maxThreadgroupWidth = 0;
    bool hasUnifiedMemory = false;

    [[nodiscard]] bool meetsMinimumMacos() const noexcept;
    [[nodiscard]] std::string macosVersion() const;
    [[nodiscard]] std::string selectedPolicy() const;
    [[nodiscard]] std::optional<std::string> validationError() const;
    [[nodiscard]] std::string json() const;

    static DeviceCapabilities querySystemDefault();
};

} // namespace gemma_runtime
