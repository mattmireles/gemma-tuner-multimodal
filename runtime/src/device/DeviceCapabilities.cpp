// Derived from incoai/splash at 298852ec603a5ef3848e8fb6d58d6d81746713ac.
// Modified for Gemma 4 E4B, Apple7-Apple10, and stable JSON diagnostics.
// Licensed under Apache-2.0; see runtime/third_party/splash/LICENSE.
#include "gemma_runtime/DeviceCapabilities.hpp"

#include <sstream>

namespace gemma_runtime {
namespace {

std::string escapeJson(const std::string &value) {
    std::string result;
    result.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') result.push_back('\\');
        result.push_back(character);
    }
    return result;
}

} // namespace

bool DeviceCapabilities::meetsMinimumMacos() const noexcept {
    return macosMajor > kMinimumMacosMajor ||
           (macosMajor == kMinimumMacosMajor && macosMinor >= kMinimumMacosMinor);
}

std::string DeviceCapabilities::macosVersion() const {
    return std::to_string(macosMajor) + '.' + std::to_string(macosMinor) + '.' +
           std::to_string(macosPatch);
}

std::string DeviceCapabilities::selectedPolicy() const {
    switch (appleGpuFamily) {
    case 7: return "apple7-conservative";
    case 8: return "apple8-conservative";
    case 9: return "apple9-conservative";
    case 10: return "apple10-conservative";
    default: return "unsupported";
    }
}

std::optional<std::string> DeviceCapabilities::validationError() const {
    if (!meetsMinimumMacos()) return "macos_15_0_required";
    if (appleGpuFamily < kMinimumAppleGpuFamily || appleGpuFamily > kMaximumAppleGpuFamily) {
        return "apple_gpu_family_7_through_10_required";
    }
    if (!physicalMemoryBytes) return "physical_memory_unavailable";
    if (!recommendedMaxWorkingSetBytes) return "recommended_working_set_unavailable";
    if (recommendedMaxWorkingSetBytes > physicalMemoryBytes) {
        return "recommended_working_set_exceeds_physical_memory";
    }
    if (!maxBufferLengthBytes) return "max_buffer_length_unavailable";
    if (maxThreadgroupMemoryBytes < 32 * 1024) return "threadgroup_memory_below_32_kib";
    if (maxThreadgroupWidth < 256) return "threadgroup_width_below_256";
    if (!hasUnifiedMemory) return "unified_memory_required";
    return std::nullopt;
}

std::string DeviceCapabilities::json() const {
    const auto failure = validationError();
    std::ostringstream stream;
    stream << "{\"device_name\":\"" << escapeJson(deviceName)
           << "\",\"macos\":\"" << macosVersion()
           << "\",\"metal_language\":\"" << kMetalLanguageVersion
           << "\",\"apple_gpu_family\":" << appleGpuFamily
           << ",\"gpu_cores\":" << gpuCoreCount
           << ",\"physical_memory_bytes\":" << physicalMemoryBytes
           << ",\"recommended_working_set_bytes\":" << recommendedMaxWorkingSetBytes
           << ",\"max_buffer_length_bytes\":" << maxBufferLengthBytes
           << ",\"max_threadgroup_memory_bytes\":" << maxThreadgroupMemoryBytes
           << ",\"max_threadgroup_width\":" << maxThreadgroupWidth
           << ",\"unified_memory\":" << (hasUnifiedMemory ? "true" : "false")
           << ",\"policy\":\"" << selectedPolicy() << "\",\"validation_error\":";
    if (failure) stream << '"' << *failure << '"';
    else stream << "null";
    stream << '}';
    return stream.str();
}

} // namespace gemma_runtime
