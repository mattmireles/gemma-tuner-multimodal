XCRUN := xcrun
CXX := $(XCRUN) -sdk macosx clang++
METAL := $(XCRUN) -sdk macosx metal
METALLIB := $(XCRUN) -sdk macosx metallib

BUILD := build
MACOS_MIN_VERSION := 26.4
TARGET_FLAG := -mmacosx-version-min=$(MACOS_MIN_VERSION)
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude $(TARGET_FLAG)
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
METALFLAGS := -std=metal4.0 -O2 -Wall -Wextra -Werror $(TARGET_FLAG)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit

CAPABILITIES_OBJECT := $(BUILD)/DeviceCapabilities.o
CAPABILITIES_QUERY_OBJECT := $(BUILD)/DeviceCapabilitiesQuery.o
CPU_SMOKE := $(BUILD)/cpu_smoke
METAL_SMOKE := $(BUILD)/metal_smoke
METAL_AIR := $(BUILD)/metal_smoke.air
METAL_LIBRARY := $(BUILD)/metal_smoke.metallib

.PHONY: all clean platform-check check-native-cpu check-native-metal

all: check-native-cpu check-native-metal

platform-check:
	@test "$$(uname -s)" = Darwin && test "$$(uname -m)" = arm64 || { \
		echo "error: native runtime requires an Apple Silicon Mac" >&2; exit 1; \
	}
	@for tool in clang++ metal metallib; do \
		$(XCRUN) -sdk macosx -f $$tool >/dev/null 2>&1 || { \
			echo "error: Xcode Metal toolchain is missing $$tool" >&2; exit 1; \
		}; \
	done

$(BUILD):
	mkdir -p $@

$(CAPABILITIES_OBJECT): src/device/DeviceCapabilities.cpp include/gemma_runtime/DeviceCapabilities.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CAPABILITIES_QUERY_OBJECT): src/device/DeviceCapabilities.mm include/gemma_runtime/DeviceCapabilities.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(BUILD)/cpu_smoke.o: tests/cpu_smoke.cpp include/gemma_runtime/DeviceCapabilities.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(CPU_SMOKE): $(CAPABILITIES_OBJECT) $(BUILD)/cpu_smoke.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(METAL_AIR): tests/metal_smoke.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(METAL_LIBRARY): $(METAL_AIR)
	$(METALLIB) $< -o $@

$(BUILD)/metal_smoke.o: tests/metal_smoke.mm include/gemma_runtime/DeviceCapabilities.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(METAL_SMOKE): $(CAPABILITIES_OBJECT) $(CAPABILITIES_QUERY_OBJECT) $(BUILD)/metal_smoke.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

check-native-cpu: platform-check $(CPU_SMOKE)
	$(CPU_SMOKE)

check-native-metal: platform-check $(METAL_LIBRARY) $(METAL_SMOKE)
	$(METAL_SMOKE) $(abspath $(METAL_LIBRARY))

clean:
	rm -rf $(BUILD)
