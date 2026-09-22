XCRUN := xcrun
CXX := $(XCRUN) -sdk macosx clang++
METAL := $(XCRUN) -sdk macosx metal
METALLIB := $(XCRUN) -sdk macosx metallib

BUILD := build
MACOS_MIN_VERSION := 15.0
METAL_LANGUAGE_VERSION := 3.1
TARGET_FLAG := -mmacosx-version-min=$(MACOS_MIN_VERSION)
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Werror -Iinclude $(TARGET_FLAG)
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc
METALFLAGS := -std=metal$(METAL_LANGUAGE_VERSION) -O2 -Wall -Wextra -Werror $(TARGET_FLAG)
FRAMEWORKS := -framework Foundation -framework Metal -framework IOKit
MPS_FRAMEWORKS := $(FRAMEWORKS) -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph

CAPABILITIES_OBJECT := $(BUILD)/DeviceCapabilities.o
CAPABILITIES_QUERY_OBJECT := $(BUILD)/DeviceCapabilitiesQuery.o
CPU_SMOKE := $(BUILD)/cpu_smoke
GEMMA4_BOUNDARIES := $(BUILD)/gemma4_boundaries_test
GEMMA4_REFERENCE := $(BUILD)/gemma4_reference_test
TENSOR_INDEX_TEST := $(BUILD)/tensor_index_test
W6_TENSOR_TEST := $(BUILD)/w6_tensor_test
TUNING_POLICY_TEST := $(BUILD)/tuning_policy_test
REFERENCE_INDEX_TOOL := $(BUILD)/validate-reference-index
W6_INDEX_TOOL := $(BUILD)/validate-w6-index
W6_BENCHMARK_TOOL := $(BUILD)/benchmark-w6-matvec
W6_GATHER_BENCHMARK_TOOL := $(BUILD)/benchmark-w6-gather
W6_MATMUL_BENCHMARK_TOOL := $(BUILD)/benchmark-w6-matmul
METAL_SMOKE := $(BUILD)/metal_smoke
METAL_AIR := $(BUILD)/metal_smoke.air
REFERENCE_METAL_AIR := $(BUILD)/gemma4_reference.air
W6_METAL_AIR := $(BUILD)/gemma4_w6.air
METAL_LIBRARY := $(BUILD)/metal_smoke.metallib
REFERENCE_METAL_TEST := $(BUILD)/gemma4_reference_metal_test
MPS_REFERENCE_TEST := $(BUILD)/mps_reference_test
W6_METAL_TEST := $(BUILD)/w6_metal_test
METAL_TENSOR_STORE_TEST := $(BUILD)/metal_tensor_store_test
W6_MAPPED_METAL_TEST := $(BUILD)/w6_mapped_metal_test
LAYER_PARITY_TEST := $(BUILD)/gemma4_layer_parity_test
PREFILL_PARITY_TEST := $(BUILD)/gemma4_prefill_parity_test
TRANSITION_PARITY_TEST := $(BUILD)/gemma4_transition_parity_test

.PHONY: all clean platform-check check-native-cpu check-native-metal check-layer-parity check-prefill-parity check-transition-parity

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

$(BUILD)/Gemma4Config.o: src/gemma4/Gemma4Config.cpp src/gemma4/Gemma4Config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/Gemma4Cache.o: src/gemma4/Gemma4Cache.cpp src/gemma4/Gemma4Cache.hpp src/gemma4/Gemma4Config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/gemma4_boundaries_test.o: tests/gemma4_boundaries_test.cpp src/gemma4/Gemma4Cache.hpp src/gemma4/Gemma4Config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(GEMMA4_BOUNDARIES): $(BUILD)/Gemma4Config.o $(BUILD)/Gemma4Cache.o $(BUILD)/gemma4_boundaries_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/Gemma4Weights.o: src/gemma4/Gemma4Weights.cpp src/gemma4/Gemma4Weights.hpp src/gemma4/Gemma4Config.hpp src/package/TensorIndex.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/Gemma4Decoder.o: src/gemma4/Gemma4Decoder.cpp src/gemma4/Gemma4Decoder.hpp src/gemma4/Gemma4Config.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/PerLayerEmbedding.o: src/gemma4/PerLayerEmbedding.cpp src/gemma4/PerLayerEmbedding.hpp src/gemma4/Gemma4Decoder.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/gemma4_reference_test.o: tests/gemma4_reference_test.cpp src/gemma4/Gemma4Decoder.hpp src/gemma4/Gemma4Weights.hpp src/gemma4/PerLayerEmbedding.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(GEMMA4_REFERENCE): $(BUILD)/Gemma4Config.o $(BUILD)/TensorIndex.o $(BUILD)/Gemma4Weights.o $(BUILD)/Gemma4Decoder.o $(BUILD)/PerLayerEmbedding.o $(BUILD)/gemma4_reference_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/TensorIndex.o: src/package/TensorIndex.cpp src/package/TensorIndex.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/TensorStore.o: src/package/TensorStore.cpp src/package/TensorStore.hpp src/package/TensorIndex.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/tensor_index_test.o: tests/tensor_index_test.cpp src/package/TensorIndex.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(TENSOR_INDEX_TEST): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/tensor_index_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/W6Tensor.o: src/package/W6Tensor.cpp src/package/W6Tensor.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/w6_tensor_test.o: tests/w6_tensor_test.cpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(W6_TENSOR_TEST): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/w6_tensor_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/TuningPolicy.o: src/device/TuningPolicy.cpp src/device/TuningPolicy.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/tuning_policy_test.o: tests/tuning_policy_test.cpp src/device/TuningPolicy.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(TUNING_POLICY_TEST): $(BUILD)/TuningPolicy.o $(BUILD)/tuning_policy_test.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/validate_reference_index.o: tools/validate_reference_index.cpp src/gemma4/Gemma4Weights.hpp src/package/TensorIndex.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(REFERENCE_INDEX_TOOL): $(BUILD)/TensorIndex.o $(BUILD)/Gemma4Config.o $(BUILD)/Gemma4Weights.o $(BUILD)/validate_reference_index.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/validate_w6_index.o: tools/validate_w6_index.cpp src/package/TensorStore.hpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(W6_INDEX_TOOL): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/validate_w6_index.o
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/benchmark_w6_matvec.o: tools/benchmark_w6_matvec.mm src/ops/W6MetalOps.hpp src/device/TuningPolicy.hpp src/package/TensorStore.hpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(W6_BENCHMARK_TOOL): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/W6MetalOps.o $(BUILD)/benchmark_w6_matvec.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/benchmark_w6_gather.o: tools/benchmark_w6_gather.mm src/package/TensorStore.hpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(W6_GATHER_BENCHMARK_TOOL): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/W6MetalOps.o $(BUILD)/benchmark_w6_gather.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/benchmark_w6_matmul.o: tools/benchmark_w6_matmul.mm src/ops/W6MetalOps.hpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(W6_MATMUL_BENCHMARK_TOOL): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/W6MetalOps.o $(BUILD)/benchmark_w6_matmul.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(METAL_AIR): tests/metal_smoke.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(REFERENCE_METAL_AIR): src/metal/gemma4_reference.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(W6_METAL_AIR): src/metal/gemma4_w6.metal | $(BUILD)
	$(METAL) $(METALFLAGS) -c $< -o $@

$(METAL_LIBRARY): $(METAL_AIR) $(REFERENCE_METAL_AIR) $(W6_METAL_AIR)
	$(METALLIB) $^ -o $@

$(BUILD)/metal_smoke.o: tests/metal_smoke.mm include/gemma_runtime/DeviceCapabilities.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(METAL_SMOKE): $(CAPABILITIES_OBJECT) $(CAPABILITIES_QUERY_OBJECT) $(BUILD)/metal_smoke.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/gemma4_reference_metal_test.o: tests/gemma4_reference_metal_test.mm | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(REFERENCE_METAL_TEST): $(BUILD)/gemma4_reference_metal_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/MpsReference.o: src/ops/MpsReference.mm src/ops/MpsReference.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/mps_reference_test.o: tests/mps_reference_test.mm src/ops/MpsReference.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(MPS_REFERENCE_TEST): $(BUILD)/MpsReference.o $(BUILD)/mps_reference_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(MPS_FRAMEWORKS) -o $@

$(BUILD)/W6MetalOps.o: src/ops/W6MetalOps.mm src/ops/W6MetalOps.hpp src/device/TuningPolicy.hpp src/package/TensorStore.hpp src/package/W6Tensor.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/w6_metal_test.o: tests/w6_metal_test.mm | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -c $< -o $@

$(W6_METAL_TEST): $(BUILD)/w6_metal_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/metal_tensor_store_test.o: tests/metal_tensor_store_test.mm src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(METAL_TENSOR_STORE_TEST): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/metal_tensor_store_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/w6_mapped_metal_test.o: tests/w6_mapped_metal_test.mm src/ops/W6MetalOps.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(W6_MAPPED_METAL_TEST): $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/W6Tensor.o $(BUILD)/W6MetalOps.o $(BUILD)/w6_mapped_metal_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(FRAMEWORKS) -o $@

$(BUILD)/Gemma4ReferenceGraph.o: src/gemma4/Gemma4ReferenceGraph.mm src/gemma4/Gemma4ReferenceGraph.hpp src/gemma4/Gemma4Config.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/gemma4_layer_parity_test.o: tests/gemma4_layer_parity_test.mm src/gemma4/Gemma4ReferenceGraph.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(LAYER_PARITY_TEST): $(BUILD)/Gemma4Config.o $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/Gemma4ReferenceGraph.o $(BUILD)/gemma4_layer_parity_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(MPS_FRAMEWORKS) -o $@

$(BUILD)/gemma4_prefill_parity_test.o: tests/gemma4_prefill_parity_test.mm src/gemma4/Gemma4Config.hpp src/gemma4/Gemma4ReferenceGraph.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(PREFILL_PARITY_TEST): $(BUILD)/Gemma4Config.o $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/Gemma4ReferenceGraph.o $(BUILD)/gemma4_prefill_parity_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(MPS_FRAMEWORKS) -o $@

$(BUILD)/gemma4_transition_parity_test.o: tests/gemma4_transition_parity_test.mm src/gemma4/Gemma4Config.hpp src/gemma4/Gemma4ReferenceGraph.hpp src/package/TensorStore.hpp | $(BUILD)
	$(CXX) $(OBJCXXFLAGS) -Isrc -c $< -o $@

$(TRANSITION_PARITY_TEST): $(BUILD)/Gemma4Config.o $(BUILD)/TensorIndex.o $(BUILD)/TensorStore.o $(BUILD)/Gemma4ReferenceGraph.o $(BUILD)/gemma4_transition_parity_test.o
	$(CXX) $(OBJCXXFLAGS) $^ $(MPS_FRAMEWORKS) -o $@

check-native-cpu: platform-check $(CPU_SMOKE) $(GEMMA4_BOUNDARIES) $(GEMMA4_REFERENCE) $(TENSOR_INDEX_TEST) $(W6_TENSOR_TEST) $(TUNING_POLICY_TEST)
	$(CPU_SMOKE)
	$(GEMMA4_BOUNDARIES)
	$(GEMMA4_REFERENCE)
	$(TENSOR_INDEX_TEST)
	$(W6_TENSOR_TEST)
	$(TUNING_POLICY_TEST)

check-native-metal: platform-check $(METAL_LIBRARY) $(METAL_SMOKE) $(REFERENCE_METAL_TEST) $(MPS_REFERENCE_TEST) $(W6_METAL_TEST) $(METAL_TENSOR_STORE_TEST) $(W6_MAPPED_METAL_TEST)
	$(METAL_SMOKE) $(abspath $(METAL_LIBRARY))
	$(REFERENCE_METAL_TEST) $(abspath $(METAL_LIBRARY))
	$(MPS_REFERENCE_TEST)
	$(W6_METAL_TEST) $(abspath $(METAL_LIBRARY))
	$(METAL_TENSOR_STORE_TEST)
	$(W6_MAPPED_METAL_TEST) $(abspath $(METAL_LIBRARY))

check-layer-parity: platform-check $(LAYER_PARITY_TEST)
	@test -n "$(MODEL_INDEX)" -a -n "$(MODEL_ROOT)" -a -n "$(ORACLE_INDEX)" -a -n "$(ORACLE_ROOT)" -a -n "$(LAYER)" || { \
		echo "error: MODEL_INDEX, MODEL_ROOT, ORACLE_INDEX, ORACLE_ROOT, and LAYER are required" >&2; exit 1; \
	}
	$(LAYER_PARITY_TEST) "$(MODEL_INDEX)" "$(MODEL_ROOT)" "$(ORACLE_INDEX)" "$(ORACLE_ROOT)" "$(LAYER)" $(MODE)

check-prefill-parity: platform-check $(PREFILL_PARITY_TEST)
	@test -n "$(MODEL_INDEX)" -a -n "$(MODEL_ROOT)" -a -n "$(ORACLE_INDEX)" -a -n "$(ORACLE_ROOT)" || { \
		echo "error: MODEL_INDEX, MODEL_ROOT, ORACLE_INDEX, and ORACLE_ROOT are required" >&2; exit 1; \
	}
	$(PREFILL_PARITY_TEST) "$(MODEL_INDEX)" "$(MODEL_ROOT)" "$(ORACLE_INDEX)" "$(ORACLE_ROOT)"

check-transition-parity: platform-check $(TRANSITION_PARITY_TEST)
	@test -n "$(MODEL_INDEX)" -a -n "$(MODEL_ROOT)" -a -n "$(ORACLE_INDEX)" -a -n "$(ORACLE_ROOT)" || { \
		echo "error: MODEL_INDEX, MODEL_ROOT, ORACLE_INDEX, and ORACLE_ROOT are required" >&2; exit 1; \
	}
	$(TRANSITION_PARITY_TEST) "$(MODEL_INDEX)" "$(MODEL_ROOT)" "$(ORACLE_INDEX)" "$(ORACLE_ROOT)"

clean:
	rm -rf $(BUILD)
