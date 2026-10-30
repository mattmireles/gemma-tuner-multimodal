#include "ops/W6MetalOps.hpp"

#include "package/W6Tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace gemma_runtime::ops {
namespace {

struct MetalTensorSlice {
    id<MTLBuffer> buffer;
    uint64_t offset;
    uint64_t length;
};

id<MTLComputePipelineState> makePipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) throw std::runtime_error("Metal library is missing W6 function: " + std::string(name.UTF8String));
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        const char *description = error.localizedDescription.UTF8String;
        throw std::runtime_error(description ? description : "unable to create W6 Metal pipeline");
    }
    return pipeline;
}

} // namespace

struct W6MetalOps::Impl {
    explicit Impl(
        id<MTLDevice> sourceDevice,
        id<MTLLibrary> library,
        package::TensorStore sourceStore)
        : device(sourceDevice),
          store(std::move(sourceStore)),
          matvecPipeline(makePipeline(device, library, @"gemma4_w6a16_matvec_bf16")),
          narrowMatvecPipeline(makePipeline(device, library, @"gemma4_w6a16_matvec_narrow_bf16")),
          matmulPipeline(makePipeline(device, library, @"gemma4_w6a16_matmul_tiled_bf16")),
          gatherPipeline(makePipeline(device, library, @"gemma4_w6_embedding_gather_bf16")) {
        if (!device || !library) throw std::invalid_argument("W6 Metal ops require a device and library");
        std::set<std::filesystem::path> paths;
        for (const auto &[name, descriptor] : store.index().entries()) {
            static_cast<void>(name);
            paths.insert(descriptor.relativePath);
        }
        for (const auto &path : paths) {
            const package::MappedFileView mapped = store.mappedFile(path);
            if (mapped.allocationLength > device.maxBufferLength) {
                throw std::runtime_error("tensor shard exceeds the Metal device buffer limit: " + path.string());
            }
            id<MTLBuffer> buffer = [device newBufferWithBytesNoCopy:const_cast<std::byte *>(mapped.data)
                                                             length:mapped.allocationLength
                                                            options:MTLResourceStorageModeShared
                                                        deallocator:nil];
            if (!buffer) throw std::runtime_error("Metal rejected no-copy tensor shard: " + path.string());
            buffers.emplace(path, buffer);
            fileBytes += mapped.byteLength;
        }
    }

    MetalTensorSlice slice(const std::string &name) const {
        const package::TensorDescriptor &descriptor = store.index().at(name);
        const auto found = buffers.find(descriptor.relativePath);
        if (found == buffers.end()) throw std::logic_error("Metal tensor shard is not resident: " + name);
        return MetalTensorSlice{
            .buffer = found->second,
            .offset = descriptor.fileOffset,
            .length = descriptor.byteLength,
        };
    }

    id<MTLDevice> device;
    package::TensorStore store;
    id<MTLComputePipelineState> matvecPipeline;
    id<MTLComputePipelineState> narrowMatvecPipeline;
    id<MTLComputePipelineState> matmulPipeline;
    id<MTLComputePipelineState> gatherPipeline;
    std::map<std::filesystem::path, id<MTLBuffer>> buffers;
    uint64_t fileBytes = 0;
};

W6MetalOps::W6MetalOps(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    package::TensorStore tensorStore)
    : impl_(std::make_unique<Impl>(device, library, std::move(tensorStore))) {}

W6MetalOps::W6MetalOps(W6MetalOps &&) noexcept = default;
W6MetalOps &W6MetalOps::operator=(W6MetalOps &&) noexcept = default;
W6MetalOps::~W6MetalOps() = default;

void W6MetalOps::encodeMatvec(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> input,
    NSUInteger inputOffset,
    id<MTLBuffer> output,
    NSUInteger outputOffset,
    const std::string &baseName,
    W6MatvecVariant variant) {
    if (!encoder || !input || !output) throw std::invalid_argument("W6 matvec received a nil Metal object");
    const package::W6Tensor tensor = package::W6Tensor::open(impl_->store, baseName);
    if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX || tensor.columns() % 256 != 0) {
        throw std::invalid_argument("W6 matvec geometry is unsupported: " + baseName);
    }
    const MetalTensorSlice weight = impl_->slice(baseName + ".weight");
    const MetalTensorSlice scales = impl_->slice(baseName + ".scales");
    const MetalTensorSlice biases = impl_->slice(baseName + ".biases");
    const uint32_t rows = static_cast<uint32_t>(tensor.rows());
    const uint32_t columns = static_cast<uint32_t>(tensor.columns());
    [encoder setComputePipelineState:
        variant == W6MatvecVariant::Wide8 ? impl_->matvecPipeline : impl_->narrowMatvecPipeline];
    [encoder setBuffer:input offset:inputOffset atIndex:0];
    [encoder setBuffer:weight.buffer offset:0 atIndex:1];
    [encoder setBuffer:scales.buffer offset:0 atIndex:2];
    [encoder setBuffer:biases.buffer offset:0 atIndex:3];
    [encoder setBuffer:output offset:outputOffset atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&columns length:sizeof(columns) atIndex:6];
    [encoder setBytes:&weight.offset length:sizeof(weight.offset) atIndex:7];
    [encoder setBytes:&scales.offset length:sizeof(scales.offset) atIndex:8];
    [encoder setBytes:&biases.offset length:sizeof(biases.offset) atIndex:9];
    [encoder dispatchThreadgroups:MTLSizeMake((rows + 7) / 8, 1, 1)
                  threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
}

void W6MetalOps::encodeMatmul(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> input,
    NSUInteger inputOffset,
    id<MTLBuffer> output,
    NSUInteger outputOffset,
    const std::string &baseName,
    uint32_t rows) {
    if (!encoder || !input || !output) throw std::invalid_argument("W6 matmul received a nil Metal object");
    const package::W6Tensor tensor = package::W6Tensor::open(impl_->store, baseName);
    if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX || tensor.columns() % 8 != 0 || rows == 0) {
        throw std::invalid_argument("W6 matmul geometry is unsupported: " + baseName);
    }
    const MetalTensorSlice weight = impl_->slice(baseName + ".weight");
    const MetalTensorSlice scales = impl_->slice(baseName + ".scales");
    const MetalTensorSlice biases = impl_->slice(baseName + ".biases");
    const uint32_t outputColumns = static_cast<uint32_t>(tensor.rows());
    const uint32_t reduction = static_cast<uint32_t>(tensor.columns());
    [encoder setComputePipelineState:impl_->matmulPipeline];
    [encoder setBuffer:input offset:inputOffset atIndex:0];
    [encoder setBuffer:weight.buffer offset:0 atIndex:1];
    [encoder setBuffer:scales.buffer offset:0 atIndex:2];
    [encoder setBuffer:biases.buffer offset:0 atIndex:3];
    [encoder setBuffer:output offset:outputOffset atIndex:4];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:5];
    [encoder setBytes:&outputColumns length:sizeof(outputColumns) atIndex:6];
    [encoder setBytes:&reduction length:sizeof(reduction) atIndex:7];
    [encoder setBytes:&weight.offset length:sizeof(weight.offset) atIndex:8];
    [encoder setBytes:&scales.offset length:sizeof(scales.offset) atIndex:9];
    [encoder setBytes:&biases.offset length:sizeof(biases.offset) atIndex:10];
    constexpr NSUInteger threadgroupBytes = 2 * 32 * 34 * sizeof(uint16_t);
    [encoder setThreadgroupMemoryLength:threadgroupBytes atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake((outputColumns + 31) / 32, (rows + 31) / 32, 1)
                  threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
}

void W6MetalOps::encodeGather(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLBuffer> tokenIds,
    NSUInteger tokenIdsOffset,
    id<MTLBuffer> output,
    NSUInteger outputOffset,
    const std::string &baseName,
    uint32_t tokenCount,
    uint32_t columnOffset,
    uint32_t outputColumns,
    float outputScale) {
    if (!encoder || !tokenIds || !output) throw std::invalid_argument("W6 gather received a nil Metal object");
    const package::W6Tensor tensor = package::W6Tensor::open(impl_->store, baseName);
    if (tensor.rows() > UINT32_MAX || tensor.columns() > UINT32_MAX || tokenCount == 0 || outputColumns == 0 ||
        static_cast<uint64_t>(columnOffset) + outputColumns > tensor.columns()) {
        throw std::invalid_argument("W6 gather geometry is unsupported: " + baseName);
    }
    if (!std::isfinite(outputScale)) throw std::invalid_argument("W6 gather scale must be finite");
    const MetalTensorSlice weight = impl_->slice(baseName + ".weight");
    const MetalTensorSlice scales = impl_->slice(baseName + ".scales");
    const MetalTensorSlice biases = impl_->slice(baseName + ".biases");
    const uint32_t vocabulary = static_cast<uint32_t>(tensor.rows());
    const uint32_t sourceColumns = static_cast<uint32_t>(tensor.columns());
    [encoder setComputePipelineState:impl_->gatherPipeline];
    [encoder setBuffer:tokenIds offset:tokenIdsOffset atIndex:0];
    [encoder setBuffer:weight.buffer offset:0 atIndex:1];
    [encoder setBuffer:scales.buffer offset:0 atIndex:2];
    [encoder setBuffer:biases.buffer offset:0 atIndex:3];
    [encoder setBuffer:output offset:outputOffset atIndex:4];
    [encoder setBytes:&tokenCount length:sizeof(tokenCount) atIndex:5];
    [encoder setBytes:&vocabulary length:sizeof(vocabulary) atIndex:6];
    [encoder setBytes:&sourceColumns length:sizeof(sourceColumns) atIndex:7];
    [encoder setBytes:&columnOffset length:sizeof(columnOffset) atIndex:8];
    [encoder setBytes:&outputColumns length:sizeof(outputColumns) atIndex:9];
    [encoder setBytes:&outputScale length:sizeof(outputScale) atIndex:10];
    [encoder setBytes:&weight.offset length:sizeof(weight.offset) atIndex:11];
    [encoder setBytes:&scales.offset length:sizeof(scales.offset) atIndex:12];
    [encoder setBytes:&biases.offset length:sizeof(biases.offset) atIndex:13];
    const NSUInteger width = std::min<NSUInteger>(impl_->gatherPipeline.maxTotalThreadsPerThreadgroup, 256);
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(tokenCount) * outputColumns, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}

const package::TensorStore &W6MetalOps::tensorStore() const noexcept { return impl_->store; }
size_t W6MetalOps::mappedShardCount() const noexcept { return impl_->buffers.size(); }
uint64_t W6MetalOps::mappedFileBytes() const noexcept { return impl_->fileBytes; }
uint64_t W6MetalOps::copiedWeightBytes() const noexcept { return 0; }

} // namespace gemma_runtime::ops
