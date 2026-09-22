#pragma once

#include "device/TuningPolicy.hpp"
#include "package/TensorStore.hpp"

#import <Metal/Metal.h>

#include <cstddef>
#include <memory>
#include <string>

namespace gemma_runtime::ops {

class W6MetalOps {
  public:
    W6MetalOps(
        id<MTLDevice> device,
        id<MTLLibrary> library,
        package::TensorStore tensorStore);
    W6MetalOps(W6MetalOps &&) noexcept;
    W6MetalOps &operator=(W6MetalOps &&) noexcept;
    W6MetalOps(const W6MetalOps &) = delete;
    W6MetalOps &operator=(const W6MetalOps &) = delete;
    ~W6MetalOps();

    void encodeMatvec(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> input,
        NSUInteger inputOffset,
        id<MTLBuffer> output,
        NSUInteger outputOffset,
        const std::string &baseName,
        W6MatvecVariant variant = W6MatvecVariant::Wide8);

    void encodeMatmul(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> input,
        NSUInteger inputOffset,
        id<MTLBuffer> output,
        NSUInteger outputOffset,
        const std::string &baseName,
        uint32_t rows);

    void encodeGather(
        id<MTLComputeCommandEncoder> encoder,
        id<MTLBuffer> tokenIds,
        NSUInteger tokenIdsOffset,
        id<MTLBuffer> output,
        NSUInteger outputOffset,
        const std::string &baseName,
        uint32_t tokenCount,
        uint32_t columnOffset,
        uint32_t outputColumns,
        float outputScale);

    [[nodiscard]] const package::TensorStore &tensorStore() const noexcept;
    [[nodiscard]] size_t mappedShardCount() const noexcept;
    [[nodiscard]] uint64_t mappedFileBytes() const noexcept;
    [[nodiscard]] uint64_t copiedWeightBytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace gemma_runtime::ops
