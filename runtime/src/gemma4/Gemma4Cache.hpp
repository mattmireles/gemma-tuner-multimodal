#pragma once

#include "gemma4/Gemma4Config.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace gemma_runtime::gemma4 {

struct KvToken {
    uint64_t position = 0;
    std::vector<float> key;
    std::vector<float> value;
};

class Gemma4Cache {
  public:
    static constexpr uint32_t kPhysicalLayers = Gemma4Config::kFirstSharedLayer;

    struct Slot {
        uint64_t logicalLength = 0;
        std::vector<KvToken> retained;
        std::vector<KvToken> attentionView;
    };

    struct Snapshot {
        std::array<Slot, kPhysicalLayers> slots;
        uint64_t revision = 0;
    };

    // Append one token to a physical cache layer and return the exact view the
    // current attention operation must see. Shared layers use view() directly.
    [[nodiscard]] const std::vector<KvToken> &append(
        uint32_t layer,
        uint64_t position,
        std::span<const float> key,
        std::span<const float> value);

    [[nodiscard]] const std::vector<KvToken> &view(uint32_t layer) const;
    [[nodiscard]] uint64_t logicalLength(uint32_t layer) const;
    [[nodiscard]] size_t retainedLength(uint32_t layer) const;

    [[nodiscard]] Snapshot snapshot() const;
    void commit(const Snapshot &snapshot) const;
    void rollback(const Snapshot &snapshot);
    void clear() noexcept;

  private:
    [[nodiscard]] static uint32_t physicalOwner(uint32_t layer);

    std::array<Slot, kPhysicalLayers> slots_{};
    uint64_t revision_ = 0;
};

} // namespace gemma_runtime::gemma4
