#include "gemma4/Gemma4Cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace gemma_runtime::gemma4 {

uint32_t Gemma4Cache::physicalOwner(uint32_t layer) {
    const LayerSpec spec = Gemma4Config::layer(layer);
    return spec.sharedOwner.value_or(layer);
}

const std::vector<KvToken> &Gemma4Cache::append(
    uint32_t layer,
    uint64_t position,
    std::span<const float> key,
    std::span<const float> value) {
    const LayerSpec spec = Gemma4Config::layer(layer);
    if (spec.sharesKeyValue) {
        throw std::invalid_argument("shared Gemma 4 layers cannot append their own K/V");
    }
    if (key.empty() || key.size() != value.size()) {
        throw std::invalid_argument("K/V token vectors must be non-empty and equal-sized");
    }
    Slot &slot = slots_[layer];
    if (position != slot.logicalLength) {
        throw std::invalid_argument("K/V token position must equal the layer logical length");
    }
    KvToken token{position, {key.begin(), key.end()}, {value.begin(), value.end()}};
    slot.attentionView = slot.retained;
    slot.attentionView.push_back(std::move(token));
    slot.logicalLength += 1;

    if (spec.attention == AttentionKind::Sliding) {
        constexpr size_t retainedLimit = Gemma4Config::kSlidingWindow - 1;
        const size_t begin = slot.attentionView.size() > retainedLimit
                                 ? slot.attentionView.size() - retainedLimit
                                 : 0;
        slot.retained.assign(slot.attentionView.begin() + static_cast<std::ptrdiff_t>(begin),
                             slot.attentionView.end());
    } else {
        slot.retained = slot.attentionView;
    }
    revision_ += 1;
    return slot.attentionView;
}

const std::vector<KvToken> &Gemma4Cache::view(uint32_t layer) const {
    return slots_[physicalOwner(layer)].attentionView;
}

uint64_t Gemma4Cache::logicalLength(uint32_t layer) const {
    return slots_[physicalOwner(layer)].logicalLength;
}

size_t Gemma4Cache::retainedLength(uint32_t layer) const {
    return slots_[physicalOwner(layer)].retained.size();
}

Gemma4Cache::Snapshot Gemma4Cache::snapshot() const {
    return Snapshot{.slots = slots_, .revision = revision_};
}

void Gemma4Cache::commit(const Snapshot &snapshot) const {
    if (snapshot.revision > revision_) {
        throw std::invalid_argument("cannot commit a cache snapshot from the future");
    }
}

void Gemma4Cache::rollback(const Snapshot &snapshot) {
    if (snapshot.revision > revision_) {
        throw std::invalid_argument("cannot roll back to a cache snapshot from the future");
    }
    slots_ = snapshot.slots;
    revision_ += 1;
}

void Gemma4Cache::clear() noexcept {
    slots_ = {};
    revision_ += 1;
}

} // namespace gemma_runtime::gemma4
