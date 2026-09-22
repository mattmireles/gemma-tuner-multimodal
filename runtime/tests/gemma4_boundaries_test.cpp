#include "gemma4/Gemma4Cache.hpp"
#include "gemma4/Gemma4Config.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

void append(gemma_runtime::gemma4::Gemma4Cache &cache, uint32_t layer, uint64_t position) {
    const float value = static_cast<float>(position);
    (void)cache.append(layer, position, std::span<const float>(&value, 1), std::span<const float>(&value, 1));
}

} // namespace

int main() {
    using namespace gemma_runtime::gemma4;

    require(Gemma4Config::attentionKind(0) == AttentionKind::Sliding, "layer 0 must be sliding");
    require(Gemma4Config::attentionKind(5) == AttentionKind::Full, "layer 5 must be full");
    require(Gemma4Config::layer(23).sharesKeyValue == false, "layer 23 must own K/V");
    require(Gemma4Config::layer(24).sharedOwner == 22, "layer 24 must reuse sliding owner 22");
    require(Gemma4Config::layer(29).sharedOwner == 23, "layer 29 must reuse full owner 23");
    require(Gemma4Config::layer(41).sharedOwner == 23, "layer 41 must reuse full owner 23");

    Gemma4Cache sliding;
    for (uint64_t position = 0; position < 511; ++position) append(sliding, 0, position);
    require(sliding.logicalLength(0) == 511, "logical length at token 511");
    require(sliding.retainedLength(0) == 511, "sliding storage at token 511");
    require(sliding.view(0).size() == 511, "attention view at token 511");
    append(sliding, 0, 511);
    require(sliding.logicalLength(0) == 512, "logical length at token 512");
    require(sliding.retainedLength(0) == 511, "sliding storage remains window minus one");
    require(sliding.view(0).size() == 512, "current attention sees the complete 512-token window");
    require(sliding.view(0).front().position == 0, "token 512 still sees position zero");
    append(sliding, 0, 512);
    require(sliding.logicalLength(0) == 513, "logical length at token 513");
    require(sliding.view(0).front().position == 1, "token 513 evicts position zero");
    require(sliding.view(0).back().position == 512, "token 513 includes itself");

    Gemma4Cache shared;
    for (uint64_t position = 0; position < 513; ++position) {
        append(shared, 22, position);
        append(shared, 23, position);
    }
    require(&shared.view(24) == &shared.view(22), "shared sliding layer must alias owner view");
    require(&shared.view(29) == &shared.view(23), "shared full layer must alias owner view");
    require(shared.view(24).size() == 512, "shared sliding view must preserve mask input window");
    require(shared.view(29).size() == 513, "shared full view must preserve full context");
    try {
        append(shared, 24, 0);
        require(false, "shared layer append must fail");
    } catch (const std::invalid_argument &) {
    }

    Gemma4Cache transactional;
    for (uint64_t position = 0; position < 3; ++position) append(transactional, 0, position);
    const auto checkpoint = transactional.snapshot();
    append(transactional, 0, 3);
    transactional.rollback(checkpoint);
    require(transactional.logicalLength(0) == 3, "rollback restores logical length");
    require(transactional.view(0).back().position == 2, "rollback restores attention view");
    transactional.commit(checkpoint);

    std::cout << "gemma4 boundaries: PASS\n";
    return EXIT_SUCCESS;
}
