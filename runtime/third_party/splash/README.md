# Splash Source Boundary

- **Upstream:** `https://github.com/incoai/splash`
- **Pinned commit:** `298852ec603a5ef3848e8fb6d58d6d81746713ac`
- **License:** Apache-2.0; see [`LICENSE`](LICENSE)
- **Upstream NOTICE:** none at the pinned commit

## Copied and Modified Inventory

Only the device-discovery substrate is derived from Splash:

| Local file | Pinned upstream source | Change |
| --- | --- | --- |
| `runtime/include/gemma_runtime/DeviceCapabilities.hpp` | `runtime/metal/DeviceCapabilities.hpp` | Renamed namespace, supports Apple7-Apple10, records the Metal language ABI, and selects a fail-closed per-family policy. |
| `runtime/src/device/DeviceCapabilities.cpp` | `runtime/metal/DeviceCapabilities.cpp` | Replaced Splash's Apple9/placement-sparse product gate with Gemma runtime validation and stable JSON diagnostics. |
| `runtime/src/device/DeviceCapabilities.mm` | `runtime/metal/MetalBackend.mm` device-query portion | Isolated public Metal, Foundation, and IOKit capability queries; removed backend, sparse-memory, server, model, and request-stack dependencies. |

No Splash server, agent, HTTP, Qwen model, scheduler, cache, speculative
decoder, or kernel source is copied in Phase 1.

## Modification Policy

Every derived source file carries a prominent modification notice. Refreshes
must pin a full upstream commit, review its license and NOTICE state, update the
table above, and retain only mechanisms still used by this runtime. Do not
replace the pin with a branch or silently absorb upstream model assumptions.
