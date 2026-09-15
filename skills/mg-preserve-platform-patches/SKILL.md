---
name: mg-preserve-platform-patches
description: Audit and preserve Megatron-LM-FL platform abstractions and multi-accelerator patches across an NVIDIA upstream upgrade. Use when PlatformBase, registered platform implementations, cur_platform call sites, device/RNG/stream/memory/graph APIs, raw torch.cuda/CUDA/NCCL/NVTX assumptions, vendor capability fallbacks, or hardware test gates may drift; use to build a dynamically discovered capability matrix and reviewed CUDA-specific allowlist before integration.
---

# Preserve Megatron Platform Patches

Audit exact classifier refs before editing. Discover platform implementations and consumers from source; never encode a fixed vendor list or mass-replace CUDA strings.

## Workflow

1. Require matching inventory and domain-routing refs.
2. Run `scripts/audit_platform_contract.py` against the fork and target.
3. Extract `PlatformBase` methods/properties, every subclass, registration key, selection order, and `cur_platform` consumer.
4. Compare method presence, sync/async/property kind, parameter order/defaults, inherited behavior, and capabilities.
5. Scan all platform-related routes for raw CUDA/NCCL/NVTX/device literals. Classify each occurrence as portable abstraction candidate, intentional backend-specific use, build/test-only use, or manual.
6. Store intentional uses in a reviewed allowlist with reason, owner, scope, and observing test. Do not treat path-level allowlisting as sufficient when only individual symbols are intentional.
7. Coordinate override dispatch with `mg-audit-plugin-overrides`, semantic features with `mg-integrate-runtime-features`, and hardware evidence with `mg-run-upgrade-test-matrix`.
8. Validate the reviewed matrix with `scripts/validate_platform_audit.py`. Create a skill gap for generated platforms, runtime monkey patches, or device mechanisms the static model cannot represent.

Read [platform-contract.md](references/platform-contract.md) and [artifact-contract.md](references/artifact-contract.md).

## Invariants

- Every discovered platform has a complete or explicitly inherited base contract.
- New target device assumptions are either represented by a compatible platform API or explicitly allowlisted.
- Vendor selection, initialization order, optional imports, fallback, and device-name identity remain coherent.
- CUDA-specific code is preserved only where semantics are genuinely CUDA/NCCL/NVTX/build specific.
- Unavailable accelerator tests remain blocked with owners.
- FlagScale markers are ownership hints, not completeness evidence.

## Completion

Deliver `platform-audit.json`, `capability-matrix.tsv`, `raw-device-assumptions.tsv`, reviewed decisions/allowlist, and `platform-validation.json`. Require zero unexplained contract gaps, uncovered platform routes, or unowned device assumptions before source integration.
