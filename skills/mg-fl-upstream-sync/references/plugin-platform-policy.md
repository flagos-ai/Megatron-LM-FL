# Plugin, Override, Marker, and Platform Policy

Compare override targets structurally: registry key, object kind, vendor, signature, defaults, annotations, async/class/static behavior, copied base logic, and fork-only branches. One overridable target may have multiple vendor implementations.

Use FlagScale Begin, End, Add, and legacy comments only as incomplete ownership hints. Where an apparent block exists, inspect its enclosing symbol, target anchor, intent, callers, and observing tests. Equal marker counts or pairing are not acceptance evidence.

Scan new target code for raw CUDA assumptions. Replace only when the platform abstraction exposes compatible semantics. Record intentional CUDA/NCCL/NVTX/build-time uses in a reviewed allowlist and exercise vendor-specific paths through their external gates.
