# Platform Contract

Compare PlatformBase and implementations structurally: method/property kind, parameters, defaults, abstractness, inheritance, optional imports, registration, selection, and initialization hooks.

Classify device assumptions at occurrence or enclosing-symbol granularity. `torch.cuda`, literal `cuda`, NCCL, NVTX, CUDA graph, CUDA RNG, and CUDA extension references may be intentional; require evidence rather than global replacement. Prefer an existing compatible `cur_platform` capability. Adding a new abstraction requires consumer and implementation coverage plus tests.

Treat inherited methods as covered only when the parent semantics apply to the child backend. Record inherited exceptions such as a CUDA-derived non-CUDA platform for explicit review.
