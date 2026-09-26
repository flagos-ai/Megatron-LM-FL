# Validation gates

Order gates: static syntax/schema; source/package manifest closure; isolated wheel build; wheel content and metadata; clean-environment import; optional dependency behavior; Docker build/context; accelerator-specific smoke.

Record unavailable compiler, network, registry, image, or accelerator as an owned external gate. Never count an unrun gate as pass. Keep build outputs outside the source tree and verify no generated artifact enters the diff.
