# Training lifecycle contract

Trace behavior across these stages:

1. Public package/API exposure.
2. CLI or programmatic argument declaration.
3. Argument validation and normalization.
4. Config/dataclass propagation and defaults.
5. Model, optimizer, scheduler, data, and parallel-state construction.
6. Train/eval step dispatch, pipeline schedule, precision and distributed execution.
7. Checkpoint, metrics, logging, and output side effects.
8. Unit, integration, functional, or external observing test.

A lifecycle row may span several stages. A same-named symbol is not proof of equivalence. Record default values, validation constraints, call sites, ordering, rank/device behavior, state persistence, and failure behavior where relevant.
