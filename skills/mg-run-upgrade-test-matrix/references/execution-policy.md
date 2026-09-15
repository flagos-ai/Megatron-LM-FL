# Execution policy

Discovery is read-only and requires no accelerator. Execution is a separate authorized phase.

Run cheapest and widest-signal tiers first: syntax/schema, import/CPU, focused unit, single accelerator, distributed, functional/golden, then FlagScale E2E. Never infer one accelerator backend from another.

An external gate must name an owner, required accelerator/runner, exact or templated command, datasets/checkpoints/services, and expected oracle. Unavailable hardware is blocked coverage, not success.

Record commit SHA, worktree state, interpreter and package environment, device inventory, command, timestamps, exit code, stdout/stderr log, and produced artifacts. Preserve initial failure evidence across reruns.
