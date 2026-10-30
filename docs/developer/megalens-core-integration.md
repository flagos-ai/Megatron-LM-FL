# Reviewing MegaLens Core integration

MegaLens observation helpers live in `megatron/megalens/probes/`. Core retains
the native computation, collective launches, waits and scheduling decisions,
with observation calls at the boundaries that define each event. Paired
`BEGIN/END MEGALENS OBSERVABILITY` comments identify those integration points.

The current integration includes main `bbc208e00f2a201783938ef549f5058d6047e4c0`.
Independent native training repairs and external fix payloads have been removed
from the integration MR; their implementations and tests live in the
[related MRs](megalens-related-mrs.md). Hybrid-CP and DualPipeV configurations
that need those repairs require their owning MRs. Unmodified upstream Core
currently imports Triton, so provide a real Triton installation for the CPU gate
until the separate optional-import repair is merged.

## Implementation ownership

| Native entry point | Observation implementation | Boundary retained in the native flow |
|---|---|---|
| `core/pipeline_parallel/p2p_communication.py` | `megalens/probes/p2p.py` | Launch, raw Work registration, wait and existing synchronization |
| `core/pipeline_parallel/schedules.py` | `megalens/probes/pipeline.py` | Forward/backward phases, workload FIFO and gradient synchronization |
| `core/pipeline_parallel/combined_1f1b.py` | `megalens/probes/pipeline.py` | Original combined function body and native FSDP hooks |
| `core/tensor_parallel/mappings.py` | `megalens/probes/tp.py` | Original equal/unequal collective branches and tensor transformations |
| `training/training.py` | `megalens/probes/training.py` | Runtime ownership and the iteration envelope containing skip/continue/break |
| `training/arguments.py` | `megalens/probes/training.py` | Existing CLI registration and validation calls |
| `core/distributed/finalize_model_grads.py` | `megalens/probes/dp.py` | Native embedding/SP all-reduce calls and tensor views |
| DP and MoE entry points | `megalens/probes/dp.py`, `megalens/probes/moe.py` | Existing coalescing, stream, handle and postprocess boundaries |

Paths in the table are relative to `megatron/`. The generic
`core/observability.py` sink/gate/scope facade remains backend independent.
Existing TP/MoE observation imports remain compatibility entry points.

P2P keeps main's explicit even/odd rank branches and `_p2p_ops` signature. Its
small `_p2p_groups` helper shares main's native PP/WORLD group selection with
enabled observation, so recorded backend fields describe the actual request
group. Observation resolves this metadata lazily. This helper adds no group
creation, collective or synchronization.

Schedule metadata builders and training lifecycle helpers stay in the probe
package. Their callers retain the local scopes needed to observe failures and
early exits. The mode-1 iteration barrier stays before scope entry; skipped
iterations close the existing trace window. Tensor-parallel helpers preserve
the single-rank AllReduce scope and each event's existing group-slot timing.

Combined 1F1B decorates the original native function directly. It retains the
two existing observation arguments after the native arguments without a second
copy of the full forwarding signature. MoE phase and EP metadata are prepared
inside the probe package. Gradient reductions keep their native calls in Core;
the probe fills peer ranks only after a successful collective.

## Keeping indentation changes small

When a probe covers the body of an existing context, use a standard multiple
context-manager statement:

```python
with (
    cur_platform.stream(self.stream),
    moe_probes.shared_expert_overlap_scope(self, "pre_forward_comm"),
):
    # Native computation keeps its original indentation.
    ...
```

Python enters the stream before evaluating the probe expression, and exits the
probe before exiting the stream. This preserves metadata preparation and scope
boundaries while avoiding an extra indentation level in five shared-expert
stages. Use the existing decorator for scopes that already cover an entire
function. Keep local `with` blocks when the measured region covers only part of
a function. Preserve normal Black formatting and readable calls.

Review remaining indentation-only changes with
`rtk proxy git diff -w b471409ac -- <path>` alongside the normal diff against that
same fixed upstream main revision. Do not change native computation, introduce
manual start/stop, or move argument evaluation outside its scope to reduce the
displayed diff. Format changed regions when an inherited file has unrelated
formatting differences.

## Keeping native functions aligned in diffs

Preserve the order of upstream functions and methods. Place added integration
helpers after the original function they support, or after the native methods
at the end of their class. P2P's `_p2p_groups` follows `_p2p_ops`; the five MoE
helpers follow `MoELayer.backward_dw`. This keeps the native `_p2p_ops` and
`postprocess` declarations aligned in the default Git diff.

Compare each path with the fixed main revision using the default diff first:

```bash
rtk proxy git diff --no-ext-diff --no-color --unified=3 b471409ac -- <path>
rtk proxy git diff --word-diff=plain b471409ac -- <path>
rtk proxy git diff -w b471409ac -- <path>
```

Retain upstream text for functions that do not need observation changes,
including their docstrings, comments, imports and call layout. Keep observation
markers around the added imports and calls. Apply normal formatting to new
regions; whole-file import sorting or formatting can reintroduce unrelated
changes to inherited code.

The `diff_review` validation entry records G1.16's review of all 142 MR paths:
36 existing files and 106 additions, including 82 added Python files checked
for syntax. It lists the existing paths individually and identifies the ten
adjusted modules. Native definition order was checked across 959 function
entries; all 864 functions with unchanged AST have identical upstream text.
The `backward_step` signature still spans changed lines because of its existing
observation keyword parameters. This is an actual signature extension to review.
MoE's diff line count can grow when helper definitions are separated from the
native `postprocess`; declaration alignment is the acceptance criterion.

## Open review TODOs

- [ ] Restore native P2P group selection inside `_p2p_ops`. The current call to
  `_p2p_groups` at line 94 extracts upstream logic to share it with observation.
  Restore `_p2p_ops` verbatim from main; put observation-only group resolution in
  `megatron/megalens/probes/p2p.py` behind the collection gate, and remove the
  Core helper/callback. Verify PP=2 versus larger groups, even/odd ranks and the
  UCC exception against the groups actually passed to native isend/irecv.
  Preserve backend fields, Work identity, request order and wait/sync counts;
  trace-off must add no group/backend queries. This is pending implementation;
  the provenance documentation update does not change the P2P code.

## Invariants when updating upstream

- Preserve native signatures, branch order, return values, exceptions and raw
  Work objects. Keep collective arguments and wait/synchronize counts unchanged.
- Keep contiguous/reshape operations and stream/coalescing contexts within their
  existing observation boundaries. Moving a scope changes the measured event.
- A closed gate skips metadata and peer construction. Core imports must not load
  the Trace runtime, Tracer, analysis modules or optional collectors.
- Retain local `with` blocks around native work. Extract reusable observation
  logic without moving native training functions into the probe package.
- Maintain event names, fields and lifecycle semantics in
  `tests/megalens/fixtures/probe_scan_gate.json` and the event catalog.

## Verification

From the repository root in a configured development environment, run the CPU
gate against MixedPara revision `12fb7169ce30fdb62b50f86b41afa09336a523ea`:

```bash
rtk proxy python -B tests/test_utils/runners/run_megalens_gate.py \
  --profile cpu --source-repo ../MixedPara --timeout 300
```

Use `--source-repo` to point at that checkout. The scanner
checks the source/target event contract; the CPU suites exercise disabled gates,
field construction, raw Work identity, exceptions and import boundaries.
The focused suites are `test_p2p_observability.py`,
`test_pipeline_schedule_observability.py`, `test_runtime_lifecycle.py` and the
three `test_tp_*_observability.py` suites under `tests/megalens/`.
MoE and gradient cleanup also uses `test_ep_primitive_observability.py`,
`test_moe_phase_observability.py`, `test_shared_expert_overlap_observability.py`,
`test_sp_layernorm_allreduce_observability.py` and
`test_embedding_grads_allreduce_observability.py`.

For a structural refactor, compare moved definitions and preserved native
functions by AST against the fixed source revisions. Compare P2P after inlining
only native group selection, and TP mappings after removing observation scope
wrappers. Build sdist/wheel and run `tests/megalens/wheel_import_check.py` in an
environment without the source checkout to verify the relocated modules ship.

CPU and AST checks do not establish GPU completion or communication timing.
Recheck affected trace-off/on training profiles and checkpoints on the actual
backend. `tests/test_utils/runners/run_p2p_order_nccl.py` verifies two-rank NCCL
launch order, request groups, payloads and raw Work identity. Performance overhead
requires a separate controlled measurement.

The [related merge requests](megalens-related-mrs.md) document owns independent
training/dependency fixes and their merge dependencies.

The [validation record](megalens-core-validation.json) fixes the reviewed code
revisions and records the source comparisons, CPU gate, packaging, installed
module origins and focused H800 training/NCCL results, with their evidence limits.
Its `cleanup` entry records the subsequent helper and indentation cleanup against
its own source revision; earlier hardware results retain their original revision.
The `diff_review` entry fixes G1.16's source identity and CPU/structural checks;
it adds no GPU, packaging or performance result for that source revision.
