<!---
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.
-->

# MegaLens Trace Quickstart

MegaLens adds training semantics to performance traces. Each rank writes its own trace shard during
training. After the run, the analyzer combines those shards into a Chrome Trace event list for
Perfetto and generates PP, DP, TP/SP, EP, and Hybrid reports when the topology supports them.

Use this guide to add MegaLens to an existing, working Megatron-LM or FlagScale training command.
Start with framework tracing and a short capture window. Add hardware counters or CUDA kernels only
when the question requires device-side evidence.

## Enable a framework trace

### FlagScale YAML

The recorded formal GPU runs use FlagScale revision
`6d775cd01d5c822f9413b9952a81652c917d273e` and the pinned environment in
[`docker/Dockerfile.work`](https://github.com/flagos-ai/Megatron-LM-FL/blob/main/docker/Dockerfile.work).
Keep that image definition as the reproduction contract for those runs.

Current dev-image acceptance uses a temporary image derived from
`harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856`, with
Megatron-LM-FL revision `13ef6ae7ed8e3ac35143f3e03526674d12194fc6` and FlagScale revision
`067efe6f1b00bc9416e22cfe46c52990e10bf045` installed.
FlagScale gives its vendored `megatron.training` package import priority. Apply the
[`docker/patches/flagscale-megalens.patch` compatibility overlay](https://github.com/flagos-ai/Megatron-LM-FL/blob/main/docker/patches/flagscale-megalens.patch)
so that this entry receives the MegaLens arguments, runtime ownership, iteration/optimizer scopes,
and shutdown lifecycle. The `utils.py` hunk carries the dense-batch metadata-broadcast correction
used by the separately validated whole-iteration CUDA Graph path. Recheck the patch when using
another FlagScale revision.

Create a writable container from the dev image and copy clean Megatron-LM-FL and FlagScale checkouts
into it:

```bash
export BASE=harbor.baai.ac.cn/flagscale/flagscale-train:dev-cu128-py3.12-20260319182856
export IMAGE=<derived-image-tag>

docker create --name megalens-dev-build "$BASE" sleep infinity
docker start megalens-dev-build
docker cp /path/to/Megatron-LM-FL megalens-dev-build:/workspace/Megatron-LM-FL
docker cp /path/to/FlagScale megalens-dev-build:/workspace/FlagScale
docker exec -it megalens-dev-build bash
```

Run the following commands inside that container. Installing with `--no-deps` preserves its CUDA,
PyTorch, Transformer Engine, and NCCL stack.

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate flagscale-train

git -C /workspace/FlagScale apply --check \
  /workspace/Megatron-LM-FL/docker/patches/flagscale-megalens.patch
git -C /workspace/FlagScale apply \
  /workspace/Megatron-LM-FL/docker/patches/flagscale-megalens.patch

python -m pip uninstall -y megatron-core
PIP_NO_INDEX=1 MAX_JOBS=4 python -m pip install \
  --no-cache-dir --no-deps --no-build-isolation --no-index \
  -e /workspace/Megatron-LM-FL
PIP_NO_INDEX=1 python -m pip install \
  --no-cache-dir --no-deps --no-build-isolation --no-index \
  -e /workspace/FlagScale
python -m pip check
python -c 'import megatron.core, megatron.megalens, megatron.training'
exit
```

Commit the prepared container as the derived image used by the acceptance command below:

```bash
docker commit megalens-dev-build "$IMAGE"
```

Add the following keys under `train.system`:

```yaml
train:
  system:
    trace: true
    trace_mode: 1
    trace_dir: /absolute/path/to/unique-run/traces
    trace_interval: 1000
    continuous_trace_iterations: 1
    trace_granularity: full
    trace_cupti_kernels: "off"
    hardware_monitor: false
```

Keep the existing FlagScale launch command. A configuration that needs trace-off and trace-on runs
can resolve the switch from an environment variable:

```yaml
trace: ${oc.decode:${oc.env:MEGALENS_TRACE,false}}
```

```bash
MEGALENS_TRACE=true flagscale run \
  --config-path=/path/to/configs \
  --config-name=train \
  --action=test
```

[`tests/megalens/fixtures/flagscale_single_node_pp2_smoke.yaml`](https://github.com/flagos-ai/Megatron-LM-FL/blob/main/tests/megalens/fixtures/flagscale_single_node_pp2_smoke.yaml)
is a two-GPU configuration that uses the same fields. The focused current-dev acceptance uses
[`flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml`](https://github.com/flagos-ai/Megatron-LM-FL/blob/main/tests/megalens/fixtures/flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml): eight H100 GPUs,
TP2×PP4×DP1, four microbatches per iteration, and two iterations. Run it from a clean
Megatron-LM-FL checkout with a new run directory for each mode:

```bash
export IMAGE=<derived-image-tag>
export RUN_ROOT=/absolute/path/to/new-run-root

for MODE in trace-off trace-on; do
  python3 tests/test_utils/runners/run_flagscale_megalens.py \
    --run-dir "$RUN_ROOT/tp2-pp4-${MODE#trace-}" \
    --input-config \
      tests/megalens/fixtures/flagscale_single_node_tp2_pp4_multimicrobatch_smoke.yaml \
    --mode "$MODE" \
    --image "$IMAGE" \
    --megatron-source-root "$(pwd)" \
    --timeout 1200
done
```

The accepted trace-off run produced zero shards. Trace-on produced eight non-empty shards with
4,864 records. Both runs completed the iteration-2 checkpoint with identical recorded losses. The
analyzer inferred PP4/TP2/DP1/EP1 and generated the applicable PP, TP, and Hybrid reports.
After the Megatron-LM-FL change merges, port the same training-entry hooks into FlagScale. Create a
dedicated image definition only when the released FlagScale dev image still requires project-specific
assembly.

### Megatron-LM CLI

Append these arguments to an existing `pretrain_gpt.py` command:

```bash
--trace \
--trace-mode 1 \
--trace-dir /absolute/path/to/unique-run/traces \
--trace-interval 1000 \
--continuous-trace-iterations 1 \
--trace-granularity full \
--trace-cupti-kernels off
```

For an eight-step demonstration that captures every step, use
`--trace-interval 1 --continuous-trace-iterations 1`.

## Choose the capture window

MegaLens uses 1-based training iteration IDs. For interval `N` and continuous count `K`, an iteration
is active when `(iteration - 1) % N < K`.

| Configuration | Captured iterations |
| --- | --- |
| `N=1000`, `K=1` | 1, 1001, 2001, ... |
| `N=100`, `K=5` | 1–5, 101–105, ... |
| `N=1`, `K=1` | Every iteration |

`continuous_trace_iterations` must be in `[1, trace_interval]`. Mode 1 aligns every executed,
non-skipped step with a WORLD barrier, including steps outside the active capture window. Use short,
controlled windows for routine inspection.

## Trace modes and optional data

| Setting | Purpose |
| --- | --- |
| `trace_mode: 1` | Framework Probe scopes in rank-local `benchmark-*.json` files; input for the current analyzer |
| `trace_mode: 0` | Lightweight step sentinel in `mode0-sentinel-*.jsonl`; records wall time, rank topology, and optional hardware samples |
| `trace_granularity: base` | Coarse training phases and communication events from the base event set |
| `trace_granularity: full` | All scopes produced by the active training path, including model-layer scopes |
| `hardware_monitor: true` | Optional `CPU_Metrics` and `GPU_Metrics`; providers come from the `megalens-monitor` extra |
| `trace_cupti_kernels: "on"` | CUDA kernel records for a short mode-1 window |

The default CUDA kernel setting is `"off"`. `"auto"` enables kernels for mode 1 with `full`
granularity, while `"on"` enables them for either granularity. Mode 0 always uses sentinel records.
The analyzer consumes mode-1 `benchmark-*.json` shards. Mode-0
`mode0-sentinel-*.jsonl` files provide sentinel evidence. CUDA kernel capture is best-effort:
confirm actual `record_type="cuda_kernel"` records in the relevant shards before making
kernel-level claims.

Install the offline report dependencies when needed:

```bash
uv pip install -e ".[megalens]"
```

Add hardware providers with:

```bash
uv pip install -e ".[megalens,megalens-monitor]"
```

## Check and aggregate the output

Mode 1 uses one file per rank with names such as:

```text
benchmark-global-3-data-1-pipeline-0-tensor-1.json
```

Use an empty, writable `trace_dir` for each run. With the default rank-local writer, a complete
mode-1 run produces exactly `WORLD_SIZE` non-empty `benchmark-*.json` shards. For multi-node or
container runs, use a shared, persistent mount visible at the same path to every rank. With
node-local storage, collect every rank shard into one persistent directory before the job or
container ends and before aggregation.

Confirm that the run produced non-empty shards and that their count matches the launch world size:

```bash
export TRACE_DIR=/absolute/path/to/unique-run/traces
export REPORT_DIR=/absolute/path/to/unique-run/reports
: "${WORLD_SIZE:?set WORLD_SIZE to the launch world size}"

find "$TRACE_DIR" -maxdepth 1 -name 'benchmark-*.json' -type f -size +0 -print
test "$(find "$TRACE_DIR" -maxdepth 1 -name 'benchmark-*.json' -type f -size +0 | wc -l)" \
  -eq "$WORLD_SIZE"
```

Aggregate the shards and run every applicable analyzer:

```bash
python -m megatron.megalens.analyzer \
  --bench-dir "$TRACE_DIR" \
  --run all \
  --output-dir "$REPORT_DIR"
```

The command writes `$REPORT_DIR/aggregated_trace.json` and creates `pp/`, `dp/`, `tp/`, `ep/`, or
`hybrid/` report directories according to the inferred parallel topology. Load the aggregated JSON
directly in Perfetto or a Chrome Trace viewer.

To separate aggregation from analysis:

```bash
python -m megatron.megalens.analyzer \
  --bench-dir "$TRACE_DIR" \
  --aggregate-only \
  --trace-output "$REPORT_DIR/aggregated_trace.json"

python -m megatron.megalens.analyzer \
  --trace "$REPORT_DIR/aggregated_trace.json" \
  --run pp dp tp ep hybrid \
  --output-dir "$REPORT_DIR"
```

`--align-framework-timeline` is opt-in and fail-closed. For every discovered iteration and TP group,
it requires at least three `tp-allreduce` spans with `ph="X"`, `op="all_reduce"`, and
`timing_phase="collective_call"` from every group member. Peer lists and group sizes must be
consistent; counts and ordered `(op, timing_phase, data_bytes, reduce_op, payload_role)` signatures
must match; and each rank must have one iteration bound plus framework events. Missing or ambiguous
anchors, conflicting shifts, and shifts that do not fit within the recorded iteration are rejected.
When these prerequisites hold, viewer-derived CUDA kernel timestamps receive the same per-rank
shift; event durations and raw profiler coordinates remain unchanged.

## Supported Probe families

Framework records commonly include `name`, `ph`, `iteration`, `g_rk`, `dp_rk`, `pp_rk`, `tp_rk`,
`dev`, and `rel_ts`. Aggregated events add Chrome Trace fields such as `pid`, `tid`, `ts`, and `dur`.
CUPTI kernel events keep their raw profiler fields and add iteration-anchored `ts` and `dur` when
`iter_rel_start_us` and `duration_us` are available.
Fields for EP, CP, virtual pipeline stages, workload, payload, or async operations appear when the
executed producer supplies them.

| Family | Representative events | What the trace can identify |
| --- | --- | --- |
| Training and model | `iteration`, `forward-step`, `backward-step`, `decoder`, `transformer_layer`, `attention`, `MLP.forward`, `loss`, `optimizer*` | Iteration, model phase, microbatch, pipeline stage, layer, workload, and optimizer boundaries |
| PP, P2P, and Bridge | `forward`, `backward`, `send-*`, `recv-*`, `p2p-launch`, `p2p-batch-complete`, `combined-forward-backward-step`, `bridge-*` | Schedule phase, peer group, payload, transport API, request or operation ID, and existing completion boundaries |
| TP and SP | `tp-allreduce`, `tp-all-gather-*`, `tp-reduce-scatter*`, `tp-linear-async-*`, `sp-layernorm-allreduce`, `embedding-grads-allreduce` | Group, payload size, collective kind, split sizes, async launch, and wait boundaries |
| DP and optimizer | `dp-allreduce`, `dp-reduce-scatter`, `dp-param-all-gather`, `dp-grad-sync-complete`, `dp-param-sync-complete`, `all-grads-sync`, `optimizer*` | Gradient or parameter role, bucket, DP group, overlap setting, dispatch, and stream dependency |
| EP and MoE | `moe-router`, `moe-dispatch`, `moe-experts`, `moe-shared-expert`, `moe-combine`, `ep-alltoall-*`, `ep-allgather-*` | Tokens, experts, routing load, dispatcher, capacity, communication payload, and async lifecycle |
| Hardware and kernels | `CPU_Metrics`, `GPU_Metrics`, `cuda_kernel` | Available CPU/GPU counters and rank-local CUDA kernel start, end, duration, device, and iteration |

Current validated DP Probe and Analysis coverage includes Standard DDP and Megatron Distributed
Optimizer. Adding Megatron-FSDP to this coverage requires FSDP-specific communication producers,
Analysis support, and GPU evidence. Torch FSDP2 acceptance follows resolution of the current
`finish_grad_sync` signature incompatibility and GPU validation.

The machine-checkable producer inventory is
`tests/megalens/fixtures/probe_scan_gate.json`. `megatron/megalens/event_catalog.py` defines typed
aggregation roles for the async and nested events that require special counting rules; it is a
focused interpretation catalog rather than the complete event list.

The validated Qwen3 CP2×DP8 Distributed Optimizer path reuses the model and DP lifecycle Probe
families. Its DP×CP process group is visible through group size, peers, payload, and completion
fields, and its offline view uses DP and Hybrid Analysis. Additional CP diagnosis can add
CP-specific producers and analysis rules.

## Interpretation and configuration notes

- Communication launch spans represent the framework or API boundary. Completion events represent
  an existing `Work.wait()` or stream dependency. Combine framework launch and completion events
  with the CUPTI device timeline when measuring physical overlap. Confirm IB/RDMA routing through
  `NET/IB` or equivalent backend routing fields.
- Choose one PyTorch-profiler owner. MegaLens kernel capture with mode 1 and `auto` or `on` is mutually
  exclusive with Megatron `profile: true` plus `use_pytorch_profiler: true`. Framework Probe remains
  available with `trace_cupti_kernels: "off"`.
- Validated CUDA Graph coverage includes local layerwise full Graph and Transformer Engine whole-layer
  Graph. Graph-external framework scopes remain visible, and CUPTI provides best-effort rank-local
  kernel capture inside replay. Confirm capture through actual `record_type="cuda_kernel"` records in
  the shard. Graph-node attribution requires a stable node ID and a kernel-mapping contract.
- Hardware fields follow the providers available on the host. Available counters continue when at
  least one provider is usable. With no usable provider, MegaLens emits a warning and framework
  tracing continues.
- FlagGems selects operator implementations, FlagCX selects the distributed backend, and MegaLens
  Trace controls observation. Keep their configuration and evidence claims separate.
- Cross-GPU CUDA kernel timestamps use rank-local profiler coordinates. Cross-device timing claims
  require an explicit device-clock calibration.

For exact CLI defaults, inspect `megatron/training/arguments.py` or use `--help` on the existing
training entry point. Analyzer options are available through:

```bash
python -m megatron.megalens.analyzer --help
```
