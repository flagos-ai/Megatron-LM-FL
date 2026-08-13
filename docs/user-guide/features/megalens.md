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

`tests/megalens/fixtures/flagscale_single_node_pp2_smoke.yaml` is a runnable two-GPU configuration
that uses the same fields.

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

Use an empty, writable `trace_dir` for each run. A shared directory lets all nodes write into one
location. With node-local storage, collect every rank shard into one directory before aggregation.

Confirm that the run produced non-empty shards:

```bash
export TRACE_DIR=/absolute/path/to/unique-run/traces
export REPORT_DIR=/absolute/path/to/unique-run/reports

find "$TRACE_DIR" -maxdepth 1 -name 'benchmark-*.json' -type f -size +0 -print
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

`--align-framework-timeline` applies an optional cross-rank framework calibration when matching
synchronous TP AllReduce anchors exist. Event durations and CUDA kernel timestamps remain unchanged.

## Supported Probe families

Framework records commonly include `name`, `ph`, `iteration`, `g_rk`, `dp_rk`, `pp_rk`, `tp_rk`,
`dev`, and `rel_ts`. Aggregated events add Chrome Trace fields such as `pid`, `tid`, `ts`, and `dur`.
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
