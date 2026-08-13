# CI Testing Guide

This guide describes the configuration-driven GitHub Actions test system, how
to add tests, and how to reproduce CI on a development machine. The commands
below are intended to exercise the same setup and runner scripts as CI.

## CI architecture

Each hardware platform has three entry points:

| Component | Location | Responsibility |
| --- | --- | --- |
| Platform workflow | `.github/workflows/all_tests_<platform>.yml` | Triggers, concurrency, and suite enablement |
| Platform config | `.github/configs/<platform>.yml` | Image, runner labels, container settings, process count, and test matrix |
| Platform setup | `.github/scripts/set_env_<platform>.sh` | Runtime variables, dependencies, device validation, and project installation |

The platform workflow calls `.github/workflows/all_tests_common.yml`, which
loads the platform config and dispatches the reusable unit and functional
workflows. Platform-specific behavior belongs in the config or setup script;
the common workflows should not need a new hardware conditional.

The execution path is:

```text
all_tests_<platform>.yml
  -> all_tests_common.yml
     -> lint_common.yml
     -> unit_tests_common.yml
        -> set_env_<platform>.sh
        -> tests/test_utils/runners/run_ci_unit_tests.sh
     -> functional_tests_common.yml
        -> set_env_<platform>.sh
        -> tests/functional_tests/shell_test_utils/run_ci_test.sh
```

Unit tests run before functional tests when both suites are enabled. A failed
unit matrix prevents the functional matrix from starting.

## Platform configuration contract

Start from `.github/configs/template.yml`. The fields used by the common
workflow are:

| Field | Meaning |
| --- | --- |
| `setup_script` | Repository-relative platform setup script |
| `test_matrix.unit.nproc_per_node` | Number of distributed processes used by each unit-test group |
| `ci_image` | Container image used by test jobs |
| `runner_labels` | Labels that select the self-hosted runner |
| `container_volumes` | Host-to-container mounts |
| `container_options` | Runtime, device, shared-memory, and security options |
| `device_types` | Device values used to expand the test matrix |
| `test_matrix.unit.groups` | Named unit-test paths |
| `test_matrix.unit.pytest_extra_args` | Additional pytest arguments |
| `test_matrix.unit.ignored_tests` | Platform-specific file or node exclusions |
| `test_matrix.functional.train` | Functional and benchmark cases |

`device_types` drives both reusable workflow matrices. Some existing platform
configs still carry a legacy `test_matrix.unit.devices` value, but the common
workflow does not read it. Do not add that field to new configs.

The setup script receives the suite and distributed process count:

```text
CI_TEST_SUITE=unit|functional|build
CI_NPROC_PER_NODE=<positive integer>
```

The unit-test runner additionally receives the platform, device, and group
variables used for test selection and coverage naming:

```text
CI_PLATFORM=<platform>
CI_DEVICE=<device type>
CI_TEST_GROUP=<unit-test group>
```

It must preserve the vendor-provided accelerator runtime and PyTorch stack,
prepare only the selected suite, install this project, and fail early when the
configured devices are unavailable. Use helpers from
`.github/scripts/set_env_common.sh`. `ci_export_env` exports values in the
current process and writes them to `GITHUB_ENV` when running in Actions.

Use an immutable, validated image tag. The image should already contain the
vendor PyTorch build, communication library, TransformerEngine implementation,
fused operators, and expensive build dependencies. Setup scripts should not
silently replace that stack with packages from public PyPI.

## Adding a unit test

There is no separate unit-test registry. A test enters CI when pytest discovers
its name and a unit group in the target platform config selects its path.

Put the test under `tests/unit_tests/` in the directory that owns the behavior.
Name files `test_*.py`, functions and methods `test_*`, and classes `Test*`.
Prefer extending the nearest existing test module instead of adding an
unrelated root-level file.

### How CI discovers and starts the test

The complete unit-test selection path is:

```text
.github/configs/<platform>.yml
  test_matrix.unit.groups[].path
    -> all_tests_common.yml reads the groups
    -> unit_tests_common.yml creates one job per device and group
    -> CI_TEST_PATH=<group path>
    -> run_ci_unit_tests.sh expands paths and platform exclusions
    -> torch.distributed.run starts CI_NPROC_PER_NODE processes
    -> every rank runs the same pytest selection
```

For example, a new file at
`tests/unit_tests/models/test_my_model.py` is automatically selected when the
platform contains:

```yaml
- name: models
  path: tests/unit_tests/models/
```

No workflow edit is needed in that case. The common directory mapping is:

| Test location | Group |
| --- | --- |
| `tests/unit_tests/test_*.py` | `core` |
| `tests/unit_tests/models/` | `models` |
| `tests/unit_tests/distributed/` | `distributed` |
| `tests/unit_tests/dist_checkpointing/` | `dist_checkpointing` |
| `tests/unit_tests/tensor_parallel/` | `tensor_parallel` |
| `tests/unit_tests/pipeline_parallel/` | `pipeline_parallel` |
| `tests/unit_tests/data/` | `data` |
| `tests/unit_tests/fusions/` | `fusions` |
| `export/`, `post_training/`, `tokenizers/`, `utils/` | `others` |

The platform config remains authoritative. A platform may omit a complete
group when its runtime cannot execute that boundary. A root glob such as
`tests/unit_tests/test_*.py` also does not include tests in subdirectories.

If a new top-level directory is not covered, add a unique group to every
platform config that should run it:

```yaml
test_matrix:
  unit:
    groups:
      - name: optimizer
        path: tests/unit_tests/optimizer/
        description: Optimizer tests
```

A group `path` may contain multiple repository-relative paths separated by
spaces. Directory paths are collected recursively.

Before submitting a new test:

1. Check the group paths in every applicable `.github/configs/<platform>.yml`.
2. Run `python3 -m pytest --collect-only <path>` in the platform container.
3. Check that `test_matrix.unit.ignored_tests` does not exclude the file or
   node ID.
4. Run the complete affected group through `run_ci_unit_tests.sh` using the
   CI-parity command below.

Running only `pytest <node>` proves the test can execute, but does not prove CI
selects it. The group run is the final collection check.

### Distributed test requirements

CI launches every unit group through `torch.distributed.run`, including tests
that do not explicitly use collectives. Every rank collects and executes the
same pytest selection. Distributed tests must:

- enter collectives and process-group creation in the same order on all ranks;
- destroy process groups in fixture teardown, including exception paths;
- avoid rank-local skips after another rank has entered a collective;
- use bounded timeouts for network, subprocess, and rendezvous operations;
- avoid multiplying large worker pools by the number of torchrun ranks; and
- use temporary paths whose ownership and synchronization are explicit.

A failure on one rank often appears on other ranks as a later rendezvous or
collective timeout. Diagnose the first rank-specific exception instead of the
final `ChildFailedError` or SIGTERM cascade.

### Platform exclusions

The CI runner accepts ignored entries as either a complete file or a pytest
node ID:

```yaml
ignored_tests:
  - tests/unit_tests/vendor_only/test_kernel.py
  - tests/unit_tests/test_optimizer.py::test_vendor_specific_path
```

Complete files become `--ignore`; node IDs become `--deselect`. Add a
platform-specific exclusion only when the platform cannot support the behavior
or a tracked infrastructure limitation makes the test unsafe. Include a short
reason next to the entry. Do not add platform conditionals to Megatron source
code to make a CI test disappear.

## Adding a functional test

A functional or benchmark case lives at:

```text
tests/functional_tests/test_cases/<model>/<test_case>/
  model_config.yaml
  golden_values_<environment>_<platform>.json
```

Use a nearby case with the same model and test type as the starting point.
`model_config.yaml` contains:

- `ENV_VARS`: variables required by the training and validation scripts.
- `MODEL_ARGS`: arguments passed to the training entry point.
- `TEST_TYPE`: usually `regular`; resume and release modes are also supported.
- `METRICS`: TensorBoard metrics validated by a regular test.

Paths embedded in `MODEL_ARGS`, such as `--data-path` and
`--tokenizer-model`, must match `container_volumes` in the platform config.
Validate both the host source and the in-container destination before running
the workflow.

Register the case in each applicable platform config:

```yaml
test_matrix:
  functional:
    train:
      - model: gpt
        test_case: my_gpt_case
        training_script: pretrain_gpt.py
        n_repeat: 1
        golden_environment: dev
        golden_platform: my_platform
        enable_lightweight_mode: false
        gpus_per_node: 8
```

Optional matrix values and their defaults are:

| Field | Default |
| --- | --- |
| `n_repeat` | `1` |
| `golden_environment` | `dev` |
| `golden_platform` | `dgx_a100` |
| `enable_lightweight_mode` | `false` |
| `data_path` | `/opt/data/datasets` |
| `data_cache_path` | `/tmp/data_cache` |
| `checkpoint_load_path` | `/tmp/checkpoints` |
| `tensorboard_subpath` | `tensorboard` |

`gpus_per_node` is required for every functional matrix entry and must be a
positive integer. Set it from the topology used by that case, not from the
unit-test process count. For example, TP1/PP1 uses one device and TP2/PP2 uses
four devices.

### Regular functional cases

A regular case extracts the configured TensorBoard metrics and compares them
with `golden_values_<environment>_<platform>.json`. Generate a baseline from a
complete, reviewed run on the target software and hardware stack. Do not update
golden values merely to hide an unexplained regression.

The exact check and the approximate check are separate. If a backend is known
to have nondeterministic reduction ordering, record that explicitly in the case
configuration and retain the approximate comparison rather than weakening the
shared checker.

### Benchmark cases

Set `ENV_VARS.BENCHMARK_TEST: 1`. A benchmark golden file is platform-specific:

```json
{
  "my_platform": {
    "my_device": {
      "elapsed time per iteration (ms):": {
        "values": [100.0, 98.0, 97.0],
        "threshold": {"type": "upper_bound", "tolerance": 0.1}
      },
      "throughput per GPU (TFLOP/s/GPU):": {
        "values": [10.0, 10.2, 10.3],
        "threshold": {"type": "lower_bound", "tolerance": 0.1}
      }
    }
  }
}
```

The benchmark checker discards the first five warmup observations before
comparing the averages. Record enough iterations to leave a representative
steady-state sample. Never reuse a performance baseline from a different
accelerator model.

## Running tests on a development machine

### Recommended: reproduce the CI container

Use the image, options, and mounts from `.github/configs/<platform>.yml`. This
is the supported way to reproduce CI because a host Python environment usually
does not contain the same vendor runtime or operator implementations.

Before starting the container:

1. Verify all host paths in `container_volumes` exist and contain the expected
   datasets, tokenizers, libraries, or checkpoints.
2. Verify the required device nodes exist and are idle.
3. Pull the configured image without changing its tag.
4. Mount the repository into the container and set it as the working directory.

For the current Hygon BW1000 config, run this at the repository root:

```bash
docker run --rm -it \
  --ipc=host \
  --shm-size=500g \
  --hostname megatron_cicd \
  --user root \
  --ulimit nofile=65535:65535 \
  --env DTK_HOME=/opt/dtk \
  --env GEMS_VENDOR=amd \
  --env ROCM_PATH=/opt/dtk \
  --env HIP_PATH=/opt/dtk/hip \
  --env HSA_PATH=/opt/dtk/hsa \
  --env HIP_CLANG_PATH=/opt/dtk/llvm/bin \
  --env DEVICE_LIB_PATH=/opt/dtk/amdgcn/bitcode \
  --device=/dev/kfd \
  --device=/dev/mkfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal \
  -v /data/megatron-ci-assets-20260726/baai_datasets:/home/gitlab-runner/data \
  -v /data/megatron-ci-assets-20260726/baai_tokenizers:/home/gitlab-runner/tokenizers \
  -v /data/megatron-ci-assets-20260726/baai_datasets:/opt/data/datasets \
  -v /data/megatron-ci-assets-20260726/baai_tokenizers:/opt/data/tokenizers \
  -v "$PWD:/workspace/Megatron-LM-FL" \
  -w /workspace/Megatron-LM-FL \
  harbor.baai.ac.cn/flagos-dev/megatron-lm-fl:manual-20260728-hygon-dev \
  bash
```

The setup script prepends the DTK executable and library directories to
`PATH` and `LD_LIBRARY_PATH`. The corresponding full values are also declared
in `.github/configs/hygon.yml`; that config remains authoritative if the DTK
layout changes.

Do not replace the Hygon device options with `--gpus all`; BW1000 is exposed by
the DTK runtime through `/dev/kfd`, `/dev/mkfd`, `/dev/dri`, and `/opt/hyhal`.
For another platform, copy the exact values from that platform config instead
of adapting this example by guesswork.

### Run a unit-test group with CI parity

Inside the container, at the repository root:

```bash
export GITHUB_WORKSPACE="$PWD"
export CI_TEST_SUITE=unit
export CI_PLATFORM=hygon
export CI_DEVICE=bw1000
export CI_NPROC_PER_NODE=8
export CI_TEST_GROUP=models
export CI_TEST_PATH='tests/unit_tests/models/'

source .github/scripts/set_env_hygon.sh

export CI_IGNORED_TESTS="$(yq -o=json -I=0 \
  '.test_matrix.unit.ignored_tests // []' .github/configs/hygon.yml)"
export CI_PYTEST_EXTRA_ARGS="$(yq -o=json -I=0 \
  '.test_matrix.unit.pytest_extra_args // []' .github/configs/hygon.yml)"

tests/test_utils/runners/run_ci_unit_tests.sh
```

Use `source`, not `bash`, for local setup so exported runtime variables remain
in the current shell. GitHub Actions invokes the script with `bash` because
`ci_export_env` writes the variables to `GITHUB_ENV` for the next step.

Change `CI_TEST_GROUP` and `CI_TEST_PATH` together to select another group. A
single test can be selected without changing the script:

```bash
export CI_TEST_GROUP=debug_one_test
export CI_TEST_PATH='tests/unit_tests/models/test_gpt_model.py'
tests/test_utils/runners/run_ci_unit_tests.sh
```

The runner still applies the platform ignore list, pytest arguments, distributed
process count, hard-timeout wrapper, and coverage settings.

### Run a functional or benchmark case with CI parity

Start a fresh container when switching from unit setup to functional setup.
Then run:

```bash
export GITHUB_WORKSPACE="$PWD"
export CI_TEST_SUITE=functional
export CI_PLATFORM=hygon
export CI_DEVICE=bw1000
export CI_NPROC_PER_NODE=1

source .github/scripts/set_env_hygon.sh

MODEL=gpt
TEST_CASE=qwen3_0p6b_mcore_te_tp1_pp1_no_mmap_bin_files_hygon
GOLDEN_VALUES_PATH=./tests/functional_tests/test_cases/gpt/$TEST_CASE/golden_values_dev_bw1000.json
GPUS_PER_NODE=1
OUTPUT_DIR="./test_output/$TEST_CASE"
mkdir -p "$OUTPUT_DIR"

bash tests/functional_tests/shell_test_utils/run_ci_test.sh \
  "DATA_PATH=/opt/data/datasets" \
  "DATA_CACHE_PATH=/tmp/data_cache" \
  "OUTPUT_PATH=$OUTPUT_DIR" \
  "TENSORBOARD_PATH=$OUTPUT_DIR/tensorboard" \
  "CHECKPOINT_SAVE_PATH=$OUTPUT_DIR/checkpoints" \
  "CHECKPOINT_LOAD_PATH=/tmp/checkpoints" \
  "TRAINING_SCRIPT_PATH=pretrain_gpt.py" \
  "TRAINING_PARAMS_PATH=./tests/functional_tests/test_cases/$MODEL/$TEST_CASE/model_config.yaml" \
  "GOLDEN_VALUES_PATH=$GOLDEN_VALUES_PATH" \
  "N_REPEAT=1" \
  "ENABLE_LIGHTWEIGHT_MODE=false" \
  "GPUS_PER_NODE=$GPUS_PER_NODE"
```

For the Hygon benchmark, start a fresh container and replace the case-specific
values above before sourcing the setup script:

```bash
TEST_CASE=qwen3_0p6b_mcore_te_tp2_pp2_benchmark_hygon
GOLDEN_VALUES_PATH=./tests/functional_tests/test_cases/gpt/$TEST_CASE/golden_values_dev_bw1000.json
export CI_NPROC_PER_NODE=4
GPUS_PER_NODE=4
```

and pass those values to the same runner command. The matrix entry in
`.github/configs/hygon.yml` is the source of truth for the case name, golden
target, repeat count, and distributed process count.

The functional runner may normalize values in `model_config.yaml` with
in-place `yq` edits. Run it from a disposable worktree or inspect `git diff`
afterward and keep only intentional test-case changes.

### Faster direct debugging

After the platform setup has succeeded, a direct pytest command can shorten the
edit/debug loop:

```bash
python3 -m torch.distributed.run --nproc_per_node=8 \
  -m pytest -v \
  tests/unit_tests/models/test_gpt_model.py::TestGPTModel::test_constructor \
  -p no:randomly
```

This is not a CI-equivalent result: it does not automatically apply the full
group ignore list, extra pytest arguments, coverage configuration, or platform
hard timeout. Run the CI runner script before submitting the change.

### Native host setup

Running outside the configured container is supported only when the host
already matches the platform image: vendor driver/runtime, vendor PyTorch,
collective library, TransformerEngine implementation, fused operators, Python,
and test tools must all be compatible. Set the same `CI_*` variables and
`source` the platform setup script. Some setup scripts install system tools or
write under `/usr/local`, so an isolated container is normally safer.

Do not use `tests/test_utils/runners/run_tests.sh` as proof of compatibility
with this workflow. It is a legacy convenience runner with a separate platform
configuration and does not reproduce every platform supported by the new
GitHub Actions path.

## Adding a new platform

1. Copy `.github/configs/template.yml` to
   `.github/configs/<platform>.yml` and fill every required field.
2. Add `.github/scripts/set_env_<platform>.sh`. Reuse
   `set_env_common.sh`, preserve the image's vendor packages, and implement the
   `unit`, `functional`, and `build` suite branches that the platform supports.
3. Add `.github/workflows/all_tests_<platform>.yml` that calls
   `all_tests_common.yml` with `platform: <platform>`.
4. Register an online self-hosted runner with every label in `runner_labels`.
5. Validate the container options, mounts, device count, one collective, one
   unit group, one regular functional case, and one benchmark independently.
6. Enable the full matrices only after those smoke tests pass.

No change to `all_tests_common.yml`, `unit_tests_common.yml`, or
`functional_tests_common.yml` should be required for a normal platform
addition.

## Failure diagnosis

Use the stage that failed to classify the problem:

| Symptom | First place to investigate |
| --- | --- |
| Job remains queued | Runner online state and exact `runner_labels` match |
| Setup import failure | Image contents and platform setup script |
| Device or collective preflight timeout | Host driver, device nodes, stale processes, and communication runtime |
| Test exits with `124` or `137` | Platform timeout, rank failure, or host/runtime recovery issue |
| Dataset/tokenizer not found | Host asset path and `container_volumes` destination |
| `yq` expression behaves unexpectedly | Confirm mikefarah `yq` v4, not the Python package |
| `uv` downloads another Python | Use the platform setup; Hygon intentionally installs a compatibility shim |
| Functional comparison fails | Training log, TensorBoard extraction, and the selected golden file |
| Benchmark comparison fails | Warmup count, complete iteration list, target hardware, and threshold direction |

Functional Actions artifacts contain the case log directory for seven days.
The rank-zero live log is not always sufficient: inspect all rank logs under:

```text
test_output/<test_case>/logs/<repeat>/<run-id>/attempt_0/<rank>/stderr.log
```

When one rank crashes, the other ranks may report only a rendezvous or
collective timeout. The first rank-specific exception is the useful failure.
If the platform preflight itself fails before pytest or training starts, treat
that as an environment/runtime failure rather than changing test expectations.

## Submission checklist

- The new test is discovered by the intended group or functional matrix.
- The platform config uses the correct image, labels, options, and mounts.
- The setup script succeeds and validates the configured device count.
- Unit tests were run through `run_ci_unit_tests.sh`.
- Functional tests were run through `run_ci_test.sh`.
- Golden values came from a reviewed run on the declared environment/platform.
- Benchmark values contain steady-state samples after warmup.
- Platform exclusions are narrow and documented.
- No vendor package was silently replaced during setup.
- The full platform workflow was run with both unit and functional suites
  enabled before merge.
