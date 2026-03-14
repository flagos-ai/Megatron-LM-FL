Megatron-LM Unit Test Authoring Guide
=====================================

This document explains how to add and integrate unit tests under `tests/unit_tests` so that they are easy to run locally and correctly discovered and executed by CI.

Directory layout and naming conventions
---------------------------------------

- **Location**: All unit tests live in `tests/unit_tests` and its subdirectories, for example:
  - `tests/unit_tests/test_basic.py`
  - `tests/unit_tests/transformer/test_attention.py`
  - `tests/unit_tests/models/test_gpt_model.py`
- **File names**:
  - Use the `test_xxx.py` naming convention (so that `pytest` can auto-discover them).
  - For an existing module, prefer adding tests in the corresponding subdirectory instead of dropping everything into the root.
- **Test names**:
  - Function names must start with `test_`, for example:

    ```python
    def test_my_feature():
        ...
    ```

  - Or use a class whose name starts with `Test`, and define `test_` methods inside:

    ```python
    class TestMyFeature:
        def test_case_1(self):
            ...
    ```

Workflow config and `.github/configs`
-------------------------------------

Unit-test GitHub workflows are parameterized by small config files under `.github/configs/`.  
For CUDA-based unit tests the relevant file is:

- `.github/configs/cuda.yml`
  - Defines:
    - `ci_image`: Docker image used to run tests.
    - `runner_labels`: labels for self-hosted runners.
    - `container_volumes` / `container_options`: how the container is started.
    - `device_types`: which device types (e.g. `a100`) are used.
  - The `test_matrix.unit.ignored_tests` section is especially relevant for unit tests:
    - It is a list of test file paths under `tests/unit_tests/...`.
    - These paths are passed into the reusable workflow `unit_tests_common.yml` as the `ignored_tests` input.
    - The workflow then converts this list into `pytest` options of the form `--deselect=<path>`, so those tests are **not** run in CI for that hardware/platform.

**How to use `ignored_tests` (when absolutely necessary):**

- Prefer to **fix** flaky tests or use `pytest` markers (`flaky`, `flaky_in_dev`, `internal`, `experimental`) first.
- If a test is known to be broken or too expensive on a specific platform and must be skipped at the CI-infra level:
  1. Add its file path (relative to repo root, e.g. `tests/unit_tests/transformer/test_attention.py`) to the `test_matrix.unit.ignored_tests` list in `.github/configs/cuda.yml`.
  2. Make sure the path matches the actual location of the test file.
  3. Commit the change so the CI workflow will start skipping it on that platform.

Running unit tests locally
--------------------------

- **Run a single test file**:

  ```bash
  torchrun --nproc_per_node=8 -m pytest tests/unit_tests/xxx.py
  ```

When in doubt, find a similar existing test in the tree and follow the same style and patterns.

