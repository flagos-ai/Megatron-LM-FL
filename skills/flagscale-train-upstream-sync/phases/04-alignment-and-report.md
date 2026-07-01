# Phase 4: Alignment & Report

Stages 7-9 of the FlagScale training upstream sync workflow.

---

### Stage 7: Precision Alignment Verification

After training passes on the dev branches, verify that the upgrade does not regress training
quality by comparing metrics between pre-upgrade (baseline) and post-upgrade (comparison) over a
short run. This catches silent numerical divergences that functional tests miss — e.g. a changed
default, a reordered reduction, or a dropped scaling factor.

#### Branch mapping

| Role | FlagScale branch | Megatron-LM-FL branch |
|------|------------------|-----------------------|
| Baseline (pre-upgrade) | `main` | `main` |
| Comparison (post-upgrade) | `$FS_DEV_BRANCH` | `$MG_DEV_BRANCH` |

`$FS_DEV_BRANCH` and `$MG_DEV_BRANCH` are the branch names specified by the user in Stage 1 Step 3.

#### Procedure

**Before each run, verify branch consistency.** Stage 7 switches branches multiple times
(main for baseline, dev for comparison). Every time you switch, confirm both repos landed on the
correct pair — a forgotten `git checkout` or `pip install -e .` in one repo silently corrupts
the comparison.

```bash
FS_BRANCH=$(cd "$FLAGSCALE_DIR" && git branch --show-current)
MG_BRANCH=$(cd "$MG_FL_DIR" && git branch --show-current)
echo "About to run: FlagScale=$FS_BRANCH, Megatron-LM-FL=$MG_BRANCH"

# For baseline: both must be "main"
# For comparison: FlagScale=dev-train-*, Megatron-LM-FL=dev-*
# Abort if the pair doesn't match the intended run
```

For **each model config** that was validated in Stage 6:

1. **Run baseline (pre-upgrade):**
   ```bash
   cd "$MG_FL_DIR" && git checkout main && pip install -e .
   cd "$FLAGSCALE_DIR" && git checkout main
   # Modify train_iters to 10 in the YAML if needed, set a unique exp_dir
   python run.py --config-path <yaml_dir> --config-name <yaml_name>
   ```
   Capture the training log. Extract per-iteration: `lm loss`, `grad norm`, `throughput (TFLOP/s/GPU or tokens/s)`, `mem (allocated)`.

2. **Run comparison (post-upgrade):**
   ```bash
   cd "$MG_FL_DIR" && git checkout $MG_DEV_BRANCH && pip install -e .
   cd "$FLAGSCALE_DIR" && git checkout $FS_DEV_BRANCH
   python run.py --config-path <yaml_dir> --config-name <yaml_name>
   ```
   Capture the same metrics.

3. **Switch back to dev branches** after both runs complete:
   ```bash
   cd "$MG_FL_DIR" && git checkout $MG_DEV_BRANCH && pip install -e .
   cd "$FLAGSCALE_DIR" && git checkout $FS_DEV_BRANCH
   ```

#### Comparison table

Produce a side-by-side table for each model:

```markdown
## <Model Name> — Precision Alignment (10 steps)

| Iter | lm loss (base) | lm loss (dev) | Δ | grad norm (base) | grad norm (dev) | throughput (base) | throughput (dev) |
|------|----------------|---------------|---|------------------|-----------------|-------------------|------------------|
| 1    |                |               |   |                  |                 |                   |                  |
| ...  |                |               |   |                  |                 |                   |                  |
| 10   |                |               |   |                  |                 |                   |                  |
```

#### Acceptance criteria

- **lm loss:** Per-step values should match within ±5% relative difference. Small divergence in
  early steps is acceptable if the trend converges. Large divergence (>10%) indicates a real
  regression — investigate before proceeding.
- **grad norm:** Should be in the same order of magnitude. Spikes in one but not the other
  warrant investigation.
- **throughput:** Should be comparable (±10%). Significant drops may indicate an unintended
  code path change (e.g. falling back to unfused kernels).

If metrics diverge beyond thresholds, do NOT proceed to the sync report. Investigate the root
cause — common culprits: changed default hyperparameters, different initialization seeds,
reordered collective operations, missing FlagScale optimizations.

---

### Stage 8: FlagScale-Specific Feature Verification (Hetero-train & Engram)

Stages 6–7 validate the core GPT training path. But FlagScale has its own features — hetero-train
(heterogeneous parallel training) and engram (n-gram embedding augmentation) — that exercise
different code paths, custom model builders, and custom argument groups. An upstream sync can
silently break these features even when standard GPT training passes. This stage catches those
regressions.

The CI/CD functional tests already define YAML configs and gold loss values for both features.
Reuse them rather than writing new configs from scratch.

#### Reference configs from CI/CD

| Feature | CI config dir | Top-level YAML | Train YAML | Entrypoint | Gold values |
|---------|--------------|----------------|------------|------------|-------------|
| Hetero-train | `tests/functional_tests/hetero_train/aquila/conf/` | `tp2pp1_tp4pp1_tp2pp1.yaml` | `train/tp2pp1_tp4pp1_tp2pp1.yaml` | `flagscale/train/megatron/train_gpt.py` | `gold_values/tp2pp1_tp4pp1_tp2pp1.json` |
| Engram | `tests/functional_tests/train/deepseek/conf/` | `tp2_pp2_ep2_engram.yaml` | `train/tp2_pp2_ep2_engram.yaml` | `flagscale/train/megatron/train_engram.py` | `gold_values/tp2_pp2_ep2_engram.json` |

The CI configs use CI-specific paths (e.g. `/home/gitlab-runner/data/...`). When running locally,
create adapted copies under `test_yamls/` that point to your local data and tokenizer paths, but
keep all model/parallelism/training parameters identical to the CI originals.

Key parameters to preserve exactly:
- Hetero-train: `hetero_pipeline_layer_split`, `hetero_process_meshes`, `hetero_device_types`,
  `recompute_*_per_stage_micro_batch`, `num_layers: 8`, `global_batch_size: 1024`
- Engram: `use_engram: true`, `engram_vocab_size`, `engram_layer_ids`, `engram_embedding_parallel_size`,
  `num_experts: 4`, `moe_layer_freq`, `num_layers: 4`

#### Step 0: Branch consistency gate

Same as Stage 6 Step 0 — verify both repos are on matching branch pairs before proceeding.

#### Step 1: Functional run on dev branches

Run each feature config on the dev branches. The goal is to confirm the feature works at all
after the upstream sync — not precision alignment yet.

```bash
# Ensure dev branches
cd "$FLAGSCALE_DIR" && git checkout $FS_DEV_BRANCH
cd "$MG_FL_DIR" && git checkout $MG_DEV_BRANCH && pip install -e .

# Hetero-train
cd "$FLAGSCALE_DIR"
python -m flagscale.run --config-path <local_hetero_yaml_dir> --config-name <hetero_yaml_name>

# Engram
python -m flagscale.run --config-path <local_engram_yaml_dir> --config-name <engram_yaml_name>
```

If either run fails:
1. Read the full error traceback.
2. Triage: is it a megatron.core/plugin issue (fix in Megatron-LM-FL) or a training/model issue
   (fix in FlagScale)? Apply Rule #6.
3. Fix, commit, and re-run until both features complete all training iterations without errors.
4. Common failure modes after upstream sync:
   - **Hetero-train:** Changed `TransformerConfig` fields, modified recompute argument handling,
     altered pipeline schedule interfaces. The hetero code patches these extensively.
   - **Engram:** Changed model builder signatures, modified `pretrain()` call interface,
     altered loss function signatures, new required arguments in `forward_step`.

#### Step 2: Precision alignment (main vs dev)

Once both features run successfully on dev, compare loss curves against main (pre-upgrade) to
detect silent numerical regressions.

For **each feature**:

1. **Run baseline (main branches):**
   ```bash
   cd "$MG_FL_DIR" && git checkout main && pip install -e .
   cd "$FLAGSCALE_DIR" && git checkout main
   # Run with the same adapted YAML (ensure train_iters=10 or train_samples produces ~10 steps)
   python -m flagscale.run --config-path <yaml_dir> --config-name <yaml_name>
   ```
   Extract per-iteration `lm loss` from the training log.

2. **Run comparison (dev branches):**
   ```bash
   cd "$MG_FL_DIR" && git checkout $MG_DEV_BRANCH && pip install -e .
   cd "$FLAGSCALE_DIR" && git checkout $FS_DEV_BRANCH
   python -m flagscale.run --config-path <yaml_dir> --config-name <yaml_name>
   ```
   Extract the same metrics.

3. **Switch back to dev branches** after both runs.

#### Comparison tables

Produce a table for each feature:

```markdown
## Hetero-train (Aquila tp2pp1_tp4pp1_tp2pp1) — Precision Alignment

| Iter | lm loss (main) | lm loss (dev) | Δ% |
|------|----------------|---------------|----|
| 1    |                |               |    |
| ...  |                |               |    |
| 10   |                |               |    |

## Engram (DeepSeek tp2_pp2_ep2_engram) — Precision Alignment

| Iter | lm loss (main) | lm loss (dev) | Δ% |
|------|----------------|---------------|----|
| 1    |                |               |    |
| ...  |                |               |    |
| 10   |                |               |    |
```

#### Acceptance criteria

Same thresholds as Stage 7:
- **lm loss:** Per-step values within ±5% relative difference. Engram's loss curve has a
  characteristic spike at iter 3 (see gold values) — this is expected behavior, not a regression.
  Compare the spike magnitude, not just the smooth steps.
- **Throughput:** Within ±10%.

If a feature diverges beyond thresholds, investigate before proceeding. Do NOT skip a failing
feature — both hetero-train and engram must pass for the upgrade to be considered complete.

#### Optional: Cross-check against CI gold values

If the dev-branch run produces loss values, you can also compare against the CI gold values
stored in `gold_values/*.json`. These are the reference values the CI pipeline uses. A match
(within `np.allclose` tolerance) confirms the upgrade preserves CI-level correctness.

---

### Stage 9: Sync Report

```markdown
# FlagScale Training Upgrade Report: → <TARGET_VERSION>

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Environment setup | ✅/❌ | |
| 2. Three-way merge refs | ✅/❌ | base=$BASE_VERSION, target=$TARGET_VERSION |
| 3. Customization diff | ✅/❌ | N unmodified, M customized, K FS-added |
| 4. Upgrade applied | ✅/❌ | N replaced, M patched, K conflicts resolved |
| 5. Stale reference fixes | ✅/❌ | N stale refs found/fixed |
| 6. Verification | ✅/❌ | syntax: OK, imports: OK, training: OK |
| 7. Precision alignment | ✅/❌ | lm loss Δ < 5%, throughput Δ < 10% |
| 8. Feature verification | ✅/❌ | hetero-train: OK, engram: OK, precision aligned |

## Files Changed
<list of modified files in flagscale/train/megatron/>

## Compatibility Status
- megatron.core imports: ✅ all resolve
- megatron.plugin imports: ✅ all resolve
- Training loop: ✅ runs without errors
- FlagScale-specific features: ✅ preserved (hetero-train, engram verified)
```

