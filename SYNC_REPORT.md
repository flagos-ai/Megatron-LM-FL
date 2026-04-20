# Upstream Sync Report: core_v0.15.0rc7 → core_v0.16.1

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Repo setup | ✅ | Branches: base (v0.15.0rc7), dev (core_v0.16.1), main (fork) |
| 2. Identify plugin changes | ✅ | PLUGIN_CHANGES.md catalogued all fork modifications |
| 3. Merge & conflict resolution | ✅ | Merge commit: 196656807 |
| 4. Plugin integrity verification | ✅ | platform: OK, override: OK, features: OK, override body sync: OK |
| 5. Stale reference fixes | ✅ | 1 stale ref fixed (ModelCommProcessGroups → ProcessGroupCollection) |
| 6. Build & import verification | ✅ | pip install: OK, import: OK, unit tests: 50+ passed |
| 7. FlagScale training upgrade handoff | ✅ | FlagScale cloned, ready for flagscale-train-upstream-sync |

## Sync Commits

```
51429fb83 fix(compat): add get_standard_config_overrides to optimizer/__init__.py
ac0a1432e fix(compat): add get_megatron_muon_optimizer backward-compat shim
1b2dee9d9 fix(plugin): remove stale ModelCommProcessGroups import
cfd2c94b5 fix(plugin): sync override bodies with upstream changes
9e06c54a9 fix(plugin): patch new torch.cuda calls and verify decorator chain
196656807 Stage 3: merge upstream and resolve all conflicts
3dbac4533 patch(plugin): add new plugin and fork-specific files
9873afcab Stage 2: Add PLUGIN_CHANGES.md cataloguing all fork modifications
ccbe78e67 Stage 1: Add SYNC_POINT.md for core_v0.16.1 upstream sync
```

## Plugin System Status

- Platform mechanism (cur_platform): ✅ intact
- Override mechanism (@overridable/@override): ✅ intact, 4 override bodies synced with upstream
- New features (dualpipev, hetero): ✅ intact
- pip install + import megatron.core: ✅ functional

## Override Body Sync Details

| Override | File | Action |
|----------|------|--------|
| `_allreduce_embedding_grad` | plugin/distributed/finalize_model_grads.py | Added MTP branch + config param |
| `_is_in_embd_group` | plugin/models/common/language_module/language_module.py | Added mtp_process early return |
| `setup_embeddings_and_output_layer` | plugin/models/common/language_module/language_module.py | Fixed post_process condition for MTP |
| `count_zeros_fp32` | plugin/optimizer/clip_grads.py | Added tp_group parameter |

## Backward-Compat Shims Added

| Shim | File | Reason |
|------|------|--------|
| `get_megatron_muon_optimizer` | core/optimizer/muon.py | Removed in upstream refactor, needed by training code |
| `get_standard_config_overrides` | core/optimizer/__init__.py | From newer upstream commit, needed by training code |

## Rollback Info

- Merge commit: `196656807`
- Rollback command: `git revert -m 1 196656807`
