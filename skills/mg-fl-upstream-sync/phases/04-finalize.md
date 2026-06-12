# Phase 4: Finalize

Stages 8-10 of the Megatron-LM-FL upstream sync workflow.

---

### Stage 8: Merge ${DEV_BRANCH} into main (Tree Replacement Strategy)

When the dev branch (e.g. `dev-0.17.0`) is a **superset** of main — meaning all main's features
have already been incorporated into the dev branch during the preceding stages — use the tree
replacement strategy to create a clean merge commit for PR.

**Prerequisite:** ${DEV_BRANCH} must be complete and verified (Stage 3–7 passed). If ${DEV_BRANCH}
is missing FlagScale features that exist on main, go back and fix it first — do NOT patch during
this stage.

#### Why not `-X theirs`?

`-X theirs` resolves **conflicts** by taking theirs, but **non-conflicting changes from both
sides are still merged**. When both branches independently added the same patch (e.g. `cur_platform`
replacements), git sees them as non-conflicting additions and keeps both copies — resulting in
duplicate imports, duplicate FlagScale Begin/End blocks, and broken code. In our case this
produced 72+ buggy files.

#### Step 1: Create merge branch with tree replacement

```bash
git checkout main
git checkout -b merge-upstream-<version>

# Create a merge commit whose tree is exactly the dev branch's tree.
# "merge -s ours" records both parents but keeps main's tree,
# then "read-tree" replaces the tree with the dev branch's content.
git merge -s ours ${DEV_BRANCH} --no-edit
git read-tree -m -u ${DEV_BRANCH}
git commit --amend --no-edit
```

After this, the working tree is **identical** to `${DEV_BRANCH}`, but the commit has both
`main` and `${DEV_BRANCH}` as parents (preserving full history).

#### Step 2: Verify tree equality

```bash
# Should produce no output — trees are identical
git diff ${DEV_BRANCH} HEAD
```

#### Step 3: Incorporate new main commits (if any)

If `origin/main` received new commits after the merge branch was created:

```bash
git checkout main
git pull origin main
git checkout merge-upstream-<version>
git merge main --no-edit
# Resolve any conflicts (typically few, since dev branch is a superset)
git commit  # if conflicts were resolved
```

#### Step 4: Final verification

```bash
# No conflict markers
grep -rn "<<<<<<" megatron/ tests/ .github/ 2>/dev/null | head

# History is correct — both parents present
git log --oneline --graph -5

# pip install still works
pip install -e . 2>&1 | tail -5
python -c "import megatron.core; import megatron.plugin; print('OK')"
```

**Success criteria:** Merge branch tree equals ${DEV_BRANCH}, no duplicate code blocks, both
parents in history, pip-installable. Push branch and open PR.

---

### Stage 9: Sync Report

Generate a summary of the entire sync operation:

```markdown
# Upstream Sync Report: ${BASE_UPSTREAM_BRANCH} → ${TARGET_UPSTREAM_BRANCH}

| Stage | Status | Notes |
|-------|--------|-------|
| 1. Repo setup | ✅/❌ | |
| 2. Identify plugin changes | ✅/❌ | |
| 3. Merge & conflict resolution | ✅/❌ | N conflicts, P0: N, P1: N, P2: N |
| 4. Plugin integrity verification | ✅/❌ | platform: OK, override: OK, features: OK |
| 5. Stale reference fixes | ✅/❌ | N stale refs found/fixed |
| 6. Build & import verification | ✅/❌ | pip install: OK, import: OK, tests: OK |
| 7. Unit test verification | ✅/❌ | N groups passed, M groups failed |
| 8. Merge into main | ✅/❌ | tree replacement merge, PR opened |
| 9. Sync report | ✅/❌ | this document |
| 10. FlagScale training upgrade handoff | ✅/❌ | → continue with flagscale-train-upstream-sync |

## Plugin System Status
- Platform mechanism (cur_platform): ✅ intact
- Override mechanism (@overridable/@override): ✅ intact
- New features (dualpipev, hetero): ✅ intact
- pip install + import megatron.core: ✅ functional
- pip install + import megatron.plugin: ✅ functional

## Rollback Info
- Merge commit: <sha>
- Rollback command: `git revert -m 1 <sha>`
```

---

### Stage 10: Hand Off to FlagScale Training Upgrade

The Megatron-LM-FL library sync is now complete. The second half of the full upgrade is syncing
FlagScale's training code at `FlagScale/flagscale/train/megatron/` with the corresponding
upstream training code changes. This is a co-equal upgrade step, not just validation.

#### Step 1: Ensure FlagScale is available

The Repo Detection Preamble already attempted to locate FlagScale. If `$FLAGSCALE_DIR` is
empty, clone it now:

```bash
if [ -z "$FLAGSCALE_DIR" ]; then
  echo "FlagScale not found. Cloning..."
  cd "$(dirname $MG_FL_DIR)"
  git clone <flagscale-repo-url> FlagScale
  FLAGSCALE_DIR=$(cd FlagScale && pwd)
  echo "FlagScale: $FLAGSCALE_DIR"
fi

# Verify FlagScale training directory exists
if [ ! -d "$FLAGSCALE_DIR/flagscale/train/megatron" ]; then
  echo "ERROR: $FLAGSCALE_DIR/flagscale/train/megatron/ not found"
  echo "Check that the FlagScale repo is correct and complete."
  exit 1
fi
```

#### Step 2: Verify Megatron-LM-FL is installed and importable

```bash
cd "$MG_FL_DIR"
pip install -e . --no-build-isolation
python3 -c "
import megatron.core
import megatron.plugin
from megatron.plugin.platform import cur_platform
print('megatron.core OK')
print('megatron.plugin OK')
print('cur_platform:', cur_platform)
"
```

#### Step 3: Hand off

Switch to the companion skill `flagscale-train-upstream-sync` to continue. That skill handles:
1. Diffing upstream training code changes between the base and target versions
2. Identifying FlagScale's local modifications to protect
3. Applying upstream training changes while preserving FlagScale customizations
4. Fixing stale references to renamed/removed megatron.core symbols
5. Verifying imports and running a training smoke test

Pass both paths to the companion skill: `MG_FL_DIR` and `FLAGSCALE_DIR`.

**Success criteria:** Megatron-LM-FL is pip-installed and importable, FlagScale repo is available, ready for FlagScale upgrade.

---
