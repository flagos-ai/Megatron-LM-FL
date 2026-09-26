# Phase 1: Setup & Analysis

Stages 1-2 of the Megatron-LM-FL upstream sync workflow.

---

### Stage 1: Repo Setup & Branch Preparation

Run the **Repo Detection Preamble** first to set `MG_FL_DIR` (and `FLAGSCALE_DIR`). If
Megatron-LM-FL doesn't exist, the preamble clones it automatically. Then add the upstream
remote and create the dev and base branches needed for the merge.

**Before any branch operations, ask the user for four parameters:**

1. **Target upstream branch** — the upstream release to sync to
2. **Target upstream commit** — (optional) specific commit SHA on the target branch to checkout;
   if empty, use the branch tip
3. **Base upstream branch** — the upstream release the fork is currently based on
4. **Base upstream commit** — (optional) specific commit SHA on the base branch to checkout;
   if empty, use the branch tip

**Prompt the user:**
> Please specify the following:
> 1. Target upstream branch to sync to (e.g. `core_v0.17.0`, `main`)
> 2. Target upstream commit (leave empty for branch tip, e.g. `abc1234`)
> 3. Base upstream branch the fork is currently based on (e.g. `core_v0.16.1`)
> 4. Base upstream commit (leave empty for branch tip, e.g. `def5678`)

Store the answers and use them throughout **all** subsequent stages:

```bash
TARGET_UPSTREAM_BRANCH="<user answer 1>"   # e.g. core_v0.17.0
TARGET_UPSTREAM_COMMIT="<user answer 2>"   # e.g. abc1234 (empty = branch tip)
BASE_UPSTREAM_BRANCH="<user answer 3>"     # e.g. core_v0.16.1
BASE_UPSTREAM_COMMIT="<user answer 4>"     # e.g. def5678 (empty = branch tip)
DEV_BRANCH="dev"
BASE_BRANCH="base"
```

#### Step 1: Add upstream remote

```bash
cd "$MG_FL_DIR"
git remote -v | grep upstream || \
  git remote add upstream https://github.com/NVIDIA/Megatron-LM.git
git fetch upstream --tags
```

#### Step 2: Create dev branch from target upstream release

The dev branch mirrors the target upstream release exactly — no fork changes.

```bash
# List available upstream core releases for reference
git branch -r | grep upstream/core

# Create dev branch from target upstream, optionally at a specific commit
if [ -n "$TARGET_UPSTREAM_COMMIT" ]; then
  git checkout -b ${DEV_BRANCH} ${TARGET_UPSTREAM_COMMIT}
  echo "Created ${DEV_BRANCH} at commit ${TARGET_UPSTREAM_COMMIT} on upstream/${TARGET_UPSTREAM_BRANCH}"
else
  git checkout -b ${DEV_BRANCH} upstream/${TARGET_UPSTREAM_BRANCH}
  echo "Created ${DEV_BRANCH} at tip of upstream/${TARGET_UPSTREAM_BRANCH}"
fi
```

Record the sync point — create `SYNC_POINT.md` at repo root:
```markdown
# Upstream Sync Point
- Upstream: NVIDIA/Megatron-LM
- Branch: ${TARGET_UPSTREAM_BRANCH}
- Commit SHA: <git rev-parse HEAD>
- Sync Date: <current date>
- Synced By: <user>
```

Verify: `git log --oneline -1 ${DEV_BRANCH}` shows the expected commit.

#### Step 3: Create base branch from fork's original upstream

The base branch represents the upstream version the fork was originally based on — needed for
accurate three-way merges.

```bash
git fetch upstream ${BASE_UPSTREAM_BRANCH}

# Create base branch, optionally at a specific commit
if [ -n "$BASE_UPSTREAM_COMMIT" ]; then
  git checkout -b ${BASE_BRANCH} ${BASE_UPSTREAM_COMMIT}
  echo "Created ${BASE_BRANCH} at commit ${BASE_UPSTREAM_COMMIT} on upstream/${BASE_UPSTREAM_BRANCH}"
else
  git checkout -b ${BASE_BRANCH} upstream/${BASE_UPSTREAM_BRANCH}
  echo "Created ${BASE_BRANCH} at tip of upstream/${BASE_UPSTREAM_BRANCH}"
fi
```

Verify: `git log --oneline -1 ${BASE_BRANCH}` shows the expected commit.

**Success criteria:** `${DEV_BRANCH}` and `${BASE_BRANCH}` branches exist and match their respective upstream releases.

#### Checkpoint: pre-commit & commit

```bash
pre-commit run --all-files
git add -A
git commit -m "Stage 1: repo setup and branch preparation for $(cat SYNC_POINT.md | grep Branch | awk '{print $NF}') upstream sync"
```

---

### Stage 2: Identify Plugin Changes

Diff `${BASE_BRANCH}..main` to catalogue everything the fork added. This becomes the protection checklist
for the merge.

```bash
# Summary of all fork changes
git diff ${BASE_BRANCH}..main --stat

# Full diff for reference
git diff ${BASE_BRANCH}..main > plugin_changes.diff

# Plugin-specific changes
git diff ${BASE_BRANCH}..main --name-status -- 'megatron/plugin/'

# Core modifications (decorated functions, cur_platform replacements)
git diff ${BASE_BRANCH}..main --name-status -- 'megatron/core/'

# CI/CD changes
git diff ${BASE_BRANCH}..main --name-status -- '.github/' 'setup.py' 'pyproject.toml'
```

Save a structured summary to `PLUGIN_CHANGES.md`:
```markdown
# Plugin Changes (${BASE_BRANCH} → main)

## New Files (added by fork)
<files only in main>

## Modified Files in megatron/core/
<files changed — focus on @overridable decorations and cur_platform replacements>

## Plugin Directory Contents
<full listing of megatron/plugin/>

## CI/CD & Build Changes
<changes to .github/, setup.py, pyproject.toml>
```

This record is critical — during Stage 3, use it to verify every plugin change survives the merge.

#### Step 2b: Print categorized diff list to console

After extracting categories, print a full inventory to the console so the user can review all
fork modifications before proceeding to Stage 3. This is the "what are we protecting" checklist.

```bash
echo "================================================================"
echo "  Stage 2: Fork Modification Inventory (main vs ${BASE_BRANCH})"
echo "================================================================"

echo ""
echo "--- Category A: New plugin files ($(wc -l < /tmp/new_plugin_files.txt)) ---"
git diff --name-only --diff-filter=A ${BASE_BRANCH}..main -- 'megatron/plugin/'

echo ""
echo "--- Category A2: New non-plugin fork files ($(wc -l < /tmp/new_other_files.txt)) ---"
git diff --name-only --diff-filter=A ${BASE_BRANCH}..main | grep -v "^megatron/plugin/"

echo ""
echo "--- Category B: Modified core files ($(wc -l < /tmp/modified_core_files.txt)) ---"
git diff --name-only --diff-filter=M ${BASE_BRANCH}..main -- 'megatron/core/'

echo ""
echo "--- Category C: Modified plugin files ---"
git diff --name-only --diff-filter=M ${BASE_BRANCH}..main -- 'megatron/plugin/'

echo ""
echo "--- Category D: Deleted files ---"
git diff --name-only --diff-filter=D ${BASE_BRANCH}..main

echo ""
echo "--- Category E: Other modified files ($(wc -l < /tmp/modified_other_files.txt)) ---"
git diff --name-only --diff-filter=M ${BASE_BRANCH}..main | grep -v "^megatron/core/" | grep -v "^megatron/plugin/"

echo ""
echo "--- Category F: Files with FlagScale Begin/End blocks on main ($(wc -l < /tmp/flagscale_block_files.txt)) ---"
cat /tmp/flagscale_block_files.txt

echo ""
echo "================================================================"
echo "  Summary"
echo "================================================================"
echo "  A  (new plugin):        $(wc -l < /tmp/new_plugin_files.txt)"
echo "  A2 (new other):         $(wc -l < /tmp/new_other_files.txt)"
echo "  B  (modified core):     $(wc -l < /tmp/modified_core_files.txt)"
echo "  C  (modified plugin):   $(git diff --name-only --diff-filter=M ${BASE_BRANCH}..main -- 'megatron/plugin/' | wc -l)"
echo "  D  (deleted):           $(git diff --name-only --diff-filter=D ${BASE_BRANCH}..main | wc -l)"
echo "  E  (other modified):    $(wc -l < /tmp/modified_other_files.txt)"
echo "  F  (FlagScale blocks):  $(wc -l < /tmp/flagscale_block_files.txt)"
echo "================================================================"
```

Review the list. If any category looks unexpected (e.g. missing files, unexpected deletes),
investigate before proceeding. This is the last checkpoint before patching begins.

**Success criteria:** `plugin_changes.diff` and `PLUGIN_CHANGES.md` generated, categorized diff
list reviewed and confirmed by user.

#### Checkpoint: pre-commit & commit

```bash
pre-commit run --all-files
git add -A
git commit -m "Stage 2: catalogue fork plugin changes in PLUGIN_CHANGES.md"
```

---

