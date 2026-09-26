# Phase 1: Setup & Branch Structure

Stages 1-2 of the FlagScale training upstream sync workflow.

---

### Stage 1: Environment & Repo Setup

Run the **Repo Detection Preamble** below to set `MG_FL_DIR` and `FLAGSCALE_DIR`. Both repos
are required for this skill. If either is missing, the preamble offers to clone it.

#### Step 1: Locate both repos

```bash
# --- Detect Megatron-LM-FL ---
MG_FL_DIR=""
for candidate in \
  "$(pwd)" \
  "$(pwd)/Megatron-LM-FL" \
  "$(pwd)/../Megatron-LM-FL" \
  "$(dirname $(pwd))/Megatron-LM-FL"; do
  if [ -d "$candidate/megatron" ] && [ -f "$candidate/setup.py" -o -f "$candidate/pyproject.toml" ]; then
    MG_FL_DIR=$(cd "$candidate" && pwd)
    break
  fi
done

if [ -z "$MG_FL_DIR" ]; then
  echo "Megatron-LM-FL not found locally. Cloning..."
  git clone https://github.com/flagos-ai/Megatron-LM-FL.git
  MG_FL_DIR=$(cd Megatron-LM-FL && pwd)
fi
echo "Megatron-LM-FL: $MG_FL_DIR"

# --- Detect FlagScale ---
# FlagScale must live as a sibling of Megatron-LM-FL in the same workspace directory.
WORKSPACE_DIR="$(dirname "$MG_FL_DIR")"
FLAGSCALE_DIR=""
for candidate in \
  "$WORKSPACE_DIR/FlagScale" \
  "$WORKSPACE_DIR/flagscale"; do
  if [ -d "$candidate/flagscale" ] && [ "$(cd "$candidate" && pwd)" != "$MG_FL_DIR" ]; then
    FLAGSCALE_DIR=$(cd "$candidate" && pwd)
    break
  fi
done

if [ -z "$FLAGSCALE_DIR" ]; then
  echo "FlagScale not found in $WORKSPACE_DIR. Cloning..."
  git clone https://github.com/flagos-ai/FlagScale.git "$WORKSPACE_DIR/FlagScale"
  FLAGSCALE_DIR="$WORKSPACE_DIR/FlagScale"
fi
echo "FlagScale: $FLAGSCALE_DIR"

# --- Verify Megatron-LM-FL is installed ---
cd "$MG_FL_DIR"
python3 -c "
import megatron.core
import megatron.plugin
from megatron.plugin.platform import cur_platform
print('megatron.core OK')
print('megatron.plugin OK')
print('cur_platform:', cur_platform)
" || {
  echo "megatron.core/plugin not importable. Installing Megatron-LM-FL..."
  pip install -e . --no-build-isolation
  python3 -c "
import megatron.core
import megatron.plugin
print('megatron.core OK')
print('megatron.plugin OK')
" || {
    echo "ERROR: megatron.core/plugin still not importable after install."
    echo "Check that pyproject.toml includes both megatron.core and megatron.plugin."
    echo "Fix Megatron-LM-FL first."
    exit 1
  }
}

echo "Both repos ready."
```

#### Step 2: Add upstream remote and fetch the target version

```bash
cd "$FLAGSCALE_DIR"
git remote -v | grep mg-upstream || \
  git remote add mg-upstream https://github.com/NVIDIA/Megatron-LM.git
git fetch mg-upstream --tags
```

#### Step 3: Confirm versions and branch names

The base and target versions must match the Megatron-LM-FL sync. Check Megatron-LM-FL's
`SYNC_POINT.md` or branch history to identify them:

```bash
cd "$MG_FL_DIR"
cat SYNC_POINT.md 2>/dev/null || echo "No SYNC_POINT.md found"
git log --oneline --all | grep -i "base\|core_v0" | head -10
```

**Before any branch operations, ask the user for six parameters:**

1. **Base upstream version** — the upstream release the fork is currently based on
2. **Base upstream commit** — (optional) specific commit SHA on the base version to use;
   if empty, use the tag/branch tip
3. **Target upstream version** — the upstream release to sync to
4. **Target upstream commit** — (optional) specific commit SHA on the target version to use;
   if empty, use the tag/branch tip
5. **FlagScale dev branch name** — the FlagScale branch for the upgraded training code
6. **Megatron-LM-FL dev branch name** — the Megatron-LM-FL dev branch (from the companion skill)

**Prompt the user:**
> Please specify the following:
> 1. Base upstream version the fork is currently based on (e.g. `core_v0.15.0rc7`)
> 2. Base upstream commit (leave empty for tag/branch tip, e.g. `abc1234`)
> 3. Target upstream version to sync to (e.g. `core_v0.16.1`)
> 4. Target upstream commit (leave empty for tag/branch tip, e.g. `def5678`)
> 5. FlagScale dev branch name (e.g. `dev-train-0.16`)
> 6. Megatron-LM-FL dev branch name (e.g. `dev-0.16.1`)

Store the answers and use them throughout **all** subsequent stages:

```bash
cd "$FLAGSCALE_DIR"
BASE_VERSION="<user answer 1>"            # e.g. core_v0.15.0rc7
BASE_VERSION_COMMIT="<user answer 2>"     # e.g. abc1234 (empty = tag/branch tip)
TARGET_VERSION="<user answer 3>"          # e.g. core_v0.16.1
TARGET_VERSION_COMMIT="<user answer 4>"   # e.g. def5678 (empty = tag/branch tip)
FS_DEV_BRANCH="<user answer 5>"          # e.g. dev-train-0.16
MG_DEV_BRANCH="<user answer 6>"          # e.g. dev-0.16.1

echo "Base version (old upstream): $BASE_VERSION"
echo "Base commit: ${BASE_VERSION_COMMIT:-tip}"
echo "Target version (new upstream): $TARGET_VERSION"
echo "Target commit: ${TARGET_VERSION_COMMIT:-tip}"
echo "FlagScale dev branch: $FS_DEV_BRANCH"
echo "Megatron-LM-FL dev branch: $MG_DEV_BRANCH"

# Verify both tags/commits exist
if [ -n "$BASE_VERSION_COMMIT" ]; then
  git rev-parse "$BASE_VERSION_COMMIT" > /dev/null 2>&1 || {
    echo "ERROR: commit $BASE_VERSION_COMMIT not found"; exit 1
  }
else
  git rev-parse mg-upstream/$BASE_VERSION > /dev/null 2>&1 || {
    echo "ERROR: mg-upstream/$BASE_VERSION not found"; exit 1
  }
fi
if [ -n "$TARGET_VERSION_COMMIT" ]; then
  git rev-parse "$TARGET_VERSION_COMMIT" > /dev/null 2>&1 || {
    echo "ERROR: commit $TARGET_VERSION_COMMIT not found"; exit 1
  }
else
  git rev-parse mg-upstream/$TARGET_VERSION > /dev/null 2>&1 || {
    echo "ERROR: mg-upstream/$TARGET_VERSION not found"; exit 1
  }
fi
```

#### Step 4: Reset main to origin/main

If local main has stale commits from previous upgrade attempts, reset it:

```bash
cd "$FLAGSCALE_DIR"

# Check if local main has diverged from origin/main
LOCAL_MAIN=$(git rev-parse main)
ORIGIN_MAIN=$(git rev-parse origin/main)

if [ "$LOCAL_MAIN" != "$ORIGIN_MAIN" ]; then
  echo "Local main ($LOCAL_MAIN) differs from origin/main ($ORIGIN_MAIN)"
  echo "Commits on local main not in origin/main:"
  git log --oneline origin/main..main
  echo ""
  echo "Resetting local main to origin/main..."
  git checkout main
  git reset --hard origin/main
  echo "main reset to origin/main: $(git rev-parse --short HEAD)"
else
  echo "main is already at origin/main: $(git rev-parse --short HEAD)"
fi
```

---

### Stage 2: Create Branch Structure for Three-Way Merge

This follows the same pattern as Megatron-LM-FL's `base`/`dev`/`main` branches. The three
branches represent:
- `base-train` — FlagScale's tree with training/legacy replaced by old upstream content
- `main` — FlagScale's current tree (with FlagScale's customizations)
- `dev-train` — FlagScale's tree with training/legacy replaced by new upstream content

The diff `base-train..main` captures FlagScale's customizations. Applying that diff to
`dev-train` produces the upgraded result.

#### Step 1: Create `base-train` branch

Start from `main` (FlagScale's current state), then replace training/legacy content with the
old upstream version. This creates a synthetic commit where the only difference from `main` is
that training/legacy matches the old upstream.

```bash
cd "$FLAGSCALE_DIR"

git branch -D base-train 2>/dev/null || true
git checkout -b base-train main

# Resolve the base ref (commit SHA if provided, otherwise tag/branch)
BASE_REF="${BASE_VERSION_COMMIT:-mg-upstream/$BASE_VERSION}"

# Replace training/ content with old upstream version
# First, remove existing training and legacy dirs
rm -rf flagscale/train/megatron/training/
rm -rf flagscale/train/megatron/legacy/

# Extract old upstream training/ into FlagScale's path
mkdir -p flagscale/train/megatron/training
git archive $BASE_REF -- megatron/training/ | \
  tar -x --strip-components=2 -C flagscale/train/megatron/training/

# Extract old upstream legacy/ into FlagScale's path (if it exists)
if git ls-tree $BASE_REF megatron/legacy/ > /dev/null 2>&1; then
  mkdir -p flagscale/train/megatron/legacy
  git archive $BASE_REF -- megatron/legacy/ | \
    tar -x --strip-components=2 -C flagscale/train/megatron/legacy/
fi

git add -A
git commit -m "base-train: replace training/legacy with upstream $BASE_VERSION content"
echo "base-train branch created (ref: $BASE_REF)"
```

#### Step 2: Create `dev-train` branch

Same approach, but with the new upstream version:

```bash
cd "$FLAGSCALE_DIR"

git branch -D dev-train 2>/dev/null || true
git checkout -b dev-train main

# Resolve the target ref (commit SHA if provided, otherwise tag/branch)
TARGET_REF="${TARGET_VERSION_COMMIT:-mg-upstream/$TARGET_VERSION}"

# Replace training/ content with new upstream version
rm -rf flagscale/train/megatron/training/
rm -rf flagscale/train/megatron/legacy/

# Extract new upstream training/ into FlagScale's path
mkdir -p flagscale/train/megatron/training
git archive $TARGET_REF -- megatron/training/ | \
  tar -x --strip-components=2 -C flagscale/train/megatron/training/

# Extract new upstream legacy/ into FlagScale's path (if it exists)
if git ls-tree $TARGET_REF megatron/legacy/ > /dev/null 2>&1; then
  mkdir -p flagscale/train/megatron/legacy
  git archive $TARGET_REF -- megatron/legacy/ | \
    tar -x --strip-components=2 -C flagscale/train/megatron/legacy/
fi

git add -A
git commit -m "dev-train: replace training/legacy with upstream $TARGET_VERSION content"
echo "dev-train branch created (ref: $TARGET_REF)"
```

#### Step 3: Verify the branch structure

```bash
cd "$FLAGSCALE_DIR"

echo "=== Branch structure ==="
echo "base-train: training/legacy = upstream $BASE_VERSION"
echo "main:       training/legacy = FlagScale's current version (with customizations)"
echo "dev-train:  training/legacy = upstream $TARGET_VERSION"

echo ""
echo "=== FlagScale customizations (diff base-train..main) ==="
git diff --stat base-train..main -- flagscale/train/megatron/training/ flagscale/train/megatron/legacy/

echo ""
echo "=== Upstream changes (diff base-train..dev-train) ==="
git diff --stat base-train..dev-train -- flagscale/train/megatron/training/ flagscale/train/megatron/legacy/
```

Return to dev-train for the merge work:

```bash
git checkout dev-train
```

#### Checkpoint: commit

No pre-commit needed here — the branches are synthetic reference points.

```bash
echo "Branch structure ready. Proceeding to Stage 3."
```

---

