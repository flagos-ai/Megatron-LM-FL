# Phase 2: Merge & Integrate

Stages 3-5 of the Megatron-LM-FL upstream sync workflow.

---

### Stage 3: Patch-Based Fork Integration

**Why patch-based instead of `git merge`?** A direct `git merge main` into the dev branch produces many
spurious conflicts because git's three-way merge sees divergent histories. The fork's changes are
well-categorized (new plugin files, cur_platform replacements, @overridable decorators, plugin
imports) and are better applied as structured patches onto the upstream base. This produces fewer
conflicts and makes each conflict meaningful.

Read `references/conflict-resolution.md` for the detailed conflict priority matrix.

#### Step 1: Prepare the dev branch

Start from a clean dev branch based on the upstream target tag:

```bash
cd "$MG_FL_DIR"
git checkout ${DEV_BRANCH}
# If a failed merge is in progress, abort it first
git merge --abort 2>/dev/null || true
# Verify dev branch is clean and based on the upstream tag
git log --oneline -1
```

#### Step 2: Extract fork patches by category

Generate the fork delta (`${BASE_BRANCH}..main`) as patches, categorized by priority. This is the heart of
the approach — instead of one monolithic merge, we apply fork changes in layers.

```bash
# Generate the full fork diff for reference
git diff ${BASE_BRANCH}..main > /tmp/fork_full_diff.patch

# --- Category A: New plugin files (P0 — just copy, no conflict possible) ---
git diff --name-only --diff-filter=A ${BASE_BRANCH}..main | grep "^megatron/plugin/" > /tmp/new_plugin_files.txt
echo "New plugin files to copy: $(wc -l < /tmp/new_plugin_files.txt)"

# --- Category B: Modified core files with cur_platform (P1) ---
git diff --name-only --diff-filter=M ${BASE_BRANCH}..main | grep "^megatron/core/" > /tmp/modified_core_files.txt
echo "Modified core files: $(wc -l < /tmp/modified_core_files.txt)"

# --- Category C: @overridable decorated files (P0) ---
# These are core files that have both cur_platform AND @overridable decorators
grep -rl "@overridable" megatron/core/ --include="*.py" 2>/dev/null | \
  sed "s|^|megatron/core/|" | sort -u > /tmp/overridable_files.txt 2>/dev/null || true
echo "Files with @overridable: $(wc -l < /tmp/overridable_files.txt)"

# --- Category D: New non-plugin files (CI/CD, tests, etc.) ---
git diff --name-only --diff-filter=A ${BASE_BRANCH}..main | grep -v "^megatron/plugin/" > /tmp/new_other_files.txt
echo "New non-plugin files: $(wc -l < /tmp/new_other_files.txt)"

# --- Category E: Modified non-core files (pyproject.toml, etc.) ---
git diff --name-only --diff-filter=M ${BASE_BRANCH}..main | grep -v "^megatron/core/" | \
  grep -v "^megatron/plugin/" > /tmp/modified_other_files.txt
echo "Modified non-core files: $(wc -l < /tmp/modified_other_files.txt)"

# --- Category F: FlagScale Begin/End feature blocks (P0 — fork-specific features) ---
# These are fork-specific feature additions wrapped in FlagScale Begin/End markers.
# They are NOT cur_platform or @overridable — they are standalone features like
# qk_layernorm_hidden_dim, use_partial_reduce_for_shared_embedding, engram/hetero
# config fields, etc. They MUST be preserved during the merge.
grep -rn "FlagScale Begin" megatron/core/ --include="*.py" -l 2>/dev/null | sort -u > /tmp/flagscale_block_files.txt
echo "Files with FlagScale Begin/End blocks: $(wc -l < /tmp/flagscale_block_files.txt)"

# Build a manifest of all FlagScale blocks on main (for audit in later stages)
echo "=== FlagScale Begin/End Block Manifest (main) ===" > /tmp/flagscale_blocks_main.txt
while IFS= read -r f; do
  echo "--- $f ---" >> /tmp/flagscale_blocks_main.txt
  grep -n "FlagScale Begin\|FlagScale End" "$f" >> /tmp/flagscale_blocks_main.txt
done < /tmp/flagscale_block_files.txt
```

#### Step 3: Apply Category A — New plugin files (zero-conflict)

These files don't exist in upstream, so they can be copied directly from main:

```bash
git checkout ${DEV_BRANCH}
while IFS= read -r f; do
  mkdir -p "$(dirname "$f")"
  git show main:"$f" > "$f"
  git add "$f"
  echo "Copied: $f"
done < /tmp/new_plugin_files.txt

# Also copy any new non-plugin fork files (CI/CD, test infra, etc.)
while IFS= read -r f; do
  # Check if file exists in dev branch — if not, safe to copy
  if ! git show ${DEV_BRANCH}:"$f" > /dev/null 2>&1; then
    mkdir -p "$(dirname "$f")"
    git show main:"$f" > "$f"
    git add "$f"
    echo "Copied new file: $f"
  else
    echo "Exists in dev branch, will patch: $f"
  fi
done < /tmp/new_other_files.txt

git commit -m "patch(plugin): add new plugin and fork-specific files" --allow-empty
```

#### Step 4: Apply Category B — Core file patches (per-file, with conflict handling)

For each modified core file, generate and apply a per-file patch. This isolates conflicts to
individual files and makes them easier to resolve:

```bash
CONFLICT_FILES=""
while IFS= read -r f; do
  # Generate per-file patch (fork's changes to this file)
  git diff ${BASE_BRANCH}..main -- "$f" > "/tmp/patch_$(echo $f | tr '/' '_').patch"
  PATCH_FILE="/tmp/patch_$(echo $f | tr '/' '_').patch"

  # Skip empty patches
  if [ ! -s "$PATCH_FILE" ]; then continue; fi

  # Try to apply with 3-way merge fallback
  if git apply --3way "$PATCH_FILE" 2>/dev/null; then
    git add "$f"
    echo "OK: $f"
  else
    echo "CONFLICT: $f"
    CONFLICT_FILES="$CONFLICT_FILES $f"
  fi
done < /tmp/modified_core_files.txt

if [ -n "$CONFLICT_FILES" ]; then
  echo ""
  echo "=== Files needing manual resolution ==="
  echo "$CONFLICT_FILES" | tr ' ' '\n' | grep -v '^$'
fi
```

For each conflicted file, resolve using the priority matrix:

| Priority | Identification | Strategy |
|----------|---------------|----------|
| P0 | File has `@overridable` or is in `PLUGIN_CHANGES.md` as decorated | Read both versions. Start from ${DEV_BRANCH} (upstream). Re-apply: (1) `from megatron.plugin` imports, (2) `@overridable` decorators, (3) `cur_platform` replacements, (4) any `FlagScale Begin/End` blocks. Verify `@override` in plugin still matches signature. |
| P0-F | File has `FlagScale Begin/End` blocks (Category F) but no @overridable | Read both versions. Start from ${DEV_BRANCH} (upstream). Re-apply each `FlagScale Begin/End` block from main onto the dev branch's code. These are standalone fork features (config fields, custom logic, etc.) — they must NOT be discarded. See "Applying FlagScale blocks" below. |
| P1 | File is in `PLUGIN_CHANGES.md` as cur_platform-modified (no FlagScale blocks) | Read both versions. Start from ${DEV_BRANCH} (upstream). Re-apply: (1) `from megatron.plugin.platform import get_platform` + `cur_platform = get_platform()`, (2) all `torch.cuda` → `cur_platform` replacements. |
| P2 | File not in `PLUGIN_CHANGES.md` AND no `FlagScale Begin/End` blocks | Accept ${DEV_BRANCH} (upstream) version: `git checkout ${DEV_BRANCH} -- "$f"` |

**CRITICAL: Never use P2 (accept ${DEV_BRANCH}) on a file that has `FlagScale Begin/End` blocks on main.**
Always check: `git show main:"$f" | grep -c "FlagScale Begin"` before classifying as P2.

**Applying FlagScale blocks onto the dev branch's base:**
Do NOT copy main's whole file — the dev branch may have upstream changes that main lacks. Instead:
1. Read the dev branch's version of the file (has latest upstream code)
2. Read main's version to identify each `FlagScale Begin/End` block
3. For each block, find the correct insertion point in the dev branch's version (match surrounding
   upstream context lines) and patch it in
4. If the surrounding context changed in upstream, adapt the block to the new context
   (e.g. new function signatures, renamed variables)

Common FlagScale block types that get lost:
- Config dataclass fields (e.g. `qk_layernorm_hidden_dim`, `use_partial_reduce_for_shared_embedding`)
- Conditional logic branches (e.g. `if not self.config.qk_layernorm_hidden_dim:`)
- Import additions for fork features
- Test skip markers

**For P1 conflicts (most common — cur_platform replacements):** The cleanest approach is:
1. Start from the ${DEV_BRANCH} (upstream) version of the file
2. Add the `cur_platform` import at the top (after existing imports)
3. Replace each `torch.cuda.xxx` call with `cur_platform.xxx`
4. This is mechanical and can be done with search-and-replace

```bash
# After resolving each file:
git add "$f"
```

Commit after all core files are resolved:
```bash
git commit -m "patch(core): apply fork modifications to megatron/core/"
```

#### Step 5: Apply Category E — Other modified files

```bash
while IFS= read -r f; do
  git diff ${BASE_BRANCH}..main -- "$f" > "/tmp/patch_$(echo $f | tr '/' '_').patch"
  PATCH_FILE="/tmp/patch_$(echo $f | tr '/' '_').patch"
  if [ ! -s "$PATCH_FILE" ]; then continue; fi

  if git apply --3way "$PATCH_FILE" 2>/dev/null; then
    git add "$f"
    echo "OK: $f"
  else
    echo "CONFLICT: $f — resolve manually"
    # For pyproject.toml: keep fork metadata + plugin entries, accept upstream deps
    # For CI/CD: keep fork version if fork-specific, accept upstream otherwise
  fi
done < /tmp/modified_other_files.txt

git commit -m "patch(other): apply fork modifications to build/CI files" --allow-empty
```

#### Step 6: Handle deleted-by-upstream files

Some files that the fork modified may have been deleted by upstream. These need special handling:

```bash
# Find files modified by fork but deleted in upstream
while IFS= read -r f; do
  if ! git show ${DEV_BRANCH}:"$f" > /dev/null 2>&1; then
    echo "DELETED BY UPSTREAM: $f"
    # Check if fork changes are cur_platform only
    FORK_CHANGES=$(git diff ${BASE_BRANCH}..main -- "$f" | grep "^+" | grep -v "^+++" | head -5)
    echo "  Fork changes: $FORK_CHANGES"
    echo "  → If only cur_platform: safe to delete (the module is gone)"
    echo "  → If @overridable or structural: flag for review"
  fi
done < /tmp/modified_core_files.txt
```

#### Step 7: Post-patch verification

```bash
# Verify no conflict markers remain
grep -rn "<<<<<<<\|=======\|>>>>>>>" megatron/ --include="*.py" && echo "CONFLICT MARKERS FOUND" || echo "OK: clean"

# Verify plugin imports survived
echo "cur_platform imports: $(grep -rn 'from megatron.plugin.platform' megatron/core/ --include='*.py' | grep -v __pycache__ | wc -l)"
echo "@overridable decorators: $(grep -rn '@overridable' megatron/core/ --include='*.py' | grep -v __pycache__ | wc -l)"

# Verify FlagScale Begin/End blocks survived — compare dev branch vs main
MAIN_FS_COUNT=$(git show main:. 2>/dev/null && git stash 2>/dev/null; \
  git checkout main -- megatron/ 2>/dev/null; \
  grep -rn "FlagScale Begin" megatron/ --include="*.py" | wc -l; \
  git checkout ${DEV_BRANCH} -- megatron/ 2>/dev/null; git stash pop 2>/dev/null)
DEV_FS_COUNT=$(grep -rn "FlagScale Begin" megatron/ --include="*.py" | wc -l)
echo "FlagScale blocks on main:        $MAIN_FS_COUNT"
echo "FlagScale blocks on ${DEV_BRANCH}: $DEV_FS_COUNT"
# dev branch must have >= main's count. If less, features were lost during patching.

# Per-file audit: find files where main has FlagScale blocks but dev branch doesn't
while IFS= read -r f; do
  DEV_BLOCKS=$(grep -c "FlagScale Begin" "$f" 2>/dev/null || echo 0)
  MAIN_BLOCKS=$(git show main:"$f" 2>/dev/null | grep -c "FlagScale Begin" || echo 0)
  if [ "$MAIN_BLOCKS" -gt "$DEV_BLOCKS" ]; then
    echo "MISSING BLOCKS: $f (main=$MAIN_BLOCKS, ${DEV_BRANCH}=$DEV_BLOCKS)"
  fi
done < /tmp/flagscale_block_files.txt
# Any file reported as MISSING BLOCKS must be fixed before proceeding.

# Syntax check all modified Python files
for f in $(git diff --name-only HEAD~3..HEAD -- '*.py'); do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "OK: $f" || echo "SYNTAX ERROR: $f"
done
```

**Success criteria:** All patches applied, conflicts resolved, no conflict markers, plugin imports
and decorators intact, FlagScale block count on ${DEV_BRANCH} >= main (no features lost).

#### Checkpoint: pre-commit & commit

Commit the final state of Stage 3 (post-patch verification clean):

```bash
pre-commit run --all-files
git add -A
git commit -m "Stage 3: merge upstream and resolve all conflicts" --allow-empty
```

Note: Stage 3 may produce multiple intermediate commits (per-category patches). The final
checkpoint ensures the stage boundary is clean. If all changes were already committed in the
per-step commits, this will be a no-op.

---

### Stage 4: Plugin Integrity Verification

Read `references/plugin-verification.md` for detailed verification procedures. The summary:

#### 4a: Platform mechanism verification

Every `torch.cuda` call in `megatron/core/` should use `cur_platform` instead. Check that
upstream didn't introduce new `torch.cuda` calls that bypass the platform mechanism:

```bash
# Find any raw torch.cuda calls in megatron/core/ (should be zero or minimal)
grep -rn "torch\.cuda" megatron/core/ --include="*.py" | grep -v "__pycache__" | grep -v "cur_platform"
```

Any hits need to be replaced with the appropriate `cur_platform` equivalent.

#### 4b: Override mechanism verification

Verify all `@overridable` decorators in `megatron/core/` survived the merge:

```bash
# Count @overridable decorators — compare with pre-merge count from PLUGIN_CHANGES.md
grep -rn "@overridable" megatron/core/ --include="*.py" | grep -v "__pycache__"

# Verify corresponding @override implementations still exist in megatron/plugin/
grep -rn "@override" megatron/plugin/ --include="*.py" | grep -v "__pycache__"
```

#### 4c: New feature integrity

```bash
# Verify dualpipev and hetero modules are intact
python3 -c "import ast; ast.parse(open('megatron/plugin/dualpipev/__init__.py').read())" 2>/dev/null && echo "dualpipev OK"
python3 -c "import ast; ast.parse(open('megatron/plugin/hetero/__init__.py').read())" 2>/dev/null && echo "hetero OK"
```

**Success criteria:** All plugin mechanisms verified intact.

#### 4d: Override body sync verification

Nearly all `@override` implementations are **complete replacements** — they copy the base
`@overridable` function body and add fork-specific logic (hetero multi-group support, CPU
communication, extra scheduler types, etc.). When upstream upgrades the base function (new
features, bug fixes, new parameters), the override silently misses those changes. This step
detects that drift and ensures overrides stay in sync.

**Why this matters:** An override that worked perfectly with the old upstream may silently drop
new upstream features. For example, if upstream adds MTP (multi-token prediction) support to
`_allreduce_embedding_grad`, but the override was copied from the old version, MTP silently
breaks in hetero mode. This is the most dangerous class of regression because it passes all
syntax checks and decorator verification — it only fails at runtime under specific conditions.

**Step 1: Diff each @overridable function between ${BASE_BRANCH} and ${DEV_BRANCH}**

For each `@overridable` function, generate the diff of what upstream changed:

```bash
# Build the list of @overridable functions and their files
grep -rn "@overridable" megatron/core/ --include="*.py" | grep -v "__pycache__" | \
  while IFS=: read -r file line _; do
    # Get the function name from the line after @overridable
    FUNC=$(sed -n "$((line+1))p" "$file" | sed 's/.*def \([a-zA-Z_]*\).*/\1/')
    echo "$file:$FUNC"
  done > /tmp/overridable_functions.txt

# For each, show what upstream changed in the function body
while IFS=: read -r file func; do
  echo "=== $func in $file ==="
  git diff ${BASE_BRANCH}..${DEV_BRANCH} -- "$file" | head -100
  echo ""
done < /tmp/overridable_functions.txt
```

**Step 2: For each override, verify it incorporates upstream changes**

Read the reference file `references/plugin-verification.md` (section "Override Body Sync") for
the detailed per-function analysis procedure. The core approach:

1. Read the `@overridable` function body in ${DEV_BRANCH} (the new upstream version)
2. Read the `@override` function body in `megatron/plugin/`
3. Identify what upstream added/changed vs ${BASE_BRANCH}
4. Check whether the override incorporates those changes

Common patterns of drift to look for:
- **Missing new branches**: upstream added `if config.new_feature:` logic that the override lacks
- **Missing new parameters**: upstream added a parameter the override doesn't accept or forward
- **Stale copied logic**: override copied old base logic that upstream has since fixed/improved
- **Missing imports**: upstream started using a new utility the override doesn't import

**Step 3: Update out-of-sync overrides**

For each override that's out of sync, the resolution depends on the nature of the drift:

| Drift type | Resolution |
|-----------|-----------|
| New feature branch in base | Add the same branch to the override, adapted for multi-group/hetero if needed |
| Bug fix in base | Apply the same fix to the override's copy of that logic |
| New parameter in base | Add the parameter to the override signature; forward it or handle it |
| Refactored base logic | Re-copy the base logic into the override, then re-apply fork modifications |

When updating an override, preserve the fork-specific modifications (the parts between
`# FlagScale Begin` / `# FlagScale End` markers or the hetero/multi-group branches) while
bringing the "copied base" portions up to date with ${DEV_BRANCH}.

**Success criteria:** Every `@override` function incorporates all new features and fixes from
the ${DEV_BRANCH} version of its corresponding `@overridable` function.

#### Checkpoint: pre-commit & commit

If any fixes were applied during verification (e.g., new `cur_platform` replacements, restored
decorators, signature updates, override body syncs), commit them now:

```bash
pre-commit run --all-files
git add -A
git commit -m "fix(plugin): patch new torch.cuda calls, verify decorator chain, sync override bodies"
```

Skip this checkpoint if no code changes were needed (verification passed cleanly).

---

### Stage 5: Detect & Fix Stale References

Upstream may have renamed, moved, or removed symbols that fork code depends on.

```bash
# Symbols removed in upstream
git diff ${BASE_BRANCH}..${DEV_BRANCH} -- '*.py' | grep -E '^\-[^-]' | grep -E '^\-(def |class )' | \
  sed 's/^-//' | sed 's/(.*//' | sed 's/def //' | sed 's/class //' | sed 's/://' | \
  tr -d ' ' | sort -u > /tmp/upstream_removed_symbols.txt

# Check fork-specific code for references to removed symbols
while IFS= read -r symbol; do
  [ -z "$symbol" ] && continue
  MATCHES=$(grep -rn "$symbol" megatron/plugin/ --include='*.py' 2>/dev/null || true)
  if [ -n "$MATCHES" ]; then
    echo "STALE: '$symbol' (removed upstream) referenced in:"
    echo "$MATCHES"
  fi
done < /tmp/upstream_removed_symbols.txt
```

For each stale reference: find the new symbol name in upstream, update the fork code.

**Success criteria:** No stale references remain in `megatron/plugin/`.

#### Checkpoint: pre-commit & commit

If any stale references were fixed, commit them:

```bash
pre-commit run --all-files
git add -A
git commit -m "fix(plugin): update stale references to renamed/removed upstream symbols"
```

Skip if no stale references were found.

---

