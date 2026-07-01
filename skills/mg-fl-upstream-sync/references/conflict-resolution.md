# Conflict Resolution Reference

Detailed procedures for Stage 3 of the Megatron-LM-FL upstream sync workflow.

## Table of Contents
- [Conflict Priority Matrix](#conflict-priority-matrix)
- [P0 Resolution: Plugin Files](#p0-resolution-plugin-files)
- [P1 Resolution: Core Files with Plugin Modifications](#p1-resolution-core-files-with-plugin-modifications)
- [P2 Resolution: Non-Plugin Files](#p2-resolution-non-plugin-files)
- [Post-Merge Verification](#post-merge-verification)

---

## Conflict Priority Matrix

### P0 — Sacred (Plugin System)

These files ARE the fork's value. The fork version is always the starting point.

**Paths:**
- `megatron/plugin/**` — entire plugin directory
- Any file in `megatron/core/` containing `@overridable` decorators
- Any file in `megatron/core/` with `from megatron.plugin` imports

**Resolution strategy:**
1. Start with the fork (main) version of the file
2. Read the upstream (dev) changes to the same file
3. If upstream made changes to the same function that has `@overridable`:
   - Keep the `@overridable` decorator
   - Integrate upstream's functional changes into the decorated function body
   - Verify the `@override` implementation in `megatron/plugin/` still matches the signature
4. If upstream added new functions to a file that also has `@overridable` functions:
   - Accept the new functions from upstream
   - Keep all existing decorators and plugin imports intact
5. If upstream deleted or renamed a function that has `@overridable`:
   - Flag for manual review — the `@override` implementation needs updating too
   - Check `megatron/plugin/` for the corresponding `@override` and update both together

**Verification after resolution:**
```bash
# For each resolved P0 file, verify decorators survived
grep -n "@overridable" <resolved-file>
grep -n "from megatron.plugin" <resolved-file>
```

### P1 — Careful (Core Files with Plugin Modifications)

These files have fork modifications (typically `cur_platform` replacements) but are not
part of the plugin directory itself.

**Paths:**
- `megatron/core/` files that appear in `PLUGIN_CHANGES.md` as modified
- `setup.py`, `pyproject.toml` — build configuration
- Files with `cur_platform` replacements (replacing `torch.cuda` calls)

**Resolution strategy:**
1. Open the three-way diff (base, main, dev)
2. Identify fork-specific changes:
   - `torch.cuda.xxx` → `cur_platform.xxx` replacements
   - `from megatron.plugin.platform import cur_platform` imports
   - Any `@overridable` decorator additions
3. Identify upstream changes:
   - New functionality, bug fixes, performance improvements
   - New `torch.cuda` calls (these need `cur_platform` conversion)
4. Merge both:
   - Keep all `cur_platform` replacements from the fork
   - Accept upstream functional changes
   - Convert any NEW `torch.cuda` calls from upstream to `cur_platform`
   - Preserve `cur_platform` import statements

**For build files (setup.py, pyproject.toml):**
1. Accept upstream dependency version bumps
2. Keep fork-specific build targets and plugin-related entries
3. Keep fork-specific metadata (package name, URLs, etc.)
4. Verify `megatron.plugin` is still included in package discovery

**Verification after resolution:**
```bash
# Check no raw torch.cuda calls leaked into resolved core files
grep -n "torch\.cuda" <resolved-file> | grep -v "cur_platform" | grep -v "#"

# Check cur_platform import is present
grep -n "cur_platform" <resolved-file>
```

### P2 — Upstream-Preferred (Non-Plugin Files)

These files have no fork-specific modifications. Accept upstream.

**Paths:**
- Documentation, README files
- Test files not related to plugin
- Example scripts
- Files NOT listed in `PLUGIN_CHANGES.md`

**Resolution strategy:**
```bash
git checkout --theirs <file>
git add <file>
```

**Safety check before accepting upstream:**
```bash
# Verify the file truly has no fork modifications
git diff base..main -- <file>
# If output is empty → safe to accept upstream
# If output is non-empty → re-classify as P1
```

---

## Post-Merge Verification

After all conflicts are resolved:

```bash
# 1. Verify no conflict markers remain
grep -rn "<<<<<<<\|=======\|>>>>>>>" megatron/ --include="*.py"

# 2. Verify all plugin imports are intact
grep -rn "from megatron.plugin" megatron/core/ --include="*.py" | grep -v "__pycache__"

# 3. Verify cur_platform imports
grep -rn "from megatron.plugin.platform import cur_platform" megatron/core/ --include="*.py" | grep -v "__pycache__"

# 4. Verify @overridable decorator count matches pre-merge
PRE_MERGE_COUNT=$(grep -c "@overridable" PLUGIN_CHANGES.md 2>/dev/null || echo "check manually")
POST_MERGE_COUNT=$(grep -rn "@overridable" megatron/core/ --include="*.py" | grep -v "__pycache__" | wc -l)
echo "Pre-merge @overridable count: $PRE_MERGE_COUNT"
echo "Post-merge @overridable count: $POST_MERGE_COUNT"

# 5. Syntax check all Python files in affected directories
for f in $(git diff --name-only HEAD -- '*.py'); do
  python3 -c "import ast; ast.parse(open('$f').read())" 2>&1 && echo "OK: $f" || echo "FAIL: $f"
done

# 6. Finalize
git add -A
git commit --no-edit
```
