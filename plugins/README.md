# Plugin System for Method Replacement

## Overview

This plugin system allows core code to be decorated with `@plugin_method`, and the actual implementation can be provided in plugins. The plugin files mirror the path structure of the original files.

## File Structure

```
Megatron-LM-FL/
├── megatron/
│   └── core/
│       └── models/
│           └── common/
│               └── language_module/
│                   └── language_module.py  # Core code (with @plugin_method)
└── plugins/
    ├── __init__.py                          # Export decorators
    ├── decorators.py                        # plugin_method + plugin_implementation
    ├── README.md                            # This file
    └── core/
        └── models/
            └── common/
                └── language_module/
                    └── language_module.py  # Plugin implementation
```

## How It Works

1. **Core code**: Add `@plugin_method` decorator (no parameters needed!)
2. **Plugin code**: Define the implementation function with `@plugin_implementation` decorator in a file that mirrors the original path
3. **Runtime**: When the method is called, the decorator automatically:
   - Detects class name and method name
   - Looks up plugin using `"ClassName.method_name"` as key
   - Uses plugin if found, otherwise uses original implementation

## Usage

### Step 1: Add decorator to core code

In `megatron/core/models/common/language_module/language_module.py`:

```python
from plugins import plugin_method

class LanguageModule(MegatronModule):
    @plugin_method  # No parameters needed!
    def _is_in_embd_group(self):
        # Original implementation (fallback if no plugin)
        ...
```

### Step 2: Create plugin implementation

In `plugins/core/models/common/language_module/language_module.py`:

```python
from plugins.decorators import plugin_implementation

@plugin_implementation("LanguageModule", "_is_in_embd_group")
def _is_in_embd_group(self):
    """fl_init implementation with list support."""
    # Full implementation here
    ...
```

### Step 3: Import plugins

```python
# In your code, import plugins to register implementations
import plugins
```

## Key Format

The method key follows the format: `"ClassName.method_name"`

- **Class name**: Automatically detected by decorator from `self.__class__.__name__`
- **Method name**: Automatically detected from the function name
- **Example**: `"LanguageModule._is_in_embd_group"`

## Advantages

✅ **Mirror path structure**: Plugin files mirror original file paths for easy navigation  
✅ **Zero parameters in core**: Decorator needs no arguments, everything is auto-detected  
✅ **Simple plugin registration**: Just use `@plugin_implementation` decorator  
✅ **Minimal invasion**: Only need to add `@plugin_method` decorator in core code  
✅ **Fallback support**: Original implementation is used if no plugin exists  
✅ **Clean separation**: All plugin logic is in plugins directory  

## Example: _is_in_embd_group

**Core code change (minimal):**
```python
# megatron/core/models/common/language_module/language_module.py
from plugins import plugin_method

@plugin_method
def _is_in_embd_group(self):
    # Original implementation
    ...
```

**Plugin implementation:**
```python
# plugins/core/models/common/language_module/language_module.py
from plugins.decorators import plugin_implementation

@plugin_implementation("LanguageModule", "_is_in_embd_group")
def _is_in_embd_group(self):
    # fl_init implementation
    ...
```

