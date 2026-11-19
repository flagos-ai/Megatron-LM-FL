# KunLunXin 后端使用与验证

本文档说明如何在 Megatron-LM-FL 中启用 KunLunXin ME-FL 后端，以及如何在运行过程中确认代码确实使用了 KunLunXin 平台和 KunLunXin override 实现。

## 1. 基本概念

KunLunXin 后端在用户侧应保持 CUDA 兼容体验：客户代码可以继续按 CUDA 使用 `torch.cuda` 等路径。ME-FL 内部后续也需要能识别当前 backend 是 KunLunXin，使 `get_platform().device_name()` 返回 `kunlunxin`；这部分 platform selection 修正会在后续 patch 中提交。

当前生产/本地 baseline 中，建议运行时确认分两层：

- CUDA 兼容层：`torch.cuda` API 可用，这是 KunLunXin 复用 CUDA key 时的预期现象。
- KunLunXin override/vendor：`megatron.plugin.decorators` 根据 `MG_FL_PREFER=kunlunxin` 选择 KunLunXin override 实现，期望命中 `megatron.plugin.kunlunxin.*`。

只看到 XME 的 `SYMBOL_REWRITE ... success` 日志，不能证明 ME-FL KunLunXin override 已经生效。当前更可靠的判断是：override 命中 `megatron.plugin.kunlunxin.*`，并且底层日志显示 KunLunXin 库已加载。`device_name: kunlunxin` 将作为后续 platform selection patch 的验证目标。

## 2. 环境变量

推荐运行前设置：

```bash
export XPU=1
export MG_FL_PREFER=kunlunxin
export PYTHONPATH=.
```

需要确认运行时是否真正进入 KunLunXin patch 时，额外打开：

```bash
export MG_FL_KUNLUNXIN_DEBUG=1
```

变量含义：

- `XPU=1`：让 KunLunXin platform 的 `is_available()` 返回可用。
- `MG_FL_PREFER=kunlunxin`：让 override 选择 KunLunXin vendor 实现。
- `MG_FL_KUNLUNXIN_DEBUG=1`：进入 KunLunXin patch 实现时输出 `[KunLunXin Override] ...` 日志。
- `PYTHONPATH=.`：确保本地 Megatron-LM-FL 源码优先被 Python 导入。

## 3. 自动平台选择的当前行为

当前 `megatron/plugin/platform/platform_manager.py` 的平台选择顺序是：

```text
cuda -> musa -> txda -> npu -> enflame -> kunlunxin -> cpu
```

KunLunXin 的 XMLIR/XPyTorch 适配会复用 `torch.cuda` API key，因此在 KunLunXin 环境中 `torch.cuda.is_available()` 返回真是预期行为。当前 platform manager 的 generic CUDA 检查排在 KunLunXin 之前，因此生产/当前 baseline 中可能先看到：

```text
Megatron-LM-FL Platform: cuda Selected
```

这不影响客户侧继续按 CUDA 使用设备，也不妨碍 `MG_FL_PREFER=kunlunxin` 选择 KunLunXin override 实现。后续 platform selection patch 会修正内部 identity，使显式 KunLunXin 环境返回：

```text
Megatron-LM-FL Platform: kunlunxin Selected
```

## 4. 验证当前自动选择结果

在 Megatron-LM-FL 仓库根目录运行：

```bash
XPU=1 MG_FL_PREFER=kunlunxin MG_FL_KUNLUNXIN_DEBUG=1 PYTHONPATH=. python - <<'PY'
import logging

from megatron.training.log_handler import CustomHandler

# Match the logging style used by the standard Megatron training entry.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO, force=True)

# Importing this module executes the centralized register(...) declarations.
import megatron.plugin.override_registry  # noqa: F401
from megatron.plugin.platform import get_platform
from megatron.plugin.decorators import get_override_method

platform = get_platform()
print("platform_type:", type(platform))
print("device_name:", platform.device_name())

print("\n[override impls]")
for key in [
    # P0 patches that are already present in production/local baseline.
    "fused_a2a.get_buffer",
    "fused_a2a.FusedDispatch",
    "fused_a2a.FusedCombine",
    "FileSystemWriterAsync.prepare_write_data",
    "filesystem_async.preload_tensors",
    "filesystem_async.write_preloaded_data",
    "random.CudaRNGStatesTracker",
]:
    impl = get_override_method(key)
    print(f"{key} => {impl} | module={getattr(impl, '__module__', None)}")

print("\n[debug log check]")
impl = get_override_method("fused_a2a.get_buffer")
ret = impl(group=None, hidden_bytes=0)
print("fused_a2a.get_buffer return:", ret)
PY
```

当前 baseline 中可能看到 `device_name: cuda`。platform selection 修正后，期望看到 `device_name: kunlunxin`。

无论当前 platform identity 是否已经修正，override 的 `module=` 应指向：

```text
megatron.plugin.kunlunxin...
```

底层日志也应能看到 KunLunXin 相关库加载，例如：

```text
[KunLunXin] Successfully loaded KunLunXin libs
Registered impl_ids: ['reference.torch', 'vendor.kunlunxin']
```

打开 `MG_FL_KUNLUNXIN_DEBUG=1` 后，实际进入 KunLunXin patch 时还应看到类似：

```text
[KunLunXin Override] transformer.moe.fused_a2a.get_buffer
```

## 5. 强制验证 KunLunXin platform 是否可用

如果只是想确认 `PlatformKunLunXin` 本身可以被设置和使用，可以手动设置当前 platform：

```bash
XPU=1 MG_FL_PREFER=kunlunxin PYTHONPATH=. python - <<'PY'
from megatron.plugin.platform.platform_register import PLATFORMS
from megatron.plugin.platform import set_platform, get_platform
from megatron.plugin.decorators import get_override_method

set_platform(PLATFORMS["kunlunxin"])

platform = get_platform()
print("platform_type:", type(platform))
print("device_name:", platform.device_name())

print("\n[override impls]")
for key in [
    # P0 patches that are already present in production/local baseline.
    "fused_a2a.get_buffer",
    "fused_a2a.FusedDispatch",
    "fused_a2a.FusedCombine",
    "FileSystemWriterAsync.prepare_write_data",
    "filesystem_async.preload_tensors",
    "filesystem_async.write_preloaded_data",
    "random.CudaRNGStatesTracker",
]:
    impl = get_override_method(key)
    print(f"{key} => {impl} | module={getattr(impl, '__module__', None)}")
PY
```

本验证可用于确认 `PlatformKunLunXin` 对象和 override 实现可以被选中。真实训练入口自动选择 KunLunXin platform 的逻辑会在后续 platform selection patch 中提交。

## 6. 运行时判断标准

当前 baseline 建议同时检查三项：

1. CUDA 兼容层可用，platform identity 当前可能仍显示为 CUDA：

```text
device_name: cuda
```

2. Override 命中 KunLunXin 实现：

```text
module=megatron.plugin.kunlunxin...
```

3. 底层 KunLunXin 库已加载，并且实际调用路径进入 KunLunXin 实现。

第三项建议通过 `MG_FL_KUNLUNXIN_DEBUG=1` 验证。该开关只在进入 `megatron.plugin.kunlunxin.*` patch 实现时打印 `[KunLunXin Override] ...`，不开启时不输出额外日志。

## 7. 常见问题

### 设置了 XPU=1 仍然选到 cuda

这是当前 baseline 的已知现象：KunLunXin 复用 CUDA API key，而 platform manager 的 generic CUDA 检查排在 KunLunXin 之前。

本次验证先以 KunLunXin override 命中和底层库加载为准。`get_platform().device_name()` 返回 `kunlunxin` 的修正会在后续 platform selection patch 中提交。

### 看到 SYMBOL_REWRITE success 是否说明 ME-FL KunLunXin 后端生效

不能。`SYMBOL_REWRITE` 是 XME 符号替换路径的日志。ME-FL KunLunXin 后端需要通过 `get_platform()` 和 `get_override_method()` 确认。

### MG_FL_PREFER=kunlunxin 有什么作用

它影响 override vendor 选择，让 `get_override_method()` 优先返回 KunLunXin vendor 的实现。它不一定影响 platform manager 当前选中的 platform。

### MG_FL_KUNLUNXIN_DEBUG=1 有什么作用

它只用于运行时确认是否实际进入 KunLunXin patch 实现。开启后，KunLunXin patch 入口会输出类似：

```text
[KunLunXin Override] transformer.moe.fused_a2a.get_buffer
```

标准 Megatron 训练入口会配置 INFO 级别日志；如果使用自定义入口或日志被外层系统过滤，可以显式设置：

```bash
MEGATRON_LOGGING_LEVEL=20
```

其中 `20` 表示 Python logging 的 `INFO` 级别。
