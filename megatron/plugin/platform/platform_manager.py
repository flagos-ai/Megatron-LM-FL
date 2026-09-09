# Copyright (c) BAAI Corporation.

from .platform_register import PLATFORMS

cur_platform = None


def is_current_platform_supported():
    return get_platform() in PLATFORMS.values()


def _select_detected_platform():
    available_platforms = {
        name: platform for name, platform in PLATFORMS.items() if platform.is_available()
    }

    # Some accelerator runtimes expose a CUDA-compatible API. In that case,
    # torch.cuda may report available even though tensors use the vendor native
    # device type (for example, GCU on Enflame). Prefer the single detected
    # native vendor backend over this CUDA compatibility claim.
    native_vendor_names = sorted(
        name for name in available_platforms if name not in ("cpu", "cuda")
    )
    if len(native_vendor_names) == 1:
        return available_platforms[native_vendor_names[0]]
    if len(native_vendor_names) > 1:
        raise RuntimeError(
            "Multiple native accelerator platforms are available: "
            f"{', '.join(native_vendor_names)}"
        )
    if "cuda" in available_platforms:
        return available_platforms["cuda"]
    if "cpu" in available_platforms:
        return available_platforms["cpu"]
    raise ValueError("No platform is available")


def get_platform():
    global cur_platform
    if cur_platform is not None:
        return cur_platform

    cur_platform = _select_detected_platform()
    platform_name = cur_platform._name
    print(f"Megatron-LM-FL Platform: {platform_name} Selected")

    if platform_name == "kunlunxin":
        # Deferred XME init after platform selection to avoid a circular import.
        cur_platform.ensure_xme_init()

    return cur_platform


def set_platform(platform_obj):
    global cur_platform
    cur_platform = platform_obj
