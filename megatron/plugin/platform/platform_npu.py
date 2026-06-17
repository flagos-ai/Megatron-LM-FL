# Adopted from DeepSpeed Accelerator, https://github.com/deepspeedai/DeepSpeed/

import os
import sys
import functools
import torch
from torch_npu.contrib import transfer_to_npu

from .platform_base import PlatformBase
import threading
import logging

logger = logging.getLogger(__name__)

import math
from functools import wraps
import torch.distributed
import importlib.metadata

import importlib
import inspect
import types
from typing import Dict, Union, List

try:
    import torch_npu
except ImportError:
    pass

# Thread-local storage for range counter and stack
_range_local = threading.local()


class PlatformNPU(PlatformBase):

    def __init__(self):
        self._name = 'npu'

    def is_available(self):
        try:
            import torch
            # Determine if we are on a NPU device.
            if torch_npu.npu.device_count() > 0 and torch_npu.npu.is_available():  # ignore-npu
                return True
            else:
                return False
        except Exception as e:
            return False

    def get_device_properties(self, device_index=None):
        return torch_npu.npu.get_device_properties(device_index)

    def get_device_capability(self, device_index=None):
        return 9, 0

    def is_synchronized_device(self):
        return False

    def use_host_timers(self):
        return self.is_synchronized_device()

    def resolves_data_dependency(self):
        return self.is_synchronized_device()

    def handles_memory_backpressure(self):
        return self.is_synchronized_device()

    # Device APIs
    def device_name(self, device_index=None):
        if device_index is None:
            return 'npu'
        return 'npu:{}'.format(device_index)

    def device(self, device_index=None):
        return torch.device('npu', device_index)

    def set_device(self, device_index):
        torch_npu.npu.set_device(device_index)

    def current_device(self):
        return torch_npu.npu.current_device()

    def current_device_name(self):
        return 'npu:{}'.format(torch_npu.npu.current_device())

    def device_count(self):
        return torch_npu.npu.device_count()

    def synchronize(self, device_index=None):
        return torch_npu.npu.synchronize(device_index)

    # RNG APIs
    def random(self):
        return torch.random

    def set_rng_state(self, new_state, device_index=None):
        if device_index is None:
            return torch_npu.npu.set_rng_state(new_state)

        return torch_npu.npu.set_rng_state(new_state, device_index)

    def get_rng_state(self, device=None):
        if device is None:
            return torch_npu.npu.get_rng_state()

        return torch_npu.npu.get_rng_state(device)

    def manual_seed(self, seed):
        return torch_npu.npu.manual_seed(seed)

    def manual_seed_all(self, seed):
        return torch_npu.npu.manual_seed_all(seed)

    def initial_seed(self):
        return torch_npu.npu.initial_seed()

    @property
    def default_generators(self):
        return torch.npu.default_generators

    # Streams/Events
    @property
    def Stream(self):
        return torch_npu.npu.Stream

    def stream(self, stream):
        return torch_npu.npu.stream(stream)

    def set_stream(self, stream):
        return torch_npu.npu.set_stream(stream)

    def current_stream(self, device_index=None):
        return torch_npu.npu.current_stream(device_index)

    def default_stream(self, device_index=None):
        return torch_npu.npu.default_stream(device_index)

    @property
    def MemPool(self):
        return torch.npu.MemPool

    def use_mem_pool(self, pool):
        return torch.npu.use_mem_pool(pool)

    @property
    def Event(self):
        return torch_npu.npu.Event

    # Memory management
    def empty_cache(self):
        return torch_npu.npu.empty_cache()

    def memory_allocated(self, device_index=None):
        return torch_npu.npu.memory_allocated(device_index)

    def max_memory_allocated(self, device_index=None):
        return torch_npu.npu.max_memory_allocated(device_index)

    def reset_max_memory_allocated(self, device_index=None):
        return torch_npu.npu.reset_max_memory_allocated(device_index)

    def memory_cached(self, device_index=None):
        return torch_npu.npu.memory_cached(device_index)

    def max_memory_cached(self, device_index=None):
        return torch_npu.npu.max_memory_cached(device_index)

    def reset_max_memory_cached(self, device_index=None):
        return torch_npu.npu.reset_max_memory_cached(device_index)

    def memory_stats(self, device_index=None):
        if hasattr(torch_npu.npu, 'memory_stats'):
            return torch_npu.npu.memory_stats(device_index)

    def reset_peak_memory_stats(self, device_index=None):
        if hasattr(torch_npu.npu, 'reset_peak_memory_stats'):
            return torch_npu.npu.reset_peak_memory_stats(device_index)

    def memory_reserved(self, device_index=None):
        if hasattr(torch_npu.npu, 'memory_reserved'):
            return torch_npu.npu.memory_reserved(device_index)

    def max_memory_reserved(self, device_index=None):
        if hasattr(torch_npu.npu, 'max_memory_reserved'):
            return torch_npu.npu.max_memory_reserved(device_index)

    def total_memory(self, device_index=None):
        return torch_npu.npu.get_device_properties(device_index).total_memory

    def available_memory(self, device_index=None):
        return self.total_memory(device_index) - self.memory_allocated(device_index)

    # Data types
    def is_bf16_supported(self):
        if not torch_npu.npu.is_available():
            return False
        return True

    def is_fp16_supported(self):
        if not torch_npu.npu.is_available():
            return False
        return True

    def supported_dtypes(self):
        supported_dtypes = [torch.float]
        if self.is_fp16_supported():
            supported_dtypes.append(torch.half)
        if self.is_bf16_supported():
            supported_dtypes.append(torch.bfloat16)
        return supported_dtypes

    # Misc
    def amp(self):
        if hasattr(torch_npu.npu, 'amp'):
            return torch_npu.npu.amp
        return None

    def range(self, msg):
        if hasattr(torch_npu.npu.mstx, 'mstx_range'):
            return torch_npu.npu.mstx.mstx_range(msg)

    def _get_thread_range_counter(self):
        if not hasattr(_range_local, 'counter'):
            _range_local.counter = 0
        _range_local.counter += 1
        return _range_local.counter

    def _get_range_stack(self):
        if not hasattr(_range_local, 'stack'):
            _range_local.stack = []
        return _range_local.stack

    def range_push(self, msg):
        if hasattr(torch_npu.npu.mstx, 'range_start'):
            # return torch_npu.npu.mstx.range_start(msg)
            range_id = self._get_thread_range_counter()
            self._get_range_stack().append(range_id)
            return torch_npu.npu.mstx.range_start(msg, range_id)

    def range_pop(self):
        if hasattr(torch_npu.npu.mstx, 'range_end'):
            stack = self._get_range_stack()
            if stack:
                range_id = stack.pop()
                return torch_npu.npu.mstx.range_end(range_id)
            else:
                import traceback

                logger.warning(
                    "Attempted to pop NVTX range from empty stack. "
                    "This indicates a mismatch between range_push and range_pop calls. "
                    "Call stack:\n%s",
                    traceback.format_stack(),
                )
                return None

    def lazy_call(self, callback):
        pass

    def is_triton_supported(self):
        pass

    # Graph operations
    def create_graph(self):
        return torch.npu.NPUGraph()

    def capture_to_graph(self, graph, pool=None, stream=None):
        return torch.npu.graph(graph, pool, stream)

    def replay_graph(self, graph):
        graph.replay()
        return

    # Tensor operations

    @property
    def BFloat16Tensor(self):
        return torch.npu.BFloat16Tensor
        # return functools.partial(torch.tensor, dtype=torch.bfloat16, device='npu')

    @property
    def ByteTensor(self):
        return torch.npu.ByteTensor
        # return functools.partial(torch.tensor, dtype=torch.uint8, device='npu')

    @property
    def DoubleTensor(self):
        return torch.npu.DoubleTensor
        # return functools.partial(torch.tensor, dtype=torch.double, device='npu')

    @property
    def FloatTensor(self):
        return torch.npu.FloatTensor
        # return functools.partial(torch.tensor, dtype=torch.float, device='npu')

    @property
    def HalfTensor(self):
        return torch.npu.HalfTensor
        # return functools.partial(torch.tensor, dtype=torch.half, device='npu')

    @property
    def IntTensor(self):
        return torch.npu.IntTensor
        # return functools.partial(torch.tensor, dtype=torch.int, device='npu')

    @property
    def LongTensor(self):
        return torch.npu.LongTensor
        # return functools.partial(torch.tensor, dtype=torch.long, device='npu')

    def pin_memory(self, tensor, align_bytes=1):
        return tensor.pin_memory()

    def is_pinned(self, tensor):
        return tensor.is_pinned()

    def on_accelerator(self, tensor):
        device_str = str(tensor.device)
        if device_str.startswith('npu:'):
            return True
        else:
            return False

    def build_extension(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def visible_devices_envs(self):
        return ['ASCEND_RT_VISIBLE_DEVICES']

    def set_visible_devices_envs(self, current_env, local_accelerator_ids):
        for env in self.visible_devices_envs():
            current_env[env] = ",".join(map(str, local_accelerator_ids))

    def get_compile_backend(self):
        pass

    def set_compile_backend(self, backend):
        pass

    def temperature(self):
        pass

    def power_draw(self):
        pass

    def utilization(self):
        pass

    def clock_rate(self):
        pass


def type_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if isinstance(res, str):
            res = res.replace('npu', 'cuda')
        return res

    return wrapper


def ensure_contiguous_wrapper(fn):
    @wraps(fn)
    def wrapper(tensor, *args, **kwargs):
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        return fn(tensor, *args, **kwargs)

    return wrapper


def lcm(a, b):
    return (a * b) // math.gcd(a, b)


def dummy_function(*args, **kwargs):
    pass


def torch_all_reduce_double_dtype_bypass_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if torch.is_tensor(args[0]) and args[0].dtype == torch.double:
            args = list(args)
            args[0] = args[0].float()
            handle = fn(*args, **kwargs)
            if handle is not None:
                handle.wait()
            args[0] = args[0].double()
            return None

        return fn(*args, **kwargs)

    return wrapper


def dummy_compile(*args, **kwargs):
    if len(args) > 0 and callable(args[0]):
        def wrapper(*fn_args, **fn_kwargs):
            return args[0](*fn_args, **fn_kwargs)

        return wrapper
    else:
        def compile_wrapper(fn):
            def wrapper(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)

            return wrapper

        return compile_wrapper


def version_wrapper(fn):
    @wraps(fn)
    def wrapper(name, *args, **kwargs):
        return '2.2.0' if name == 'transformer-engine' else fn(name, *args, **kwargs)

    return wrapper


def get_func_name(func):
    if isinstance(func, str):
        return func
    return '.'.join((func.__module__, func.__qualname__))


def dummy_function_wrapper(func_name):
    def dummy_function(*args, **kwargs):
        raise RuntimeError('function {} no exist'.format(func_name))

    return dummy_function


class Patch:
    def __init__(self, orig_func_name, new_func, create_dummy):
        split_name = orig_func_name.rsplit('.', 1)
        if len(split_name) == 1:
            self.orig_module_name, self.orig_func_name = orig_func_name, None
        else:
            self.orig_module_name, self.orig_func_name = split_name
        self.orig_module = None
        self.orig_func = None
        self.patch_func = None
        self.final_patch_func = None
        self.wrappers = []
        if new_func is None:
            new_func = dummy_function_wrapper(orig_func_name)
        self.set_patch_func(new_func)
        self.is_applied = False
        self.create_dummy = create_dummy

    @property
    def orig_func_id(self):
        return id(self.orig_func)

    @property
    def patch_func_id(self):
        return id(self.patch_func)

    def set_patch_func(self, new_func, force_patch=False):
        if hasattr(new_func, '__name__') and new_func.__name__.endswith(('wrapper', 'decorator')):
            if new_func not in self.wrappers:
                self.wrappers.append(new_func)
        else:
            if self.patch_func and not force_patch:
                raise RuntimeError('the patch of {} exist !'.format(self.orig_func_name))
            self.patch_func = new_func
        self.is_applied = False

    def remove_wrappers(self, wrapper_names: Union[str, List[str]] = None):
        if wrapper_names is None:
            self.wrappers.clear()
            return
        if isinstance(wrapper_names, str):
            wrapper_names = [wrapper_names]
        for name in wrapper_names:
            i = 0
            while i < len(self.wrappers):
                if self.wrappers[i].__name__ == name:
                    self.wrappers.pop(i)
                else:
                    i += 1

    def remove_patch(self):
        for key, value in sys.modules.copy().items():
            if 'megatron_adaptor' in key or 'mindspeed' in key or 'torch.classes' == key:
                continue
            if inspect.isclass(self.orig_module) and hasattr(value, self.orig_module_name.split('.')[-1]):
                value = getattr(value, self.orig_module_name.split('.')[-1])
            if self.orig_func_name is not None and hasattr(value, self.orig_func_name) \
                    and id(getattr(value, self.orig_func_name)) == id(self.final_patch_func):
                setattr(value, self.orig_func_name, self.orig_func)
        self.patch_func = None
        self.final_patch_func = None
        self.is_applied = False

    def apply_patch(self):
        if self.is_applied:
            return
        current_module, current_func = Patch.parse_path(self.orig_module_name, self.orig_func_name, self.create_dummy)
        if self.orig_module is None:
            self.orig_module, self.orig_func = current_module, current_func
        final_patch_func = self.orig_func
        if self.patch_func is not None:
            final_patch_func = self.patch_func
        for wrapper in self.wrappers:
            final_patch_func = wrapper(final_patch_func)
        if self.orig_func_name is not None:
            setattr(self.orig_module, self.orig_func_name, final_patch_func)
        for _, value in sys.modules.copy().items():
            if self.orig_func_name is not None and hasattr(value, self.orig_func_name) \
                    and id(getattr(value, self.orig_func_name)) == id(current_func):
                setattr(value, self.orig_func_name, final_patch_func)
        self.is_applied = True
        self.final_patch_func = final_patch_func

    @staticmethod
    def parse_path(module_path, function_name, create_dummy):
        from importlib.machinery import ModuleSpec
        modules = module_path.split('.')
        for i in range(1, len(modules) + 1):
            parent = '.'.join(modules[:i - 1])
            path = '.'.join(modules[:i])
            try:
                importlib.import_module(path)
            except ModuleNotFoundError as e:
                if not parent or not hasattr(importlib.import_module(parent), modules[i - 1]):
                    if not create_dummy:
                        raise ModuleNotFoundError(e) from e
                    sys.modules[path] = types.ModuleType(path)
                    sys.modules[path].__file__ = 'mindspeed.dummy_module.py'
                    sys.modules[path].__spec__ = ModuleSpec(path, None)
                    if parent:
                        setattr(importlib.import_module(parent), modules[i - 1], sys.modules[path])
                else:
                    module = getattr(importlib.import_module(parent), modules[i - 1])
                    if hasattr(module, function_name):
                        return module, getattr(module, function_name)
                    elif create_dummy:
                        return module, dummy_function_wrapper(function_name)
                    else:
                        raise RuntimeError('no exist {} of {}'.format(function_name, module))

        if function_name is not None and not hasattr(sys.modules[module_path], function_name):
            setattr(sys.modules[module_path], function_name, None)
        return sys.modules[module_path], getattr(sys.modules[module_path],
                                                 function_name) if function_name is not None else None


class PatchesManager:
    patches_info: Dict[str, Patch] = {}

    @staticmethod
    def register_patch(orig_func_name, new_func=None, force_patch=False, create_dummy=False):
        """Patch registration method. When this method is executed, the patch does not take effect in real time.
        It takes effect only after the apply_patches method is invoked. Other details are as follows:

        1. If `orig_func_name` does not exist and create_dummy is set to True, a dummy function is created to ensure
        that the import is normal.
        2. If `orig_func_name` is not None, `orig_func_name` is replaced with `new_func`.
        3. If the `new_func` function name ends with `wrapper` or `decorator`, then `new_func` is decorated on
        `orig_func_name` as a decorator, and the decorator can be superimposed repeatedly.
        4. When force_patch=False, a function cannot be replaced repeatedly (but can be decorated repeatedly),
        otherwise the replacement is overwritten.
        """
        if orig_func_name not in PatchesManager.patches_info:
            PatchesManager.patches_info[orig_func_name] = Patch(orig_func_name, new_func, create_dummy)
        else:
            PatchesManager.patches_info.get(orig_func_name).set_patch_func(new_func, force_patch)

    @staticmethod
    def remove_wrappers(orig_func_name, wrappers_name, remove_check=True):
        """Remove wrapper registered in orig_func_name."""
        if orig_func_name not in PatchesManager.patches_info:
            raise ValueError('The function <{}> not exist.'.format(orig_func_name))
        patch = PatchesManager.patches_info.get(orig_func_name)
        wrappers_len = len(patch.wrappers)
        patch.remove_wrappers(wrappers_name)
        if remove_check and wrappers_len == len(patch.wrappers):
            raise RuntimeError('Remove wrappers has not remove anything.')

    @staticmethod
    def remove_patches():
        for patch in PatchesManager.patches_info.values():
            patch.remove_patch()
            patch.remove_wrappers()

    @staticmethod
    def apply_patches():
        for patch in PatchesManager.patches_info.values():
            patch.apply_patch()

    @staticmethod
    def get_patch(orig_func_name):
        return PatchesManager.patches_info.get(orig_func_name)


def registry_patch():
    PatchesManager.register_patch('torch.compile', dummy_compile)
    PatchesManager.register_patch('torch.jit.script', dummy_compile)
    PatchesManager.register_patch('torch.nn.parameter.Parameter.type', type_wrapper)
    PatchesManager.register_patch('torch.Tensor.type', type_wrapper)
    PatchesManager.register_patch('torch.Tensor.view', ensure_contiguous_wrapper)
    PatchesManager.register_patch('torch.distributed.all_reduce', torch_all_reduce_double_dtype_bypass_wrapper)
    PatchesManager.register_patch('torch._C._jit_set_nvfuser_enabled', dummy_function, create_dummy=True)

    if sys.version_info < (3, 9):
        PatchesManager.register_patch('math.lcm', lcm, create_dummy=True)

    PatchesManager.register_patch('importlib.metadata.version', version_wrapper)

    PatchesManager.apply_patches()
