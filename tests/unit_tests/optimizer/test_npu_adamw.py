"""Dependency-free contract tests; these do not claim NPU numerical validation.

Run directly with Python. The real override dispatcher and source functions are
loaded in isolation, with hardware and tensor operations replaced by test doubles.
"""

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
CORE = ROOT / "megatron/core/optimizer/__init__.py"
DISTRIBUTED = ROOT / "megatron/core/optimizer/distrib_optimizer.py"
PLUGIN = ROOT / "megatron/plugin/Ascend/optimizer/adamw.py"


class Scalar:
    def __init__(self, value, dtype="float32", device="cpu"):
        self.value, self.dtype, self.device = value, dtype, device

    def item(self):
        return self.value

    def detach(self):
        return self

    def clone(self):
        return copy.copy(self)


class Parameter:
    def __init__(self, device="npu"):
        self.device = SimpleNamespace(type=device)
        self.data = self


class Adam:
    def __init__(self, values=(3, 3)):
        self.state = {index: {"step": Scalar(value)} for index, value in enumerate(values)}
        self.param_groups = [{"params": list(self.state)}]

    def state_dict(self):
        return copy.deepcopy({"state": self.state, "param_groups": self.param_groups})

    def load_state_dict(self, state):
        self.state = state["state"]
        self.param_groups = state["param_groups"]


class AdamW(Adam):
    pass


class FusedAdam:
    def state_dict(self):
        return {"state": {}, "param_groups": [{"params": [0], "step": 3}]}


def source_function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def execute_function(node, namespace):
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), "<source>", "exec"), namespace)
    return namespace[node.name]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class NpuAdamWContractTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"MG_FL_PREFER": "npu"})
        self.env.start()
        self.addCleanup(self.env.stop)
        fake_torch = ModuleType("torch")
        fake_torch.optim = SimpleNamespace(Adam=Adam, AdamW=AdamW)
        fake_torch.npu = SimpleNamespace(is_available=lambda: True)
        fake_torch.float32 = fake_torch.float = "float32"
        fake_torch.tensor = lambda value, dtype: Scalar(value, dtype)
        fake_torch.zeros = lambda shape, dtype, device: Scalar(0, dtype, device)
        fake_torch.zeros_like = lambda value: Scalar(0)
        self.torch = fake_torch
        packages = {"torch": fake_torch}
        for name in ("megatron", "megatron.plugin", "megatron.core", "megatron.plugin.Ascend",
                     "megatron.plugin.Ascend.optimizer", "megatron.plugin.override_registry"):
            packages[name] = ModuleType(name)
            packages[name].__path__ = []
        self.modules = patch.dict(sys.modules, packages)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.dispatcher = load_module("megatron.plugin.decorators", ROOT / "megatron/plugin/decorators.py")
        self.core = ModuleType("megatron.core.optimizer")
        self.core.__dict__.update(torch=fake_torch, overridable=self.dispatcher.overridable,
                                  OptimizerConfig=object, Dict=dict, Any=object,
                                  USING_PYTORCH_OPTIMIZER=False, Adam=FusedAdam)
        sys.modules[self.core.__name__] = self.core
        execute_function(source_function(CORE, "_get_adam_class"), self.core.__dict__)
        self.plugin = load_module("megatron.plugin.Ascend.optimizer.adamw", PLUGIN)
        registry = ast.parse((ROOT / "megatron/plugin/override_registry.py").read_text(encoding="utf-8"))
        registration = next(
            node for node in registry.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and any(k.arg == "target" and isinstance(k.value, ast.Constant)
                    and k.value.value == "megatron.core.optimizer._get_adam_class"
                    for k in node.value.keywords)
        )
        exec(compile(ast.Module(body=[registration], type_ignores=[]), "<registry>", "exec"),
             {"register": self.dispatcher.register})
        self.config = SimpleNamespace(decoupled_weight_decay=True, optimizer_cuda_graph=False,
                                      use_precision_aware_optimizer=False, optimizer_cpu_offload=False)
        self.kwargs = {"params": [{"params": [Parameter()]}], "capturable": False}

    def test_npu_dispatch_preserves_native_class_and_exact_options(self):
        self.assertIs(self.core._get_adam_class(self.config, self.kwargs), AdamW)
        self.assertEqual({k: self.kwargs[k] for k in ("fused", "foreach", "capturable")},
                         {"fused": True, "foreach": False, "capturable": False})
        self.assertNotIn("adam_w_mode", self.kwargs)

    def test_explicit_default_uses_original_te_selection(self):
        os.environ["MG_FL_PREFER"] = "default"
        self.assertIs(self.core._get_adam_class(self.config, self.kwargs), FusedAdam)
        self.assertTrue(self.kwargs["adam_w_mode"])
        self.assertNotIn("fused", self.kwargs)

    def test_native_default_selects_adam_or_adamw(self):
        self.core.USING_PYTORCH_OPTIMIZER = True
        original = self.core._get_adam_class.__wrapped__
        self.assertIs(original(self.config, self.kwargs), AdamW)
        self.config.decoupled_weight_decay = False
        self.assertIs(original(self.config, self.kwargs), Adam)

    def test_no_npu_falls_back_without_vendor_options(self):
        del self.torch.npu
        self.assertIs(self.plugin.get_adam_class(self.config, self.kwargs), FusedAdam)
        self.assertNotIn("fused", self.kwargs)

    def test_unavailable_npu_falls_back(self):
        self.torch.npu.is_available = lambda: False
        self.assertIs(self.plugin.get_adam_class(self.config, self.kwargs), FusedAdam)

    def test_unsupported_modes_preserve_original_path(self):
        for key, value in (("decoupled_weight_decay", False), ("optimizer_cuda_graph", True),
                           ("use_precision_aware_optimizer", True), ("optimizer_cpu_offload", True)):
            with self.subTest(key=key):
                config = copy.copy(self.config)
                setattr(config, key, value)
                kwargs = copy.deepcopy(self.kwargs)
                self.assertIs(self.plugin.get_adam_class(config, kwargs), FusedAdam)
                self.assertNotIn("fused", kwargs)

    def test_cpu_mixed_and_empty_params_do_not_select_npu(self):
        for params in ([Parameter("cpu")], [Parameter(), Parameter("cpu")], []):
            with self.subTest(params=params):
                kwargs = {"params": [{"params": params}]}
                self.assertIs(self.plugin.get_adam_class(self.config, kwargs), FusedAdam)

    def distributed_function(self, name, have_te=True):
        namespace = {"torch": self.torch, "HAVE_APEX_OR_TE": have_te,
                     "HybridDeviceOptimizer": type("HybridDeviceOptimizer", (), {}),
                     "USING_TE_OPTIMIZER": have_te, "USING_APEX_OPTIMIZER": False,
                     "param_group_identifier_keys": []}
        return execute_function(source_function(DISTRIBUTED, name), namespace)

    def test_native_save_uses_instance_with_and_without_te(self):
        for have_te in (True, False):
            for cls in (Adam, AdamW):
                with self.subTest(have_te=have_te, optimizer=cls):
                    state = self.distributed_function("state_dict", have_te)(
                        SimpleNamespace(optimizer=cls(), grad_scaler=None))
                    self.assertEqual(state["optimizer"]["param_groups"][0]["step"], 3)

    def test_empty_native_state_can_be_saved_at_step_zero(self):
        state = self.distributed_function("state_dict")(
            SimpleNamespace(optimizer=AdamW(values=()), grad_scaler=None))
        self.assertEqual(state["optimizer"]["param_groups"][0]["step"], 0)

    def test_inconsistent_parameter_steps_are_rejected(self):
        with self.assertRaises(AssertionError):
            self.distributed_function("state_dict")(
                SimpleNamespace(optimizer=AdamW(values=(2, 3)), grad_scaler=None))

    def test_te_save_still_reads_group_step(self):
        state = self.distributed_function("state_dict")(
            SimpleNamespace(optimizer=FusedAdam(), grad_scaler=None))
        self.assertEqual(state["optimizer"]["param_groups"][0]["step"], 3)

    def test_native_load_restores_distinct_step_scalars(self):
        optimizer = AdamW(values=(0, 0))
        receiver = SimpleNamespace(optimizer=optimizer, grad_scaler=None,
                                   ddp_config=SimpleNamespace(use_megatron_fsdp=False),
                                   config=SimpleNamespace(fp16=False))
        self.distributed_function("load_state_dict")(
            receiver, {"optimizer": {"param_groups": [{"step": 3}]}})
        steps = [state["step"] for state in optimizer.state.values()]
        self.assertEqual([step.item() for step in steps], [3, 3])
        self.assertIsNot(steps[0], steps[1])
        self.assertEqual(steps[0].dtype, "float32")

    def test_native_init_creates_step_before_moments(self):
        init = execute_function(source_function(CORE, "init_state_fn"), {"torch": self.torch})
        for fused in (True, False):
            optimizer, param = AdamW(values=()), Parameter()
            optimizer.param_groups = [{"params": [param], "fused": fused}]
            optimizer.state = {param: {}}
            init(optimizer)
            state = optimizer.state[param]
            self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq"})
            self.assertEqual(state["step"].item(), 0)
            self.assertEqual(state["step"].device, param.device if fused else "cpu")

    def test_sharded_formats_already_handle_step_separately(self):
        for name in ("sharded_param_state_fs_model_space",
                     "load_parameter_state_from_fs_model_space",
                     "load_parameter_state_from_fully_reshardable"):
            with self.subTest(name=name):
                node = source_function(DISTRIBUTED, name)
                self.assertTrue(any(
                    isinstance(branch, ast.If)
                    and any(isinstance(value, ast.Constant) and value.value == "step"
                            for value in ast.walk(branch.test))
                    and any(isinstance(child, ast.Continue) for child in branch.body)
                    for branch in ast.walk(node)
                ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
