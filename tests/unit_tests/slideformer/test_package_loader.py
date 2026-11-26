"""Never mix installed and in-tree runtime modules or silently fall back."""
from types import SimpleNamespace

import pytest

from megatron.plugin import slideformer_loader as loader
from megatron.plugin.slideformer.config import MegatronSlideFormerConfig


@pytest.fixture(autouse=True)
def clear_environment(monkeypatch):
    for name in ("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "MEGATRON_SLIDEFORMER_ENABLE",
                 "FLAGSCALE_SLIDEFORMER_MEGATRON"):
        monkeypatch.delenv(name, raising=False)


def test_explicit_disable_ignores_legacy_enable_and_missing_wheel(monkeypatch):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "package")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ENABLE", "0")
    monkeypatch.setenv("FLAGSCALE_SLIDEFORMER_MEGATRON", "1")
    seen = []
    monkeypatch.setattr(loader, "import_module", lambda name: seen.append(name) or object())
    loader.load_slideformer()
    assert seen == ["megatron.plugin.slideformer"]
    assert not MegatronSlideFormerConfig.from_env().enable


def test_all_package_submodules_use_same_namespace(monkeypatch):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "package")
    monkeypatch.setenv("FLAGSCALE_SLIDEFORMER_MEGATRON", "1")
    seen = []
    monkeypatch.setattr(loader, "import_module", lambda name: seen.append(name) or SimpleNamespace(API_VERSION=1))
    loader.load_slideformer("gpu_gradient_pipeline")
    loader.load_slideformer("sharded_cpu_optimizer")
    assert seen == ["slideformer_megatron", "slideformer_megatron.gpu_gradient_pipeline",
                    "slideformer_megatron", "slideformer_megatron.sharded_cpu_optimizer"]


def test_missing_requested_wheel_fails_without_fallback(monkeypatch):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "package")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ENABLE", "1")
    def missing(name):
        assert name == "slideformer_megatron"
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(loader, "import_module", missing)
    with pytest.raises(ImportError, match="requested SlideFormer implementation"):
        loader.load_slideformer()


def test_incompatible_api_rejected(monkeypatch):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "package")
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_ENABLE", "1")
    monkeypatch.setattr(loader, "import_module", lambda name: SimpleNamespace(API_VERSION=0))
    with pytest.raises(RuntimeError, match="API_VERSION=1"):
        loader.load_slideformer()


def test_unknown_mode_is_not_silently_accepted(monkeypatch):
    monkeypatch.setenv("MEGATRON_SLIDEFORMER_IMPLEMENTATION", "packge")
    with pytest.raises(ValueError, match="in_tree or package"):
        loader.load_slideformer()
