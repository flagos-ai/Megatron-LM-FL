import pytest


def pytest_addoption(parser):
    """Register pytest options for test configuration and environment."""
    opts = [
        ("--path", "path", "Base directory path for test cases"),
        ("--task", "task", "Task type (train/inference/hetero_train/rl/serve)"),
        ("--model", "model", "Model name (aquila/deepseek/mixtral)"),
        ("--case", "case", "Specific test case configuration"),
        (
            "--platform",
            "platform",
            "Platform type (cuda, etc.) - see tests/test_utils/config/platforms/",
        ),
        ("--device", "device", "Device type (a100/a800/etc.)"),
    ]
    for opt, name, help_text in opts:
        parser.addoption(opt, action="store", default="none", help=help_text)

    # Functional test options — registered here so they are available regardless of
    # which conftest.py is loaded first (pytest_addoption must be called during startup).
    parser.addoption(
        "--allow-nondeterministic-algo",
        action="store_true",
        default=False,
        help="If set, test system checks for approximate results.",
    )
    parser.addoption("--golden-values-path", action="store", default=None, help="Path to golden values")
    parser.addoption("--actual-values-path", action="store", default=None, help="Path to actual values")
    parser.addoption("--actual-values-first-run-path", action="store", default=None, help="Path to actual values (first run)")
    parser.addoption("--actual-values-second-run-path", action="store", default=None, help="Path to actual values (second run)")
    parser.addoption("--scope", action="store", default=None, help="Test scope (MR, weekly, prerelease, release)")
    parser.addoption("--train-iters", action="store", default=100, help="Number of train iters", type=int)
    parser.addoption("--test-values-path", action="store", default=None, help="Path to tensorboard records")
    parser.addoption("--tensorboard-path", action="store", default=None, help="Path to tensorboard records")
    parser.addoption("--logs-dir", action="store", default=None, help="Path to torchrun output logs")
    parser.addoption("--model-config-path", action="store", default=None, help="Path to model_config.yaml")


@pytest.fixture
def path(request):
    return request.config.getoption("--path")


@pytest.fixture
def task(request):
    return request.config.getoption("--task")


@pytest.fixture
def model(request):
    return request.config.getoption("--model")


@pytest.fixture
def case(request):
    return request.config.getoption("--case")


@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")


@pytest.fixture
def device(request):
    return request.config.getoption("--device")
