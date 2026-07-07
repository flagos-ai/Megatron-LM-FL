import pytest

from tests.functional_tests.python_test_utils import common


@pytest.fixture
def compare_approximate_results(request) -> bool:
    """Simple fixture returning whether to check against results approximately."""
    return request.config.getoption("--allow-nondeterministic-algo") is True


@pytest.fixture
def golden_values_path(request):
    """Simple fixture returning golden values."""
    return request.config.getoption("--golden-values-path")


@pytest.fixture
def golden_values(request):
    """Simple fixture returning golden values."""
    return common.read_golden_values_from_json(request.config.getoption("--golden-values-path"))


@pytest.fixture
def actual_values(request):
    """Simple fixture returning golden values."""
    return common.read_golden_values_from_json(request.config.getoption("--actual-values-path"))


@pytest.fixture
def actual_values_first_run(request):
    """Simple fixture returning actual values."""
    return common.read_golden_values_from_json(
        request.config.getoption("--actual-values-first-run-path")
    )


@pytest.fixture
def actual_values_second_run(request):
    """Simple fixture returning actual values."""
    return common.read_golden_values_from_json(
        request.config.getoption("--actual-values-second-run-path")
    )


@pytest.fixture
def scope(request):
    """Simple fixture returning golden values."""
    return request.config.getoption("--scope")


@pytest.fixture
def train_iters(request):
    """Simple fixture returning number of train iters."""
    return request.config.getoption("--train-iters")


@pytest.fixture
def tensorboard_logs(request, train_iters):
    """Simple fixture returning tensorboard metrics."""
    return common.read_tb_logs_as_list(
        request.config.getoption("--tensorboard-path"), train_iters=train_iters
    )


@pytest.fixture
def test_values_path(request):
    return request.config.getoption("--test-values-path")


@pytest.fixture
def tensorboard_path(request):
    """Simple fixture returning path to tensorboard logs."""
    return request.config.getoption("--tensorboard-path")


@pytest.fixture
def logs_dir(request):
    """Simple fixture returning path to torchrun output logs."""
    return request.config.getoption("--logs-dir")


@pytest.fixture
def model_config_path(request):
    """Simple fixture returning path to model_config.yaml."""
    return request.config.getoption("--model-config-path")
