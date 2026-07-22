"""KunLunXin RNG tracker override matching XME v0.17 behavior."""

import contextlib
import logging
import os

import torch

from megatron.core.tensor_parallel.random import (
    CudaRNGStatesTracker as _CoreCudaRNGStatesTracker,
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
)


def _gpu_aligned_rng_enabled() -> bool:
    """Check whether GPU-aligned RNG initialization is enabled via env var."""
    value = os.getenv("TORCH_NN_INIT_IS_GPU_ALIGNED", "0")
    return value.upper() in ("1", "TRUE", "YES", "Y")


class CudaRNGStatesTrackerKunlunxin(_CoreCudaRNGStatesTracker):
    """Cuda RNG tracker using XMLIR mock RNG state when XME enables it."""

    def add(self, name, seed):
        """Track the rng state using XMLIR mock RNG when GPU-aligned.

        Args:
            name: Unique name for the rng state.
            seed: Integer seed for the rng state.
        """
        if not _gpu_aligned_rng_enabled():
            return super().add(name, seed)

        from torch_xmlir.symbrewrite.plugins.torch.mock_torch_init import (
            MockRNGStateFinal,
            get_global_mock_rng_state_final,
            set_global_mock_rng_state_final,
        )

        self._is_initialized = True
        if seed in self.seeds_:
            raise Exception('seed {} already exists'.format(seed))
        self.seeds_.add(seed)
        if name in self.states_:
            raise Exception('cuda rng state {} already exists'.format(name))

        orig_rng_state = get_global_mock_rng_state_final()
        new_state = MockRNGStateFinal()
        set_global_mock_rng_state_final(new_state)

        torch.cuda.manual_seed(seed)
        self.states_[name] = get_global_mock_rng_state_final()

        set_global_mock_rng_state_final(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state using XMLIR mock RNG when GPU-aligned.

        Args:
            name: Name of the previously added rng state to fork.
        """
        if not _gpu_aligned_rng_enabled():
            with super().fork(name):
                yield
            return

        from torch_xmlir.symbrewrite.plugins.torch.mock_torch_init import (
            get_global_mock_rng_state_final,
            set_global_mock_rng_state_final,
        )

        if name not in self.states_:
            raise Exception('cuda rng state {} is not added'.format(name))
        orig_rng_state = get_global_mock_rng_state_final()
        set_global_mock_rng_state_final(self.states_[name])
        cpu_rng_state = torch.get_rng_state()
        try:
            yield
        finally:
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning(
                    'CPU RNG state changed within GPU RNG context'
                )
            self.states_[name] = get_global_mock_rng_state_final()
            set_global_mock_rng_state_final(orig_rng_state)
