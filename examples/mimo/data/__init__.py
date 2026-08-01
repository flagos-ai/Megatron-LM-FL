"""Data providers for the MiMo examples."""

__all__ = ["VisionAudioQASample"]


def __getattr__(name: str):
    """Load the AVLM sample type only when that optional provider is selected."""

    if name == "VisionAudioQASample":
        from .energon_avlm_task_encoder import VisionAudioQASample

        return VisionAudioQASample
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
