from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MegatronSlideFormerConfig:
    enable: bool = False
    mode: str = "true_slideformer"
    activation_offload: bool = False
    activation_offload_min_numel: int = 1
    activation_offload_max_dim: int = 99
    activation_offload_max_tensors_per_layer: int = -1
    param_prefetch: bool = True
    chunk_size_mb: int = 32
    nvme_offload_fraction: float = 0.0
    offload_dir: str | None = None
    double_buffer: bool = False
    gpu_buffer_count: int = 2
    window_size: int = 1
    activation_backend: str = "slideformer-slot"
    activation_slot_prefetch: bool = True
    activation_slot_gpu_window: int = 0
    unified_h2d_scheduler: bool = True
    max_outstanding_h2d: int = 3
    overlap_grad_d2h_cpu_adam: bool = True
    optimizer_pipeline_depth: int = 2
    optimizer_worker_count: int = 1
    cpu_adam_per_layer_optimizer: bool = False
    shared_cpu_buffers: bool = True
    cpu_grad_buffer_count: int = 0
    cpu_param_staging_buffer_count: int = 1
    te_fused_main_grad: bool = True
    fp32_master_params: bool = True
    kernel_policy: str = "auto"
    attention_backend: str = "auto"
    mlp_backend: str = "auto"
    split_swiglu_threshold_gib: float = 0.5
    loss_backend: str = "auto"
    norm_backend: str = "auto"
    rope_backend: str = "auto"
    strict_kernels: bool = True
    strict: bool = True

    @classmethod
    def from_env(cls) -> "MegatronSlideFormerConfig":
        return cls(
            enable=os.getenv("MEGATRON_SLIDEFORMER_ENABLE", "0").lower()
            in {"1", "true", "yes", "on"}
            or os.getenv("FLAGSCALE_SLIDEFORMER_MEGATRON", "0").lower()
            in {"1", "true", "yes", "on"},
            mode=os.getenv(
                "MEGATRON_SLIDEFORMER_MODE",
                os.getenv("FLAGSCALE_SLIDEFORMER_MODE", "true_slideformer"),
            ),
            activation_offload=os.getenv(
                "MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD",
                os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_OFFLOAD", "0"),
            ).lower()
            in {"1", "true", "yes", "on"},
            activation_offload_min_numel=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD_MIN_NUMEL",
                    os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_OFFLOAD_MIN_NUMEL", "1"),
                )
            ),
            activation_offload_max_dim=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD_MAX_DIM",
                    os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_OFFLOAD_MAX_DIM", "99"),
                )
            ),
            activation_offload_max_tensors_per_layer=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_ACTIVATION_OFFLOAD_MAX_TENSORS_PER_LAYER",
                    os.getenv(
                        "FLAGSCALE_SLIDEFORMER_ACTIVATION_OFFLOAD_MAX_TENSORS_PER_LAYER", "-1"
                    ),
                )
            ),
            param_prefetch=os.getenv(
                "MEGATRON_SLIDEFORMER_PARAM_PREFETCH",
                os.getenv("FLAGSCALE_SLIDEFORMER_PARAM_PREFETCH", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            chunk_size_mb=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_CHUNK_SIZE_MB",
                    os.getenv("FLAGSCALE_SLIDEFORMER_CHUNK_SIZE_MB", "32"),
                )
            ),
            nvme_offload_fraction=float(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_NVME_OFFLOAD_FRACTION",
                    os.getenv("FLAGSCALE_SLIDEFORMER_NVME_OFFLOAD_FRACTION", "0.0"),
                )
            ),
            offload_dir=os.getenv(
                "MEGATRON_SLIDEFORMER_OFFLOAD_DIR",
                os.getenv("FLAGSCALE_SLIDEFORMER_OFFLOAD_DIR", ""),
            )
            or None,
            double_buffer=os.getenv(
                "MEGATRON_SLIDEFORMER_DOUBLE_BUFFER",
                os.getenv("FLAGSCALE_SLIDEFORMER_DOUBLE_BUFFER", "0"),
            ).lower()
            not in {"0", "false", "no", "off"},
            gpu_buffer_count=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_GPU_BUFFER_COUNT",
                    os.getenv("FLAGSCALE_SLIDEFORMER_GPU_BUFFER_COUNT", "2"),
                )
            ),
            window_size=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_WINDOW_SIZE",
                    os.getenv("FLAGSCALE_SLIDEFORMER_WINDOW_SIZE", "1"),
                )
            ),
            activation_backend=os.getenv(
                "MEGATRON_SLIDEFORMER_ACTIVATION_BACKEND",
                os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_BACKEND", "slideformer-slot"),
            ),
            activation_slot_prefetch=os.getenv(
                "MEGATRON_SLIDEFORMER_ACTIVATION_SLOT_PREFETCH",
                os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_SLOT_PREFETCH", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            activation_slot_gpu_window=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_ACTIVATION_SLOT_GPU_WINDOW",
                    os.getenv("FLAGSCALE_SLIDEFORMER_ACTIVATION_SLOT_GPU_WINDOW", "0"),
                )
            ),
            unified_h2d_scheduler=os.getenv(
                "MEGATRON_SLIDEFORMER_UNIFIED_H2D_SCHEDULER",
                os.getenv("FLAGSCALE_SLIDEFORMER_UNIFIED_H2D_SCHEDULER", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            max_outstanding_h2d=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_MAX_OUTSTANDING_H2D",
                    os.getenv("FLAGSCALE_SLIDEFORMER_MAX_OUTSTANDING_H2D", "3"),
                )
            ),
            overlap_grad_d2h_cpu_adam=os.getenv(
                "MEGATRON_SLIDEFORMER_OVERLAP_GRAD_D2H_CPU_ADAM",
                os.getenv("FLAGSCALE_SLIDEFORMER_OVERLAP_GRAD_D2H_CPU_ADAM", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            optimizer_pipeline_depth=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_OPTIMIZER_PIPELINE_DEPTH",
                    os.getenv("FLAGSCALE_SLIDEFORMER_OPTIMIZER_PIPELINE_DEPTH", "2"),
                )
            ),
            optimizer_worker_count=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_OPTIMIZER_WORKER_COUNT",
                    os.getenv("FLAGSCALE_SLIDEFORMER_OPTIMIZER_WORKER_COUNT", "1"),
                )
            ),
            cpu_adam_per_layer_optimizer=os.getenv(
                "MEGATRON_SLIDEFORMER_CPU_ADAM_PER_LAYER_OPTIMIZER",
                os.getenv("FLAGSCALE_SLIDEFORMER_CPU_ADAM_PER_LAYER_OPTIMIZER", "0"),
            ).lower()
            not in {"0", "false", "no", "off"},
            shared_cpu_buffers=os.getenv(
                "MEGATRON_SLIDEFORMER_SHARED_CPU_BUFFERS",
                os.getenv("FLAGSCALE_SLIDEFORMER_SHARED_CPU_BUFFERS", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            cpu_grad_buffer_count=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT",
                    os.getenv("FLAGSCALE_SLIDEFORMER_CPU_GRAD_BUFFER_COUNT", "0"),
                )
            ),
            cpu_param_staging_buffer_count=int(
                os.getenv(
                    "MEGATRON_SLIDEFORMER_CPU_PARAM_STAGING_BUFFER_COUNT",
                    os.getenv("FLAGSCALE_SLIDEFORMER_CPU_PARAM_STAGING_BUFFER_COUNT", "1"),
                )
            ),
            te_fused_main_grad=os.getenv(
                "MEGATRON_SLIDEFORMER_TE_FUSED_MAIN_GRAD",
                os.getenv("FLAGSCALE_SLIDEFORMER_TE_FUSED_MAIN_GRAD", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            fp32_master_params=os.getenv(
                "MEGATRON_SLIDEFORMER_FP32_MASTER_PARAMS",
                os.getenv("FLAGSCALE_SLIDEFORMER_FP32_MASTER_PARAMS", "1"),
            ).lower()
            not in {"0", "false", "no", "off"},
            kernel_policy=os.getenv("MEGATRON_SLIDEFORMER_KERNEL_POLICY", "auto"),
            attention_backend=os.getenv("MEGATRON_SLIDEFORMER_ATTENTION_BACKEND", "auto"),
            mlp_backend=os.getenv("MEGATRON_SLIDEFORMER_MLP_BACKEND", "auto"),
            split_swiglu_threshold_gib=float(
                os.getenv("MEGATRON_SLIDEFORMER_SPLIT_SWIGLU_THRESHOLD_GIB", "0.5")
            ),
            loss_backend=os.getenv("MEGATRON_SLIDEFORMER_LOSS_BACKEND", "auto"),
            norm_backend=os.getenv("MEGATRON_SLIDEFORMER_NORM_BACKEND", "auto"),
            rope_backend=os.getenv("MEGATRON_SLIDEFORMER_ROPE_BACKEND", "auto"),
            strict_kernels=os.getenv("MEGATRON_SLIDEFORMER_STRICT_KERNELS", "1").lower()
            not in {"0", "false", "no", "off"},
            strict=os.getenv(
                "MEGATRON_SLIDEFORMER_STRICT", os.getenv("FLAGSCALE_SLIDEFORMER_STRICT", "1")
            ).lower()
            not in {"0", "false", "no", "off"},
        )

    def validate(
        self,
        *,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        data_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        num_microbatches: int = 1,
        recompute_granularity: str | None = None,
        recompute_method: str | None = None,
    ) -> None:
        if not self.enable:
            return
        if self.mode != "true_slideformer":
            raise ValueError("Megatron-LM-FL SlideFormer currently supports mode=true_slideformer")
        if tensor_model_parallel_size != 1:
            raise ValueError("Megatron-LM-FL SlideFormer currently supports TP=1 only")
        if pipeline_model_parallel_size != 1:
            raise ValueError("Megatron-LM-FL SlideFormer currently supports PP=1 only")
        if data_parallel_size != 1:
            raise ValueError("Megatron-LM-FL SlideFormer currently supports DP=1 only")
        if expert_model_parallel_size != 1:
            raise ValueError("Megatron-LM-FL SlideFormer currently supports EP=1 only")
        if num_microbatches != 1:
            raise ValueError(
                "Megatron-LM-FL SlideFormer currently supports num_microbatches=1 only"
            )
        if recompute_granularity is not None or recompute_method is not None:
            raise ValueError(
                "Megatron-LM-FL SlideFormer activation offload/checkpointing is incompatible "
                "with Megatron native recompute. Disable --recompute-* options."
            )
        if self.window_size < 1:
            raise ValueError("Megatron-LM-FL SlideFormer window_size must be at least 1")
        if self.activation_backend not in {"checkpoint", "slideformer-slot"}:
            raise ValueError(
                "SlideFormer activation_backend must be checkpoint or slideformer-slot"
            )
        if self.activation_backend == "slideformer-slot" and not self.activation_offload:
            raise ValueError("slideformer-slot activation backend requires activation_offload=true")
        if self.optimizer_pipeline_depth < 1 or self.optimizer_worker_count < 1:
            raise ValueError(
                "SlideFormer optimizer pipeline depth and worker count must be positive"
            )
        if self.cpu_grad_buffer_count < 0 or self.cpu_param_staging_buffer_count < 1:
            raise ValueError(
                "SlideFormer CPU grad buffer count must be non-negative and parameter "
                "staging buffer count must be positive"
            )
        if self.kernel_policy not in {"auto", "off"}:
            raise ValueError("SlideFormer kernel_policy must be auto or off")
        if self.attention_backend not in {"auto", "flash", "megatron"}:
            raise ValueError("SlideFormer attention_backend must be auto, flash, or megatron")
        if self.mlp_backend not in {"auto", "liger", "megatron"}:
            raise ValueError("SlideFormer mlp_backend must be auto, liger, or megatron")
        if self.split_swiglu_threshold_gib < 0:
            raise ValueError("SlideFormer split SwiGLU threshold must be non-negative")
        if self.loss_backend not in {"auto", "legacy", "liger", "megatron"}:
            raise ValueError("SlideFormer loss_backend must be auto, legacy, liger, or megatron")
        if self.norm_backend not in {"auto", "liger", "megatron"}:
            raise ValueError("SlideFormer norm_backend must be auto, liger, or megatron")
        if self.rope_backend not in {"auto", "flash", "megatron"}:
            raise ValueError("SlideFormer rope_backend must be auto, flash, or megatron")
