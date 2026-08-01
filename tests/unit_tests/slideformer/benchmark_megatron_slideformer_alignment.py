from __future__ import annotations

import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.distributed as dist

from megatron.core import parallel_state as mpu
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.slideformer import (
    MegatronSlideFormerEngineConfig,
    apply_true_megatron_slideformer,
)


def _build_model(args: argparse.Namespace) -> GPTModel:
    params_dtype = torch.bfloat16 if args.bf16 else torch.float32
    transformer_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=args.bf16,
        fp16=False,
        params_dtype=params_dtype,
        pipeline_dtype=params_dtype,
        attention_backend=getattr(AttnBackend, args.attention_backend),
        recompute_granularity=args.recompute_granularity,
    )
    model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=args.vocab_size,
        max_sequence_length=args.seq_len,
    ).cuda()
    model.eval()
    return model


def _execution_model(model: GPTModel, args: argparse.Namespace) -> torch.nn.Module:
    if not args.bf16:
        return model
    return Float16Module(model.config, model)


def _run_baseline(
    model: GPTModel,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    exec_model = _execution_model(model, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    losses: list[float] = []
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = exec_model(tokens, position_ids, attention_mask).float().pow(2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = exec_model(tokens, position_ids, attention_mask).float().pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize()
    return {
        "step_time_s": (time.perf_counter() - start) / args.steps,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "losses": losses,
    }


def _run_slideformer(
    model: GPTModel,
    tokens: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, object]:
    exec_model = _execution_model(model, args)
    engine = apply_true_megatron_slideformer(
        model,
        config=MegatronSlideFormerEngineConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            activation_offload=args.activation_offload,
            offload_after_forward=args.activation_offload,
            prefetch=True,
        ),
    )
    losses: list[float] = []
    try:
        for _ in range(args.warmup_steps):
            engine.zero_unmanaged_grads()
            loss = exec_model(tokens, position_ids, attention_mask).float().pow(2).mean()
            loss.backward()
            engine.step_unmanaged_params()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            engine.zero_unmanaged_grads()
            loss = exec_model(tokens, position_ids, attention_mask).float().pow(2).mean()
            loss.backward()
            engine.step_unmanaged_params()
            losses.append(float(loss.detach().cpu()))
        torch.cuda.synchronize()
        return {
            "step_time_s": (time.perf_counter() - start) / args.steps,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "losses": losses,
        }
    finally:
        engine.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--activation-offload", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "flash", "fused", "unfused", "local"],
        default="auto",
    )
    parser.add_argument("--recompute-granularity", choices=["full", "selective"], default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        torch.cuda.set_device(0)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl",
            init_method=f"file://{Path(tmpdir) / 'rdzv'}",
            rank=0,
            world_size=1,
        )
        mpu.initialize_model_parallel()
        model_parallel_cuda_manual_seed(2026)
        try:
            baseline = _build_model(args)
            initial_state = {
                name: tensor.detach().cpu().clone() if torch.is_tensor(tensor) else tensor
                for name, tensor in baseline.state_dict().items()
            }
            tokens = torch.randint(
                0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda"
            )
            position_ids = (
                torch.arange(args.seq_len, device="cuda").unsqueeze(0).expand(args.batch_size, -1)
            )
            attention_mask = torch.ones(
                (args.batch_size, 1, args.seq_len, args.seq_len), dtype=torch.bool, device="cuda"
            )
            baseline_result = _run_baseline(baseline, tokens, position_ids, attention_mask, args)

            del baseline
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            slideformer = _build_model(args)
            slideformer.load_state_dict(initial_state)
            slideformer_result = _run_slideformer(
                slideformer, tokens, position_ids, attention_mask, args
            )
            loss_diffs = [
                abs(float(a) - float(b))
                for a, b in zip(
                    baseline_result["losses"], slideformer_result["losses"], strict=True
                )
            ]
            result = {
                "config": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "baseline": baseline_result,
                "slideformer": slideformer_result,
                "max_loss_diff": max(loss_diffs) if loss_diffs else 0.0,
                "speed_ratio_slideformer_over_baseline": baseline_result["step_time_s"]
                / slideformer_result["step_time_s"],
                "memory_ratio_slideformer_over_baseline": slideformer_result["peak_memory_gb"]
                / baseline_result["peak_memory_gb"],
            }
            text = json.dumps(result, indent=2)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text + "\n")
            print(text)
        finally:
            mpu.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
