"""Compare Slide CPU gradients and updates with Megatron's native DDP path.

SLIDEFORMER_GPU_IDS=4,5,6,7 python -m torch.distributed.run --standalone \
  --nproc_per_node=4 tests/functional_tests/slideformer/test_native_gradient_parity.py --tp 2
"""
import argparse
import os
from pathlib import Path
import sys

visible = os.environ["SLIDEFORMER_GPU_IDS"].split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = visible[int(os.environ["LOCAL_RANK"])]
os.environ["LOCAL_RANK"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.plugin.slideformer import apply_true_megatron_slideformer, MegatronSlideFormerEngineConfig
from megatron.plugin.slideformer.layer_adam import LayerAdam

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--activation-backend", choices=["checkpoint", "slideformer-slot"], default="checkpoint")
parser.add_argument("--outstanding", action="store_true", help="Queue both forwards before backward, as in PP warmup")
parser.add_argument("--te-fused-main-grad", action="store_true", help="Exercise the leased GPU gradient pool")
args = parser.parse_args()
torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
ps.initialize_model_parallel(tensor_model_parallel_size=args.tp)
torch.manual_seed(1234)
model_parallel_cuda_manual_seed(1234)
config = TransformerConfig(
    num_layers=2, hidden_size=128, ffn_hidden_size=384, num_attention_heads=4,
    num_query_groups=2, kv_channels=32, params_dtype=torch.bfloat16, bf16=True,
    tensor_model_parallel_size=args.tp, sequence_parallel=args.tp > 1,
    use_cpu_initialization=True, gradient_accumulation_fusion=False,
    normalization="RMSNorm", qk_layernorm=True, add_bias_linear=False,
    gated_linear_unit=True, activation_func=torch.nn.functional.silu,
    attention_dropout=0, hidden_dropout=0, calculate_per_token_loss=True,
)


def build():
    return GPTModel(
        config=config, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(qk_layernorm=True),
        vocab_size=1024, max_sequence_length=128, position_embedding_type="rope",
        parallel_output=True, share_embeddings_and_output_weights=False,
        post_process=not args.outstanding,
    )


reference = build().cuda()
model = build()
model.load_state_dict(reference.state_dict())
reference = DistributedDataParallel(
    config, DistributedDataParallelConfig(grad_reduce_in_fp32=True, overlap_grad_reduce=False), reference,
)
params = dict(reference.module.named_parameters())
masters = {name: torch.nn.Parameter(param.detach().float().cpu().clone()) for name, param in params.items()}
# Use the same CPU Adam kernel to isolate gradient transport and scaling from
# differences in torch/AVX optimizer arithmetic.
optimizer = LayerAdam(lr=2e-4, weight_decay=0)
optimizer.add_layer_params(0, masters.values())
engine = apply_true_megatron_slideformer(
    model, device=torch.device("cuda:0"), config=MegatronSlideFormerEngineConfig(
        lr=2e-4, weight_decay=0, activation_offload=True, offload_after_forward=True,
        prefetch=True, defer_optimizer_step=True, qkv_layout_backend="megatron",
        activation_backend=args.activation_backend,
        activation_slot_prefetch=args.activation_backend == "slideformer-slot",
        te_fused_main_grad=args.te_fused_main_grad,
    ),
)
owners = {id(param): owner for owner in engine.managed_layers for param in owner.params}
try:
    for step in range(args.steps):
        reference.zero_grad_buffer()
        model.zero_grad(set_to_none=True)
        engine.zero_unmanaged_grads(num_microbatches=2)
        pending = []
        for micro in range(2):
            generator = torch.Generator().manual_seed(4000 + step * 10 + micro + ps.get_data_parallel_rank() * 100)
            tokens = torch.randint(0, 1024, (1, 128), generator=generator).cuda()
            labels = torch.randint(0, 1024, (1, 128), generator=generator).cuda()
            positions = torch.arange(128, device="cuda").view(1, -1)
            expected = reference(tokens, positions, None, labels=labels)
            actual = model(tokens, positions, None, labels=labels)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.005)
            if args.outstanding:
                pending.append((expected, actual))
            else:
                expected.float().sum().backward()
                actual.float().sum().backward()
        for expected, actual in pending:
            expected.float().sum().backward()
            actual.float().sum().backward()
        finalize_model_grads([reference], num_tokens=torch.tensor(256, device="cuda"))
        engine.finalize_data_parallel_gradients(
            group=ps.get_data_parallel_group(with_context_parallel=True),
            num_tokens=torch.tensor(256, device="cuda"), tp_group=ps.get_tensor_model_parallel_group(),
        )
        max_grad_error = 0.0
        for name, param in model.named_parameters():
            actual = engine._accumulated_grads[owners[id(param)].layer_idx][param]
            expected = params[name].main_grad.detach().cpu()
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            max_grad_error = max(max_grad_error, (actual - expected).abs().max().item())
            masters[name].grad = expected
        engine.step_unmanaged_params()
        optimizer.step_with_grad_views(0, {p: p.grad for p in masters.values()}, {p: p for p in masters.values()})
        max_param_error = 0.0
        for name, param in model.named_parameters():
            actual = owners[id(param)].cpu_params[param]
            torch.testing.assert_close(actual, masters[name].detach(), atol=1e-7, rtol=1e-6)
            max_param_error = max(max_param_error, (actual - masters[name].detach()).abs().max().item())
            params[name].data.copy_(masters[name].detach().to(device="cuda", dtype=params[name].dtype))
        print(f"NATIVE_PARITY rank={dist.get_rank()} tp={args.tp} step={step + 1} "
              f"grad_error={max_grad_error} param_error={max_param_error}", flush=True)
finally:
    engine.close()
    ps.destroy_model_parallel()
    dist.destroy_process_group()
