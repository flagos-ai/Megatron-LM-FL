"""Full installed-Session DP transport comparison with native Megatron DDP.

Uses the same CPU Adam kernel, comparing every consumed gradient shard,
all FP32 masters and both moment shards, across all trainable owners.
Launch through the delivery guard and rank entry wrapper; GAS=1, B>=16.
"""
import argparse
import faulthandler
import inspect
import json
import os
from pathlib import Path
import sys

if os.environ.get("SSDP_DIAGNOSTIC_STACK_SECONDS"):
    faulthandler.dump_traceback_later(int(os.environ["SSDP_DIAGNOSTIC_STACK_SECONDS"]), repeat=True)

visible = os.environ["SLIDEFORMER_GPU_IDS"].split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = visible[int(os.environ["LOCAL_RANK"])]
os.environ["LOCAL_RANK"] = "0"

import torch
# Publish each admitted rank's CUDA context before slow model-library imports,
# so other shared-host launchers can see that this device is now in use.
# The outer guard still rejects any foreign context that appears afterward.
torch.cuda.set_device(0)
device_probe = torch.empty(1, dtype=torch.uint8, device="cuda:0")
del device_probe
print(f"VALIDATION_DEVICE_READY rank={os.environ.get('RANK')} "
      f"visible={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
import torch.distributed as dist
from megatron.core import parallel_state as ps
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from slideformer_megatron import FullFineTuneSession, MegatronSlideFormerEngineConfig, DataParallelTransport
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_batch_on_this_cp_rank
from slideformer_megatron.layer_adam import LayerAdam

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--gradient-reduce", choices=("hsr", "gpu_reduce"), required=True)
parser.add_argument("--parameter-transfer", choices=("full", "shard"), required=True)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--cp", type=int, default=1)
parser.add_argument("--ep", type=int, default=1)
parser.add_argument("--moe", action="store_true")
parser.add_argument("--steps", type=int, default=3)
parser.add_argument("--layers", type=int, default=2, help="Use >=4 to exercise GPU slot reuse within one backward")
parser.add_argument("--microbatches", type=int, default=2)
parser.add_argument("--batch", type=int, default=1, help="Per-rank batch size")
parser.add_argument("--activation-backend", choices=["checkpoint", "slideformer-slot"], default="checkpoint")
parser.add_argument("--outstanding", action="store_true", help="Queue both forwards before backward, as in PP warmup")
parser.add_argument("--te-fused-main-grad", action="store_true", help="Exercise the leased GPU gradient pool")
parser.add_argument("--projection-layout", action="store_true",
                    help="Compare adaptive TP1 projection storage with native sequence-major FA2")
parser.add_argument("--precompute-tokens", action="store_true", help="Supply the actual step token count before backward for layerwise Adam")
args = parser.parse_args()
if args.microbatches != 1 or args.tp != 1 or args.cp != 1 or args.ep != 1 or args.moe or args.outstanding:
    parser.error("This transport probe requires dense DP and one batch per update")
args.precompute_tokens = True
if args.microbatches < 1 or args.batch < 1:
    parser.error("Positive batch and microbatches required")
local_tokens = args.microbatches * args.batch * 128 // args.cp
torch.cuda.set_device(0)
dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
ps.initialize_model_parallel(tensor_model_parallel_size=args.tp, context_parallel_size=args.cp,
                             expert_model_parallel_size=args.ep, expert_tensor_parallel_size=args.tp)
groups = ProcessGroupCollection.use_mpu_process_groups()
torch.manual_seed(1234)
model_parallel_cuda_manual_seed(1234)
config = TransformerConfig(
    num_layers=args.layers, hidden_size=128, ffn_hidden_size=384, num_attention_heads=4,
    num_query_groups=2, kv_channels=32, params_dtype=torch.bfloat16, bf16=True,
    tensor_model_parallel_size=args.tp, sequence_parallel=args.tp > 1,
    context_parallel_size=args.cp, expert_model_parallel_size=args.ep, expert_tensor_parallel_size=args.tp,
    num_moe_experts=4 if args.moe or args.ep > 1 else None,
    moe_grouped_gemm=args.moe or args.ep > 1, moe_router_topk=2,
    moe_token_dispatcher_type="alltoall", moe_aux_loss_coeff=0.01 if args.moe or args.ep > 1 else 0,
    use_cpu_initialization=True, gradient_accumulation_fusion=False,
    normalization="RMSNorm", qk_layernorm=True, add_bias_linear=False,
    gated_linear_unit=True, activation_func=torch.nn.functional.silu,
    attention_dropout=0, hidden_dropout=0, calculate_per_token_loss=True,
)


def build():
    return GPTModel(
        config=config, transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            qk_layernorm=True, num_experts=config.num_moe_experts, moe_grouped_gemm=config.moe_grouped_gemm),
        vocab_size=1024, max_sequence_length=128, position_embedding_type="rope",
        parallel_output=True, share_embeddings_and_output_weights=False,
        post_process=not args.outstanding,
    )


reference = build().cuda()
model = build()
model.load_state_dict(reference.state_dict())
if args.projection_layout:
    from slideformer_megatron.kernels import FlashAttentionCore, apply_tp1_batch_major_attention, restore_tp1_batch_major_attention
    for index, layer in enumerate(reference.decoder.layers):
        previous = layer.self_attention.core_attention
        layer.self_attention.core_attention = FlashAttentionCore(
            config, index + 1, previous.attn_mask_type).cuda()
    # Session intentionally preserves its caller's native operators. Exercise
    # the explicit external layout adapter without widening that API contract.
    assert apply_tp1_batch_major_attention(model) == args.layers
reduce_precision = os.getenv("MEGATRON_SLIDEFORMER_GPU_REDUCE_DTYPE", "bf16")
use_bf16 = args.gradient_reduce == "gpu_reduce" and reduce_precision == "bf16"
reference = DistributedDataParallel(
    config, DistributedDataParallelConfig(grad_reduce_in_fp32=not use_bf16, overlap_grad_reduce=False), reference,
)
params = dict(reference.module.named_parameters())
masters = {name: torch.nn.Parameter(param.detach().float().cpu().clone()) for name, param in params.items()}
# Use the same CPU Adam kernel to isolate gradient transport and scaling from
# differences in torch/AVX optimizer arithmetic.
optimizer = LayerAdam(lr=2e-4, weight_decay=0)
optimizer.add_layer_params(0, masters.values())
session = FullFineTuneSession(
    model, groups=groups, device=torch.device("cuda:0"),
    transport=DataParallelTransport(args.gradient_reduce, args.parameter_transfer), config=MegatronSlideFormerEngineConfig(
        lr=2e-4, weight_decay=0, activation_offload=True, offload_after_forward=True,
        gpu_reduce_dtype=reduce_precision,
        prefetch=True, defer_optimizer_step=True,
        qkv_layout_backend="megatron",
        activation_backend=args.activation_backend,
        activation_slot_prefetch=args.activation_backend == "slideformer-slot",
        te_fused_main_grad=args.te_fused_main_grad,
    ),
)
engine = session.engine
projection_inputs, projection_gradients = [], []
if args.projection_layout:
    for index, layer in enumerate(engine.layout.layers):
        projection = layer.self_attention.linear_proj
        original = projection._slideformer_tp1_projection_original_forward
        def observe(inputs, *values, _original=original, _index=index, **kwargs):
            projection_inputs.append(dict(layer=_index, shape=list(inputs.shape),
                                          stride=list(inputs.stride()), contiguous=inputs.is_contiguous()))
            result = _original(inputs, *values, **kwargs)
            if result[0].requires_grad:
                result[0].register_hook(lambda gradient: projection_gradients.append(gradient.is_contiguous()))
            return result
        projection._slideformer_tp1_projection_original_forward = observe
sharded = engine._sharded_opt
captured = {}
original_adam = sharded.adam_owner
# Qualification-only interception at the actual CPU Adam consumption boundary.
# This covers the reduced/scaled shard, including the asynchronous GPU path.
def capture_adam(owner, grad_slice=None):
    lo, hi = sharded.bounds[owner.layer_idx][sharded.rank]
    grad = grad_slice if grad_slice is not None else sharded.acc[owner.layer_idx][sharded.rank][lo:hi]
    captured[owner.layer_idx] = grad.detach().clone()
    original_adam(owner, grad_slice)
sharded.adam_owner = capture_adam
names = {id(p): n for n, p in model.named_parameters()}
print("PARALLEL_GROUPS " + json.dumps({name: dist.get_process_group_ranks(getattr(groups, name))
      for name in ("dp_cp", "tp", "cp", "ep", "expt_dp", "expt_tp")}), flush=True)
if config.num_moe_experts:
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device="cuda"))
owners = {id(param): owner for owner in engine.managed_layers for param in owner.params}
try:
    for step in range(args.steps):
        projection_inputs.clear()
        projection_gradients.clear()
        reference.zero_grad_buffer()
        if args.precompute_tokens:
            session.begin_step(num_microbatches=args.microbatches, num_tokens=torch.tensor(local_tokens, device="cuda"))
        else:
            session.begin_step(num_microbatches=args.microbatches)
        pending = []
        for micro in range(args.microbatches):
            generator = torch.Generator().manual_seed(4000 + step * 10 + micro + ps.get_data_parallel_rank() * 100)
            tokens = torch.randint(0, 1024, (args.batch, 128), generator=generator).cuda()
            labels = torch.randint(0, 1024, (args.batch, 128), generator=generator).cuda()
            positions = torch.arange(128, device="cuda").view(1, -1).expand(args.batch, -1).contiguous()
            batch = dict(tokens=tokens, labels=labels, position_ids=positions, attention_mask=None)
            if args.cp > 1:
                kwargs = {"cp_group": groups.cp}
                if "dynamic_cp" in inspect.signature(get_batch_on_this_cp_rank).parameters:
                    batch["cu_seqlens"] = None
                    kwargs["dynamic_cp"] = False
                batch = get_batch_on_this_cp_rank(batch, **kwargs)
            tokens, labels, positions = batch["tokens"], batch["labels"], batch["position_ids"]
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
        # Drain the candidate's background collectives before the independent
        # reference issues NCCL finalization on its own groups. Interleaving
        # two training engines here creates a wait cycle absent from the CLI.
        session.finalize_gradients(num_tokens=torch.tensor(local_tokens, device="cuda"))
        finalize_model_grads([reference], num_tokens=torch.tensor(local_tokens, device="cuda"))
        max_grad_error = 0.0
        assert set(captured) == {o.layer_idx for o in sharded.owners}
        covered = 0
        for owner in sharded.owners:
            lo, hi = sharded.bounds[owner.layer_idx][sharded.rank]
            expected = torch.cat([params[names[id(p)]].main_grad.detach().cpu().view(-1) for p in owner.params])[lo:hi]
            actual = captured[owner.layer_idx]
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            if actual.numel():
                max_grad_error = max(max_grad_error, (actual - expected).abs().max().item())
            covered += actual.numel()
        consumed_dtypes = sorted({str(t.dtype) for t in captured.values()})
        assert consumed_dtypes == (["torch.bfloat16"] if use_bf16 else ["torch.float32"])
        captured.clear()
        total_covered = torch.tensor(covered, device="cuda", dtype=torch.int64)
        dist.all_reduce(total_covered, group=groups.dp_cp)
        assert total_covered.item() == sum(p.numel() for p in model.parameters())
        for name, param in params.items():
            masters[name].grad = param.main_grad.detach().float().cpu()
        session.step()
        if getattr(engine, "layer_update_pipeline", None) is not None:
            if os.getenv("MEGATRON_SLIDEFORMER_ASYNC_LAYER_CAPTURE") == "1" and args.microbatches == 1 and args.tp == args.cp == args.ep == 1:
                assert engine.layer_update_pipeline.stats["async_gradient_captures"] == len(engine.layer_update_pipeline.owners)
                assert not engine.layer_update_pipeline.leases
            print("LAYER_UPDATE_STATS " + json.dumps(dict(rank=dist.get_rank(), step=step + 1,
                  **engine.layer_update_pipeline.stats)), flush=True)
        optimizer.step_with_grad_views(0, {p: p.grad for p in masters.values()}, {p: p for p in masters.values()})
        max_param_error = 0.0
        for name, param in model.named_parameters():
            actual = owners[id(param)].cpu_params[param]
            torch.testing.assert_close(actual, masters[name].detach(), atol=1e-7, rtol=1e-6)
            max_param_error = max(max_param_error, (actual - masters[name].detach()).abs().max().item())
            params[name].data.copy_(masters[name].detach().to(device="cuda", dtype=params[name].dtype))
        max_moment_error = 0.0
        for owner in sharded.owners:
            lo, hi = sharded.bounds[owner.layer_idx][sharded.rank]
            assert owner.step == step + 1
            for key, actual in (("exp_avg", sharded.exp_avg[owner.layer_idx]),
                                ("exp_avg_sq", sharded.exp_avg_sq[owner.layer_idx])):
                expected = torch.cat([optimizer.state[masters[names[id(p)]]][key].view(-1) for p in owner.params])[lo:hi]
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)
                if actual.numel():
                    max_moment_error = max(max_moment_error, (actual - expected).abs().max().item())
        print("SESSION_TRANSPORT_PARITY " + json.dumps(dict(
            rank=dist.get_rank(), step=step+1, gradient_reduce=args.gradient_reduce,
            parameter_transfer=args.parameter_transfer, gpu_reduce_dtype=reduce_precision,
            consumed_grad_dtypes=consumed_dtypes,
            grad_error=max_grad_error,
            param_error=max_param_error, moment_error=max_moment_error,
            global_covered_elements=total_covered.item(), owners=len(sharded.owners))), flush=True)
        if args.projection_layout:
            assert len(projection_inputs) >= args.layers and all(item["contiguous"] for item in projection_inputs)
            assert len(projection_gradients) >= args.layers and all(projection_gradients)
            print("PROJECTION_LAYOUT_AUDIT " + json.dumps(dict(rank=dist.get_rank(), step=step + 1,
                inputs=projection_inputs, gradient_count=len(projection_gradients),
                all_input_and_output_gradients_contiguous=True)), flush=True)

finally:
    session.close()
    if args.projection_layout:
        assert restore_tp1_batch_major_attention(model) == args.layers
    ps.destroy_model_parallel()
    dist.destroy_process_group()
