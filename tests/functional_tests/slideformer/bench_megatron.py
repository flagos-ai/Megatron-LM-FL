import json
import os
import sys
import time
from pathlib import Path

started = time.time()
from bench_runtime import configure_worker

configure_worker()
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import torch
import pretrain_gpt as pg
import megatron.training.training as training
from megatron.training import get_args, set_startup_timestamps
from mbridge import AutoBridge
from bench_common import SEQ, MBS, STEPS, sample, log_input, start_step, end_step, memory

original_apply = training._apply_slideformer_if_enabled


def audited_apply(model):
    engines = original_apply(model)
    for engine in engines or []:
        opt = engine.layer_optimizer
        sh = engine._sharded_opt
        nbytes = lambda tensors: sum(t.numel() * t.element_size() for t in tensors)
        print(
            "REVIEW_OPT_MEMORY "
            + json.dumps(
                {
                    "rank": torch.distributed.get_rank(),
                    "active_optimizer_moments_bytes": nbytes(
                        list(opt.exp_avg_flat.values()) + list(opt.exp_avg_sq_flat.values())
                    ),
                    "shard_alias_moments_bytes": (
                        nbytes(list(sh.exp_avg.values()) + list(sh.exp_avg_sq.values()))
                        if sh
                        else 0
                    ),
                }
            ),
            flush=True,
        )
    return engines


training._apply_slideformer_if_enabled = audited_apply


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        tokens = sample(idx)
        labels = torch.roll(tokens, -1)
        labels[-1] = -100
        mask = torch.ones(SEQ)
        mask[-1] = 0
        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": mask,
            "position_ids": torch.arange(SEQ),
        }


def datasets(counts):
    return Dataset(counts[0]), None, None


datasets.is_distributed = True
first = True


def batch(data_iterator, vp_stage=None):
    global first
    if get_args().tensor_model_parallel_size > 1:
        from megatron.training.utils import get_batch_on_this_tp_rank

        b = get_batch_on_this_tp_rank(data_iterator)
    else:
        b = next(data_iterator)
    if first:
        log_input(b["tokens"])
        first = False
    b = {k: v.cuda(non_blocking=True) if v is not None else None for k, v in b.items()}
    return b["tokens"], b["labels"], b["loss_mask"], None, b["position_ids"], None


pg.get_batch = batch


def provider(pre_process=True, post_process=True, **kwargs):
    args = get_args()
    bridge = AutoBridge.from_pretrained(os.environ["SLIDEFORMER_REAL_MODEL"])
    bridge.set_extra_args(
        gradient_accumulation_fusion=False,
        calculate_per_token_loss=True,
        sequence_parallel=args.sequence_parallel,
        attention_backend="flash",
        deterministic_mode=False,
        bias_activation_fusion=args.bias_swiglu_fusion,
        use_te_activation_func=args.use_te_activation_func,
        apply_rope_fusion=args.apply_rope_fusion,
    )
    model = bridge._model_provider([])(pre_process, post_process)
    if torch.distributed.get_rank() == 0:
        print(
            "REVIEW_MODEL_CONFIG "
            + json.dumps(
                {
                    key: getattr(model.config, key)
                    for key in (
                        "bias_activation_fusion",
                        "use_te_activation_func",
                        "apply_rope_fusion",
                        "gradient_accumulation_fusion",
                        "sequence_parallel",
                        "tensor_model_parallel_size",
                        "calculate_per_token_loss",
                    )
                }
            ),
            flush=True,
        )
    mapping = bridge._weight_name_mapping_mcore_local_to_global(model)
    assert set(dict(model.named_parameters())) <= {k for k in mapping if "_extra_state" not in k}
    with torch.no_grad():
        bridge.load_weights([model], os.environ["SLIDEFORMER_REAL_MODEL"], memory_efficient=True)
    return model


orig = training.train_step
step = 0


def timed(*args, **kwargs):
    global step
    profiler = None
    profile_dir = os.getenv("SLIDE_BENCH_PROFILE_DIR")
    if profile_dir and step == 4 and torch.distributed.get_rank() == 0:
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        )
        profiler.start()
    if step == 0:
        torch.cuda.reset_peak_memory_stats()
    t = start_step()
    result = orig(*args, **kwargs)
    for chunk in args[2]:
        engine = getattr(chunk, "_megatron_slideformer_engine", None)
        if engine:
            engine.wait_for_completion()
    step += 1
    end_step(t, step)
    if profiler is not None:
        profiler.stop()
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(Path(profile_dir) / "megatron_rank0_step5.json"))
    if step == STEPS:
        memory()
        for chunk in args[2]:
            engine = getattr(chunk, "_megatron_slideformer_engine", None)
            if engine:
                print("REVIEW_PIPELINE " + json.dumps({
                    "rank": torch.distributed.get_rank(),
                    "traffic": engine.traffic_counters,
                    "overlap": engine.config.overlap_config_summary(),
                }), flush=True)
    return result


training.train_step = timed
set_startup_timestamps(program_start=started, main_entry=time.time())
try:
    pg.pretrain(
        datasets,
        provider,
        pg.ModelType.encoder_or_decoder,
        pg.forward_step,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        get_embedding_ranks=pg.get_embedding_ranks,
    )
except BaseException:
    import traceback

    traceback.print_exc()
    raise
finally:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
