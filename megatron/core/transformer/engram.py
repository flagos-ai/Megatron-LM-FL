# Copyright (c) 2026, FlagScale CORPORATION. All rights reserved.

## built-in
import copy
from typing import Optional, Callable, Tuple
import math

## third-party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sympy import isprime
from transformers import AutoTokenizer

from tokenizers import Regex, normalizers

# megatron-core
from megatron.core import tensor_parallel
from megatron.core.utils import (
    get_pg_size,
    get_pg_rank,
    get_tensor_model_parallel_group_if_none,
)
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_cpu
from megatron.core import parallel_state
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.utils import nvtx_range_push, nvtx_range_pop
from megatron.core.transformer.utils import sharded_state_dict_default
from megatron.core.transformer.transformer_config import TransformerConfig


def _vocab_size_with_padding(orig_vocab_size, tp_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = tp_size
    after = int(math.ceil(after / multiple) * multiple)
    return after


def _initialize_engram_weight_gpu_with_seed(
    weight, init_method, local_init_seed, partition_dim=0, stride=1
):
    tensor_parallel.set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )
    with torch.random.fork_rng(devices=[weight.device]):
        torch.manual_seed(local_init_seed)
        init_method(weight)


class EngramMemory(nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default values are kept.

    Unlike to the MCore VocabParallelEmbedding, the embedding parallel use parallelism like expert parallel.
    The parallel group is the subset of data parallel, which is given as the engram_model_parallel_size.
    Input of each rank is different, when forwarding, the input will be transmit to other rank using an All2All operator.

    Args:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.

    Keyword Args:
        init_method: A Callable.
        config: A DeepSeekConfig object.
        embedding_parallel_group: vocab parallel group, a torch.distributed.ProcessGroup object.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        reduce_scatter_embeddings: bool = False,
        config: ModelParallelConfig,
        embedding_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.reduce_scatter_embeddings = reduce_scatter_embeddings
        self.embedding_parallel_group = embedding_parallel_group
        if self.embedding_parallel_group is None:
            self.embedding_parallel_size = 1
            self.embedding_parallel_rank = 0
        else:
            self.embedding_parallel_size = get_pg_size(self.embedding_parallel_group)
            self.embedding_parallel_rank = get_pg_rank(self.embedding_parallel_group)

        (self.vocab_start_index, self.vocab_end_index) = (
            VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                self.embedding_parallel_rank,
                self.embedding_parallel_size,
            )
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )
        self.deterministic_mode = config.deterministic_mode

        # Allocate weights and initialize on GPU only.
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                    rank=self.embedding_parallel_rank,
                    world_size=self.embedding_parallel_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                engram_seed = int(getattr(config, "engram_seed", 0))
                pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                local_init_seed = (
                    2718
                    + engram_seed
                    + pp_rank * 100
                    + int(self.embedding_parallel_rank)
                )
                _initialize_engram_weight_gpu_with_seed(
                    self.weight, init_method, local_init_seed, partition_dim=0, stride=1
                )

    def enable_parallel(self):
        if self.embedding_parallel_size > 1:
            setattr(self.weight, "is_engram_embedding", True)
            setattr(self.weight, "allreduce", False)

    def enable_offloading(self):
        setattr(self.weight, "is_offloading_candidate", True)

    def _dispatch(self, input_ids):
        nvtx_range_push("engram::embedding_dispatch")
        self.hidden_shape = input_ids.shape
        input_ids = input_ids.view(-1)
        routing_map = input_ids // self.num_embeddings_per_partition
        # [num_partitions], number of tokens assigned to each partition from the current rank's input.
        num_tokens_per_partition = torch.bincount(
            routing_map,
            minlength=self.embedding_parallel_size,
        ).to(dtype=torch.int64)
        # Reorder the token indices to match the order of partitions.
        # Shape = (batch * seqlen, ).
        token_indices_partitions_sorted = torch.argsort(routing_map, stable=True)
        # Shape = (batch * seqlen, ).
        routed_input = input_ids[token_indices_partitions_sorted]
        # Use to unsort.
        self._token_unsort_indices = torch.empty_like(token_indices_partitions_sorted)
        self._token_unsort_indices[token_indices_partitions_sorted] = torch.arange(
            token_indices_partitions_sorted.size(0),
            device=token_indices_partitions_sorted.device,
        )
        # generate the input splits and output splits for all-to-all
        with torch.no_grad():
            output_splits_cuda = tensor_parallel.all_to_all(
                self.embedding_parallel_group,
                num_tokens_per_partition,
                None,
                None,
            )
            # Need to wait explicitly because it is used by a triton kernel later
            # which doesn't realize that AsyncCollectiveTensor needs unwrapping
            output_splits_cuda = torch.ops._c10d_functional.wait_tensor(
                output_splits_cuda
            )
            input_splits = (
                num_tokens_per_partition.view(self.embedding_parallel_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=True)
            )
            # NOTE: this would incur a device-to-host sync
            output_splits = (
                output_splits_cuda.view(self.embedding_parallel_size, -1)
                .sum(dim=1)
                .to(torch.device("cpu"), non_blocking=False)
            )
            self.input_splits = input_splits.tolist()
            self.output_splits = output_splits.tolist()

        # perform all-to-all
        routed_input = tensor_parallel.all_to_all(
            self.embedding_parallel_group,
            routed_input,
            self.output_splits,
            self.input_splits,
        )
        routed_input = routed_input - self.vocab_start_index
        nvtx_range_pop()
        return routed_input

    def _combine(self, hidden_states: torch.Tensor):
        nvtx_range_push("engram::embedding_combine")
        routed_hidden_states = tensor_parallel.all_to_all(
            self.embedding_parallel_group,
            hidden_states,
            self.input_splits,
            self.output_splits,
        )
        routed_hidden_states = routed_hidden_states[self._token_unsort_indices]
        hidden_states = routed_hidden_states.view(*self.hidden_shape, -1)
        nvtx_range_pop()
        return hidden_states

    def forward(self, input_: torch.Tensor):
        """Forward.

        Args:
            input_ (torch.Tensor): Input tensor, shape (b, s), dtype = torch.int64.
        """
        nvtx_range_push("engram::embedding_forward")
        if self.reduce_scatter_embeddings:
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            num_tokens_per_sp_rank = input_.shape[1] // tp_size
            if tp_rank < tp_size - 1:
                input_ = input_[
                    :,
                    tp_rank * num_tokens_per_sp_rank : (tp_rank + 1)
                    * num_tokens_per_sp_rank,
                ]
            else:
                input_ = input_[:, tp_rank * num_tokens_per_sp_rank :]
            input_ = input_.contiguous()
        if self.embedding_parallel_size > 1:
            input_ = self._dispatch(input_)
        # Get the embeddings.
        if self.deterministic_mode:
            output = self.weight[input_]
        else:
            # F.embedding currently has a non-deterministic backward function
            output = F.embedding(input_, self.weight)
        # Get the complete output embedding
        if self.embedding_parallel_size > 1:
            output = self._combine(output)
        if self.reduce_scatter_embeddings:
            output = output.transpose(0, 1).contiguous()
        nvtx_range_pop()
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int]] = (),
        metadata: Optional[dict] = None,
        **kwargs,
    ):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        weight_prefix = f"{prefix}weight"
        prepend_axis_num = len(sharded_offsets)
        new_offsets = []
        tp_rank = self.embedding_parallel_rank
        tp_size = self.embedding_parallel_size
        dp_replica_id = get_pg_rank(parallel_state.get_engram_data_parallel_group())
        new_offsets.append((prepend_axis_num, tp_rank, tp_size))

        replica_id = (0, 0, dp_replica_id)
        sharded_tensor = ShardedTensor.from_rank_offsets(
            weight_prefix,
            state_dict["weight"],
            *sharded_offsets,
            *new_offsets,
            replica_id=replica_id,
            prepend_axis_num=prepend_axis_num,
            allow_shape_mismatch=True,
            **kwargs,
        )
        return {weight_prefix: sharded_tensor}


class MultiHeadEmbedding(nn.Module):
    def __init__(self, config, list_of_N: list[int], D: int):
        super().__init__()
        self.config = config
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)

        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)

        # embeddings (parallel).
        if self.config.engram_embedding_parallel_method == "allreduce":
            self.tp_group = get_tensor_model_parallel_group_if_none(tp_group=None)
            self.reduce_scatter_embeddings = self.config.sequence_parallel

            padded_total_N = _vocab_size_with_padding(
                total_N, get_pg_size(self.tp_group)
            )
            print(
                f"Engram multi-head embedding: pad total_n from {total_N} to {padded_total_N}"
            )

            self.memory = tensor_parallel.VocabParallelEmbedding(
                num_embeddings=padded_total_N,
                embedding_dim=D,
                init_method=self.config.embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=self.config,
                tp_group=self.tp_group,
            )
        else:
            self.embedding_parallel_group = (
                parallel_state.get_engram_embedding_parallel_group()
            )
            self.reduce_scatter_embeddings = self.config.sequence_parallel
            padded_total_N = _vocab_size_with_padding(
                total_N, get_pg_size(self.embedding_parallel_group)
            )
            print(
                f"Engram multi-head embedding: pad total_n from {total_N} to {padded_total_N}"
            )
            self.memory = EngramMemory(
                num_embeddings=padded_total_N,
                embedding_dim=D,
                init_method=self.config.embedding_init_method,
                reduce_scatter_embeddings=self.reduce_scatter_embeddings,
                config=self.config,
                embedding_parallel_group=self.embedding_parallel_group,
            )
            if self.config.engram_embedding_parallel_method == "alltoall":
                self.memory.enable_parallel()
                if self.config.engram_offload_embedding_optimizer_states:
                    self.memory.enable_offloading()
            else:
                raise ValueError(
                    f"Unsupported engram_embedding_parallel_method: {self.config.engram_embedding_parallel_method}"
                )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        output = self.memory(shifted_input_ids)

        if not self.reduce_scatter_embeddings:
            output = output.transpose(0, 1).contiguous()
        return output

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ):
        sharded_dict = {}
        memory_prefix = f"{prefix}memory."
        memory_sharded_dict = self.memory.sharded_state_dict(
            memory_prefix, sharded_offsets, metadata
        )
        sharded_dict.update(memory_sharded_dict)
        return sharded_dict


_HASH_MAPPING_CACHE = {}


# Ensures that an NgramHashMapping with identical configuration is created only once.
def get_or_create_hash_mapping(
    engram_vocab_size,
    max_ngram_size,
    n_embed_per_ngram,
    n_head_per_ngram,
    layer_ids,
    tokenizer_name_or_path,
    pad_id,
    seed,
):
    cache_key = (
        tuple(engram_vocab_size),
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        tuple(layer_ids),
        tokenizer_name_or_path,
        pad_id,
        seed,
    )

    if cache_key not in _HASH_MAPPING_CACHE:
        _HASH_MAPPING_CACHE[cache_key] = NgramHashMapping(
            engram_vocab_size=engram_vocab_size,
            max_ngram_size=max_ngram_size,
            n_embed_per_ngram=n_embed_per_ngram,
            n_head_per_ngram=n_head_per_ngram,
            layer_ids=layer_ids,
            tokenizer_name_or_path=tokenizer_name_or_path,
            pad_id=pad_id,
            seed=seed,
        )

    return _HASH_MAPPING_CACHE[cache_key]


class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\ue000"
        self.normalizer = normalizers.Sequence(
            [
                normalizers.NFKC(),
                normalizers.NFD(),
                normalizers.StripAccents(),
                normalizers.Lowercase(),
                normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
                normalizers.Replace(Regex(r"^ $"), SENTINEL),
                normalizers.Strip(),
                normalizers.Replace(SENTINEL, " "),
            ]
        )

        self.lookup_table, self.num_new_token = self._build_lookup_table()

    def __len__(self):
        return self.num_new_token

    def _build_lookup_table(self):
        old2new = {}
        key2new = {}
        new_tokens = []

        from megatron.training import get_args

        args = get_args()
        vocab_size = args.vocab_size
        print(f"CompressedTokenizer: vocab_size: {vocab_size}")
        # vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "�" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup_list = [0] * vocab_size
        for tid in range(vocab_size):
            lookup_list[tid] = old2new[tid]

        lookup = torch.tensor(lookup_list, dtype=torch.long)

        return lookup, len(new_tokens)

    def _compress(self, input_ids):
        x = input_ids.to(dtype=torch.long)
        if self.lookup_table.device != x.device:
            self.lookup_table = self.lookup_table.to(x.device)

        vocab_size = len(self.lookup_table)
        pos_mask = (x >= 0) & (x < vocab_size)
        # # cut here to reduce device-to-host memcpy
        # if not pos_mask.any():
        #     return x
        out = x.clone()
        valid_ids = out[pos_mask]
        mapped = self.lookup_table[valid_ids]
        out[pos_mask] = mapped
        return out

    def __call__(self, input_ids):
        return self._compress(input_ids)


def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    def __init__(
        self,
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        max_long = torch.iinfo(torch.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            gen = torch.Generator(device="cpu")
            gen.manual_seed(base_seed)
            r = torch.randint(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=torch.long,
                generator=gen,
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self._layer_multipliers_per_device = {}

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}

        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []

                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1

                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, seen_primes
                    )
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime

                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes

        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        assert input_ids is not None, "input_ids can not be None in NgramHashMapping"

        x = input_ids.to(dtype=torch.long)
        device = x.device
        B, T = x.shape

        # multipliers = self.layer_multipliers[layer_id].to(device=device, dtype=torch.long)
        key = (layer_id, str(device))
        if key not in self._layer_multipliers_per_device:
            self._layer_multipliers_per_device[key] = self.layer_multipliers[
                layer_id
            ].to(device=device, dtype=torch.long)
        multipliers = self._layer_multipliers_per_device[key]

        def shift_k(k: int) -> torch.Tensor:
            if k == 0:
                return x
            pad = torch.full((B, k), self.pad_id, dtype=torch.long, device=device)
            shifted = torch.cat([pad, x], dim=1)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]

            mix = tokens[0] * multipliers[0]

            for k in range(1, n):
                mix = torch.bitwise_xor(mix, tokens[k] * multipliers[k])

            num_heads_for_this_ngram = self.n_head_per_ngram
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]

            for j in range(num_heads_for_this_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = torch.remainder(mix, mod).to(dtype=torch.long)
                all_hashes.append(head_hash)

        hashes = torch.stack(all_hashes, dim=2)
        return hashes

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(
                input_ids, layer_id=layer_id
            )
        return hash_ids_for_all_layers


class ShortConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )

        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (L,B,HC_MULT,D)
        Output: (L,B,HC_MULT,D)
        """
        T, B, G, C = x.shape
        x = x.permute(1, 0, 2, 3)  # (B, L, G, C)

        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))

        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]

        if self.activation:
            y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()  # (B, L, G, C)

        return y.permute(1, 0, 2, 3).contiguous()  # (L, B, G, C)


class EngramModule(nn.Module):
    def __init__(self, config: TransformerConfig, layer_id):
        super().__init__()
        self.config = config
        self.enable_mhc = self.config.enable_hyper_connections
        if self.enable_mhc:
            self.hc_mult = self.config.num_residual_streams
        else:
            self.hc_mult = 1

        self.layer_id = layer_id
        global_hash_mapping = get_or_create_hash_mapping(
            engram_vocab_size=config.engram_vocab_size,
            max_ngram_size=config.max_ngram_size,
            n_embed_per_ngram=config.n_embed_per_ngram,
            n_head_per_ngram=config.n_head_per_ngram,
            layer_ids=config.engram_layer_ids,
            tokenizer_name_or_path=config.engram_tokenizer_name_or_path,
            pad_id=config.engram_pad_id,
            seed=config.engram_seed,
        )
        self.memory = MultiHeadEmbedding(
            config,
            list_of_N=[
                x
                for y in global_hash_mapping.vocab_size_across_layers[self.layer_id]
                for x in y
            ],
            D=config.n_embed_per_ngram // config.n_head_per_ngram,
        )
        self.embedding_cache = None  # Cache for pre-computed embeddings
        self.embedding_stream = None  # Stream for pre-computing embeddings
        if torch.cuda.is_available():
            self.embedding_stream = torch.cuda.Stream()
        self.short_conv = ShortConv(
            hidden_size=self.config.hidden_size,
            kernel_size=config.engram_kernel_size,
            dilation=config.max_ngram_size,
            hc_mult=self.hc_mult,
        )
        engram_hidden_size = (
            config.max_ngram_size - 1
        ) * config.n_embed_per_ngram
        self.value_proj = nn.Linear(
            engram_hidden_size, self.config.hidden_size
        )
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, self.config.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [
                nn.RMSNorm(self.config.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )
        self.norm2 = nn.ModuleList(
            [
                nn.RMSNorm(self.config.hidden_size)
                for _ in range(self.hc_mult)
            ]
        )

    def forward(self, hidden_states, hash_input_ids):
        """
        # hidden_states: [L, B, HC_MULT * D] or [L/tp_size, B, HC_MULT * D] if SP is on
        input_ids: [B, L] or [B, L/tp_size] if SP is on

        # return: [L, B, HC_MULT * D] or [L/tp_size, B, HC_MULT * D] if SP is on
        """
        # When sequence parallel is enabled, we need to gather the full sequence
        # before entering engram, because:
        # 1. Short conv operates on sequence dimension and needs full sequence
        # 2. Hash input_ids are split and need to be gathered
        # 3. Engram parameters will need all-reduce in backward on tp_group
        sp_enabled = self.config.sequence_parallel
        if sp_enabled:
            # Gather hidden_states from [L/tp_size, B, D] to [L, B, D]
            # Backward: split (scatter without reduce), since engram computes identically
            # on all ranks and d_input is the same everywhere.
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,
                tensor_parallel_output_grad=False,
            )
            # Gather hash_input_ids from [B, L/tp_size] to [B, L]
            if hash_input_ids is not None:
                # hash_input_ids: [B, L/tp_size] -> [L/tp_size, B] -> [L, B] -> [B, L]
                hash_input_ids = hash_input_ids.transpose(0, 1)  # [L/tp_size, B]
                hash_input_ids = tensor_parallel.gather_from_sequence_parallel_region(
                    hash_input_ids,
                    tensor_parallel_output_grad=False,
                )
                hash_input_ids = hash_input_ids.transpose(0, 1).contiguous()  # [B, L]

        # [B, L, N_GRAM * N_HEADS_PER_GRAM]
        # fake hyper-connection
        seq_len, batch_size, expanded_hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(
            seq_len, batch_size, self.hc_mult, -1
        )
        if self.embedding_cache is not None:
            embeddings, embedding_event = self.embedding_cache
            if embedding_event is not None:
                torch.cuda.current_stream().wait_event(
                    embedding_event
                )  # Ensure pre-computed embeddings are ready
            self.embedding_cache = None  # Clear cache after use
            del embedding_event  # Free the event
        else:
            assert hash_input_ids is not None, (
                "If there is no embedding cache, hash input ids can not be None for Engram"
            )
            embeddings = self.memory(hash_input_ids).flatten(start_dim=-2)
        # [L/tp_size, B, N_GRAM * N_HEADS_PER_GRAM, N_EMBED_PER_GRAM // N_HEADS_PER_GRAM]
        # [L/tp_size, B, N_GRAM * N_EMBED_PER_NGRAM]

        # Pre-compute scaling factor for efficiency
        scale = 1.0 / math.sqrt(self.config.hidden_size)
        gates = []
        for hc_idx in range(self.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            # [L/tp_size, B, HIDDEN_SIZE]
            normed_key = self.norm1[hc_idx](key)

            query = hidden_states[:, :, hc_idx, :]
            # [L, B, HIDDEN_SIZE]
            normed_query = self.norm2[hc_idx](query)

            # Compute scaled dot product similarity
            gate = torch.sum(normed_key * normed_query, dim=-1, keepdim=True) * scale
            # Apply smooth absolute value transformation: sign(x) * sqrt(|x|)
            # This is equivalent to: abs().clamp_min(1e-6).sqrt() * sign()
            gate = torch.sign(gate) * torch.sqrt(torch.abs(gate).clamp_min(1e-6))
            gate = torch.sigmoid(gate)
            # [L, B, 1]

            gates.append(gate)
        gates = torch.stack(gates, dim=2)
        # [L, B, HC_MULT, 1]

        value = gates * self.value_proj(embeddings).unsqueeze(2)
        # [L, B, HC_MULT, HIDDEN_SIZE]
        output = value + self.short_conv(value)
        # [L, B, HC_MULT * HIDDEN_SIZE]
        output = output.view(seq_len, batch_size, expanded_hidden_size)

        # When sequence parallel is enabled, scatter back to [L/tp_size, B, D]
        # Forward: scatter (split output, since all ranks computed identical results)
        # Backward: all-gather (gradients from all chunks flow back to engram params)
        if sp_enabled:
            output = tensor_parallel.scatter_to_sequence_parallel_region(output)

        return output

    def pre_compute_embedding(self, input_ids: torch.Tensor):
        """
        Pre-compute the multi-head embedding for the given input IDs.
        This can be called before the forward pass to warm up the embedding cache.
        """
        assert input_ids is not None, "Input ids can not be None for EngramModel"
        # When sequence parallel is enabled, input_ids is [B, L/tp_size].
        # We need to gather to full [B, L] before computing embeddings.
        if self.config.sequence_parallel:
            # input_ids: [B, L/tp_size] -> [L/tp_size, B] -> gather -> [L, B] -> [B, L]
            input_ids_t = input_ids.transpose(0, 1).contiguous()  # [L/tp_size, B]
            input_ids_t = tensor_parallel.gather_from_sequence_parallel_region(
                input_ids_t,
                tensor_parallel_output_grad=False,
            )
            input_ids = input_ids_t.transpose(0, 1).contiguous()  # [B, L]
        self.embedding_stream.synchronize()  # Ensure previous computations on the stream are finished
        with torch.cuda.stream(self.embedding_stream):
            embedding_result = self.memory(input_ids).flatten(start_dim=-2)
        embedding_event = torch.cuda.Event()
        embedding_event.record(self.embedding_stream)
        self.embedding_cache = (embedding_result, embedding_event)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: tuple = (),
        metadata: dict | None = None,
    ):
        sharded_dict = {}
        memory_prefix = f"{prefix}memory."
        sharded_dict.update(
            self.memory.sharded_state_dict(memory_prefix, sharded_offsets, metadata)
        )
        conv_prefix = f"{prefix}short_conv."
        sharded_dict.update(
            sharded_state_dict_default(
                self.short_conv, conv_prefix, sharded_offsets, metadata
            )
        )
        value_proj_prefix = f"{prefix}value_proj."
        sharded_dict.update(
            sharded_state_dict_default(
                self.value_proj, value_proj_prefix, sharded_offsets, metadata
            )
        )
        key_projs_prefix = f"{prefix}key_projs."
        sharded_dict.update(
            sharded_state_dict_default(
                self.key_projs, key_projs_prefix, sharded_offsets, metadata
            )
        )
        norm1_prefix = f"{prefix}norm1."
        sharded_dict.update(
            sharded_state_dict_default(
                self.norm1, norm1_prefix, sharded_offsets, metadata
            )
        )
        norm2_prefix = f"{prefix}norm2."
        sharded_dict.update(
            sharded_state_dict_default(
                self.norm2, norm2_prefix, sharded_offsets, metadata
            )
        )
        return sharded_dict
