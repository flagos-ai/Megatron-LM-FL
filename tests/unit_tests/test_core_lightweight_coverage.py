# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import importlib
import json
import logging
import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.export.export_config import ExportConfig
from megatron.core.inference import inference_request as inference_request_module
from megatron.core.inference import utils as inference_utils
from megatron.core.inference.batch_dimensions_utils import (
    CUDAGraphBatchDimensionBuilder,
    InferenceBatchDimensions,
)
from megatron.core.inference.inference_request import (
    DynamicInferenceEvent,
    DynamicInferenceEventType,
    InferenceRequest,
    Status,
    compute_block_hashes_batched,
    deserialize_tensor,
    serialize_tensor,
    unwrap_serialized_tensors,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.models.mimo.config.role import (
    MIMO_LANGUAGE_MODULE_KEY,
    ModuleLayout,
    RankRole,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.tokenizers.text.models.bert_tokenizer import BertTokenizer
from megatron.core.tokenizers.text.models.default_tokenizer import DefaultTokenizerText
from megatron.core.tokenizers.text.models.gpt_tokenizer import GPTTokenizer
from megatron.core.tokenizers.text.models.mamba_tokenizer import MambaTokenizer
from megatron.core.tokenizers.vision.libraries.null_multimodal_tokenizer import (
    MegatronNullMultimodalTokenizer,
)
from megatron.core.tokenizers.vision.vision_tokenizer import MegatronTokenizerVision
from megatron.core.tokenizers.text.parsers.deepseek_r1_reasoning_parser import (
    DeepSeekR1ReasoningParser,
)
from megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser import (
    Qwen3CoderToolParser,
    _Qwen3CoderToolParser,
)


def test_export_config_defaults_and_deprecated_embedding_warning():
    assert ExportConfig().inference_tp_size == 1

    with pytest.warns(DeprecationWarning, match="use_embedding_sharing is deprecated"):
        config = ExportConfig(
            inference_tp_size=2,
            inference_pp_size=4,
            use_parallel_embedding=True,
            use_embedding_sharing=False,
        )

    assert config.inference_pp_size == 4
    assert config.use_parallel_embedding is True


def test_mimo_model_config_defaults_emit_experimental_warning():
    module = importlib.import_module("megatron.core.models.mimo.config.base_configs")
    with pytest.warns(UserWarning, match="experimental"):
        module = importlib.reload(module)

    language_spec = ModuleSpec(module=object)
    vision_spec = ModuleSpec(module=object)
    config = module.MimoModelConfig(
        language_model_spec=language_spec,
        modality_submodules_spec={"vision": vision_spec},
        special_token_ids={"vision": -200},
    )
    assert config.kv_format == "sbhd"
    assert config.language_model_spec is language_spec
    assert config.special_token_ids == {"vision": -200}
    assert config.modality_submodules_spec == {"vision": vision_spec}


def test_rank_role_unified_properties_and_stage_queries():
    role = RankRole.unified(["vision", MIMO_LANGUAGE_MODULE_KEY])

    assert role.mode is ModuleLayout.UNIFIED
    assert role.has_modality_modules is True
    assert role.has_language_module is True
    assert role.modality_module_names == ["vision"]
    assert role.is_first_stage("vision") is True
    assert role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY) is True
    assert role.is_first_stage("audio") is False


def test_rank_role_from_grid_map_validates_keys_and_rank_membership(monkeypatch):
    class FakePipelineGroup:
        def __init__(self, rank, size):
            self._rank = rank
            self._size = size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    class FakeGrid:
        def __init__(self, rank_offset, size, pp_rank=None, pp_size=None):
            self.rank_offset = rank_offset
            self.size = size
            self.dim_names = [] if pp_rank is None else ["pp"]
            self._pg = FakePipelineGroup(pp_rank or 0, pp_size or 1)

        def get_pg(self, name):
            assert name == "pp"
            return self._pg

    grids = {
        "vision": FakeGrid(rank_offset=0, size=2, pp_rank=1, pp_size=2),
        MIMO_LANGUAGE_MODULE_KEY: FakeGrid(rank_offset=4, size=1),
    }

    monkeypatch.setattr("megatron.core.models.mimo.config.role.dist.get_rank", lambda: 1)
    role = RankRole.from_grid_map(grids, ["vision"])
    assert role.mode is ModuleLayout.NON_COLOCATED
    assert role.has_modality_modules is True
    assert role.has_language_module is False
    assert role.is_first_stage("vision") is False
    assert role.is_last_stage("vision") is True

    with pytest.raises(ValueError, match="Missing"):
        RankRole.from_grid_map({MIMO_LANGUAGE_MODULE_KEY: FakeGrid(0, 1)}, ["vision"])

    monkeypatch.setattr("megatron.core.models.mimo.config.role.dist.get_rank", lambda: 9)
    with pytest.raises(RuntimeError, match="not in any module grid"):
        RankRole.from_grid_map(grids, ["vision"])


def test_deepseek_reasoning_parser_extracts_present_and_inferred_think_tags():
    text, metadata = DeepSeekR1ReasoningParser.parse("prefix <think>reason</think> answer")
    assert text == "prefix  answer"
    assert metadata == {"reasoning": "reason"}

    text, metadata = DeepSeekR1ReasoningParser.parse("hidden</think> visible")
    assert text == " visible"
    assert metadata == {"reasoning": "hidden"}

    assert DeepSeekR1ReasoningParser.parse("plain response") == ("plain response", {})


def test_qwen3_tool_parser_extracts_typed_tool_arguments(monkeypatch):
    monkeypatch.setattr(
        "megatron.core.tokenizers.text.parsers.qwen3_coder_tool_parser.uuid.uuid4",
        lambda: SimpleNamespace(hex="1234567890abcdef1234567890abcdef"),
    )
    tools = [
        {
            "type": "function",
            "function": SimpleNamespace(
                name="search",
                parameters={
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "enabled": {"type": "boolean"},
                        "payload": {"type": "object"},
                        "optional": {"type": "string"},
                    }
                },
            ),
        }
    ]
    response = (
        "before <tool_call><function=search>"
        "<parameter=query>\nhello\n</parameter>"
        "<parameter=limit>3</parameter>"
        "<parameter=ratio>2.5</parameter>"
        "<parameter=enabled>true</parameter>"
        "<parameter=payload>{\"a\": 1}</parameter>"
        "<parameter=optional>null</parameter>"
        "</function></tool_call>"
    )

    content, metadata = Qwen3CoderToolParser.parse(response, tools=tools)

    assert content == "before "
    tool_call = metadata["tool_calls"][0]
    assert tool_call["id"] == "call_1234567890abcdef12345678"
    assert tool_call["function"]["name"] == "search"
    args = json.loads(tool_call["function"]["arguments"])
    assert args == {
        "query": "hello",
        "limit": 3,
        "ratio": 2.5,
        "enabled": True,
        "payload": {"a": 1},
        "optional": None,
    }


def test_qwen3_tool_parser_fallbacks_and_conversion_edges(caplog):
    parser = _Qwen3CoderToolParser()

    assert parser.extract_tool_calls("plain text", tools=None) == {
        "tools_called": False,
        "tool_calls": [],
        "content": "plain text",
    }
    assert parser._convert_param_value("bad", "limit", {"limit": {"type": "int"}}, "fn") == "bad"
    assert parser._convert_param_value("not-bool", "flag", {"flag": {"type": "bool"}}, "fn") is False
    assert parser._convert_param_value("[1, 2]", "items", {"items": {"type": "array"}}, "fn") == [
        1,
        2,
    ]
    assert parser._convert_param_value("{'a': 1}", "items", {"items": {"type": "dict"}}, "fn") == {
        "a": 1
    }
    assert Qwen3CoderToolParser.parse("<tool_call><function=broken") == (
        "<tool_call><function=broken",
        {},
    )


def test_inference_request_serialization_deserialization_and_events():
    params = SamplingParams(top_k=5, top_n_logprobs=2, skip_prompt_log_probs=False)
    request = InferenceRequest(
        request_id=7,
        prompt="hello",
        sampling_params=params,
        status=Status.ACTIVE_AND_GENERATING_TOKENS,
        prompt_tokens=[1, 2],
        generated_length=3,
    )

    serialized = request.serialize()
    assert serialized["status"] == "ACTIVE_AND_GENERATING_TOKENS"
    assert serialized["sampling_params"]["top_k"] == 5
    assert unwrap_serialized_tensors({"x": ("tensor", [1, 2]), "y": "plain"}) == {
        "x": [1, 2],
        "y": "plain",
    }

    restored = InferenceRequest.deserialize(serialized)
    assert restored.status is Status.ACTIVE_AND_GENERATING_TOKENS
    assert restored.sampling_params.top_n_logprobs == 2
    assert restored.prompt_tokens == [1, 2]

    with pytest.warns(UserWarning, match="renamed to `sampling_params`"):
        legacy = InferenceRequest(
            request_id=8,
            prompt="legacy",
            inference_parameters=SamplingParams(temperature=0.5),
        )
    assert legacy.sampling_params.temperature == 0.5

    event = DynamicInferenceEvent(
        type=DynamicInferenceEventType.GENERATED_TOKEN, payload={"token_id": 42}
    )
    assert event.type is DynamicInferenceEventType.GENERATED_TOKEN
    assert "token=42" in str(event)

    pause_event = DynamicInferenceEvent(type=DynamicInferenceEventType.PAUSE)
    assert pause_event.payload is None


def test_inference_request_tensor_and_hash_helpers(monkeypatch):
    calls = []
    fake_nvtx = SimpleNamespace(
        range_push=lambda name: calls.append(("push", name)),
        range_pop=lambda: calls.append(("pop", None)),
    )
    monkeypatch.setattr(inference_request_module.torch.cuda, "nvtx", fake_nvtx, raising=False)

    serialized = serialize_tensor(torch.tensor([[1, 2], [3, 4]]))
    assert serialized == [[1, 2], [3, 4]]
    assert torch.equal(deserialize_tensor(serialized), torch.tensor([[1, 2], [3, 4]]))
    assert calls == [("push", "serialize_tensor"), ("pop", None)]

    inference_request_module._hash_powers = None
    hashes = compute_block_hashes_batched(torch.tensor([1, 2, 3, 4, 5]), block_size=2)
    assert len(hashes) == 2
    assert hashes[0] != hashes[1]
    assert compute_block_hashes_batched(torch.tensor([1]), block_size=2) == []


def test_inference_utils_counter_and_flashinfer_cache_checks(monkeypatch, caplog):
    counter = inference_utils.Counter(start=5)
    assert next(counter) == 5
    assert next(counter) == 6
    counter.reset()
    assert next(counter) == 0

    caplog.set_level(logging.INFO)
    monkeypatch.setattr(inference_utils, "FLASHINFER_JIT_CACHE_VERSION", "1.2.3")
    inference_utils.check_flashinfer_jit_cache_installed(log_version=True)
    assert "flashinfer-jit-cache 1.2.3" in caplog.text

    monkeypatch.setattr(inference_utils, "FLASHINFER_JIT_CACHE_VERSION", None)
    monkeypatch.setattr(inference_utils.torch.version, "cuda", "12.9")
    with pytest.raises(RuntimeError, match="flashinfer-jit-cache"):
        inference_utils.check_flashinfer_jit_cache_installed()

    monkeypatch.setattr(inference_utils.torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="required for expert parallel inference"):
        inference_utils.check_flashinfer_jit_cache_installed()


def test_inference_batch_dimensions_validity_matching_and_generation(monkeypatch):
    dim = InferenceBatchDimensions(token_count=16, prefill_req_count=2, decode_req_count=4)
    assert str(dim) == "[16]: 2 P + 4 D"
    assert dim.req_count == 6
    assert dim.is_valid(max_requests=8, max_sequence_length=16, num_speculative_tokens=0)
    assert not InferenceBatchDimensions(2, 2, 4).is_valid(8, 16, 0)
    assert not InferenceBatchDimensions(100, 1, 0).is_valid(8, 16, 0)

    decode_graph = InferenceBatchDimensions(8, 0, 8)
    assert decode_graph.is_applicable_for_batch_dim(InferenceBatchDimensions(4, 0, 4))
    assert not decode_graph.is_applicable_for_batch_dim(InferenceBatchDimensions(4, 1, 3))

    mixed_graph = InferenceBatchDimensions(12, 3, 3)
    assert mixed_graph.is_applicable_for_batch_dim(InferenceBatchDimensions(10, 2, 4))
    assert not mixed_graph.is_applicable_for_batch_dim(
        InferenceBatchDimensions(10, 2, 4), strict=True
    )

    assert CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(2, 1, 9) == [8]
    auto_counts = CUDAGraphBatchDimensionBuilder._calculate_cuda_graph_token_counts(4, -1, 20)
    assert auto_counts[0] == 20
    assert auto_counts[-1] == 4

    dims, token_counts = CUDAGraphBatchDimensionBuilder.generate_cuda_graph_batch_dimensions_list(
        tp_size=2,
        num_cuda_graphs=2,
        cuda_graph_max_tokens=8,
        cuda_graph_mixed_prefill_request_count=2,
        max_requests=8,
        max_tokens=8,
        max_sequence_length=16,
        use_cuda_graphs_for_non_decode_steps=True,
    )
    assert token_counts == sorted(token_counts, reverse=True)
    assert any(item.prefill_req_count > 0 for item in dims)

    monkeypatch.setattr(
        "megatron.core.inference.batch_dimensions_utils.get_pg_size", lambda group=None: 1
    )
    best = CUDAGraphBatchDimensionBuilder.match_graph_config(
        real_batch_dim=InferenceBatchDimensions(4, 0, 4),
        cuda_graph_batch_dimensions_list=[InferenceBatchDimensions(8, 0, 8)],
        smallest_non_decode_cuda_graph_size=8,
    )
    assert best == InferenceBatchDimensions(8, 0, 8)
    assert CUDAGraphBatchDimensionBuilder.match_graph_config(
        real_batch_dim=InferenceBatchDimensions(99, 0, 99),
        cuda_graph_batch_dimensions_list=[InferenceBatchDimensions(8, 0, 8)],
        smallest_non_decode_cuda_graph_size=8,
    ) is None


def test_null_multimodal_tokenizer_text_image_and_special_token_paths(monkeypatch):
    fake_llava = SimpleNamespace(IMAGE_TOKEN="<image>", DEFAULT_IMAGE_TOKEN_INDEX=-200)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.multimodal.llava_model",
        fake_llava,
    )

    tokenizer = MegatronNullMultimodalTokenizer(vocab_size=10)

    assert tokenizer.tokenize("1 2 3") == [1, 2, 3]
    assert tokenizer.detokenize([1, 2, 3]) == "1 2 3"
    assert tokenizer.offsets([10, 2], "10 2") == [0, 3]
    assert tokenizer.convert_tokens_to_ids("1  2") == [1, 2]
    assert tokenizer.convert_tokens_to_ids("<image>") == -200
    assert tokenizer.vocab_size == 11
    assert tokenizer.eod == 10
    assert tokenizer.cls == tokenizer.sep == tokenizer.mask == -1
    assert tokenizer.additional_special_tokens_ids is None

    custom = MegatronNullMultimodalTokenizer(
        vocab_size=5, image_token="<pic>", image_token_id=99
    )
    assert custom.convert_tokens_to_ids("<pic>") == 99


def test_text_tokenizer_wrappers_use_null_text_library_and_expose_properties():
    tokenizer_classes = [DefaultTokenizerText, GPTTokenizer, BertTokenizer, MambaTokenizer]

    for tokenizer_cls in tokenizer_classes:
        config = {"library": "null-text"}
        tokenizer = tokenizer_cls(path=None, config=config, vocab_size=12)

        assert config["class_name"] == tokenizer_cls.__name__
        assert tokenizer.tokenize("1 2") == [1, 2]
        assert tokenizer.detokenize([1, 2]) == "1 2"
        assert tokenizer.offsets([10, 2], "10 2") == [0, 3]
        assert tokenizer.vocab_size == 13
        assert tokenizer.eod == 12
        assert tokenizer.additional_special_tokens_ids is None
        assert tokenizer.unique_identifiers["class"].endswith(tokenizer_cls.__name__)
        assert tokenizer.vocab_file is None
        assert tokenizer.merges_file is None
        with pytest.raises(NotImplementedError):
            _ = tokenizer.space_sensitive
        with pytest.raises(ValueError, match="save_pretrained"):
            tokenizer.save_pretrained("/tmp/unused")
        with pytest.raises(NotImplementedError, match="SFTTokenizer"):
            tokenizer.tokenize_conversation([], False, False)


def test_vision_tokenizer_delegates_to_null_multimodal_tokenizer(monkeypatch):
    fake_llava = SimpleNamespace(IMAGE_TOKEN="<image>", DEFAULT_IMAGE_TOKEN_INDEX=-200)
    monkeypatch.setitem(
        sys.modules,
        "megatron.core.models.multimodal.llava_model",
        fake_llava,
    )
    tokenizer = MegatronTokenizerVision(
        path=None,
        config={"library": "null-multimodal"},
        vocab_size=7,
    )

    assert tokenizer.tokenize("3 4") == [3, 4]
    assert tokenizer.detokenize([3, 4]) == "3 4"
    assert tokenizer.convert_tokens_to_ids("<image>") == -200
    assert tokenizer.offsets([3, 44], "3 44") == [0, 2]
    assert tokenizer.vocab_size == 8
    assert tokenizer.eod == 7

    with pytest.raises(NotImplementedError, match="vision tokenizers"):
        tokenizer.apply_chat_template()
