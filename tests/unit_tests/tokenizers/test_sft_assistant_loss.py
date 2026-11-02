# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import argparse
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from megatron.core.tokenizers.text.libraries import sft_tokenizer
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import _add_sft_args
from megatron.training.datasets.sft_dataset import SFTDataset

CHAT_TEMPLATE = """{{ bos_token }}{% for message in messages %}
{{ '<' + message['role'] + '>' }}{% if message['role'] == 'assistant' %}
{% generation %}{{ message['content'] + eos_token }}{% endgeneration %}
{% else %}{{ message['content'] }}{% endif %}{% endfor %}
{% if add_generation_prompt %}{{ '<assistant>' }}{% endif %}"""
CONVERSATION = [
    {"role": "system", "content": "instructions"},
    {"role": "user", "content": "answer one"},
    {"role": "assistant", "content": "answer one"},
    {"role": "tool", "content": "tool result"},
    {"role": "assistant", "content": "answer two"},
    {"role": "user", "content": "question"},
]


@pytest.fixture
def hf_tokenizer(monkeypatch):
    special = ["<pad>", "<unk>", "<eos>", "<bos>", "<system>", "<user>", "<assistant>", "<tool>"]
    words = ["instructions", "answer", "one", "two", "tool", "result", "question"]
    vocab = {token: index for index, token in enumerate(special + words)}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
        bos_token="<bos>",
        additional_special_tokens=special[4:],
        chat_template=CHAT_TEMPLATE,
    )
    monkeypatch.setattr(
        sft_tokenizer.transformers.AutoTokenizer, "from_pretrained", lambda **kwargs: tokenizer
    )
    return tokenizer


def test_default_masks_only_assistant_bodies_and_eos(hf_tokenizer):
    tokenizer = sft_tokenizer.SFTTokenizer("unused", "default")
    tokens, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    supervised = targets[targets != sft_tokenizer.IGNORE_INDEX].tolist()
    assert hf_tokenizer.convert_ids_to_tokens(supervised) == [
        "answer",
        "one",
        "<eos>",
        "answer",
        "two",
        "<eos>",
    ]
    np.testing.assert_array_equal(tokens[targets != sft_tokenizer.IGNORE_INDEX], supervised)
    assert hf_tokenizer.convert_tokens_to_ids("<assistant>") not in supervised
    # The user repeats the same answer text: masking must depend on position, not text.
    assert (tokens == hf_tokenizer.convert_tokens_to_ids("one")).sum() == 2
    assert (targets == hf_tokenizer.convert_tokens_to_ids("one")).sum() == 1


@pytest.mark.parametrize("add_generation_prompt", [False, True])
def test_switch_does_not_change_inputs_or_inference(hf_tokenizer, add_generation_prompt):
    enabled = sft_tokenizer.SFTTokenizer("unused", "default")
    disabled = sft_tokenizer.SFTTokenizer("unused", "default", assistant_only_loss=False)
    tokens, _ = enabled.tokenize_conversation(CONVERSATION, True, add_generation_prompt)
    old_tokens, old_targets = disabled.tokenize_conversation(
        CONVERSATION, True, add_generation_prompt
    )
    np.testing.assert_array_equal(tokens, old_tokens)
    np.testing.assert_array_equal(old_tokens, old_targets)
    np.testing.assert_array_equal(
        tokens, enabled.tokenize_conversation(CONVERSATION, False, add_generation_prompt)
    )


def test_missing_generation_spans_fail_only_when_masking(hf_tokenizer):
    hf_tokenizer.chat_template = CHAT_TEMPLATE.replace("{% generation %}", "").replace(
        "{% endgeneration %}", ""
    )
    enabled = sft_tokenizer.SFTTokenizer("unused", "default")
    with pytest.raises(ValueError, match="requires.*generation"):
        enabled.tokenize_conversation(CONVERSATION, True, False)
    assert enabled.tokenize_conversation(CONVERSATION, False, False).size > 0
    disabled = sft_tokenizer.SFTTokenizer("unused", "default", assistant_only_loss=False)
    tokens, targets = disabled.tokenize_conversation(CONVERSATION, True, False)
    np.testing.assert_array_equal(tokens, targets)


def test_empty_generation_mask_is_reported(hf_tokenizer):
    hf_tokenizer.chat_template = (
        "{% generation %}{% endgeneration %}"
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    tokenizer = sft_tokenizer.SFTTokenizer("unused", "default")
    with pytest.raises(ValueError, match="no assistant tokens"):
        tokenizer.tokenize_conversation(CONVERSATION, True, False)


def test_no_assistant_turn_has_no_targets(hf_tokenizer):
    tokenizer = sft_tokenizer.SFTTokenizer("unused", "default")
    _, targets = tokenizer.tokenize_conversation(CONVERSATION[:2], True, True)
    assert (targets == sft_tokenizer.IGNORE_INDEX).all()


@pytest.mark.parametrize("prompt_format", ["identity", "nemotron-nano-v2", "nemotron-h-aligned"])
def test_other_prompt_formats_are_unchanged(hf_tokenizer, prompt_format):
    # These formats use their existing templates and masking rules, regardless of the flag.
    conversation = CONVERSATION[:3]
    enabled = sft_tokenizer.SFTTokenizer("unused", prompt_format)
    disabled = sft_tokenizer.SFTTokenizer("unused", prompt_format, assistant_only_loss=False)
    for return_target in (False, True):
        actual = enabled.tokenize_conversation(conversation, return_target, False)
        expected = disabled.tokenize_conversation(conversation, return_target, False)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("option", [None, True, False])
def test_factory_passes_switch_and_defaults_on(hf_tokenizer, option):
    args = SimpleNamespace(
        tokenizer_type="SFTTokenizer",
        tokenizer_model="unused",
        sft_tokenizer_prompt_format="default",
        metadata_path=None,
        pad_vocab_size=False,
    )
    if option is not None:
        args.sft_assistant_only_loss = option
    tokenizer = build_tokenizer(args)
    tokens, targets = tokenizer.tokenize_conversation(CONVERSATION, True, False)
    assert (targets == sft_tokenizer.IGNORE_INDEX).any() == (option is not False)
    if option is False:
        np.testing.assert_array_equal(tokens, targets)


@pytest.mark.parametrize(
    "argv,expected",
    [([], True), (["--sft-assistant-only-loss"], True), (["--no-sft-assistant-only-loss"], False)],
)
def test_training_argument_switch(argv, expected):
    parser = _add_sft_args(argparse.ArgumentParser())
    assert parser.parse_args(argv).sft_assistant_only_loss is expected


@pytest.mark.parametrize("sequence_length", [12, 128])
def test_dataset_alignment_truncation_and_padding(hf_tokenizer, sequence_length):
    tokenizer = sft_tokenizer.SFTTokenizer("unused", "default")
    # Two packed conversations also exercise the per-conversation CP padding path.
    messages = CONVERSATION[:3] + CONVERSATION[:3]
    dataset = SFTDataset.__new__(SFTDataset)
    dataset.config = SimpleNamespace(
        tokenizer=SimpleNamespace(
            tokenize_conversation=tokenizer.tokenize_conversation,
            eod=hf_tokenizer.eos_token_id,
            pad=hf_tokenizer.pad_token_id,
        ),
        sequence_length=sequence_length,
        reset_position_ids=False,
        context_parallel_size=2,
        preserve_packed_sequence_lengths=False,
        create_attention_mask=False,
        reset_attention_mask=False,
    )
    dataset.dataset = [messages]
    dataset.indices = np.array([0])
    sample = dataset[0]
    labels = sample['labels'].numpy()
    mask = sample['loss_mask'].numpy()
    np.testing.assert_array_equal(mask, (labels != -100) & (labels != hf_tokenizer.pad_token_id))
    expected = hf_tokenizer.convert_tokens_to_ids(["answer", "one", "<eos>"])
    assert labels[mask.astype(bool)].tolist() == expected * (1 if sequence_length == 12 else 2)
    # Every supervised label predicts the next assistant token, including the first answer token.
    for position in np.flatnonzero(mask):
        if position + 1 < sequence_length:
            assert labels[position] == sample['tokens'][position + 1].item()
    assert sample['tokens'].numel() == sequence_length
