# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import json
import fcntl
import os
import runpy
import sys
import tempfile

import nltk
import pytest
import requests

from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MEGATRON_CONFIG_MAP
from tools.merge_datasets import main as merge_main
from tools.preprocess_data import Encoder
from tools.preprocess_data import get_args as build_args
from tools.preprocess_data import main as build_main

__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB = (
    "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
)

__LOCAL_BERT_VOCAB = "/home/gitlab-runner/data/bert_data/vocab.txt"

__LOCAL_GPT2_MERGE = "/home/gitlab-runner/data/gpt3_data/gpt2-merges.txt"

__LOCAL_GPT2_VOCAB = "/home/gitlab-runner/data/gpt3_data/gpt2-vocab.json"

__OPT_DATA_BERT_VOCAB = "/opt/data/tokenizers/bert/vocab.txt"

__OPT_DATA_GPT2_MERGE = "/opt/data/tokenizers/megatron/gpt2-merges.txt"

__OPT_DATA_GPT2_VOCAB = "/opt/data/tokenizers/megatron/gpt2-vocab.json"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _first_existing_path(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def _download_once(url, filename):
    cache_dir = os.path.join(tempfile.gettempdir(), "megatron_unit_test_assets")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, filename)
    lock_path = path + ".lock"

    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(path):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            tmp_path = f"{path}.{os.getpid()}.tmp"
            with open(tmp_path, "wb") as writer:
                writer.write(response.content)
            os.replace(tmp_path, path)
    return path


def dummy_jsonl(odir):
    # numbers
    list_numbers = [json.dumps({"text": str(i + 1)}) + "\n" for i in range(100)]
    with open(os.path.join(odir, "numbers.jsonl"), "w") as writer:
        writer.writelines(list_numbers)
    # numbers ascending
    list_numbers_ascending = [
        json.dumps({"text": " ".join([str(j + 1) for j in range(i + 1)])}) + "\n"
        for i in range(100)
    ]
    with open(os.path.join(odir, "numbers_ascending.jsonl"), "w") as writer:
        writer.writelines(list_numbers_ascending)
    # test
    list_test = []
    with open(__file__) as reader:
        for line in reader:
            list_test.append(json.dumps({"text": line}) + "\n")
    with open(os.path.join(odir, "test.jsonl"), "w") as writer:
        writer.writelines(list_test)


def build_datasets(idir, odir, extra_args=[]):
    for name in os.listdir(idir):
        sys.argv = [
            sys.argv[0],
            "--input",
            os.path.join(idir, name),
            "--output-prefix",
            os.path.join(odir, os.path.splitext(name)[0]),
        ] + extra_args
        build_main()


def merge_datasets(idir):
    sys.argv = [sys.argv[0], "--input", idir, "--output-prefix", os.path.join(idir, "merge")]
    merge_main()


def do_test_preprocess_data(temp_dir, extra_args=[]):
    # set the default nltk data path
    os.environ["NLTK_DATA"] = os.path.join(temp_dir, "nltk_data")
    nltk.data.path.append(os.environ["NLTK_DATA"])

    path_to_raws = os.path.join(temp_dir, "sample_raws")
    path_to_data = os.path.join(temp_dir, "sample_data")
    os.mkdir(path_to_raws)
    os.mkdir(path_to_data)

    # create the dummy resources
    dummy_jsonl(path_to_raws)

    # build the datasets
    build_datasets(path_to_raws, path_to_data, extra_args=extra_args)

    # merge the datasets
    merge_datasets(path_to_data)

    sys.argv = [sys.argv[0], "--input", None, "--output-prefix", None] + extra_args
    encoder = Encoder(build_args())
    encoder.initializer()

    def tokens_to_string(toks):
        for option in ["decode", "detokenize"]:
            try:
                return getattr(encoder.tokenizer, option)(toks)
            except:
                continue
        raise RuntimeError(f"{type(encoder.tokenizer)} tokenizer cannot decode or detokenize")

    merged_index = 0
    merged_dataset = IndexedDataset(os.path.join(path_to_data, "merge"))

    # sorted to ensure ordering matches merged dataset
    basenames = sorted(
        [
            name
            for name in os.listdir(path_to_data)
            if name.endswith(".idx") and not name.startswith("merge")
        ]
    )

    # index into the merged document index
    merged_doc_index_index = 0

    for basename in basenames:
        realpath_raw = f"{os.path.join(path_to_raws, '_'.join(basename.split('_')[:-2]))}.jsonl"
        realpath_doc = os.path.join(path_to_data, basename.split(".")[-2])

        dataset_index = 0
        dataset = IndexedDataset(realpath_doc)

        merged_doc_idx = merged_dataset.document_indices[
            merged_doc_index_index : merged_doc_index_index + len(dataset.document_indices)
        ]
        merged_doc_idx = merged_doc_idx - merged_doc_idx[0]

        assert (
            dataset.document_indices == merged_doc_idx
        ).all(), f"ERROR: {basename.split('_')[:-2]}: merged dataset document indices mismatch"

        merged_doc_index_index += len(dataset.document_indices) - 1

        with open(realpath_raw, "rt") as reader:
            for json_line in reader:
                toks = encoder.encode(json_line)[0]["text"]

                raw = tokens_to_string(toks)

                processed_toks = []
                while len(processed_toks) < len(toks):
                    processed_toks.extend(dataset[dataset_index])
                    dataset_index += 1
                processed = tokens_to_string(processed_toks)

                assert (
                    raw == processed
                ), f"ERROR: {basename.split('_')[:-2]}: raw and processed documents do not match"

                merged_toks = []
                while len(merged_toks) < len(toks):
                    merged_toks.extend(merged_dataset[merged_index])
                    merged_index += 1
                merged = tokens_to_string(merged_toks)

                assert (
                    raw == merged
                ), f"ERROR: {basename.split('_')[:-2]}: raw and merged documents do not match"

        print(
            f"INFO: {''.join(basename.split('_')[:-2])}: raw, processed, and merged documents match!"
        )

    print("INFO: Success!")


def gpt2_vocab(odir):
    local_path = _first_existing_path(__LOCAL_GPT2_VOCAB, __OPT_DATA_GPT2_VOCAB)
    if local_path is not None:
        return local_path
    return local_gpt2_tokenizer(odir)[1]


def gpt2_merge(odir):
    local_path = _first_existing_path(__LOCAL_GPT2_MERGE, __OPT_DATA_GPT2_MERGE)
    if local_path is not None:
        return local_path
    return local_gpt2_tokenizer(odir)[2]


def _gpt2_byte_encoder():
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(161, 173))
        + list(range(174, 256))
    )
    unicode_values = byte_values[:]
    next_unicode = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            unicode_values.append(256 + next_unicode)
            next_unicode += 1
    return dict(zip(byte_values, [chr(value) for value in unicode_values]))


def local_gpt2_tokenizer(odir):
    tokenizer_dir = os.path.join(odir, "gpt2_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    vocab_file = os.path.join(tokenizer_dir, "vocab.json")
    merges_file = os.path.join(tokenizer_dir, "merges.txt")
    byte_encoder = _gpt2_byte_encoder()
    vocab = {token: index for index, token in enumerate(byte_encoder.values())}
    vocab["<|endoftext|>"] = len(vocab)

    with open(vocab_file, "w", encoding="utf-8") as writer:
        json.dump(vocab, writer)

    with open(merges_file, "w", encoding="utf-8") as writer:
        writer.write("#version: 0.2\n")

    with open(os.path.join(tokenizer_dir, "config.json"), "w", encoding="utf-8") as writer:
        json.dump({"model_type": "gpt2"}, writer)

    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as writer:
        json.dump(
            {
                "model_max_length": 1024,
                "tokenizer_class": "GPT2Tokenizer",
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            },
            writer,
        )

    with open(
        os.path.join(tokenizer_dir, "special_tokens_map.json"), "w", encoding="utf-8"
    ) as writer:
        json.dump(
            {
                "bos_token": "<|endoftext|>",
                "eos_token": "<|endoftext|>",
                "unk_token": "<|endoftext|>",
            },
            writer,
        )

    return tokenizer_dir, vocab_file, merges_file


def test_preprocess_data_gpt():
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_dir, vocab_file, merges_file = local_gpt2_tokenizer(temp_dir)

        # gpt specific args
        gpt_args = [
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            gpt2_vocab(temp_dir) or vocab_file,
            "--merge-file",
            gpt2_merge(temp_dir) or merges_file,
            "--append-eod",
            "--workers",
            "10",
            "--log-interval",
            "1",
        ]

        original_tokenizer_name = MEGATRON_CONFIG_MAP["GPT2BPETokenizer"][
            "tokenizer_name"
        ]
        MEGATRON_CONFIG_MAP["GPT2BPETokenizer"]["tokenizer_name"] = tokenizer_dir
        try:
            do_test_preprocess_data(temp_dir, extra_args=gpt_args)
        finally:
            MEGATRON_CONFIG_MAP["GPT2BPETokenizer"][
                "tokenizer_name"
            ] = original_tokenizer_name


def test_preprocess_data_gpt_optimal_workers():
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_dir, _, _ = local_gpt2_tokenizer(temp_dir)
        input_path = os.path.join(temp_dir, "optimal_workers.jsonl")
        with open(input_path, "w") as writer:
            for i in range(1002):
                writer.write(json.dumps({"text": f"document {i}"}) + "\n")

        # gpt specific args
        gpt_args = [
            "--input",
            input_path,
            "--output-prefix",
            f"{temp_dir}/optimal_workers",
            "--tokenizer-type",
            "GPT2BPETokenizer",
            "--vocab-file",
            gpt2_vocab(temp_dir),
            "--merge-file",
            gpt2_merge(temp_dir),
            "--append-eod",
            "--workers",
            "2",
            "--log-interval",
            "1",
            "--find-optimal-num-workers",
            "--workers-to-check",
            "2",
            "4",
            "8",
            "--max-documents",
            "1002",
        ]
        preprocess_data_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../tools/preprocess_data.py")
        )
        original_tokenizer_name = MEGATRON_CONFIG_MAP["GPT2BPETokenizer"][
            "tokenizer_name"
        ]
        MEGATRON_CONFIG_MAP["GPT2BPETokenizer"]["tokenizer_name"] = tokenizer_dir
        try:
            sys.argv = [preprocess_data_path] + gpt_args
            runpy.run_path(preprocess_data_path, run_name="__main__")
        finally:
            MEGATRON_CONFIG_MAP["GPT2BPETokenizer"][
                "tokenizer_name"
            ] = original_tokenizer_name


def bert_vocab(odir):
    local_path = _first_existing_path(__LOCAL_BERT_VOCAB, __OPT_DATA_BERT_VOCAB)
    if local_path is not None:
        return local_path
    return _download_once(__HUGGINGFACE_BERT_BASE_UNCASED_VOCAB, "bert-base-uncased-vocab.txt")


def local_bert_tokenizer(odir):
    tokenizer_dir = os.path.join(odir, "bert_tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)

    vocab_file = os.path.join(tokenizer_dir, "vocab.txt")
    vocab_tokens = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "text",
        "document",
        "test",
        "numbers",
        "ascending",
        "import",
        "def",
        "with",
        "for",
        "in",
        "range",
        "json",
        "line",
        ".",
        ",",
        ":",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
        "_",
        "-",
    ] + [str(i) for i in range(1003)]
    with open(vocab_file, "w", encoding="utf-8") as writer:
        writer.write("\n".join(vocab_tokens))

    with open(os.path.join(tokenizer_dir, "config.json"), "w", encoding="utf-8") as writer:
        json.dump({"model_type": "bert"}, writer)

    with open(
        os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8"
    ) as writer:
        json.dump(
            {"do_lower_case": True, "model_max_length": 512, "tokenizer_class": "BertTokenizer"},
            writer,
        )

    with open(
        os.path.join(tokenizer_dir, "special_tokens_map.json"), "w", encoding="utf-8"
    ) as writer:
        json.dump(
            {
                "unk_token": "[UNK]",
                "sep_token": "[SEP]",
                "pad_token": "[PAD]",
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
            },
            writer,
        )

    return tokenizer_dir, vocab_file


@pytest.mark.flaky
@pytest.mark.flaky_in_dev
def test_preprocess_data_bert():
    with tempfile.TemporaryDirectory() as temp_dir:
        tokenizer_dir, vocab_file = local_bert_tokenizer(temp_dir)

        # bert specific args
        bert_args = [
            "--tokenizer-type",
            "BertWordPieceLowerCase",
            "--vocab-file",
            vocab_file,
            "--tokenizer-hf-no-use-fast",
            "--workers",
            "2",
            "--log-interval",
            "1",
            "--partitions",
            "2",
            "--keep-sequential-samples",
        ]

        original_tokenizer_name = MEGATRON_CONFIG_MAP["BertWordPieceLowerCase"][
            "tokenizer_name"
        ]
        MEGATRON_CONFIG_MAP["BertWordPieceLowerCase"]["tokenizer_name"] = tokenizer_dir
        try:
            do_test_preprocess_data(temp_dir, extra_args=bert_args)
        finally:
            MEGATRON_CONFIG_MAP["BertWordPieceLowerCase"][
                "tokenizer_name"
            ] = original_tokenizer_name


if __name__ == "__main__":
    test_preprocess_data_gpt()
    test_preprocess_data_bert()
    test_preprocess_data_gpt_optimal_workers()
