# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generate deterministic, offline BERT smoke-test inputs."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Mapping, Sequence

DATASET_PREFIX = "bert_smoke_text_sentence"
TOKENIZER_DIRECTORY = "bert_smoke_tokenizer"
GENERATED_RELATIVE_PATHS = (
    Path(f"{DATASET_PREFIX}.bin"),
    Path(f"{DATASET_PREFIX}.idx"),
    Path(TOKENIZER_DIRECTORY) / "config.json",
    Path(TOKENIZER_DIRECTORY) / "tokenizer_config.json",
    Path(TOKENIZER_DIRECTORY) / "special_tokens_map.json",
    Path(TOKENIZER_DIRECTORY) / "vocab.txt",
)

_INDEX_HEADER = b"MMIDIDX\x00\x00"
_INDEX_VERSION = 1
_INT32_DTYPE_CODE = 4
_DOCUMENT_COUNT = 16
_SENTENCES_PER_DOCUMENT = 4
_TOKENS_PER_SENTENCE = 32
_VOCABULARY_SIZE = 128
_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sentences() -> tuple[tuple[int, ...], ...]:
    ordinary_token_count = _VOCABULARY_SIZE - len(_SPECIAL_TOKENS)
    sentences = []
    for document in range(_DOCUMENT_COUNT):
        for sentence in range(_SENTENCES_PER_DOCUMENT):
            ordinal = (
                document * _SENTENCES_PER_DOCUMENT + sentence
            ) * _TOKENS_PER_SENTENCE
            sentences.append(
                tuple(
                    len(_SPECIAL_TOKENS)
                    + (ordinal + position) % ordinary_token_count
                    for position in range(_TOKENS_PER_SENTENCE)
                )
            )
    return tuple(sentences)


def _dataset_bytes() -> tuple[bytes, bytes]:
    sentences = _sentences()
    binary = b"".join(
        struct.pack(f"<{len(sentence)}i", *sentence) for sentence in sentences
    )
    sequence_lengths = (_TOKENS_PER_SENTENCE,) * len(sentences)
    sequence_pointers = tuple(
        index * _TOKENS_PER_SENTENCE * struct.calcsize("<i")
        for index in range(len(sentences))
    )
    document_indices = tuple(
        document * _SENTENCES_PER_DOCUMENT
        for document in range(_DOCUMENT_COUNT + 1)
    )
    index = b"".join(
        (
            _INDEX_HEADER,
            struct.pack("<Q", _INDEX_VERSION),
            struct.pack("<B", _INT32_DTYPE_CODE),
            struct.pack("<Q", len(sequence_lengths)),
            struct.pack("<Q", len(document_indices)),
            struct.pack(f"<{len(sequence_lengths)}i", *sequence_lengths),
            struct.pack(f"<{len(sequence_pointers)}q", *sequence_pointers),
            struct.pack(f"<{len(document_indices)}q", *document_indices),
        )
    )
    return binary, index


def _vocabulary_bytes() -> bytes:
    vocabulary = (
        *_SPECIAL_TOKENS,
        *(
            f"token{token_id:03d}"
            for token_id in range(len(_SPECIAL_TOKENS), _VOCABULARY_SIZE)
        ),
    )
    return ("\n".join(vocabulary) + "\n").encode("utf-8")


def generate_inputs(output_directory: Path) -> tuple[Path, ...]:
    """Write the six files consumed by the locked BERT smoke profile."""

    output_directory.mkdir(parents=True, exist_ok=True)
    tokenizer_directory = output_directory / TOKENIZER_DIRECTORY
    tokenizer_directory.mkdir()

    binary, index = _dataset_bytes()
    payloads = {
        Path(f"{DATASET_PREFIX}.bin"): binary,
        Path(f"{DATASET_PREFIX}.idx"): index,
        Path(TOKENIZER_DIRECTORY) / "config.json": _json_bytes(
            {
                "model_type": "bert",
                "vocab_size": _VOCABULARY_SIZE,
            }
        ),
        Path(TOKENIZER_DIRECTORY) / "tokenizer_config.json": _json_bytes(
            {
                "do_lower_case": True,
                "model_max_length": 32,
                "tokenizer_class": "BertTokenizer",
            }
        ),
        Path(TOKENIZER_DIRECTORY) / "special_tokens_map.json": _json_bytes(
            {
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
            }
        ),
        Path(TOKENIZER_DIRECTORY) / "vocab.txt": _vocabulary_bytes(),
    }
    for relative_path in GENERATED_RELATIVE_PATHS:
        (output_directory / relative_path).write_bytes(payloads[relative_path])
    return tuple(output_directory / path for path in GENERATED_RELATIVE_PATHS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args(argv)
    generate_inputs(args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
