# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict


class NullTokenizer:
    """
    Synthetic tokenizer for performance benchmarking and debugging

    Args:
        vocab_size: vocabulary size for embedding
    """

    def __init__(self, vocab_size):
        """ """
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def text_to_ids(self, text):
        """Converts text to ids."""
        return [int(x) for x in text.split(' ')]

    def ids_to_text(self, ids):
        """Converts ids to text."""
        text = [str(x) for x in ids]
        return ' '.join(text)

    def tokens_to_ids(self, tokens):
        """Converts tokens to ids."""
        return [int(x) for x in tokens]

    def ids_to_tokens(self, ids):
        """Converts ids to tokens."""
        return [str(x) for x in ids]

    def offsets(self, ids: list[int], text: str) -> list[int]:
        """Returns offsets."""
        offsets, start_idx = [], 0
        for id_ in ids:
            offsets.append(start_idx)
            start_idx += 1 + len(str(id_))
        return offsets

    @property
    def unique_identifiers(self) -> OrderedDict:
        """Property required for use with megatron-core datasets."""
        return OrderedDict({"class": f"{type(self).__module__}.{type(self).__qualname__}"})

    @property
    def vocab_size(self):
        """Returns vocab size."""
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        """ """
        raise NotImplementedError

    @property
    def inv_vocab(self):
        """ """
        raise NotImplementedError

    @property
    def cls(self):
        """Returns cls token."""
        return -1

    @property
    def sep(self):
        """Returns sep token."""
        return -1

    @property
    def mask(self):
        """Returns mask token."""
        return -1

    @property
    def eod(self):
        """Returns eod token."""
        return self._eod_id

    @property
    def pad_id(self):
        """Returns id of padding token. NullTokenizer has none -> eod."""
        return self._eod_id

    @property
    def pad(self):
        """Returns padding token id (needed by RL trajectory padding)."""
        return self._eod_id

    @property
    def bos_id(self):
        """Returns id of beginning of sentence token. NullTokenizer has none."""
        return None

    @property
    def bos(self):
        """Returns beginning of sentence token id."""
        return None

    @property
    def eos_id(self):
        """Returns id of end of sentence token."""
        return self._eod_id

    @property
    def eos(self):
        """Returns end of sentence token id."""
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        """ """
        return None
