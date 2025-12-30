import re
import sys

from megatron.training.tokenizer.gpt2_tokenization import GPT2Tokenizer

from plugin.training.tokenizer.tokenization_utils import Trie


class AquilaTokenizer(GPT2Tokenizer):
    def __init__(self, vocab_file, merges_file, errors='replace',
                 special_tokens=None, max_len=None):
        super().__init__(vocab_file, merges_file, errors=errors,
                         special_tokens=special_tokens, max_len=max_len)

        self.tokens_trie = Trie()
        if len(self.special_tokens) > 0:
            for token in self.special_tokens.keys():
                self.tokens_trie.add(token)

        for k, v in self.special_tokens_decoder.items():
            self.decoder[k] = v
            self.encoder[v] = k

    def _tokenize(self, text):
        """ Tokenize a string. """
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            if sys.version_info[0] == 2:
                token = ''.join(self.byte_encoder[ord(b)] for b in token)
            else:
                token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def tokenize(self, text):
        tokens = self.tokens_trie.split(text)

        bpe_tokens = []
        for token in tokens:
            if not token:
                continue
            if token in self.special_tokens:
                bpe_tokens.append(token)
            else:
                bpe_tokens.extend(self._tokenize(token))
        return bpe_tokens

    def decode(self, tokens):
        text = []
        for token in tokens:
            if token in self.special_tokens_decoder:
                text.append(self.special_tokens_decoder[token])
            else:
                text.append(self.decoder[token])
        text = ''.join(text)
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
