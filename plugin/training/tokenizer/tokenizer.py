from megatron.core.datasets.megatron_tokenizer import MegatronLegacyTokenizer

from plugin.training.tokenizer.gpt2_tokenization import AquilaTokenizer


class _AquilaTokenizerFS(MegatronLegacyTokenizer):
    """Aquila tokenizer."""

    def __init__(self, vocab_file, merge_file, special_tokens_file):
        super().__init__(vocab_file, merge_file, special_tokens_file)

        special_tokens = []
        if special_tokens_file:
            special_tokens = open(special_tokens_file, encoding='utf-8').read().split('\n')[:-1]

        self.tokenizer = AquilaTokenizer(vocab_file, merge_file, errors='replace',
                                            special_tokens=special_tokens, max_len=None)
        self.eod_id = self.tokenizer.encoder['</s>']
        self.cls_id = self.tokenizer.encoder['[CLS]']
        self.pad_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _HFTokenizerFS(MegatronLegacyTokenizer):
    """Huggingface tokenizer."""

    def __init__(self, tokenizer_path):
        name = 'HFTokenizer'
        super().__init__(name)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.eod_id = self.tokenizer.eos_token_id
        self.cls_id = self.tokenizer.bos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab()
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def cls(self):
        return self.cls_id

    @property
    def pad(self):
        return self.pad_id


class _Llama3TokenizerFS(_HFTokenizerFS):

    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + len(self.tokenizer.get_added_vocab())


class _QwenTokenizerFS(_HFTokenizerFS):
    """Adapted Qwen tokenizer."""
    
    def __init__(self, tokenizer_path):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]


class _HFTokenizersTokenizerFS(MegatronLegacyTokenizer):
    """Tokenizer from HuggingFace Tokenizers."""

    def __init__(self, json_file):
        super().__init__(json_file)

        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(json_file)

        print(f"Vocab size: {self.tokenizer.get_vocab_size()}")

        self.eod_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_id = self.tokenizer.token_to_id("<|padding|>")

        self._inv_vocab = None

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size() 

    @property
    def vocab(self):
        return self.tokenizer.get_vocab()

    @property
    def inv_vocab(self):
        # return self.tokenizer.decoder
        vocab = self.vocab()
        if self._inv_vocab is None:
            self._inv_vocab = {v: k for k, v in vocab.items()}
        return self._inv_vocab 

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id

    @property
    def pad(self):
        return self.pad_id


class _Qwen2TokenizerFS(_HFTokenizerFS):
    """Adapted Qwen tokenizer."""

    def __init__(self, tokenizer_path, args):
        super().__init__(tokenizer_path)
        self.eod_id = self.tokenizer.encode('<|extra_204|>')[0]
        self.cls_id = self.tokenizer.encode('<|extra_203|>')[0]
        self.pad_id = self.tokenizer.encode('<|endoftext|>')[0]
        assert args.vocab_size is not None
        self._vocab_size = args.vocab_size

    @property
    def vocab_size(self):
        return self._vocab_size


class _Qwen2VLTokenizer(MegatronLegacyTokenizer):
    def __init__(self, tokenizer_path, extra_vocab_size):
        super().__init__(tokenizer_path)
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            padding_side="right",
            use_fast=True,
            split_special_tokens=False,
            trust_remote_code=True,
            revision = "main",
            token = None,
        )
        self.extra_vocab_size = extra_vocab_size
        self.special_tokens_map = {k:v for k, v in zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids)}
        self.image_token = '<|image_pad|>'
        self.video_token = '<|video_pad|>'
        self.vision_start_token = '<|vision_start|>'
        self.vision_end_token = '<|vision_end|>'

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            tokenizer_path,
            revision = "main",
            token = None,
        )
        # NOTE: In Qwen2-VL, template in chat_template.json is same within tokenizer_config.json and both can be used.
        # However, in Qwen 2.5-VL, the two templates are different and only the one in chat_template.json is OK.
        self.chat_template = self.processor.chat_template

    def __call__(self, text, return_tensors=None,
                    padding=None, max_length=None, truncation=None, add_special_tokens=None):

        return self.tokenizer(text, return_tensors=return_tensors, padding=padding,
                max_length=max_length, truncation=truncation, add_special_tokens=add_special_tokens)

    def apply_chat_template(self, conversations, tokenize:bool=True, **kwargs):
        return self.tokenizer.apply_chat_template(conversations, tokenize=tokenize, chat_template=self.chat_template, **kwargs)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size + self.extra_vocab_size

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.tokenizer.eos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def image_token_id(self):
        return self.special_tokens_map[self.image_token]
    
    @property
    def video_token_id(self):
        return self.special_tokens_map[self.video_token]
    
    @property
    def vision_start_token_id(self):
        return self.special_tokens_map[self.vision_start_token]
    
    @property
    def vision_end_token_id(self):
        return self.special_tokens_map[self.vision_end_token]
    
    def encode(self, x):
        return self.tokenizer.encode(x)
