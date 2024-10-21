from .BPE import BPETokenizer

SUPPORTED_TOKENIZERS = {
    "bpe": BPETokenizer
}

def build_tokenizer(args):
    if args["args"].pretrained_path:
        if args.tokenizer_type in SUPPORTED_TOKENIZERS:
            return SUPPORTED_TOKENIZERS[args.tokenizer_type].from_pretrained(args["args"].pretrained_path)
        else:
            raise ValueError(f"Tokenizer type {args.tokenizer_type} not supported")
    else:
        print("Building tokenizer from scratch")
        if args.tokenizer_type in SUPPORTED_TOKENIZERS:
            tokenizer = SUPPORTED_TOKENIZERS[args.tokenizer_type](args["args"])
            tokenizer.build_vocab()
        else:
            raise ValueError(f"Tokenizer type {args.tokenizer_type} not supported")
        