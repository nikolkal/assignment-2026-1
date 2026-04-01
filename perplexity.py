import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "facebook/opt-125m"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("out_file")
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--begin-context-tokens", type=int, default=512)
    return parser.parse_args()


def load_text_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        tie_word_embeddings=False
    )
    model.eval()
    return tokenizer, model


def count_windows(num_tokens, n_ctx, stride, begin_context_tokens):
    if num_tokens <= 0:
        return 0

    if begin_context_tokens <= 0:
        raise ValueError("begin_context_tokens must be positive")

    if stride <= 0:
        raise ValueError("stride must be positive")

    first_predicted_tokens = min(begin_context_tokens, num_tokens)
    remaining_tokens = num_tokens - first_predicted_tokens

    windows = 1
    if remaining_tokens > 0:
        windows += (remaining_tokens + stride - 1) // stride

    return windows


def main():
    args = get_args()

    print(f"Computing perplexity for {args.input_file}...")
    print("Tokenizing text...")

    text = load_text_file(args.input_file)
    tokenizer, model = initialize_model()

    tokens = tokenizer(text).input_ids
    num_tokens = len(tokens)

    print(f"Found {num_tokens} tokens")

    window_count = count_windows(
        num_tokens,
        args.n_ctx,
        args.stride,
        args.begin_context_tokens,
    )

    print(f"Processing {num_tokens} tokens in {window_count} window(s).")

    # προσωρινό output file
    Path(args.out_file).write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
