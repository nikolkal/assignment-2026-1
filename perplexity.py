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


def read_input_file(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return path.read_text(encoding="utf-8")


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        tie_word_embeddings=False
    )
    model.eval()
    return tokenizer, model


def main():
    args = get_args()

    print(f"Computing perplexity for {args.input_file}...")
    print("Tokenizing text...")

    text = read_input_file(args.input_file)
    tokenizer, model = load_model()

    tokens = tokenizer(text).input_ids

    print(f"Found {len(tokens)} tokens")

    # προσωρινό output file
    Path(args.out_file).write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
