import argparse
from pathlib import Path
import math

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
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        tie_word_embeddings=False,
        local_files_only=True,
        use_safetensors=False
    )
    model.eval()
    return tokenizer, model


def build_windows(num_tokens, n_ctx, stride, begin_context_tokens):
    if num_tokens <= 0:
        return []

    if n_ctx <= 0:
        raise ValueError("n_ctx must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if begin_context_tokens <= 0:
        raise ValueError("begin_context_tokens must be positive")

    windows = []

    first_end = min(n_ctx, num_tokens)
    first_target_len = min(begin_context_tokens, first_end)
    first_target_start = first_end - first_target_len

    windows.append({
        "start": 0,
        "end": first_end,
        "target_start": first_target_start,
        "target_end": first_end,
    })

    predicted_until = first_end

    while predicted_until < num_tokens:
        target_start = predicted_until
        target_end = min(predicted_until + stride, num_tokens)

        end = target_end
        start = max(0, end - n_ctx)

        windows.append({
            "start": start,
            "end": end,
            "target_start": target_start,
            "target_end": target_end,
        })

        predicted_until = target_end

    return windows


def compute_window_nll(
    window_tokens,
    target_start_in_window,
    target_end_in_window,
    model,
    bos_token,
):
    model_input = [bos_token] + window_tokens
    window_tensor = torch.tensor([model_input], dtype=torch.long)

    with torch.no_grad():
        logits = model(window_tensor).logits

    total_nll = 0.0

    for token_pos in range(target_start_in_window, target_end_in_window):
        row_index = token_pos
        row = logits[0, row_index].tolist()

        max_val = max(row)
        shifted = [x - max_val for x in row]
        log_sum_exp = math.log(sum(math.exp(x) for x in shifted))
        log_probs = [x - log_sum_exp for x in shifted]

        target_token = window_tokens[token_pos]
        total_nll += -log_probs[target_token]

    return total_nll


def main():
    args = get_args()

    text = load_text_file(args.input_file)
    tokenizer, model = initialize_model()
    bos_token = tokenizer.bos_token_id

    tokens = tokenizer(text).input_ids

    if tokens and tokens[0] == bos_token:
        tokens = tokens[1:]

    num_tokens = len(tokens)

    effective_n_ctx = args.n_ctx - 1
    if effective_n_ctx <= 0:
        raise ValueError("n_ctx must be at least 2 so there is room for BOS")

    windows = build_windows(
        num_tokens,
        effective_n_ctx,
        args.stride,
        args.begin_context_tokens,
    )

    total_nll = 0.0
    total_predicted_tokens = 0

    for window in windows:
        window_tokens = tokens[window["start"]:window["end"]]

        target_start_in_window = window["target_start"] - window["start"]
        target_end_in_window = window["target_end"] - window["start"]
        predicted_tokens = target_end_in_window - target_start_in_window

        window_nll = compute_window_nll(
            window_tokens,
            target_start_in_window,
            target_end_in_window,
            model,
            bos_token,
        )

        total_nll += window_nll
        total_predicted_tokens += predicted_tokens

    if total_predicted_tokens == 0:
        raise ValueError("No tokens were predicted; check input and parameters")

    mean_nll = total_nll / total_predicted_tokens
    perplexity = math.exp(mean_nll)

    print(f"Perplexity: {perplexity:.2f}")

    Path(args.out_file).write_text(
        f"Perplexity: {perplexity:.2f}\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
