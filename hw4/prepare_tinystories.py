import argparse
import json
import os
import sys
import time
from array import array
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from hw1.tokenizer import Tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_txt", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="uint16")
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"])
    parser.add_argument("--flush_every", type=int, default=200000)
    parser.add_argument("--progress_every_lines", type=int, default=50000)
    return parser.parse_args()


def load_fixed_tokenizer(vocab_path: str, merges_path: str, special_tokens: list[str]) -> Tokenizer:
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)

    offset = len(special_tokens)
    for i in range(256):
        byte_val = bytes([i])
        token_id = offset + i
        tokenizer.id_to_token[token_id] = byte_val
        tokenizer.token_to_id[byte_val] = token_id

    return tokenizer


def get_array_code(dtype_name: str) -> str:
    dtype = np.dtype(dtype_name)
    if dtype == np.uint16:
        return "H"
    raise ValueError(f"Unsupported dtype for binary token export: {dtype_name}")


def main():
    args = get_args()
    dtype = np.dtype(args.dtype)
    max_token_id = np.iinfo(dtype).max
    array_code = get_array_code(args.dtype)

    tokenizer = load_fixed_tokenizer(args.vocab_path, args.merges_path, args.special_tokens)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_tokens = 0
    total_lines = 0
    start_time = time.time()
    buffer = array(array_code)

    with open(args.input_txt, "r", encoding="utf-8") as src, open(args.output_path, "wb") as dst:
        for line in src:
            token_ids = tokenizer.encode(line)
            if token_ids:
                line_max = max(token_ids)
                if line_max > max_token_id:
                    raise ValueError(
                        f"Token id {line_max} exceeds dtype {args.dtype} capacity ({max_token_id})"
                    )
                buffer.extend(token_ids)
                total_tokens += len(token_ids)

            total_lines += 1

            if len(buffer) >= args.flush_every:
                buffer.tofile(dst)
                buffer = array(array_code)

            if total_lines % args.progress_every_lines == 0:
                elapsed = time.time() - start_time
                print(
                    f"processed {total_lines} lines | "
                    f"{total_tokens} tokens | "
                    f"{elapsed:.2f}s"
                )

        if buffer:
            buffer.tofile(dst)

    meta = {
        "input_txt": args.input_txt,
        "output_path": args.output_path,
        "dtype": args.dtype,
        "total_tokens": total_tokens,
        "total_lines": total_lines,
        "special_tokens": args.special_tokens,
    }
    meta_path = f"{args.output_path}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"finished {args.output_path}")
    print(f"tokens: {total_tokens}")
    print(f"lines: {total_lines}")
    print(f"time: {elapsed:.2f}s")
    print(f"meta: {meta_path}")


if __name__ == "__main__":
    main()
