import argparse
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from hw1.tokenizer import Tokenizer
from hw2.blocks import transformer_lm
from hw3.training_blocks import initialize_transformer_lm_for_training
from hw5.decoding import generate


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", default="outputs/tinystories_vocab.json")
    parser.add_argument("--merges", default="outputs/tinystories_merges.txt")
    parser.add_argument("--prompt", default="<|endoftext|>")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])
    endoftext_id = tokenizer.token_to_id[b"<|endoftext|>"]

    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    )
    initialize_transformer_lm_for_training(model, args.d_model, args.d_ff, device=device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    input_ids = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        context_length=args.context_length,
        temperature=args.temperature,
        top_p=args.top_p,
        endoftext_id=endoftext_id,
    )[0].tolist()

    new_ids = output_ids[len(input_ids[0]) :]
    if new_ids and new_ids[-1] == endoftext_id:
        new_ids = new_ids[:-1]

    print(tokenizer.decode(new_ids))
    print("\n---")
    print(f"generated_tokens={len(new_ids)}")
    print(f"checkpoint_iteration={checkpoint.get('iteration')}")
    print(f"temperature={args.temperature}")
    print(f"top_p={args.top_p}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
