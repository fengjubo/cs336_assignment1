import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def get_args():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--data_dtype", type=str, default="uint16")

    # checkpoint
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    # model
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # optimization
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # logging and evaluation
    parser.add_argument("--run_name", type=str, default="run")
    parser.add_argument("--metrics_jsonl", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)

    # runtime
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_memmap(path, dtype):
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def append_metric(metrics_path: str | None, payload: dict) -> None:
    if metrics_path is None:
        return
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def grads_are_finite(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return False
    return True


@torch.no_grad()
def estimate_valid_loss(model, valid_data, args):
    from hw3.training_blocks import cross_entropy
    from hw4.loop_blocks import data_loading

    model.eval()

    losses = []
    for _ in range(args.eval_iters):
        x, y = data_loading(
            valid_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        logits = model(x)

        bsz, seq_len, vocab = logits.shape
        loss = cross_entropy(
            logits.reshape(bsz * seq_len, vocab),
            y.reshape(bsz * seq_len),
        )
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


def main():
    args = get_args()
    args.device = resolve_device(args.device)

    from hw2.blocks import transformer_lm
    from hw3.training_blocks import (
        AdamW,
        cross_entropy,
        gradient_clipping,
        initialize_transformer_lm_for_training,
        learning_rate_schedule,
    )
    from hw4.loop_blocks import data_loading, load_checkpoint, save_checkpoint

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.metrics_jsonl is None:
        args.metrics_jsonl = os.path.join(args.checkpoint_dir, f"{args.run_name}_metrics.jsonl")

    train_data = load_memmap(args.train_data, args.data_dtype)
    valid_data = load_memmap(args.valid_data, args.data_dtype)

    model = transformer_lm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=args.device,
    )
    initialize_transformer_lm_for_training(model, args.d_model, args.d_ff, device=args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume is not None:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from checkpoint {args.resume} at iteration {start_iter}")

    model.train()
    start_time = time.time()
    status = "completed"
    last_completed_iter = start_iter

    print(f"Using device: {args.device}")

    for it in range(start_iter, args.max_iters):
        lr = learning_rate_schedule(
            it=it,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.cosine_cycle_iters,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loading(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )

        logits = model(x)

        bsz, seq_len, vocab = logits.shape
        loss = cross_entropy(
            logits.reshape(bsz * seq_len, vocab),
            y.reshape(bsz * seq_len),
        )

        if not torch.isfinite(loss):
            status = "diverged"
            elapsed = time.time() - start_time
            print(f"iter {it} | non-finite train loss encountered; stopping run")
            append_metric(
                args.metrics_jsonl,
                {
                    "run_name": args.run_name,
                    "iter": it,
                    "split": "train",
                    "loss": None,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "status": status,
                    "event": "non_finite_loss",
                },
            )
            break

        optimizer.zero_grad()
        loss.backward()

        if not grads_are_finite(model):
            status = "diverged"
            elapsed = time.time() - start_time
            print(f"iter {it} | non-finite gradients encountered; stopping run")
            append_metric(
                args.metrics_jsonl,
                {
                    "run_name": args.run_name,
                    "iter": it,
                    "split": "train",
                    "loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "status": status,
                    "event": "non_finite_grad",
                },
            )
            break

        if args.grad_clip is not None:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()
        last_completed_iter = it + 1

        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            append_metric(
                args.metrics_jsonl,
                {
                    "run_name": args.run_name,
                    "iter": it,
                    "split": "train",
                    "loss": float(loss.item()),
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "status": "running",
                    "event": "log",
                },
            )
            print(
                f"iter {it} | "
                f"train loss {loss.item():.4f} | "
                f"lr {lr:.6e} | "
                f"time {elapsed:.2f}s"
            )

        if it % args.eval_interval == 0 and it > 0:
            valid_loss = estimate_valid_loss(model, valid_data, args)
            elapsed = time.time() - start_time
            append_metric(
                args.metrics_jsonl,
                {
                    "run_name": args.run_name,
                    "iter": it,
                    "split": "valid",
                    "loss": float(valid_loss),
                    "lr": lr,
                    "elapsed_sec": elapsed,
                    "status": "running",
                    "event": "eval",
                },
            )
            print(f"iter {it} | valid loss {valid_loss:.4f}")

        if it % args.save_interval == 0 and it > 0:
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            step_path = os.path.join(args.checkpoint_dir, f"ckpt_{it}.pt")

            save_checkpoint(model, optimizer, it, latest_path)
            save_checkpoint(model, optimizer, it, step_path)

            print(f"saved checkpoint to {latest_path}")

    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, last_completed_iter, final_path)
    append_metric(
        args.metrics_jsonl,
        {
            "run_name": args.run_name,
            "iter": last_completed_iter,
            "split": "summary",
            "loss": None,
            "lr": None,
            "elapsed_sec": time.time() - start_time,
            "status": status,
            "event": "finished",
        },
    )
    print(f"saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
