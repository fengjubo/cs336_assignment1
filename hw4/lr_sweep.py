import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()

    # sweep config
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--learning_rates", type=float, nargs="*", default=None)
    parser.add_argument(
        "--coarse_grid",
        type=float,
        nargs="*",
        default=[1e-4, 3e-4, 6e-4, 1e-3, 2e-3, 5e-3, 1e-2],
    )
    parser.add_argument("--skip_refine", action="store_true")
    parser.add_argument("--skip_edge", action="store_true")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--edge_multiplier", type=float, default=1.5)
    parser.add_argument("--max_edge_runs", type=int, default=5)

    # training args
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--valid_data", type=str, required=True)
    parser.add_argument("--data_dtype", type=str, default="uint16")
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument("--warmup_iters", type=int, default=None)
    parser.add_argument("--cosine_cycle_iters", type=int, default=None)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.99)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def dedupe_preserve_order(values: list[float]) -> list[float]:
    seen = set()
    out = []
    for value in values:
        rounded = f"{value:.12g}"
        if rounded in seen:
            continue
        seen.add(rounded)
        out.append(value)
    return out


def lr_to_name(lr: float) -> str:
    return f"{lr:.2e}".replace("+", "")


def get_metrics_summary(metrics_path: str) -> dict:
    final_valid_loss = None
    best_valid_loss = None
    status = "unknown"

    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            if event["split"] == "valid" and event["loss"] is not None:
                final_valid_loss = event["loss"]
                if best_valid_loss is None or event["loss"] < best_valid_loss:
                    best_valid_loss = event["loss"]
            if event["split"] == "summary":
                status = event["status"]

    return {
        "status": status,
        "final_valid_loss": final_valid_loss,
        "best_valid_loss": best_valid_loss,
    }


def run_single_lr(args, max_lr: float, stage: str) -> dict:
    run_name = f"{stage}_lr_{lr_to_name(max_lr)}"
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    warmup_iters = args.warmup_iters
    if warmup_iters is None:
        warmup_iters = max(1, int(args.max_iters * args.warmup_ratio))

    cosine_cycle_iters = args.cosine_cycle_iters
    if cosine_cycle_iters is None:
        cosine_cycle_iters = args.max_iters

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "training_together.py"),
        "--train_data",
        args.train_data,
        "--valid_data",
        args.valid_data,
        "--data_dtype",
        args.data_dtype,
        "--checkpoint_dir",
        run_dir,
        "--run_name",
        run_name,
        "--metrics_jsonl",
        metrics_path,
        "--vocab_size",
        str(args.vocab_size),
        "--context_length",
        str(args.context_length),
        "--d_model",
        str(args.d_model),
        "--num_layers",
        str(args.num_layers),
        "--num_heads",
        str(args.num_heads),
        "--d_ff",
        str(args.d_ff),
        "--rope_theta",
        str(args.rope_theta),
        "--batch_size",
        str(args.batch_size),
        "--max_iters",
        str(args.max_iters),
        "--max_lr",
        str(max_lr),
        "--min_lr",
        str(max_lr * args.min_lr_ratio),
        "--warmup_iters",
        str(warmup_iters),
        "--cosine_cycle_iters",
        str(cosine_cycle_iters),
        "--adam_beta1",
        str(args.adam_beta1),
        "--adam_beta2",
        str(args.adam_beta2),
        "--adam_eps",
        str(args.adam_eps),
        "--weight_decay",
        str(args.weight_decay),
        "--grad_clip",
        str(args.grad_clip),
        "--log_interval",
        str(args.log_interval),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_iters",
        str(args.eval_iters),
        "--save_interval",
        str(args.save_interval),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]

    print(f"[{stage}] running lr={max_lr:.6g}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Training run failed for lr={max_lr} with code {completed.returncode}")

    summary = get_metrics_summary(metrics_path)
    result = {
        "stage": stage,
        "run_name": run_name,
        "max_lr": max_lr,
        "min_lr": max_lr * args.min_lr_ratio,
        "status": summary["status"],
        "final_valid_loss": summary["final_valid_loss"],
        "best_valid_loss": summary["best_valid_loss"],
        "run_dir": run_dir,
        "metrics_jsonl": metrics_path,
    }
    print(
        f"[{stage}] lr={max_lr:.6g} | "
        f"status={result['status']} | "
        f"final_valid_loss={result['final_valid_loss']}"
    )
    return result


def best_stable_lr(results: list[dict]) -> float | None:
    stable = [r for r in results if r["status"] != "diverged" and r["final_valid_loss"] is not None]
    if not stable:
        return None
    stable.sort(key=lambda r: r["final_valid_loss"])
    return stable[0]["max_lr"]


def write_summary(output_root: str, results: list[dict]) -> None:
    json_path = os.path.join(output_root, "sweep_summary.json")
    csv_path = os.path.join(output_root, "sweep_summary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "stage",
        "run_name",
        "max_lr",
        "min_lr",
        "status",
        "final_valid_loss",
        "best_valid_loss",
        "run_dir",
        "metrics_jsonl",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


def main():
    args = get_args()
    os.makedirs(args.output_root, exist_ok=True)

    results = []
    executed_lrs = set()

    if args.learning_rates is not None and len(args.learning_rates) > 0:
        initial_lrs = dedupe_preserve_order(args.learning_rates)
    else:
        initial_lrs = dedupe_preserve_order(args.coarse_grid)

    for lr in initial_lrs:
        results.append(run_single_lr(args, lr, stage="coarse"))
        executed_lrs.add(f"{lr:.12g}")
        write_summary(args.output_root, results)

    best_lr = best_stable_lr(results)

    if best_lr is not None and not args.skip_refine and args.learning_rates is None:
        refine_lrs = dedupe_preserve_order([0.5 * best_lr, 0.75 * best_lr, best_lr, 1.25 * best_lr])
        for lr in refine_lrs:
            if f"{lr:.12g}" in executed_lrs:
                continue
            results.append(run_single_lr(args, lr, stage="refine"))
            executed_lrs.add(f"{lr:.12g}")
            write_summary(args.output_root, results)
        best_lr = best_stable_lr(results)

    if best_lr is not None and not args.skip_edge and args.learning_rates is None:
        edge_lrs = [best_lr, best_lr * args.edge_multiplier, best_lr * (args.edge_multiplier ** 2)]
        edge_lrs = dedupe_preserve_order(edge_lrs)
        edge_runs = 0

        for lr in edge_lrs:
            if edge_runs >= args.max_edge_runs:
                break
            if f"{lr:.12g}" in executed_lrs:
                continue
            results.append(run_single_lr(args, lr, stage="edge"))
            executed_lrs.add(f"{lr:.12g}")
            edge_runs += 1
            write_summary(args.output_root, results)

        while edge_runs < args.max_edge_runs and not any(r["status"] == "diverged" and r["stage"] == "edge" for r in results):
            next_lr = best_lr * (args.edge_multiplier ** (edge_runs + 2))
            if f"{next_lr:.12g}" in executed_lrs:
                edge_runs += 1
                continue
            results.append(run_single_lr(args, next_lr, stage="edge"))
            executed_lrs.add(f"{next_lr:.12g}")
            edge_runs += 1
            write_summary(args.output_root, results)

    write_summary(args.output_root, results)


if __name__ == "__main__":
    main()
