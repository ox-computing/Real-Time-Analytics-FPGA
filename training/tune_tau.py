"""Sweep the GroupSum softmax temperature (tau) for a fixed DWN architecture.

Once the architecture sweep has picked a config, this tunes tau for it, since
the paper flags the GroupSum temperature as crucial for accuracy -- and smaller
models are more tau-sensitive, so the value cannot just be inherited. Selection
is on the validation set only; the 10k test set is reported for reference but
never used to pick tau. The best tau's checkpoint is saved, and that checkpoint
is what gets exported to RTL.

The architecture is given as a layer-width list (--layers 400 200), so this
works at any depth.

Shared data/model/training logic is imported from train_mnist so the two stay
in lockstep (same seeding, thermometer fit, split, and training loop).
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_mnist import set_seed, load_mnist, build_model, train_one


def plot_tau(results, arch_label, out_path):
    taus = [r["tau"] for r in results]
    best_val = [r["best_val_acc"] for r in results]
    best_test = [r["best_test_acc"] for r in results]

    plt.figure(figsize=(7, 4.5))
    plt.plot(taus, best_val, "o-", label="best val")
    plt.plot(taus, best_test, "s--", label="best test", alpha=0.6)
    # Mark the selected (best-val) point.
    bi = max(range(len(results)), key=lambda i: results[i]["best_val_acc"])
    plt.scatter([taus[bi]], [best_val[bi]], s=140, facecolors="none",
                edgecolors="red", linewidths=2, zorder=5, label="selected")
    plt.xlabel("tau (GroupSum temperature)")
    plt.ylabel("accuracy")
    plt.title(f"tau sweep - DWN MNIST ({arch_label}; z=3, n=4)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="tau sweep for one DWN MNIST architecture")
    parser.add_argument("--tau-values", type=str, default="1,1.2,1.4,1.6,1.8,2",
                        help="comma-separated tau values to try")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        metavar="WIDTH",
                        help="LUT-layer widths, e.g. --layers 400 200 (any depth)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-bits", type=int, default=3)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--results-root", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    tau_values = [float(t) for t in args.tau_values.split(",")]

    if args.layers[-1] % 10:
        raise SystemExit(f"final layer {args.layers[-1]} must divide into 10 GroupSum classes")
    arch_slug = "_".join(str(s) for s in args.layers)
    arch_label = ", ".join(str(s) for s in args.layers)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.results_root) / f"tau-sweep_{arch_slug}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Architecture: {arch_label}  ({len(args.layers)} LUT layers, n={args.n})")
    print(f"tau values: {tau_values}")

    # Data is identical across tau values, so load it once (seed first for the split).
    set_seed(args.seed)
    train_set, val_set, test_set, thermometer = load_mnist(
        args.data_root, args.num_bits, args.val_size, args.seed)
    input_size = train_set[0].size(1)
    torch.save(thermometer, run_dir / "thermometer.pt")

    results = []
    best = None
    for tau in tau_values:
        print(f"\n=== tau = {tau} ===")
        # Re-seed per run so only tau differs between models.
        set_seed(args.seed)
        model = build_model(input_size, args.layers, args.n, tau).cuda()

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard" / f"tau_{tau}"))
        t0 = time.time()
        history = train_one(model, train_set, val_set, test_set,
                            args.epochs, args.batch_size, args.lr, writer=writer)
        elapsed = time.time() - t0
        writer.close()

        best_val = max(history["val_acc"])
        best_test = max(history["test_acc"])
        record = {
            "tau": tau,
            "best_val_acc": best_val,
            "best_test_acc": best_test,
            "final_val_acc": history["val_acc"][-1],
            "final_test_acc": history["test_acc"][-1],
            "elapsed_sec": round(elapsed, 1),
            "history": history,
        }
        with open(run_dir / f"tau_{tau}_metrics.json", "w") as f:
            json.dump(record, f, indent=2)
        results.append(record)
        print(f"    tau={tau}  best_val={best_val:.4f}  best_test={best_test:.4f}  ({elapsed:.1f}s)")

        # Keep the checkpoint of the best-by-validation model only.
        if best is None or best_val > best["best_val_acc"]:
            best = record
            torch.save(model.state_dict(), run_dir / "best_checkpoint.pt")

    plot_tau(results, arch_label, run_dir / "tau_sweep.png")

    summary = {
        "run_id": run_id,
        "architecture": {"layer_sizes": args.layers, "num_layers": len(args.layers),
                         "n": args.n, "num_bits": args.num_bits},
        "args": vars(args),
        "selected_tau": best["tau"],
        "selected_best_val_acc": best["best_val_acc"],
        "selected_best_test_acc": best["best_test_acc"],
        "results": [{k: r[k] for k in (
            "tau", "best_val_acc", "best_test_acc", "final_test_acc", "elapsed_sec")}
            for r in results],
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== tau sweep summary ===")
    for r in results:
        marker = "  <- selected" if r["tau"] == best["tau"] else ""
        print(f"  tau={r['tau']:<6} best_val={r['best_val_acc']:.4f}  "
              f"best_test={r['best_test_acc']:.4f}{marker}")
    print(f"\nSelected tau={best['tau']} "
          f"(val {best['best_val_acc']:.4f}, test {best['best_test_acc']:.4f})")
    print(f"Best checkpoint + plots written to: {run_dir}")


if __name__ == "__main__":
    main()
