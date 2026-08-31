"""Train DWN models on MNIST across several architecture sizes.

Downward architecture sweep for the iCE40 (LUT-4) target. The 1500/750 model
routes badly: nextpnr overuse is 67% concentrated in the pixels -> layer1 input
crossbar, which is a random 1093 -> 1500 permutation carrying layer1*n = 6000
nets. Placement, pipelining, relabeling, and BRAM-backed inputs all relocate
that permutation rather than remove it, and n is already at the fabric optimum
(n=4 maps 1:1 onto a LUT4; n=6 costs 9.8 LUT4s/node under synth_ice40). The one
remaining lever on crossbar congestion is the net count itself, which is linear
in layer1. So this sweep walks the layer sizes *down* to find the smallest model
with acceptable accuracy, and thus the fewest input nets.

Fixed choices for this sweep:
  - z = 3 thermometer bits per pixel (input width = 28*28*3 = 2352 bits)
  - n = 4 inputs per LUT, to match the iCE40 LUT-4 fabric one-to-one
  - learnable mapping on the first layer only (random mapping afterwards)
  - each layer half the width of the one before, holding the topology's shape
    fixed so the only variable is scale
  - tau = 1.6, the value chosen by the earlier tau sweep. The first architecture
    sweep ran at a tau=6.25 placeholder; small models are more tau-sensitive, so
    reusing that here would understate them.

Depth is a free variable, not a fixed choice: a config is a list of layer widths,
so `--configs 400-200-100` trains a 3-layer model. The default list below is the
original 2-layer width sweep, which is what the shipped model came from. Note that
only the first layer's width drives input-crossbar nets (layer1 * n), so depth is
close to free on the axis that actually blocks routing -- which is the point of
being able to vary it.

Accuracy at the small end is what decides the design, so configs are trained for
enough epochs to separate a real accuracy cliff from an undertrained one.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt

import torch_dwn as dwn


# --- Reproducibility --------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix every RNG that influences this pipeline.

    Covers Python's RNG, NumPy (used by the thermometer fit and any shuffling),
    and both PyTorch CPU and CUDA generators (weight init, learnable-mapping
    init, batch permutation). Note: the EFD backward pass is a custom CUDA
    kernel and may still use non-deterministic atomics, so run-to-run results
    can differ slightly even with this set. The saved checkpoint -- not a
    re-run -- is the artifact we later verify bit-exact against the RTL.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Data -------------------------------------------------------------------

def load_mnist(data_root: str, num_bits: int, val_size: int, seed: int):
    """Load MNIST, split off a validation set, and thermometer-binarize.

    The 10k canonical test set is left untouched. A validation set is carved
    from the 60k training split so architecture selection never peeks at test.
    The thermometer is fit on the training portion only.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Materialise the whole dataset as tensors (MNIST is small enough).
    x_train_full = torch.stack([train_full[i][0] for i in range(len(train_full))])
    y_train_full = torch.tensor([train_full[i][1] for i in range(len(train_full))])
    x_test = torch.stack([test_set[i][0] for i in range(len(test_set))])
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Deterministic train/val split.
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x_train_full.size(0), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]
    x_train, y_train = x_train_full[train_idx], y_train_full[train_idx]
    x_val, y_val = x_train_full[val_idx], y_train_full[val_idx]

    # Fit the thermometer on the training portion only, then binarize all splits.
    thermometer = dwn.DistributiveThermometer(num_bits).fit(x_train)
    x_train = thermometer.binarize(x_train).flatten(start_dim=1)
    x_val = thermometer.binarize(x_val).flatten(start_dim=1)
    x_test = thermometer.binarize(x_test).flatten(start_dim=1)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test), thermometer


# --- Model ------------------------------------------------------------------

def build_model(input_size: int, layer_sizes: list, n: int, tau: float) -> nn.Module:
    """A stack of LUT layers followed by GroupSum, at any depth.

    Only the first layer gets a learnable mapping -- it is the one reading the
    thermometer-encoded image, where which input bits a LUT sees is worth
    learning. Deeper layers read another LUT layer's output, whose bits carry no
    inherent order, so a fixed random mapping is as good and costs no training.
    Mirrored by verification/check_parity.py's build_torch_model; the two must
    agree or the parity check compares different architectures.
    """
    layers, in_size = [], input_size
    for i, size in enumerate(layer_sizes):
        layers.append(dwn.LUTLayer(in_size, size, n=n,
                                   mapping="learnable" if i == 0 else "random"))
        in_size = size
    layers.append(dwn.GroupSum(k=10, tau=tau))
    return nn.Sequential(*layers)


def parse_configs(spec: str) -> list:
    """'400-200,600-300-150' -> [('400_200', [400, 200]), ('600_300_150', [...])].

    The name doubles as the results subdirectory, so it uses underscores to match
    the existing run layout (results/mnist-sweep_*/400_200/).
    """
    configs = []
    for item in spec.split(","):
        sizes = [int(s) for s in item.strip().split("-") if s]
        if not sizes:
            raise SystemExit(f"empty config in --configs: {spec!r}")
        configs.append(("_".join(str(s) for s in sizes), sizes))
    return configs


# --- Train / evaluate -------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
    model.eval()
    correct = 0
    for i in range(0, x.size(0), batch_size):
        logits = model(x[i:i + batch_size].cuda())
        correct += (logits.argmax(dim=1).cpu() == y[i:i + batch_size]).sum().item()
    return correct / y.size(0)


def train_one(model, train_set, val_set, test_set, epochs, batch_size, lr, writer=None):
    """Train a single model, returning per-epoch history and final accuracies.

    If `writer` (a SummaryWriter) is given, per-epoch scalars are logged live so
    runs can be watched in TensorBoard instead of waiting for the final plot.
    """
    x_train, y_train = train_set
    x_val, y_val = val_set
    x_test, y_test = test_set

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Staged decay approximating the paper's schedule (1e-2 -> 1e-3 -> 1e-4 -> 1e-5).
    milestones = [int(epochs * 0.3), int(epochs * 0.6), int(epochs * 0.9)]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    history = {"train_acc": [], "val_acc": [], "test_acc": [], "loss": []}
    n_samples = x_train.size(0)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(n_samples)
        correct = 0
        last_loss = 0.0
        for i in range(0, n_samples, batch_size):
            idx = permutation[i:i + batch_size]
            batch_x, batch_y = x_train[idx].cuda(), y_train[idx].cuda()
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = cross_entropy(logits, batch_y)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            last_loss = loss.item()
        scheduler.step()

        train_acc = correct / n_samples
        val_acc = accuracy(model, x_val, y_val, batch_size)
        test_acc = accuracy(model, x_test, y_test, batch_size)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["test_acc"].append(test_acc)
        history["loss"].append(last_loss)
        if writer is not None:
            writer.add_scalar("accuracy/train", train_acc, epoch)
            writer.add_scalar("accuracy/val", val_acc, epoch)
            writer.add_scalar("accuracy/test", test_acc, epoch)
            writer.add_scalar("loss", last_loss, epoch)
        print(f"    epoch {epoch + 1:3d}/{epochs}  loss {last_loss:.4f}  "
              f"train {train_acc:.4f}  val {val_acc:.4f}  test {test_acc:.4f}")

    return history


# --- Plotting ---------------------------------------------------------------

def plot_curve(history, title, out_path):
    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.plot(epochs, history["test_acc"], label="test")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_comparison(results, out_path):
    """Best validation accuracy vs total LUT-node count for each config."""
    names = [r["name"] for r in results]
    luts = [r["total_luts"] for r in results]
    best_val = [max(r["history"]["val_acc"]) for r in results]

    # Sort by size so the line reads left-to-right even when configs are given
    # in an arbitrary order (a depth sweep is not monotonic in node count).
    order = sorted(range(len(results)), key=lambda i: luts[i])
    names = [names[i] for i in order]
    luts = [luts[i] for i in order]
    best_val = [best_val[i] for i in order]

    plt.figure(figsize=(7, 4.5))
    plt.plot(luts, best_val, "o-")
    for name, x, y in zip(names, luts, best_val):
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=9)
    plt.xlabel("total DWN LUT nodes (all layers)")
    plt.ylabel("best validation accuracy")
    plt.title("MNIST DWN architecture sweep (z=3, n=4)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


# --- Main -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DWN MNIST architecture sweep")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num-bits", type=int, default=3, help="thermometer bits per pixel (z)")
    parser.add_argument("--n", type=int, default=4, help="inputs per LUT (LUT-n)")
    parser.add_argument("--tau", type=float, default=1.6,
                        help="GroupSum softmax temperature (from the tau sweep)")
    parser.add_argument("--val-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "data"))
    parser.add_argument("--results-root", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "results"))
    # Width sweep by default: 1500/750 is the baseline that fails to route and
    # the rest walk down from it. Input-crossbar nets = layer1 * n, so the first
    # layer is the axis that matters -- 6000 nets at 1500, 1600 at 400. Override
    # to sweep depth instead, e.g. --configs 400-200,400-200-100,400-200-100-50.
    parser.add_argument("--configs", type=str,
                        default="1500-750,1200-600,1000-500,800-400,600-300,400-200",
                        help="comma-separated layer stacks, widths joined by '-' "
                             "(e.g. '400-200,400-200-100')")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    configs = parse_configs(args.configs)
    for name, sizes in configs:
        if sizes[-1] % 10:
            raise SystemExit(f"config {name}: final layer {sizes[-1]} must divide "
                             f"into 10 GroupSum classes")

    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.results_root) / f"mnist-sweep_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Seed once, then load and binarize data (shared thermometer across configs).
    set_seed(args.seed)
    train_set, val_set, test_set, thermometer = load_mnist(
        args.data_root, args.num_bits, args.val_size, args.seed)
    input_size = train_set[0].size(1)
    print(f"Input size after z={args.num_bits} thermometer: {input_size} bits")

    # Save the thermometer so binarization can be reproduced exactly at verification time.
    torch.save(thermometer, run_dir / "thermometer.pt")

    results = []
    for name, layer_sizes in configs:
        print(f"\n=== config {name}  (LUTs: {', '.join(str(s) for s in layer_sizes)}; "
              f"n={args.n}, tau={args.tau:.4f}) ===")
        # Re-seed before each build+train so configs are compared on equal footing.
        set_seed(args.seed)
        model = build_model(input_size, layer_sizes, args.n, args.tau).cuda()

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard" / name))
        t0 = time.time()
        history = train_one(model, train_set, val_set, test_set,
                            args.epochs, args.batch_size, args.lr, writer=writer)
        elapsed = time.time() - t0
        writer.close()

        config_dir = run_dir / name
        config_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), config_dir / "checkpoint.pt")
        plot_curve(history, f"MNIST DWN {name} (z={args.num_bits}, n={args.n})",
                   config_dir / "training_curve.png")

        record = {
            "name": name,
            "layer_sizes": layer_sizes,
            "num_layers": len(layer_sizes),
            "total_luts": sum(layer_sizes),
            "n": args.n,
            "num_bits": args.num_bits,
            "tau": args.tau,
            "input_size": input_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "elapsed_sec": round(elapsed, 1),
            "final_train_acc": history["train_acc"][-1],
            "final_val_acc": history["val_acc"][-1],
            "final_test_acc": history["test_acc"][-1],
            "best_val_acc": max(history["val_acc"]),
            "best_test_acc": max(history["test_acc"]),
            "history": history,
        }
        with open(config_dir / "metrics.json", "w") as f:
            json.dump(record, f, indent=2)
        results.append(record)
        print(f"    done in {elapsed:.1f}s  best val {record['best_val_acc']:.4f}  "
              f"best test {record['best_test_acc']:.4f}")

    # Cross-config summary and comparison plot.
    plot_comparison(results, run_dir / "comparison.png")
    summary = [{k: r[k] for k in (
        "name", "layer_sizes", "num_layers", "total_luts", "tau",
        "best_val_acc", "best_test_acc", "final_test_acc", "elapsed_sec")} for r in results]
    with open(run_dir / "summary.json", "w") as f:
        json.dump({"run_id": run_id, "args": vars(args), "configs": summary}, f, indent=2)

    print("\n=== summary ===")
    for s in summary:
        print(f"  {s['name']:<10} LUTs={s['total_luts']:<5} "
              f"best_val={s['best_val_acc']:.4f}  best_test={s['best_test_acc']:.4f}")
    print(f"\nResults written to: {run_dir}")


if __name__ == "__main__":
    main()
