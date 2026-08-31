"""Check the pure-numpy hardware model against the live PyTorch DWN.

Runs both on identical raw MNIST test images and asserts their predicted
classes match exactly, layer by layer. This separates two concerns: if this
passes, training correctness and the export are sound, so any later RTL mismatch
must be in the Verilog itself -- not in the weights or the reference math.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms

import torch_dwn as dwn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw_model import HWModel


def build_torch_model(input_size, layer_sizes, lut_n, k, tau):
    """Mirror of training/train_mnist.py's build_model: learnable mapping on the
    first LUT layer, fixed random mapping on the rest, then GroupSum."""
    layers, in_size = [], input_size
    for i, size in enumerate(layer_sizes):
        mapping = "learnable" if i == 0 else "random"
        layers.append(dwn.LUTLayer(in_size, size, n=lut_n, mapping=mapping))
        in_size = size
    layers.append(dwn.GroupSum(k=k, tau=tau))
    return nn.Sequential(*layers)


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="PyTorch vs hardware-model parity check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thermometer", required=True)
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--tau", type=float, default=1.6,
                        help="only rescales the scores; any value gives the same "
                             "comparison, since it is applied and undone here")
    parser.add_argument("--data-root", type=str, default=str(repo / "data"))
    args = parser.parse_args()

    # Architecture comes from the export's meta -- never from a flag. (lut_n, not
    # n: `n` is the sample count below.)
    with open(args.model_json) as f:
        meta = json.load(f)["meta"]
    layer_sizes, lut_n, k = meta["layer_sizes"], meta["n"], meta["k"]

    # Raw MNIST test images (no thermometer yet): (N, 784) float in [0,1].
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    test_set = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
    n = min(args.num_samples, len(test_set))
    x_raw = torch.stack([test_set[i][0] for i in range(n)])   # (n, 784)
    y_true = torch.tensor([test_set[i][1] for i in range(n)]).numpy()

    # --- PyTorch path: capture every intermediate, not just the prediction ---
    thermometer = torch.load(args.thermometer, map_location="cpu", weights_only=False)
    input_size = thermometer.thresholds.shape[0] * thermometer.num_bits
    assert input_size == meta["input_size"], \
        f"thermometer input {input_size} != exported input {meta['input_size']}"
    model = build_torch_model(input_size, layer_sizes, lut_n, k, args.tau).cuda()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cuda"))
    model.eval()
    with torch.no_grad():
        x_bin = thermometer.binarize(x_raw).flatten(start_dim=1).cuda()
        # Step through the LUT layers one at a time, keeping each layer's 0/1
        # output (post-STE), then GroupSum -- model[-1] -- on the last of them.
        t_layer_bits, x = [], x_bin
        for layer in model[:-1]:
            x = layer(x)
            t_layer_bits.append(x.cpu().numpy().astype(np.int64))
        raw_scores = model[-1](x)                  # (n, k) == popcount / tau
        # Recover integer popcounts from the tau-scaled scores.
        t_scores = np.rint(raw_scores.cpu().numpy() * args.tau).astype(np.int64)
        torch_pred = raw_scores.argmax(dim=1).cpu().numpy()

    # --- Hardware model path (pure numpy, from raw images) ---
    hw = HWModel(args.model_json)
    out = hw.infer(x_raw.numpy(), return_all=True)
    h_layer_bits = out["layer_bits"]
    h_scores, hw_pred = out["scores"], out["preds"]
    assert len(h_layer_bits) == len(t_layer_bits), "layer count disagrees between the two models"

    # --- Bit-level comparison at every stage ---
    def report(name, a, b):
        eq = (a == b)
        n_bad = int((~eq).any(axis=1).sum())          # samples with >=1 mismatch
        n_bits_bad = int((~eq).sum())                 # total mismatched elements
        print(f"  {name:<24} samples OK: {n - n_bad}/{n}   "
              f"elements OK: {a.size - n_bits_bad}/{a.size}")
        return n_bad == 0

    print(f"Samples compared:      {n}")
    print(f"Architecture:          {'/'.join(str(s) for s in layer_sizes)} "
          f"(n={lut_n}, k={k}, {len(layer_sizes)} LUT layers)")
    print(f"PyTorch test accuracy: {100.0 * (torch_pred == y_true).mean():.2f}%")
    print(f"Hardware test accuracy:{100.0 * (hw_pred == y_true).mean():.2f}%")
    print("\nBit-level parity (most rigorous -> least):")

    # Ordered earliest-stage-first, so the first failure is the earliest divergence
    # and thus where any bug actually lives.
    stages = [(f"layer{i + 1} bits ({size})", h, t)
              for i, (size, h, t) in enumerate(zip(layer_sizes, h_layer_bits, t_layer_bits))]
    stages.append((f"popcount scores ({k})", h_scores, t_scores))
    stages.append(("predictions (1)", hw_pred.reshape(-1, 1), torch_pred.reshape(-1, 1)))

    results = [(name, report(name, h, t)) for name, h, t in stages]
    if not all(ok for _, ok in results):
        for name, ok in results:
            if not ok:
                print(f"\nFIRST DIVERGENCE at: {name}")
                break
        raise SystemExit(1)

    print("\nPARITY OK: hardware model matches PyTorch bit-for-bit at every layer.")


if __name__ == "__main__":
    main()
