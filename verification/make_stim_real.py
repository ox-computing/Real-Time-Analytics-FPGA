#!/usr/bin/env python3
"""Generate stimulus + golden outputs for the DWN RTL tests, from real images.

Thermometer-encodes actual MNIST test images and writes their post-thermometer
bits, replacing the earlier random-vector generator. Real inputs are what the
hardware sees, and their (sparse) switching is what a power estimate must be
driven by -- random bits toggle far more and inflate it. The same vectors drive
the correctness tests, so `make sim`'s golden diff and the power benches share
one stimulus file.

Produces two files (in the current directory), same format as before:

  stim.hex      NUM_VECS input vectors, one hex line each, LSB of the hex word
                = pixels_thermo[0] ($readmemh format)
  expected.txt  per vector: "<predicted class> <score0> ... <score9>", computed
                by running the same bits through verification/hw_model.py

  python verification/make_stim_real.py [--model JSON] [-n NUM_VECS] [--data-root DIR]

Needs numpy + torchvision (conda env `dwn`). MNIST is loaded from --data-root
(default data/, downloaded if absent). Structural note: real images are sparse
and correlated, so they cover fewer LUT input addresses than random vectors did;
they still exercise the full pipeline, but a targeted-coverage check would want
random stimulus on top.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "verification"))
from hw_model import HWModel  # noqa: E402

DEFAULT_MODEL = REPO / "verification/exported/dwn_400_200_tau1.3.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("-n", "--num-vecs", type=int, default=200,
                        help="number of test images to encode (default 200)")
    parser.add_argument("--data-root", type=Path, default=REPO / "data")
    parser.add_argument("--out", type=Path, default=Path("stim.hex"))
    parser.add_argument("--expected", type=Path, default=Path("expected.txt"))
    args = parser.parse_args()

    m = HWModel(args.model)
    in_bits = m.meta["input_size"]
    hex_digits = (in_bits + 3) // 4

    # Raw MNIST test images, flattened to (N, 784) float in [0,1] -- the same
    # pipeline check_parity.py feeds the model.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.flatten(x)),
    ])
    test_set = datasets.MNIST(root=str(args.data_root), train=False,
                              download=True, transform=transform)
    n = min(args.num_vecs, len(test_set))
    images = torch.stack([test_set[i][0] for i in range(n)]).numpy()  # (n, 784)

    bits = m.binarize(images)                       # (n, in_bits) 0/1
    assert bits.shape[1] == in_bits

    with open(args.out, "w") as f:
        for row in bits:
            v = 0
            for i, b in enumerate(row):
                v |= int(b) << i
            f.write(f"{v:0{hex_digits}x}\n")

    # Golden reference: same pipeline as HWModel.infer, starting from the bits
    # just written -- so stim.hex and expected.txt are guaranteed consistent.
    x = bits
    for layer in m.layers:
        x = m._lut_layer(x, layer)
    scores = x.reshape(n, m.k, m.group_size).sum(axis=2)
    preds = scores.argmax(axis=1)

    with open(args.expected, "w") as f:
        for p, s in zip(preds, scores):
            f.write(str(p) + " " + " ".join(str(int(v)) for v in s) + "\n")

    print(f"wrote {args.out} + {args.expected} ({n} real MNIST vectors, "
          f"model {args.model.name}); mean bit density {bits.mean():.3f}")


if __name__ == "__main__":
    main()
