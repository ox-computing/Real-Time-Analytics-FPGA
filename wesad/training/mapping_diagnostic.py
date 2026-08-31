"""What does the learnable layer-1 mapping actually converge to?

A DWN's mapping is its only inductive bias, and it is free in silicon -- so before
constraining it, it is worth knowing what it picks on its own. This trains the locked
config the way --fit-all does (all subjects, no LOSO) at several seeds and reports, per
seed, how layer 1 spends its 400 pins: how many of the input bits it reads at all, which
features it never reads, how its fan-in is distributed across the modalities against
their share of the feature count, and how many modalities a LUT4 spans.

The reference point is torch-dwn's own random mapping, which is a truncated randperm and
therefore reads 400 DISTINCT bits -- not an i.i.d. draw. Any concentration below that is
the learnable mapping choosing to re-read bits instead of covering the input.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dwn import (  # noqa: E402
    binarize_fold, build_model, class_weights, load_cache, select_columns, set_seed,
    train_one,
)
from torch_dwn.mapping import LearnableMapping, layer_mapping  # noqa: E402


def layer0_pins(model):
    """(n_luts, n) input-bit indices layer 0 settled on."""
    layer = model[0]
    mapping = layer.mapping
    if isinstance(mapping, LearnableMapping):
        pins = mapping.weights.argmax(dim=0).detach().cpu().numpy()
        return pins.reshape(layer.output_size, layer.n)
    return mapping.detach().cpu().numpy()


def mapping_stats(pins, feature_names, num_bits, input_size):
    """Coverage / concentration / modality-balance summary for one mapping."""
    mods = [str(n).split("_")[0] for n in feature_names]
    flat = pins.reshape(-1)
    feats = flat // num_bits
    per_lut_mods = [len({mods[b // num_bits] for b in lut}) for lut in pins]
    per_lut_feats = [len({b // num_bits for b in lut}) for lut in pins]
    fan = collections.Counter(mods[f] for f in feats)
    share = collections.Counter(mods)
    read = set(feats.tolist())
    return {
        "bits_read": int(len(set(flat.tolist()))),
        "input_bits": int(input_size),
        "feats_read": int(len(read)),
        "num_features": int(len(feature_names)),
        "dead_features": [str(n) for i, n in enumerate(feature_names) if i not in read],
        "mods_per_lut": dict(collections.Counter(per_lut_mods)),
        "feats_per_lut": dict(collections.Counter(per_lut_feats)),
        "fanin_share": {m: fan[m] / len(flat) for m in share},
        "feature_share": {m: share[m] / len(feature_names) for m in share},
    }


def main():
    root = Path(__file__).resolve().parent.parent.parent
    ap = argparse.ArgumentParser(description="Layer-1 mapping diagnostic for the WESAD DWN")
    ap.add_argument("--cache", default=str(root / "data" / "wesad_cache"
                                           / "wesad_features_dt2048k13_6mod.npz"))
    ap.add_argument("--task", default="multi", choices=["multi", "binary"])
    ap.add_argument("--drop", nargs="*", default=["Temp"])
    ap.add_argument("--drop-feat", nargs="*", dest="drop_feat",
                    default=["skew", "kurtosis", "cvrr"])
    ap.add_argument("--config", default="100-51")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--num-bits", type=int, default=4)
    ap.add_argument("--tau", type=float, default=3.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required: torch-dwn's EFD kernel has no CPU implementation.")

    k = 3 if args.task == "multi" else 2
    layer_sizes = [int(s) for s in args.config.split("-") if s]
    X, y, groups, feature_names, _raw = load_cache(args.cache, args.task)
    mask = select_columns(feature_names, None, args.drop, args.drop_feat)
    X, feature_names = X[:, mask], [str(n) for n in feature_names[mask]]
    b_all, _, _ = binarize_fold(X, X, args.num_bits)
    input_size = int(b_all.size(1))

    runs, dead_counter = [], collections.Counter()
    for seed in range(args.seeds):
        set_seed(seed)
        model = build_model(input_size, layer_sizes, args.n, args.tau, k).cuda()
        train_one(model, b_all, torch.from_numpy(y).long(),
                  args.epochs, args.batch_size, args.lr, class_weights(y, k))
        st = mapping_stats(layer0_pins(model), feature_names, args.num_bits, input_size)
        runs.append(st)
        dead_counter.update(st["dead_features"])
        print(f"  seed {seed:>2d}  bits {st['bits_read']:>3d}/{st['input_bits']}  "
              f"feats {st['feats_read']:>3d}/{st['num_features']}  "
              f"dead {len(st['dead_features']):>2d}  "
              f"mods/LUT {dict(sorted(st['mods_per_lut'].items()))}")

    # The library's random mapping is the honest reference: a truncated randperm.
    torch.manual_seed(0)
    ref = mapping_stats(layer_mapping(input_size, args.n, layer_sizes[0], random=True).numpy(),
                        feature_names, args.num_bits, input_size)

    bits = np.array([r["bits_read"] for r in runs])
    feats = np.array([r["feats_read"] for r in runs])
    print(f"\nbits read : learned {bits.mean():.0f}+/-{bits.std():.0f} / {input_size}"
          f"   random {ref['bits_read']} / {input_size}")
    print(f"feats read: learned {feats.mean():.1f}+/-{feats.std():.1f} / {len(feature_names)}"
          f"   random {ref['feats_read']} / {len(feature_names)}")

    mods = sorted(runs[0]["fanin_share"])
    print("\nfan-in share vs feature share (learned, mean over seeds):")
    for m in mods:
        f = np.mean([r["fanin_share"][m] for r in runs])
        s = runs[0]["feature_share"][m]
        print(f"  {m:<5} fan-in {f:.3f}  features {s:.3f}  ratio {f / s:.2f}")

    print(f"\nfeatures never read, by how many of the {args.seeds} seeds:")
    for name, c in dead_counter.most_common():
        if c > 1:
            print(f"  {c:>2d}/{args.seeds}  {name}")

    out = args.out or str(root / "results" / "mapping_diagnostic.json")
    with open(out, "w") as fh:
        json.dump({"args": vars(args), "runs": runs, "random_reference": ref,
                   "dead_counts": dict(dead_counter)}, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
