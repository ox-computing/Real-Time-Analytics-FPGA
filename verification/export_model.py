"""Export a trained DWN checkpoint to a plain JSON the hardware model can read.

Reduces the PyTorch checkpoint to exactly what inference needs, with all
training-time machinery resolved away:

  - learnable mapping  -> argmax over the weight matrix, giving one fixed
    input-bit index per LUT input pin (this is what becomes fixed wiring on the
    FPGA; it costs no logic at inference)
  - fixed random mapping -> already integer indices, copied as-is
  - LUT tables         -> thresholded at 0 (the STE inference rule
    output = 1 if entry > 0) into 0/1 truth tables
  - thermometer thresholds -> carried through so binarization is reproducible

Works for any number of LUT layers. The model's shape is *read out of the
checkpoint*, never passed in: a Sequential of LUTLayers stores layer i as
`i.luts` of shape (layer_size, 2**n), plus either `i.mapping.weights` (learnable)
or `i.mapping` (fixed indices). Layer count, per-layer width, n, and input size
all follow from those, so there is no shape flag to get wrong -- an export can
never silently disagree with the checkpoint it came from.

Address convention (verified empirically against the CUDA kernel): the n bits
of a LUT address are little-endian, i.e. input pin k carries weight 2**k.

The resulting JSON is consumed by both hw_model.py (the bit-exact reference)
and rtl/generate_rtl.py.
"""

import argparse
import json
from pathlib import Path

import torch


def layer_indices(sd):
    """Indices of the LUT layers in the checkpoint, in Sequential order.

    GroupSum carries no parameters, so the `<i>.luts` keys are exactly the LUT
    layers. Sorted numerically -- state-dict key order is insertion order, which
    would put "10" before "2".
    """
    idx = [int(k.split(".", 1)[0]) for k in sd if k.split(".", 1)[-1] == "luts"]
    if not idx:
        raise SystemExit("no '<i>.luts' keys in checkpoint -- not a DWN LUTLayer stack?")
    return sorted(idx)


def resolve_mapping(sd, i, num_luts, n):
    """One layer's mapping -> (num_luts, n) integer input indices.

    A learnable mapping is stored as a (in_bits, num_luts * n) weight matrix and
    is resolved by argmax over the input axis: each LUT pin picks the input bit
    it weights most, which is the fixed wire the FPGA gets. A fixed mapping is
    already indices. Layer 0 also carries a `_LUTLayer__dummy_mapping` buffer,
    which is training scaffolding and is deliberately ignored.
    """
    if f"{i}.mapping.weights" in sd:
        w = sd[f"{i}.mapping.weights"]                       # (in_bits, num_luts * n)
        return w.argmax(dim=0).reshape(num_luts, n).to(torch.int64)
    if f"{i}.mapping" in sd:
        return sd[f"{i}.mapping"].reshape(num_luts, n).to(torch.int64)
    raise SystemExit(f"layer {i}: neither '{i}.mapping.weights' nor '{i}.mapping' in checkpoint")


def main():
    parser = argparse.ArgumentParser(description="Export DWN checkpoint to JSON")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thermometer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=10, help="number of output classes/groups")
    args = parser.parse_args()

    sd = torch.load(args.checkpoint, map_location="cpu")
    thermometer = torch.load(args.thermometer, map_location="cpu", weights_only=False)

    indices = layer_indices(sd)

    # Shape comes from the tensors themselves: luts is (layer_size, 2**n).
    layers, layer_sizes = [], []
    n = None
    for i in indices:
        luts = (sd[f"{i}.luts"] > 0).to(torch.int64)
        num_luts, table_size = luts.shape
        n_i = table_size.bit_length() - 1
        assert 1 << n_i == table_size, f"layer {i}: LUT table {table_size} is not a power of two"
        if n is None:
            n = n_i
        assert n_i == n, f"layer {i}: n={n_i} differs from layer {indices[0]}'s n={n}"

        mapping = resolve_mapping(sd, i, num_luts, n)
        layers.append({"mapping": mapping.tolist(), "luts": luts.tolist()})
        layer_sizes.append(num_luts)

    # The first layer reads the thermometer-encoded input; its mapping indexes it.
    input_size = thermometer.thresholds.shape[0] * thermometer.num_bits
    first = sd.get(f"{indices[0]}.mapping.weights")
    if first is not None:
        assert first.size(0) == input_size, \
            f"checkpoint input {first.size(0)} != thermometer input {input_size}"

    out_bits = layer_sizes[-1]
    assert out_bits % args.k == 0, \
        f"final layer {out_bits} must divide evenly into k={args.k} groups"

    model = {
        "meta": {
            "checkpoint": str(args.checkpoint),
            "input_size": input_size,
            "n": n,
            "k": args.k,
            "group_size": out_bits // args.k,
            "layer_sizes": layer_sizes,
            "address_order": "little_endian (pin k -> weight 2**k)",
            "num_bits": thermometer.num_bits,
        },
        # thermometer thresholds: (num_features, num_bits)
        "thermometer_thresholds": thermometer.thresholds.tolist(),
        "layers": layers,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(model, f)
    print(f"Exported to {out_path}")
    print(f"  input_size={input_size}, layers={'/'.join(str(s) for s in layer_sizes)}, "
          f"n={n}, k={args.k}, group_size={out_bits // args.k}")


if __name__ == "__main__":
    main()
