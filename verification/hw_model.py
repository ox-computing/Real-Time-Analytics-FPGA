"""Pure-numpy, hardware-style DWN inference from an exported JSON model.

Deliberately uses only plain integer/bit operations -- no PyTorch, no learned
or differentiable ops -- so it mirrors what the Verilog will do and serves as
the golden reference the RTL is checked against. Every step here maps directly
onto a hardware construct:

  binarize  -> thermometer comparators
  gather    -> fixed input wiring (the resolved mapping)
  address   -> concatenating n wires into an n-bit index (little-endian)
  lookup    -> the LUT truth table
  popcount  -> the per-class adder tree (GroupSum)
  argmax    -> the output comparator

This is the reference model, not the RTL; the RTL will reproduce it bit-for-bit.
"""

import json

import numpy as np


class HWModel:
    def __init__(self, model_json_path):
        with open(model_json_path) as f:
            m = json.load(f)
        self.meta = m["meta"]
        self.n = m["meta"]["n"]
        self.k = m["meta"]["k"]
        self.group_size = m["meta"]["group_size"]
        # thresholds: (num_features, num_bits)
        self.thresholds = np.asarray(m["thermometer_thresholds"], dtype=np.float64)
        self.layers = [
            {"mapping": np.asarray(l["mapping"], dtype=np.int64),
             "luts": np.asarray(l["luts"], dtype=np.int64)}
            for l in m["layers"]
        ]
        # Little-endian pin weights: pin k contributes 2**k.
        self.pin_weight = (1 << np.arange(self.n, dtype=np.int64))

    def binarize(self, images):
        """Thermometer-binarize raw images. images: (N, num_features) float in [0,1].

        Returns (N, num_features * num_bits) as 0/1 int, matching the training
        thermometer: bit is 1 where pixel value > threshold.
        """
        images = np.asarray(images, dtype=np.float64)
        # (N, features, 1) > (features, bits) -> (N, features, bits)
        bits = (images[:, :, None] > self.thresholds[None, :, :]).astype(np.int64)
        return bits.reshape(images.shape[0], -1)

    def _lut_layer(self, x, layer):
        """Apply one LUT layer. x: (N, in_bits) 0/1 -> (N, num_luts) 0/1."""
        mapping = layer["mapping"]      # (num_luts, n)
        luts = layer["luts"]            # (num_luts, 2**n)
        num_luts = mapping.shape[0]

        gathered = x[:, mapping]                       # (N, num_luts, n)
        addr = (gathered * self.pin_weight).sum(axis=2)  # (N, num_luts), little-endian
        # index each LUT's truth table by its per-sample address
        out = luts[np.arange(num_luts)[None, :], addr]   # (N, num_luts)
        return out

    def infer(self, images, return_scores=False, return_all=False):
        """Raw images -> predicted classes (and optionally scores/intermediates).

        return_all yields every intermediate for bit-level checking:
          {"bin": (N, in_bits), "layer_bits": [ (N, l1), (N, l2) ],
           "scores": (N, k), "preds": (N,)}
        """
        bin_bits = self.binarize(images)
        x = bin_bits
        layer_bits = []
        for layer in self.layers:
            x = self._lut_layer(x, layer)               # final: (N, layer2_bits)
            layer_bits.append(x)

        # GroupSum: contiguous groups of `group_size` bits, popcount per class.
        n_samples = x.shape[0]
        scores = x.reshape(n_samples, self.k, self.group_size).sum(axis=2)  # (N, k)
        preds = scores.argmax(axis=1)
        if return_all:
            return {"bin": bin_bits, "layer_bits": layer_bits,
                    "scores": scores, "preds": preds}
        if return_scores:
            return preds, scores
        return preds
