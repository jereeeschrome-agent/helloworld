"""Model cross-entropy (bits/byte) vs rounds K on sampled windows: fast, no entropy coding.

    python -m jevzip.evalk FILE [FILE ...] [--weights W] [--rounds 8 16 32 64 512] [--device mps]

Matches real compressed sizes to within ~0.5% (bench.py measures those).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from .__main__ import DEFAULT_WEIGHTS
from .model import JevZip
from .train import loss_bits, make_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 16, 32, 64, 128, 512])
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    model = JevZip.load(args.weights).to(args.device)
    print(f"{args.weights.name}: {model.cfg}")
    for path in args.files:
        data = np.memmap(path, dtype=np.uint8, mode="r")
        row = []
        for k in args.rounds:
            bits = []
            with torch.no_grad():
                for i in range(0, args.windows, 16):
                    batch = make_batch(data, model.cfg, min(16, args.windows - i), np.random.default_rng(i), p_noctx=0, p_trunc=0, rounds=k)
                    bits.append(loss_bits(model, batch, args.device).item())
            row.append(f"K={k}: {np.mean(bits):.3f}")
        print(f"{path.name}  " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
