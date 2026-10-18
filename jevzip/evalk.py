"""Model cross-entropy (bits/byte) vs rounds K: exact, over the whole file, no entropy coding.

    python -m jevzip.evalk FILE [FILE ...] [--weights W] [--rounds 16 32 128] [--windows N]
        [--threads 4] [--bf16] [--device cpu]

A thin wrapper over codelength.cross_entropy: every block of the file is scored
with its true preceding context, so this is the model's full-context
cross-entropy, the floor compress() approaches (its real sizes add chain-start
blocks coded without context, quantization and the rANS flush: python -m
jevzip.codelength --oracle-context --encode splits them). fp32 by default, as
the codec.

It used to average 64 random 512-byte windows under bf16 autocast. On
networkx.py.txt (1 MB) that sample sat 1.5% (K=32) and 2.0% (K=128) below the
full-file value: sampling noise, as 64 windows of a mixed file spread by ~3%
(1 sigma); bf16 moved it by < 0.01%. results/gap/gap.md has the decomposition.
--windows N still scores only N random blocks (seeded, with their true
context) for a quick estimate on big files.
"""
import argparse
from pathlib import Path

import numpy as np

from .codelength import cross_entropy
from .model import JevZip


def main():
    from .fetch import DEFAULT_MODEL, resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--weights", default=DEFAULT_MODEL, help="path or name (see python -m jevzip.fetch --list)")
    ap.add_argument("--rounds", type=int, nargs="+", default=[16, 32, 128])
    ap.add_argument("--windows", type=int, default=0, help="score only this many random blocks (0 = the whole file)")
    ap.add_argument("--seed", type=int, default=0, help="block sample for --windows")
    ap.add_argument("--device", default="cpu", help="cpu or mps (mps: estimates only)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast (default fp32, as the codec)")
    args = ap.parse_args()

    weights = resolve(args.weights)
    model = JevZip.load(weights).to(args.device)
    print(f"{weights.name}: {model.cfg}")
    for path in args.files:
        data = path.read_bytes()
        nb = -(-len(data) // model.cfg.block)
        blocks = None
        if 0 < args.windows < nb:
            blocks = np.sort(np.random.default_rng(args.seed).choice(nb, args.windows, replace=False))
        row = [f"K={k}: {cross_entropy(data, model, k, args.threads, args.bf16, blocks).bpb('bits_float'):.4f}" for k in args.rounds]
        scope = f"{len(blocks)} of {nb} blocks" if blocks is not None else f"{nb} blocks"
        print(f"{path.name} ({scope})  " + "  ".join(row), flush=True)


if __name__ == "__main__":
    main()
