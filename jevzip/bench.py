"""Benchmark jevzip against standard compressors.

    python -m jevzip.bench FILE [FILE ...] [--rounds 8 16 32 64] [--verify]
"""
import argparse
import bz2
import gzip
import lzma
import time
from pathlib import Path

from .__main__ import DEFAULT_WEIGHTS
from .codec import Plan, compress, decompress
from .model import JevZip
from .schedule import reveal_plan


def baselines(data: bytes) -> dict[str, int]:
    out = {
        "gzip -9": len(gzip.compress(data, 9)),
        "bzip2 -9": len(bz2.compress(data, 9)),
        "xz -9e": len(lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)),
    }
    try:
        import zstandard

        out["zstd -19"] = len(zstandard.ZstdCompressor(level=19).compress(data))
    except ImportError:
        pass
    try:
        import brotli

        out["brotli -q11"] = len(brotli.compress(data, quality=11))
    except ImportError:
        pass
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 16, 32, 64], help="trained for 8..128")
    ap.add_argument("--streams", type=int, nargs="+", default=[0], help="0 = default (blocks/8)")
    ap.add_argument("--verify", action="store_true", help="also decompress and time it")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    model = JevZip.load(args.weights)
    weight_bytes = args.weights.stat().st_size
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, {weight_bytes / 1e6:.1f} MB on disk\n")
    for path in args.files:
        data = path.read_bytes()
        n = len(data)
        if n == 0:
            print(f"### {path.name}: empty, skipped\n")
            continue
        nb = -(-n // model.cfg.block)
        print(f"### {path.name} ({n / 1e6:.2f} MB)\n")
        print("| method | bytes | bits/byte | forward calls | enc KB/s | dec KB/s |")
        print("|---|---:|---:|---:|---:|---:|")
        for name, size in baselines(data).items():
            print(f"| {name} | {size} | {8 * size / n:.3f} | | | |", flush=True)
        for streams in args.streams:
            plan = Plan(n, model.cfg.block, streams or max(1, nb // 8))
            for k in args.rounds:
                # sequential model calls this implementation makes (both sides make the same ones)
                cfg = model.cfg
                calls = sum(len(reveal_plan(cfg.order, cfg.block, plan.length(b[0][0]), k, cfg.first)[1]) - 1 for b in plan.steps())
                t = time.time()
                blob = compress(data, model, k, plan.streams, args.threads)
                enc = n / (time.time() - t) / 1e3
                dec = ""
                if args.verify:
                    t = time.time()
                    out = decompress(blob, model)
                    dt = time.time() - t
                    if out != data:
                        raise RuntimeError(f"round-trip failed: {path} K={k} streams={plan.streams}")
                    dec = f"{n / dt / 1e3:.1f}"
                label = f"jevzip K={k} streams={plan.streams}" + (" (no ctx)" if plan.depth == 1 else "")
                print(f"| {label} | {len(blob)} | {8 * len(blob) / n:.3f} | {calls} | {enc:.1f} | {dec} |", flush=True)
        print()


if __name__ == "__main__":
    main()
