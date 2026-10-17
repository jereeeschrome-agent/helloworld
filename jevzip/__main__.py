"""jevzip CLI.

    python -m jevzip c  INPUT OUTPUT [--rounds K] [--streams S] [--weights W]
    python -m jevzip d  INPUT OUTPUT
"""
import argparse
import sys
import time
from pathlib import Path

from .codec import compress, decompress
from .model import JevZip

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "jevzip-lanes.pt"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevzip", description="parallel-round neural byte compressor")
    ap.add_argument("mode", choices=["c", "d"], help="compress or decompress")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--rounds", type=int, default=128, help="parallel passes per block (fewer = faster, larger; 512 = sequential)")
    ap.add_argument("--streams", type=int, default=None, help="independent block chains (default: blocks/8)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fp32", action="store_true", help="fp32 inference instead of bf16")
    args = ap.parse_args(argv)

    model = JevZip.load(args.weights)
    data = args.input.read_bytes()
    t = time.time()
    if args.mode == "c":
        out = compress(data, model, args.rounds, args.streams, args.threads, not args.fp32)
    else:
        out = decompress(data, model)
    dt = time.time() - t
    args.output.write_bytes(out)
    raw = len(data) if args.mode == "c" else len(out)
    packed = len(out) if args.mode == "c" else len(data)
    bpb = 8 * packed / raw if raw else 0.0
    print(f"{raw} -> {packed} bytes ({bpb:.3f} bits/byte) in {dt:.1f}s ({raw / dt / 1e3 if dt else 0:.1f} KB/s)", file=sys.stderr)


if __name__ == "__main__":
    main()
