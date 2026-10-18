"""jevzip CLI.

    jevzip c  INPUT OUTPUT [--rounds K] [--streams S] [--threads T] [--weights NAME|PATH] [--bf16]
    jevzip d  INPUT OUTPUT [--weights NAME|PATH]

(or python -m jevzip ...). --weights takes a manifest name (python -m jevzip.fetch --list)
or a path; a name is looked up in the checkout's weights/, then the download cache.
Inference is fp32 by default; --bf16 is faster on CPUs with native bf16 but the
output then decodes only on a machine with the same bf16 kernels.
"""
import argparse
import sys
import time
from pathlib import Path

from .codec import compress, decompress
from .fetch import DEFAULT_MODEL, REPO, resolve
from .model import JevZip

DEFAULT_WEIGHTS = REPO / "weights" / DEFAULT_MODEL  # deprecated (checkout path for old importers): use fetch.resolve(DEFAULT_MODEL)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jevzip", description="parallel-round neural byte compressor")
    ap.add_argument("mode", choices=["c", "d"], help="compress or decompress")
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--weights", default=DEFAULT_MODEL, help=f"manifest name or path (default: {DEFAULT_MODEL})")
    ap.add_argument("--rounds", type=int, default=128, help="parallel passes per block (fewer = faster, larger; 512 = sequential)")
    ap.add_argument("--streams", type=int, default=None, help="independent block chains (default: blocks/8)")
    ap.add_argument("--threads", type=int, default=4)
    prec = ap.add_mutually_exclusive_group()
    prec.add_argument("--bf16", dest="bf16", action="store_true", help="bf16 inference (default: fp32); decoding needs the same CPU bf16 kernels")
    prec.add_argument("--fp32", dest="bf16", action="store_false", help="fp32 inference: the default, kept for old command lines")
    args = ap.parse_args(argv)

    try:
        model = JevZip.load(resolve(args.weights))
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    data = args.input.read_bytes()
    t = time.time()
    if args.mode == "c":
        out = compress(data, model, rounds=args.rounds, streams=args.streams, threads=args.threads, bf16=args.bf16)
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
