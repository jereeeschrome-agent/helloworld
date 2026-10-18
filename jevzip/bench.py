"""Benchmark jevzip against the baseline compressors (jevzip/baselines.py): one markdown table per file.

    python -m jevzip.bench FILE [FILE ...] [--rounds 8 16 32 64] [--baselines fast|all|none]
                           [--train-corpus TRAIN.bin] [--weights NAME|PATH] [--verify] [--json OUT]

jevzip rows report the forward calls coding takes (the sequential model.step calls, the same on
both sides), which set its speed. Inference is fp32 unless --bf16 (then the output decodes only with
the same CPU bf16 kernels). --train-corpus adds the dictionary and primed baselines, the ones
that, like jevzip, learn from the training corpus; their reference size is shown next to them.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from . import baselines as bl
from .codec import Plan, compress, decompress, default_streams, forward_calls
from .fetch import DEFAULT_MODEL, resolve
from .model import JevZip


def jevzip_rows(data: bytes, model: JevZip, rounds, streams, threads, bf16, verify, log=print) -> list[dict]:
    """Compress `data` once per (streams, rounds); with verify also decompress, compare and time it."""
    n, cfg = len(data), model.cfg
    nb = -(-n // cfg.block)
    rows = []
    for s in streams:
        plan = Plan(n, cfg.block, s or default_streams(nb))
        for k in rounds:
            t = time.time()
            blob = compress(data, model, k, plan.streams, threads, bf16=bf16)
            row = dict(name=f"jevzip K={k} streams={plan.streams}" + (" (no ctx)" if plan.depth == 1 else ""), kind="jevzip",
                       bytes=len(blob), bits_per_byte=8 * len(blob) / n, enc_s=time.time() - t, forward_calls=forward_calls(plan, cfg, k),
                       rounds=k, streams=plan.streams, threads=threads, bf16=bf16, verified=None)
            if verify:
                t = time.time()
                row["verified"] = decompress(blob, model) == data
                row["dec_s"] = time.time() - t
                if not row["verified"]:
                    raise RuntimeError(f"round-trip failed: K={k} streams={plan.streams}")
            rows.append(row)
            log(row)
    return rows


def table(n: int, rows: list[dict]) -> str:
    out = ["| method | bytes | bits/byte | ref bytes | forward calls | enc KB/s | dec KB/s |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if "bytes" not in r or "error" in r:
            why = r.get("skipped") or r.get("error", "")
            out.append(f"| {r['name']} | {'skipped' if 'skipped' in r else 'ERROR'}: {why[:120]} | | | | | |")
            continue
        speed = lambda s: f"{n / s / 1e3:.1f}" if s else ""  # noqa: E731
        name = r["name"] + (" (marginal)" if r["kind"] == "marginal" else "")
        out.append(f"| {name} | {r['bytes']} | {8 * r['bytes'] / n:.3f} | {r.get('ref_bytes') or ''} | {r.get('forward_calls', '')} | "
                   f"{speed(r.get('enc_s'))} | {speed(r.get('dec_s'))} |")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--weights", default=DEFAULT_MODEL, help=f"name or path (default {DEFAULT_MODEL}); see python -m jevzip.fetch --list")
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 16, 32, 64], help="trained for 8..128")
    ap.add_argument("--streams", type=int, nargs="+", default=[0], help="0 = default_streams")
    ap.add_argument("--baselines", choices=["fast", "all", "none"], default="fast", help="all adds paq8px, cmix, 7zz PPMd (slow)")
    ap.add_argument("--train-corpus", type=Path, help="adds the dictionary / primed baselines built from this corpus")
    ap.add_argument("--no-jevzip", action="store_true", help="baselines only")
    ap.add_argument("--verify", action="store_true", help="also decompress jevzip output, compare and time it")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", help="bf16 inference (default fp32): faster only on CPUs with native bf16")
    ap.add_argument("--json", type=Path, help="also write every row here")
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    out = dict(command=" ".join([sys.executable, "-m", "jevzip.bench", *(sys.argv[1:] if argv is None else argv)]),
               torch=torch.__version__, environment=bl.environment(), results=[])
    model = None
    if not args.no_jevzip:
        wpath = resolve(args.weights)
        model = JevZip.load(wpath)
        out["weights"] = dict(path=str(wpath), bytes=wpath.stat().st_size, sha256=bl._file_sha(wpath), params=sum(p.numel() for p in model.parameters()))
        print(f"model: {wpath.name}, {out['weights']['params'] / 1e6:.2f}M params, {wpath.stat().st_size / 1e6:.1f} MB on disk, "
              f"{'bf16' if args.bf16 else 'fp32'}, {args.threads} threads\n")
    chosen = bl.select(args.baselines)
    if args.baselines != "none" and args.train_corpus is None:
        chosen = [b for b in chosen if b.kind == "standalone"]
    ctx = bl.Ctx(args.train_corpus)
    try:
        for path in args.files:
            data = path.read_bytes()
            if not data:
                print(f"### {path.name}: empty, skipped\n")
                continue
            res = dict(file=str(path), size=len(data), sha256=hashlib.sha256(data).hexdigest(), rows=[])
            out["results"].append(res)
            print(f"### {path.name} ({len(data) / 1e6:.2f} MB)\n", flush=True)
            log = lambda r: print("  " + bl.describe(r), file=sys.stderr, flush=True)  # noqa: E731
            res["rows"] += bl.run(data, chosen, ctx, log)
            if model is not None:
                res["rows"] += jevzip_rows(data, model, args.rounds, args.streams, args.threads, args.bf16, args.verify,
                                           lambda r: print(f"  {r['name']}: {r['bytes']} B", file=sys.stderr, flush=True))
            print(table(len(data), res["rows"]) + "\n", flush=True)
    finally:
        bl.shutil.rmtree(ctx.work, ignore_errors=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
