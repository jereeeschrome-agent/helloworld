"""Exact codelength of compress()'s schedule, without entropy coding.

    python -m jevzip.codelength FILE [FILE ...] [--weights W] [--rounds 32 128] [--streams 0 128 ...]
        [--oracle-context] [--unquantized] [--encode] [--per-byte DIR] [--json OUT]
        [--device cpu|mps] [--threads 4] [--bf16]

Runs compress()'s Plan, block batches, reveal plan and quantization, and sums
-log2(freq / 2^24) over the actual symbols: header + bits / 8 is the
compressed size up to the rANS flush (4..8 bytes, in 4-byte words; --encode
also runs the coder and gives compress()'s exact size). From the same forward
calls it sums -log2 p under the float probabilities (--unquantized reports
that one). --oracle-context reruns the chain-start batches with their true
preceding bytes as context, which no decoder has: that is the model's
full-context cross-entropy, the same for every streams value (evalk reports
it). On the CPU with the same threads and bf16 flag the probabilities are
bit-identical to compress()'s; --device mps is for quick estimates only.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .codec import HEADER, Plan, _encode, _run_batch, _symbols, default_streams, forward_calls
from .model import JevZip
from .rans import PROB_BITS, quantize

TINY = float(np.finfo(np.float32).tiny)  # float probabilities can underflow; quantized ones can't


@dataclass
class Codelength:
    """Per-byte costs in bits, indexed by byte position (NaN where a byte was not scored)."""

    n: int
    rounds: int
    streams: int
    depth: int  # steps: blocks per longest chain
    calls: int  # sequential model.step calls of the coding pass (compress and decompress each make these)
    header: int  # container header bytes
    bits: np.ndarray  # under the quantized CDFs: what rANS pays
    bits_float: np.ndarray  # under the float probabilities
    starts: list  # chain-start blocks: coded with PAD as context
    oracle_bits: np.ndarray | None = None  # every block after the first sees its true context
    oracle_bits_float: np.ndarray | None = None
    body: int | None = None  # rANS body bytes (encode=True)

    @property
    def size(self) -> int | None:
        """compress()'s output size, raw fallback included (encode=True only)."""
        return None if self.body is None else self.header + (self.body if self.body < self.n else self.n)

    def bpb(self, which: str = "bits") -> float:
        """Bits per scored byte of one cost array: "bits", "bits_float", "oracle_bits" or "oracle_bits_float"."""
        a = getattr(self, which)
        return float(np.nansum(a) / max(1, np.isfinite(a).sum()))


def codelength(data: bytes, model: JevZip, rounds: int = 32, streams: int | None = None, threads: int = 4, bf16: bool = False,
               oracle: bool = False, encode: bool = False, blocks=None, true_context: bool = False) -> Codelength:
    """Cost of every byte under compress(data, model, rounds, streams, threads, bf16)'s schedule.

    oracle: also rerun the chain-start batches (same batches, so the same
    batching numerics) with their true context; costs the extra calls of one
    step per chain-start batch. blocks: score only these block indices (others
    stay NaN); with oracle that is a whole-context sample of the file. encode:
    rANS-code the symbols too (all blocks only). true_context: give every
    block its true context in the plain pass itself (not decodable; see
    cross_entropy)."""
    cfg = model.cfg
    n = len(data)
    rounds = max(1, min(rounds, cfg.block))  # as compress() clamps it
    nb = -(-n // cfg.block)
    plan = Plan(max(n, 1), cfg.block, streams or default_streams(max(nb, 1)))
    out = Codelength(n, rounds, plan.streams, plan.depth, forward_calls(plan, cfg, rounds) if n else 0, HEADER.size,
                     np.full(n, np.nan), np.full(n, np.nan), [c[0] for c in plan.chains])
    if oracle:
        out.oracle_bits, out.oracle_bits_float = np.full(n, np.nan), np.full(n, np.nan)
    if n == 0:
        out.body = 0 if encode else None
        return out
    if encode and blocks is not None:
        raise ValueError("encode needs every block")
    keep = None if blocks is None else set(int(b) for b in blocks)
    torch.set_num_threads(threads)
    buf = np.frombuffer(data, np.uint8)
    syms = [] if encode else None

    def scorer(q, f, syms=None):
        def emit(probs, where):
            s = buf[where].astype(np.int64)
            sym = _symbols(quantize(probs), s)
            q[where] = PROB_BITS - np.log2(sym[:, 1])
            f[where] = -np.log2(np.maximum(probs[np.arange(len(s)), s].astype(np.float64), TINY))
            if syms is not None:
                syms.append(sym)
        return emit

    plain = scorer(out.bits, out.bits_float, syms)
    if oracle:
        rerun = scorer(out.oracle_bits, out.oracle_bits_float)
    for batch in plan.steps():
        if keep is not None:
            batch = [x for x in batch if x[0] in keep]
            if not batch:
                continue
        if true_context:
            batch = [(b, b > 0) for b, _ in batch]
        _run_batch(model, buf, plan, batch, rounds, bf16, plain)
        if oracle and any(b > 0 and not has_ctx for b, has_ctx in batch):
            _run_batch(model, buf, plan, [(b, b > 0) for b, _ in batch], rounds, bf16, rerun)
    if oracle:  # every other block already had its true context
        for o, a in ((out.oracle_bits, out.bits), (out.oracle_bits_float, out.bits_float)):
            np.copyto(o, a, where=np.isnan(o))
    if encode:
        out.body = len(_encode(syms))
    return out


def cross_entropy(data: bytes, model: JevZip, rounds: int = 32, threads: int = 4, bf16: bool = False, blocks=None) -> Codelength:
    """The model's full-context cost of every byte: each block sees its true
    preceding cfg.ctx bytes (the first one PAD), all blocks in one step, so in
    batches of CHUNK. These are codelength(oracle=True)'s oracle_bits* up to
    batching numerics (bit-identical on networkx.py.txt, CPU fp32), in one pass and the fewest calls; the
    limit compress() approaches as chain starts get rarer. blocks: score only
    these block indices (a sample)."""
    nb = -(-len(data) // model.cfg.block)
    return codelength(data, model, rounds, max(nb, 1), threads, bf16, blocks=blocks, true_context=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    """Versions that decide the bits: decoding needs the same torch build and CPU kernels."""
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "machine": platform.machine(), "platform": platform.platform()}


def main():
    from .fetch import DEFAULT_MODEL, resolve

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--weights", default=DEFAULT_MODEL, help="path or name (see python -m jevzip.fetch --list)")
    ap.add_argument("--rounds", type=int, nargs="+", default=[32])
    ap.add_argument("--streams", type=int, nargs="+", default=[0], help="0 = compress()'s default")
    ap.add_argument("--oracle-context", action="store_true", help="also score chain starts with their true context")
    ap.add_argument("--unquantized", action="store_true", help="report and save the cost under the float probabilities")
    ap.add_argument("--encode", action="store_true", help="also run the rANS coder: compress()'s exact size")
    ap.add_argument("--per-byte", type=Path, metavar="DIR", help="save per-byte bits as float32 .npy files")
    ap.add_argument("--json", type=Path, metavar="OUT", help="write every result as JSON")
    ap.add_argument("--device", default="cpu", help="cpu (bit-identical to the codec) or mps (estimates)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast, as compress(bf16=True)")
    args = ap.parse_args()

    weights = resolve(args.weights)
    model = JevZip.load(weights).to(args.device)
    kind = "bits_float" if args.unquantized else "bits"
    meta = {"command": [sys.executable, "-m", "jevzip.codelength", *sys.argv[1:]], "env": environment(),
            "weights": {"path": str(weights), "sha256": sha256(weights)}, "config": vars(model.cfg) | {},
            "device": args.device, "threads": args.threads, "bf16": args.bf16, "results": []}
    print(f"{weights.name}: {model.cfg}", file=sys.stderr)
    for path in args.files:
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        for k in args.rounds:
            for s in args.streams:
                t = time.time()
                cl = codelength(data, model, k, s or None, args.threads, args.bf16, args.oracle_context, args.encode)
                dt = time.time() - t
                rec = {"file": str(path), "sha256": digest, "n": cl.n, "rounds": cl.rounds, "streams": cl.streams,
                       "depth": cl.depth, "calls": cl.calls, "header_bytes": cl.header, "seconds": round(dt, 1),
                       "bits": float(np.sum(cl.bits)), "bits_float": float(np.sum(cl.bits_float)),
                       "bpb": cl.bpb(), "bpb_float": cl.bpb("bits_float")}
                line = f"{path.name} K={cl.rounds} streams={cl.streams}: {cl.bpb(kind):.4f} bpb"
                if args.oracle_context:
                    rec |= {"oracle_bits": float(np.sum(cl.oracle_bits)), "oracle_bits_float": float(np.sum(cl.oracle_bits_float)),
                            "oracle_bpb": cl.bpb("oracle_bits"), "oracle_bpb_float": cl.bpb("oracle_bits_float")}
                    line += f", oracle context {cl.bpb('oracle_' + kind):.4f}"
                predicted = cl.header + np.sum(cl.bits) / 8
                rec["predicted_bytes"] = float(predicted)
                line += f", ~{predicted:.0f} bytes"
                if args.encode:
                    rec |= {"body_bytes": cl.body, "size": cl.size}
                    line += f" ({cl.size} encoded)"
                print(line + f", {cl.calls} calls, {dt:.0f}s", flush=True)
                if args.per_byte:
                    args.per_byte.mkdir(parents=True, exist_ok=True)
                    stem = f"{path.name}-k{cl.rounds}-s{cl.streams}" + ("-float" if args.unquantized else "")
                    np.save(args.per_byte / f"{stem}.npy", getattr(cl, kind).astype(np.float32))
                    if args.oracle_context:
                        np.save(args.per_byte / f"{stem}-oracle.npy", getattr(cl, "oracle_" + kind).astype(np.float32))
                meta["results"].append(rec)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
