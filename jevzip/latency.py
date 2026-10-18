"""Wall-clock decode latency vs rounds K, on cpu / mps / cuda.

    python -m jevzip.latency FILE [--weights W] [--rounds 16 32 64 128 512] [--sizes 64K 1M]
                             [--device mps] [--dtype fp32] [--runs 3] [--threads 4] [--out results/latency]

Parallel rounds pay off as fewer *sequential* model calls. A compute-bound CPU hides
that, so this times decoding where a round's batch runs in parallel. Timed: exactly
the forward calls a decoder makes, i.e. model.step once per round with the KV cache,
the upload of the bytes the previous round revealed, and the device-to-host copy of
the round's probabilities (the decoder needs them before it can go on), with a sync
before the clock stops. The true bytes are fed back (teacher forcing: the same
compute as decoding). The entropy decoder is timed separately on the CPU: quantize +
rANS decode of the same symbols, as codec.decompress does it. End-to-end = model +
entropy, run one after the other as in the codec.

Layouts:
  parallel  all blocks at once, no previous-block context (streams = blocks): the
            dependency depth is K for the whole file. The blocks go into each forward
            call together, in waves of at most --max-batch (fp32 KV cache: ~19 MB a block).
  codec     the codec's default: codec.default_streams chains, at most codec.CHUNK blocks a call.

The AR baseline with a KV cache is K=512 on a lanes model (one lane: one byte per
round per block, all blocks batched). The trained AR control (linear order, same 11M
architecture) is --weights jevzip-ar-11m-enwik8.pt --rounds 512.

Also run: a true round trip on the device (encode with the device's probabilities,
then decode on the same device with the decoded bytes fed back), the SHA-256 of all
probabilities in every timed run (bit-stable or not), and an overhead probe that
shows what bounds the time of one round. Results: <out>/latency-<device>.json (one
entry per weights/file/dtype) and <out>/latency.md, rebuilt from every such JSON.
"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .codec import CHUNK, HEADER, Plan, default_streams
from .fetch import DEFAULT_MODEL, REPO, resolve
from .model import PAD, JevZip
from .rans import Decoder, Encoder, quantize
from .schedule import reveal_plan

DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def parse_size(s: str) -> int:
    s = s.strip().upper()
    mult = {"K": 1 << 10, "M": 1 << 20}.get(s[-1:], 1)
    return int(float(s.rstrip("KM")) * mult)


def sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def batches(n: int, block: int, layout: str, max_batch: int):
    """[(block index, has context), ...] per forward-call batch, in decode order; and the dependency depth in steps."""
    if layout == "codec":
        nb = -(-n // block)
        plan = Plan(n, block, default_streams(nb))  # codec.compress's default
        return list(plan.steps()), plan.depth
    full = [(b, False) for b in range(-(-n // block)) if min(block, n - b * block) == block]
    part = [(b, False) for b in range(-(-n // block)) if min(block, n - b * block) != block]
    out = []
    for group in (full, part):
        if group:
            size = -(-len(group) // -(-len(group) // max_batch))  # equal waves of <= max_batch
            out += [group[i : i + size] for i in range(0, len(group), size)]
    return out, 1


class Job:
    """One batch's decode schedule, with everything that doesn't depend on decoded bytes already on the device."""

    def __init__(self, model, n, batch, rounds, device):
        cfg = model.cfg
        C, B = cfg.ctx, cfg.block
        self.batch = batch
        self.len = min(B, n - batch[0][0] * B)
        order, bounds = reveal_plan(cfg.order, B, self.len, rounds, cfg.first)
        starts = np.array([b * B for b, _ in batch])
        self.ctx_from = [b * B - C if has else None for b, has in batch]
        self.pad_ctx = torch.full((len(batch), C), PAD, dtype=torch.long, device=device) if not any(h for _, h in batch) else None
        self.ctx_pos = torch.arange(C, device=device).expand(len(batch), C).contiguous()
        self.rounds = []
        for r in range(len(bounds) - 1):
            slots = order[bounds[r] : bounds[r + 1]]
            qpos = torch.from_numpy(np.ascontiguousarray(np.broadcast_to(C + slots, (len(batch), len(slots))))).to(device)
            self.rounds.append((qpos, (starts[:, None] + slots[None, :]).reshape(-1)))

    def ctx(self, buf, C, device):
        if self.pad_ctx is not None:
            return self.pad_ctx
        ctx = np.full((len(self.batch), C), PAD, np.int64)
        for i, a in enumerate(self.ctx_from):
            if a is not None:
                ctx[i] = buf[a : a + C]
        return torch.from_numpy(ctx).to(device)


def decode_pass(model, jobs, buf, device, amp, pop=None):
    """The decoder's forward calls. pop=None: teacher forcing (buf holds the true bytes);
    else pop(cdfs, where) entropy-decodes a round into buf. Returns per-round probabilities
    (host float32) and per-call seconds (each call ends in a blocking device-to-host copy)."""
    C = model.cfg.ctx
    probs_out, secs = [], []
    sync(device)
    t = time.perf_counter()
    with amp:
        for job in jobs:
            cache, tok, pos = None, job.ctx(buf, C, device), job.ctx_pos
            for r, (qpos, where) in enumerate(job.rounds):
                cache, logits = model.step(cache, tok, pos, qpos)
                probs = torch.softmax(logits.float(), -1).reshape(-1, logits.shape[-1]).cpu().numpy()
                if pop is not None:
                    pop(quantize(probs), where)
                probs_out.append(probs)
                if r + 1 < len(job.rounds):
                    tok = torch.from_numpy(buf[where].reshape(len(job.batch), -1).astype(np.int64)).to(device)
                    pos = qpos
                now = time.perf_counter()
                secs.append(now - t)
                t = now
    return probs_out, secs


def encode(probs, jobs, data):
    """rANS-encode the true bytes under the given per-round probabilities (what an encoder emits)."""
    syms = []
    rounds = [w for job in jobs for _, w in job.rounds]
    for p, where in zip(probs, rounds):
        cdfs, s = quantize(p), data[where].astype(np.int64)
        rows = np.arange(len(s))
        syms.append(np.stack([cdfs[rows, s], cdfs[rows, s + 1] - cdfs[rows, s]], 1))
    enc = Encoder()
    for start, freq in np.concatenate(syms)[::-1].tolist():
        enc.push(start, freq)
    return enc.finish()


def popper(dec, buf):
    def pop(cdfs, where):  # codec.decompress's emit
        for i, p in enumerate(where.tolist()):
            buf[p] = dec.pop(cdfs[i])

    return pop


def entropy_decode(probs, jobs, body, n):
    """Time quantize + rANS decode of every symbol, on the CPU, in decode order."""
    buf = np.zeros(n, np.uint8)
    dec = Decoder(body)
    pop = popper(dec, buf)
    rounds = [w for job in jobs for _, w in job.rounds]
    t = time.perf_counter()
    for p, where in zip(probs, rounds):
        pop(quantize(p), where)
    return time.perf_counter() - t, buf, dec.exhausted()


def digest(probs) -> str:
    h = hashlib.sha256()
    for p in probs:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()


def measure(model, data, layout, rounds, device, amp, runs, max_batch):
    """Median timings of the model's forward calls and of the entropy decoder for one (size, layout, K)."""
    n = len(data)
    plan, depth = batches(n, model.cfg.block, layout, max_batch)
    jobs = [Job(model, n, b, rounds, device) for b in plan]
    calls = sum(len(j.rounds) for j in jobs)
    seen = set()
    warm = [j for j in jobs if (len(j.batch), j.len) not in seen and not seen.add((len(j.batch), j.len))]
    decode_pass(model, warm, data, device, amp)  # warm-up: one batch of each shape (kernels, allocator)
    model_s, first_s, hashes, keep = [], [], [], None
    for i in range(runs):
        probs, secs = decode_pass(model, jobs, data, device, amp)
        model_s.append(sum(secs))
        k = 0
        firsts = []
        for j in jobs:
            firsts.append(secs[k])
            k += len(j.rounds)
        first_s.append(sum(firsts))
        hashes.append(digest(probs))
        keep = keep or probs
    body = encode(keep, jobs, data)
    ent_s, out, exhausted = entropy_decode(keep, jobs, body, n)
    assert exhausted and np.array_equal(out, data), "entropy decoder did not reproduce the input"
    m, e = statistics.median(model_s), ent_s
    rest = [a - b for a, b in zip(model_s, first_s)]
    return {
        "layout": layout, "K": rounds, "ar_baseline": rounds >= model.cfg.block,
        "blocks": sum(len(j.batch) for j in jobs), "batches": len(jobs), "max_blocks_per_call": max(len(j.batch) for j in jobs),
        "forward_calls": calls, "dependency_depth_rounds": depth * max(len(j.rounds) for j in jobs),
        "model_ms": 1e3 * m, "model_ms_runs": [1e3 * x for x in model_s],
        "per_round_ms": 1e3 * m / calls,
        "first_call_ms": 1e3 * statistics.median(first_s) / len(jobs),
        "later_round_ms": 1e3 * statistics.median(rest) / max(1, calls - len(jobs)),
        "entropy_ms": 1e3 * e, "e2e_ms": 1e3 * (m + e), "kb_per_s": n / (m + e) / 1e3,
        "model_share": m / (m + e),
        "coded_bytes": HEADER.size + len(body), "bits_per_byte": 8 * (HEADER.size + len(body)) / n,
        "probs_sha256": hashes, "deterministic": len(set(hashes)) == 1,
    }, (jobs, keep, body)


def roundtrip(model, data, jobs, body, device, amp):
    """Decode `body` (encoded with this device's probabilities) on the same device, feeding back decoded bytes."""
    n = len(data)
    buf = np.zeros(n, np.uint8)
    dec = Decoder(body)
    t = time.perf_counter()
    err = None
    try:
        decode_pass(model, jobs, buf, device, amp, pop=popper(dec, buf))
    except ValueError as e:  # rANS stream exhausted: the decoder diverged
        err = str(e)
    secs = time.perf_counter() - t
    bad = np.flatnonzero(buf != np.frombuffer(data, np.uint8))
    return {"identical": err is None and not len(bad) and dec.exhausted(), "first_mismatch": int(bad[0]) if len(bad) else None,
            "error": err, "e2e_ms_interleaved": 1e3 * secs, "kb_per_s": n / secs / 1e3}


class OpCounter(torch.utils._python_dispatch.TorchDispatchMode):
    """Counts aten ops (~ kernel launches on a GPU)."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def overhead(model, device, amp, sizes=(1, 8, 32, 128, 256), rounds=48, reps=50):
    """What one round costs: a bare sync, and K=512-style rounds (1 new byte + 1 slot per block)
    for growing batches, with and without the per-round copy to the host."""
    cfg = model.cfg
    out = {}
    x = torch.zeros(1, device=device)
    for _ in range(5):
        (x + 1).cpu()
    t = []
    for _ in range(reps):
        s = time.perf_counter()
        (x + 1).cpu()
        t.append(time.perf_counter() - s)
    out["tiny_op_and_copy_ms"] = 1e3 * statistics.median(t)

    def run(bs, copy):
        cache = None
        tok = torch.full((bs, cfg.ctx), PAD, dtype=torch.long, device=device)
        pos = torch.arange(cfg.ctx, device=device).expand(bs, cfg.ctx).contiguous()
        qs = [torch.full((bs, 1), cfg.ctx + i, dtype=torch.long, device=device) for i in range(rounds + 1)]
        one = torch.zeros(bs, 1, dtype=torch.long, device=device)
        with amp:
            cache, logits = model.step(cache, tok, pos, qs[0])
            logits.cpu()
            t = []
            sync(device)
            s0 = s = time.perf_counter()
            for i in range(rounds):
                cache, logits = model.step(cache, one, qs[i], qs[i + 1])
                p = torch.softmax(logits.float(), -1)
                if copy:
                    p.cpu()
                    now = time.perf_counter()
                    t.append(now - s)
                    s = now
            sync(device)
            total = time.perf_counter() - s0
        return 1e3 * (statistics.median(t) if copy else total / rounds)

    rows = []
    for bs in sizes:
        run(bs, True)  # warm-up for this shape
        rows.append({"blocks": bs, "round_ms": run(bs, True), "round_ms_no_copy": run(bs, False)})
    out["k512_rounds"] = rows
    out["rounds_per_measure"] = rounds
    try:
        with amp, OpCounter() as c:
            cache, _ = model.step(None, torch.full((1, cfg.ctx), PAD, dtype=torch.long, device=device),
                                  torch.arange(cfg.ctx, device=device)[None], torch.full((1, 1), cfg.ctx, dtype=torch.long, device=device))
            c.n = 0
            model.step(cache, torch.zeros(1, 1, dtype=torch.long, device=device), torch.full((1, 1), cfg.ctx, dtype=torch.long, device=device),
                       torch.full((1, 1), cfg.ctx + 1, dtype=torch.long, device=device))
        out["aten_ops_per_round"] = c.n
    except Exception as e:  # private API: informational only
        out["aten_ops_per_round"] = None
        out["aten_ops_error"] = repr(e)
    return out


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def environment(device: torch.device) -> dict:
    env = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "torch_threads": torch.get_num_threads()}
    try:
        env["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        env["cpu"] = platform.processor()
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    code = Path(__file__).parent
    env["code_sha256"] = {f: sha256(code / f)[:16] for f in ("model.py", "latency.py", "codec.py", "schedule.py", "rans.py")}
    return env


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevzip.latency", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="input: a path or a name fetch.resolve knows (slices are taken from its start)")
    ap.add_argument("--weights", default=DEFAULT_MODEL, help="a path or a manifest name")
    ap.add_argument("--rounds", type=int, nargs="+", default=[16, 32, 64, 128, 512])
    ap.add_argument("--sizes", nargs="+", default=["64K", "1M"], help="slices of FILE (K = 1024, M = 1024^2)")
    ap.add_argument("--layouts", nargs="+", choices=["parallel", "codec"], default=["parallel", "codec"])
    ap.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp32", help="autocast dtype (fp16: GPUs only)")
    ap.add_argument("--runs", type=int, default=3, help="timed runs after a warm-up; the median is reported")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-batch", type=int, default=256, help="parallel layout: most blocks in one forward call")
    ap.add_argument("--roundtrip", type=int, nargs="*", help="K values for the true round trip (parallel layout, first size; default: all --rounds; none: skip)")
    ap.add_argument("--no-overhead", action="store_true", help="skip the per-round overhead probe")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "latency")
    args = ap.parse_args(argv)
    if args.runs < 1:
        ap.error("--runs must be >= 1")

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cpu" and args.dtype == "fp16":
        ap.error("fp16 autocast is for GPUs; use fp32 or bf16 on the CPU")
    amp = torch.autocast(device.type, dtype=DTYPES[args.dtype]) if DTYPES[args.dtype] else nullcontext()
    wpath, fpath = resolve(args.weights), resolve(args.file)
    model = JevZip.load(wpath).to(device)
    full = np.fromfile(fpath, np.uint8)
    entry = {
        # one entry per configuration: re-running it replaces the entry, other configurations stay
        "key": f"{wpath.name}|{fpath.name}|{args.dtype}|{','.join(args.sizes)}|{','.join(args.layouts)}|{','.join(map(str, args.rounds))}",
        "command": " ".join([Path(sys.executable).name, "-m", "jevzip.latency", *(argv if argv is not None else sys.argv[1:])]),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"), "device": str(device), "dtype": args.dtype,
        "weights": {"path": str(wpath), "sha256": sha256(wpath), "params": sum(p.numel() for p in model.parameters()), "config": vars(model.cfg)},
        "input": {"path": str(fpath), "bytes": len(full), "sha256": sha256(fpath)},
        "runs": args.runs, "max_batch": args.max_batch, "codec_chunk": CHUNK, "env": environment(device),
        "ar_note": f"K={model.cfg.block} is the AR baseline with a KV cache (one byte per round per block, all blocks batched)"
                   + (" on a model trained in linear (AR) order" if model.cfg.order == "linear" else f" run on a model trained in {model.cfg.order} order"),
        "results": [], "roundtrip": [],
    }
    print(f"{wpath.name} ({entry['weights']['params'] / 1e6:.2f}M, order {model.cfg.order}) on {fpath.name}, {device} {args.dtype}, "
          f"{args.threads} threads, median of {args.runs}", flush=True)
    if not args.no_overhead:
        # the probe's first call prefills ctx bytes per block: cheap on a GPU, seconds per 32 blocks on a CPU
        entry["overhead"] = overhead(model, device, amp, sizes=(1, 8, 32) if device.type == "cpu" else (1, 8, 32, 128, 256))
        print(json.dumps(entry["overhead"]), flush=True)
    rt_k = args.rounds if args.roundtrip is None else args.roundtrip
    for si, size in enumerate(args.sizes):
        data = full[: parse_size(size)]
        if len(data) < parse_size(size):
            print(f"{fpath.name} has only {len(full)} bytes: slice {size} is {len(data)} bytes", flush=True)
        sl = {"size": size, "bytes": len(data), "sha256": hashlib.sha256(data.tobytes()).hexdigest()}
        for layout in args.layouts:
            for k in args.rounds:
                res, (jobs, _, body) = measure(model, data, layout, k, device, amp, args.runs, args.max_batch)
                entry["results"].append({**sl, **res})
                print(f"{size} {layout:8} K={k:<4} calls {res['forward_calls']:6}  model {res['model_ms']:9.1f} ms "
                      f"({res['per_round_ms']:.2f}/round)  entropy {res['entropy_ms']:8.1f} ms  e2e {res['kb_per_s']:7.1f} KB/s  "
                      f"{res['bits_per_byte']:.3f} bpb  deterministic={res['deterministic']}", flush=True)
                if si == 0 and layout == "parallel" and k in rt_k:
                    rt = {**sl, "layout": layout, "K": k, **roundtrip(model, data, jobs, body, device, amp)}
                    entry["roundtrip"].append(rt)
                    print(f"  round trip on {device}: identical={rt['identical']}  {rt['e2e_ms_interleaved']:.0f} ms", flush=True)
    entry["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(entry, args.out, device.type)


def save(entry, out: Path, device: str):
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"latency-{device}.json"
    doc = json.loads(path.read_text()) if path.exists() else {"device": device, "entries": []}
    doc["entries"] = [e for e in doc["entries"] if e["key"] != entry["key"]] + [entry]
    path.write_text(json.dumps(doc, indent=1) + "\n")
    docs = [json.loads(p.read_text()) for p in sorted(out.glob("latency-*.json"))]
    (out / "latency.md").write_text(report(docs))
    print(f"wrote {path} and {out / 'latency.md'}", flush=True)


def report(docs) -> str:
    """latency.md from every latency-<device>.json."""
    lines = ["# Decode latency vs rounds K", "", "Generated by `python -m jevzip.latency` from the latency-*.json files next to it; "
             "they hold the exact commands, versions and input SHA-256s. Times are medians of the timed runs. "
             "*model*: the decoder's forward calls (step + KV cache, per-round upload of revealed bytes, per-round copy of the "
             "probabilities to the host, teacher-forced). *entropy*: quantize + rANS decode on the CPU. *e2e* = model + entropy. "
             "K=512 on a lanes model is the AR baseline with a KV cache. *parallel*: all blocks at once without previous-block "
             "context (depth K); *codec*: the codec's default layout (codec.default_streams chains, <= 128 blocks a call).", ""]
    for doc in docs:
        for e in doc["entries"]:
            env = e["env"]
            lines += [f"## {Path(e['weights']['path']).name} (order {e['weights']['config']['order']}) on {Path(e['input']['path']).name}: "
                      f"{e['device']}, {e['dtype']}", "",
                      f"`{e['command']}`  ", f"{env.get('cpu', '')}, torch {env['torch']}, {env['torch_threads']} threads, "
                      f"{e['runs']} timed runs, {e['started']} to {e.get('finished', '?')}", "",
                      "| size | layout | K | forward calls | depth | model ms | ms/round | entropy ms | e2e ms | KB/s | model share | bits/byte | vs K=512 (model) | bit-stable |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            ar = {(r["size"], r["layout"]): r["model_ms"] for r in e["results"] if r["ar_baseline"]}
            for r in e["results"]:
                base = ar.get((r["size"], r["layout"]))
                speed = f"{base / r['model_ms']:.1f}x" if base else ""
                label = f"{r['K']} (AR)" if r["ar_baseline"] else str(r["K"])
                lines.append(f"| {r['size']} | {r['layout']} | {label} | {r['forward_calls']} | {r['dependency_depth_rounds']} | {r['model_ms']:.0f} | "
                             f"{r['per_round_ms']:.2f} | {r['entropy_ms']:.0f} | {r['e2e_ms']:.0f} | {r['kb_per_s']:.1f} | {100 * r['model_share']:.0f}% | "
                             f"{r['bits_per_byte']:.3f} | {speed} | {'yes' if r['deterministic'] else 'NO'} |")
            if e["roundtrip"]:
                ok = all(r["identical"] for r in e["roundtrip"])
                lines += ["", f"True round trip on {e['device']} (encode with the device's probabilities, decode on the device feeding back "
                          f"decoded bytes): **{'all identical' if ok else 'FAILED'}**. " + ", ".join(
                              f"K={r['K']}: {'ok' if r['identical'] else 'mismatch at ' + str(r['first_mismatch'])} ({r['kb_per_s']:.1f} KB/s interleaved)"
                              for r in e["roundtrip"])]
            if "overhead" in e:
                lines += ["", explain(e)]
            lines.append("")
    return "\n".join(lines) + "\n"


def explain(e) -> str:
    """What bounds one round, in measured numbers."""
    o = e["overhead"]
    rows = {r["blocks"]: r for r in o["k512_rounds"]}
    one, big = rows[min(rows)], rows[max(rows)]
    ops = o.get("aten_ops_per_round")
    per_op = f" ({ops} aten ops, ~{1e3 * one['round_ms_no_copy'] / ops:.0f} us each)" if ops else ""
    table = " / ".join(f"{b}: {r['round_ms']:.2f}" for b, r in sorted(rows.items()))
    return (f"What one round costs on {e['device']} (K=512-style round: 1 new byte + 1 slot per block, cache ~{e['weights']['config']['ctx']} deep, "
            f"median of {o['rounds_per_measure']}): a bare op + copy to host takes {o['tiny_op_and_copy_ms']:.3f} ms; a 1-block round takes "
            f"{one['round_ms']:.2f} ms with its probability copy and {one['round_ms_no_copy']:.2f} ms when rounds are queued without waiting{per_op}. "
            f"ms per round by blocks per call: {table}. A round costs ~the same from 1 to {max(b for b, r in rows.items() if r['round_ms'] < 1.5 * one['round_ms'])} "
            f"blocks: below that it is bound by launching ~all of its kernels and one sync, not by arithmetic; "
            f"{big['blocks']} blocks take {big['round_ms'] / one['round_ms']:.1f}x the 1-block round.")


if __name__ == "__main__":
    main()
