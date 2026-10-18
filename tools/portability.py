"""Portability of jevzip decoding: does a .jz made here still decode under other conditions?

    python tools/portability.py run   --tag torch-2.14.1 [--cases k32-fp32,...] [--variants control,t1,t2,t8,cross,mps]
    OLD/bin/python -I tools/portability.py run --tag torch-2.8.0 --variants control,t1
    python tools/portability.py merge results/portability/raw/*.json   # -> portability.json + markdown table

Inputs are the pre-refactor reference files in runs/ref: networkx-64k.bin compressed with
weights/jevzip-lanes-11m-code.pt on this machine (torch 2.14.1, CPU, threads 4, default
streams, K = 32 and 128, fp32 and bf16). A variant changes one thing and runs the real codec
(codec.compress / codec.decompress) with hooks around it:

* control - nothing changed (in another venv: only the torch build changes);
* t1, t2, t8 - torch.set_num_threads is overridden (the header pins 4);
* cross - the header's bf16 flag is flipped (fp32 stream decoded with bf16 autocast and back);
* mps - the model's forward calls run on the mps device (bf16 streams under mps autocast).

Two measurements per (case, variant):

* decode: decompress() under the variant. ok, or how it failed (exception text, which check
  caught it) and the first wrong byte: its index in decode order and its file offset.
* teacher forcing: compress() of the original under the variant, compared forward call by
  forward call with the encoder's own (control) probabilities: fraction of quantized CDF
  entries that differ, rows with any difference, coded-symbol intervals that differ (only
  those break rANS: the decoder stays in sync exactly as long as the coded symbol's
  [start, start + freq) matches), max |dp| of the float probabilities, and the same counts
  if the CDFs were quantized to fewer than 24 bits.

The encoder's probabilities come from a control run in the environment that made runs/ref
(it must reproduce the .jz byte for byte) and are cached as .npy under --ref-cache so other
venvs can compare against them. The codec is imported from --pkg-root (default: this repo).
"""
import argparse
import contextlib
import copy
import hashlib
import json
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BITS = (24, 20, 16, 14, 12, 10)  # CDF precisions compared (rans.PROB_BITS = 24)
VARIANTS = {  # name -> (threads override, device, flip bf16 flag)
    "control": (None, "cpu", False),
    "t1": (1, "cpu", False),
    "t2": (2, "cpu", False),
    "t8": (8, "cpu", False),
    "cross": (None, "cpu", True),
    "mps": (None, "mps", False),
}


class Stop(Exception):
    """Raised by a hook to end a run early (call or GPU budget, decode stopped after divergence)."""


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def platform_info(torch, np) -> dict:
    cpu = platform.processor()
    with contextlib.suppress(Exception):
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        elif Path("/proc/cpuinfo").exists():
            cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name"))
    cfg = torch.__config__.show()
    blas = next((s.split("=", 1)[1] for s in cfg.replace("\n", ",").split(",") if s.strip().startswith("BLAS_INFO=")), "?")
    return {
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "os": platform.platform(), "machine": platform.machine(), "cpu": cpu, "blas": blas,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(), "mps_available": torch.backends.mps.is_available(),
    }


def quantize_bits(probs, bits, np):
    """rans.quantize at another precision (same rounding rule)."""
    scale = 1 << bits
    p = np.asarray(probs, np.float64)
    k = p.shape[-1]
    freq = np.floor(p / p.sum(-1, keepdims=True) * (scale - k)).astype(np.int64) + 1
    rows = np.arange(len(freq))
    freq[rows, freq.argmax(-1)] += scale - freq.sum(-1)
    cdf = np.zeros((len(freq), k + 1), np.int64)
    np.cumsum(freq, -1, out=cdf[:, 1:])
    return cdf


class Hooks:
    """Patches the codec for one run: threads override, model calls on a device, a per-call
    callback on_call(probs, cdfs, where, buf) after every emit, and a per-symbol decode check."""

    def __init__(self, codec, torch, model, threads=None, device="cpu", bf16=False, budget=None):
        self.codec, self.torch, self.model = codec, torch, model
        self.threads, self.device, self.bf16, self.budget = threads, device, bf16, budget
        self.on_call = self.on_pop = None
        self.where = None
        self.device_seconds = 0.0

    def __enter__(self):
        codec, torch = self.codec, self.torch
        self.saved = torch.set_num_threads, codec.quantize, codec._run_batch, codec.Decoder
        set_threads, quantize, run_batch, decoder = self.saved
        last = {}
        hooks = self

        def set_threads_(n):
            set_threads(self.threads or n)

        def quantize_(probs):
            last["probs"] = probs
            return quantize(probs)

        def run_batch_(*args):
            *head, emit = args

            def emit_(arg, where):
                # pre-refactor codec: emit(cdfs, where) after its own quantize(probs);
                # refactored codec: emit(probs, where), which quantize()s inside
                hooks.where, hooks.row = where, 0
                emit(arg, where)
                if hooks.on_call:
                    float_arg = arg.dtype.kind == "f"
                    probs = arg if float_arg else last["probs"]
                    hooks.on_call(probs, quantize(probs) if float_arg else arg, where, head[1])
                last.pop("probs", None)

            return run_batch(*head, emit_)

        class Tracked(decoder):
            def pop(self, cdf):
                s = super().pop(cdf)
                if hooks.on_pop:
                    hooks.on_pop(s, int(hooks.where[hooks.row]))
                hooks.row += 1
                return s

        torch.set_num_threads, codec.quantize, codec._run_batch, codec.Decoder = set_threads_, quantize_, run_batch_, Tracked
        if self.device != "cpu":
            self._device_step()
        return self

    def _device_step(self):
        torch, dev = self.torch, self.device
        dev_model = copy.deepcopy(self.model).to(dev)  # the CPU model stays for fingerprint()

        def step(cache, ctok, cpos, qpos):
            t = time.perf_counter()
            # default device: masks built inside model.step without a device land on dev too
            with torch.device(dev), torch.autocast(dev, dtype=torch.bfloat16, enabled=self.bf16):
                cache, logits = dev_model.step(cache, ctok.to(dev), cpos.to(dev), None if qpos is None else qpos.to(dev))
                logits = None if logits is None else logits.float().cpu()
            self.device_seconds += time.perf_counter() - t
            if self.budget is not None and self.budget.spend(time.perf_counter() - t):
                raise Stop("GPU time budget used up")
            return cache, logits

        self.model.step = step

    def __exit__(self, *exc):
        codec, torch = self.codec, self.torch
        torch.set_num_threads, codec.quantize, codec._run_batch, codec.Decoder = self.saved
        self.model.__dict__.pop("step", None)
        torch.set_num_threads(4)
        return False


class Budget:
    def __init__(self, seconds):
        self.left = seconds

    def spend(self, s) -> bool:
        self.left -= s
        return self.left <= 0


def header(codec, blob) -> dict:
    keys = "magic version mode model_id n crc rounds streams threads bf16".split()
    return dict(zip(keys, codec.HEADER.unpack_from(blob)))


def flip_bf16(codec, blob) -> bytes:
    f = list(codec.HEADER.unpack_from(blob))
    f[-1] = 1 - f[-1]
    return codec.HEADER.pack(*f) + blob[codec.HEADER.size :]


def teacher_forced(codec, model, hooks, data, h, np, ref=None, max_calls=0):
    """compress() under `hooks`; with ref, compare every forward call with the encoder's arrays.

    Returns (blob or None if stopped early, arrays if ref is None else stats)."""
    buf = np.frombuffer(data, np.uint8)
    calls, probs, cdfs, wheres = [], [], [], []
    st = {"calls": 0, "rows": 0, "cdf_entries_differ": 0, "rows_differ": 0, "rows_bitwise_equal": 0,
          "interval_mismatches": 0, "first_interval_mismatch": None, "max_abs_dp": 0.0, "sum_abs_dp": 0.0,
          "first_call": None, "coarser": {b: {"rows_differ": 0, "interval_mismatches": 0} for b in BITS}}
    row0 = [0]

    def on_call(p, c, where, _buf):
        i = len(calls)
        calls.append(len(where))
        if ref is None:
            probs.append(np.array(p, np.float32))
            cdfs.append(np.array(c, np.int32))
            wheres.append(np.array(where, np.int64))
        else:
            a, b = row0[0], row0[0] + len(where)
            if not np.array_equal(ref["where"][a:b], where):
                raise RuntimeError(f"call {i}: positions differ from the reference run (codec plan changed?)")
            rp, rc, sym = ref["probs"][a:b], ref["cdfs"][a:b].astype(np.int64), buf[where].astype(np.int64)
            dp = np.abs(np.asarray(p, np.float64) - rp)
            diff = c[:, 1:-1] != rc[:, 1:-1]
            rows = np.arange(len(sym))
            bad = (c[rows, sym] != rc[rows, sym]) | (c[rows, sym + 1] != rc[rows, sym + 1])
            st["cdf_entries_differ"] += int(diff.sum())
            st["rows_differ"] += int(diff.any(1).sum())
            st["rows_bitwise_equal"] += int((np.asarray(p, np.float32) == rp).all(1).sum())
            st["interval_mismatches"] += int(bad.sum())
            if bad.any() and st["first_interval_mismatch"] is None:
                j = int(np.argmax(bad))
                st["first_interval_mismatch"] = {"decode_index": a + j, "offset": int(where[j]), "call": i}
            st["max_abs_dp"] = max(st["max_abs_dp"], float(dp.max()))
            st["sum_abs_dp"] += float(dp.sum())
            if i == 0:
                st["first_call"] = {"rows": len(sym), "cdf_entries_differ_frac": float(diff.mean()), "max_abs_dp": float(dp.max())}
            for bits in BITS:
                qa, qb = quantize_bits(p, bits, np), quantize_bits(rp, bits, np)
                cb = st["coarser"][bits]
                cb["rows_differ"] += int((qa != qb).any(1).sum())
                cb["interval_mismatches"] += int(((qa[rows, sym] != qb[rows, sym]) | (qa[rows, sym + 1] != qb[rows, sym + 1])).sum())
        row0[0] += len(where)
        st["calls"] = len(calls)
        st["rows"] = row0[0]
        if max_calls and len(calls) >= max_calls:
            raise Stop(f"stopped after {max_calls} forward calls")

    hooks.on_call = on_call
    blob = None
    with contextlib.suppress(Stop):
        blob = codec.compress(data, model, h["rounds"], h["streams"], h["threads"], bool(h["bf16"]))
    hooks.on_call = None
    if ref is None:
        return blob, {"probs": np.concatenate(probs), "cdfs": np.concatenate(cdfs), "where": np.concatenate(wheres)}
    n = max(1, st["rows"])
    st["cdf_entries_differ_frac"] = st["cdf_entries_differ"] / (n * 255)  # interior entries: cdf[0], cdf[256] are fixed
    st["rows_differ_frac"] = st["rows_differ"] / n
    st["rows_bitwise_equal_frac"] = st["rows_bitwise_equal"] / n
    st["mean_abs_dp"] = st.pop("sum_abs_dp") / (n * 256)
    st["coarser"] = {str(b): v for b, v in st["coarser"].items()}
    return blob, st


def decode(codec, model, hooks, blob, data, stop_after=None):
    """decompress() under `hooks`: outcome, first wrong byte, symbols decoded."""
    first, count, wrong = [None], [0], [0]

    def on_pop(s, offset):
        if s != data[offset]:
            wrong[0] += 1
            if first[0] is None:
                first[0] = {"decode_index": count[0], "offset": offset, "block": offset // 512, "in_block": offset % 512}
        count[0] += 1
        if first[0] is not None and stop_after is not None and count[0] - first[0]["decode_index"] > stop_after:
            raise Stop(f"stopped {stop_after} symbols after the first wrong byte")

    hooks.on_pop = on_pop
    t = time.time()
    out, err = None, None
    try:
        out = codec.decompress(blob, model)
    except Stop as e:
        err = ("stopped", str(e))
    except Exception as e:  # report every failure mode, the point of the exercise
        err = (type(e).__name__, str(e))
    hooks.on_pop = None
    r = {"ok": out == data, "seconds": round(time.time() - t, 2), "symbols_decoded": count[0], "of": len(data),
         "wrong_bytes_seen": wrong[0], "first_wrong": first[0]}
    if out == data:
        r["outcome"] = "ok"
    elif err is None:
        r["outcome"] = "wrong output, no error raised"
    else:
        msg = err[1]
        r["error"] = f"{err[0]}: {msg}"
        r["outcome"] = ("stopped early" if err[0] == "stopped" else
                        "CRC mismatch" if "CRC" in msg else
                        "rANS end state (trailing data)" if "trailing" in msg else
                        "rANS stream exhausted" if "exhausted" in msg else
                        "model fingerprint mismatch" if "different model" in msg else
                        f"exception ({err[0]})")
    return r


def run(args):
    sys.path.insert(0, str(Path(args.pkg_root).resolve()))
    import numpy as np
    import torch

    from jevzip import codec
    from jevzip.model import JevZip

    torch.set_num_threads(4)
    pkg = Path(codec.__file__).parent
    weights, data_path, ref_dir = Path(args.weights), Path(args.ref_dir) / "networkx-64k.bin", Path(args.ref_dir)
    data = data_path.read_bytes()
    model = JevZip.load(weights)
    probe = np.random.default_rng(0).dirichlet(np.ones(256), 64)
    from jevzip.rans import quantize
    assert np.array_equal(quantize_bits(probe, 24, np), quantize(probe)), "quantize_bits(24) must match rans.quantize"
    out = {
        "tag": args.tag, "command": shlex.join([sys.executable, *sys.argv]), "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform_info(torch, np),
        "code": {"pkg": str(pkg), **{f: sha256(pkg / f) for f in ("codec.py", "model.py", "rans.py", "schedule.py")}},
        "inputs": {str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p): sha256(p) for p in (weights, data_path)},
        "runs": [],
    }
    out_path = Path(args.out or REPO / "results/portability/raw" / f"{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.ref_cache)
    cache.mkdir(parents=True, exist_ok=True)
    budget = Budget(args.gpu_budget)
    for case in args.cases.split(","):
        ref_path = ref_dir / f"networkx-64k-{case}-t4.jz"
        blob = ref_path.read_bytes()
        out["inputs"][str(ref_path.relative_to(REPO)) if ref_path.is_relative_to(REPO) else str(ref_path)] = sha256(ref_path)
        h = header(codec, blob)
        key = cache / f"{case}-{sha256(ref_path)[:12]}"
        if not key.with_suffix(".probs.npy").exists():  # the encoder's own probabilities: control run here, must reproduce the .jz
            t = time.time()
            with Hooks(codec, torch, model) as hk:
                enc, ref = teacher_forced(codec, model, hk, data, h, np)
            if enc != blob:
                sys.exit(f"{case}: this environment does not reproduce {ref_path.name}; run the encoding environment first to fill {cache}")
            for k, v in ref.items():
                np.save(key.with_suffix(f".{k}.npy"), v)
            sizes = {}
            rows = np.arange(len(ref["where"]))
            s = np.frombuffer(data, np.uint8)[ref["where"]].astype(np.int64)
            for bits in BITS:
                q = quantize_bits(ref["probs"], bits, np)
                sizes[str(bits)] = float(-np.log2((q[rows, s + 1] - q[rows, s]) / (1 << bits)).sum() / 8)
            (key.with_suffix(".sizes.json")).write_text(json.dumps(sizes))
            print(f"{case}: reference probabilities cached ({time.time() - t:.0f}s)", flush=True)
        ref = {k: np.load(key.with_suffix(f".{k}.npy"), mmap_mode="r") for k in ("probs", "cdfs", "where")}
        sizes = json.loads(key.with_suffix(".sizes.json").read_text())
        for name in args.variants.split(","):
            threads, device, flip = VARIANTS[name]
            if device == "mps" and not torch.backends.mps.is_available():
                out["runs"].append({"case": case, "variant": name, "skipped": "mps not available"})
                continue
            bf16 = bool(h["bf16"]) != flip
            hh = dict(h, bf16=int(bf16))
            gpu = device != "cpu"
            r = {"case": case, "variant": name, "threads": threads or h["threads"], "device": device, "bf16": bf16,
                 "header": {k: h[k] for k in ("n", "rounds", "streams", "threads", "bf16")}, "code_bytes_at_bits": sizes}
            t = time.time()
            try:
                with Hooks(codec, torch, model, threads, device, bf16 and gpu, budget if gpu else None) as hk:
                    enc, st = teacher_forced(codec, model, hk, data, hh, np, ref, args.mps_calls if gpu else args.tf_calls)
                    r["teacher_forced"] = dict(st, encode_identical=None if enc is None else enc == (flip_bf16(codec, blob) if flip else blob))
                    r["teacher_forced"]["seconds"] = round(time.time() - t, 2)
                    if not args.no_decode:
                        r["decode"] = decode(codec, model, hk, flip_bf16(codec, blob) if flip else blob, data, args.mps_stop_after if gpu else None)
                    if gpu:
                        r["device_seconds"] = round(hk.device_seconds, 2)
            except Exception as e:
                r["error"] = f"{type(e).__name__}: {e}"
            out["runs"].append(r)
            tf, dc = r.get("teacher_forced", {}), r.get("decode", {})
            print(f"{case:10s} {name:8s} decode={dc.get('outcome')} first_wrong={dc.get('first_wrong')} "
                  f"tf: calls={tf.get('calls')} rows_differ={tf.get('rows_differ_frac')} interval_mism={tf.get('interval_mismatches')} "
                  f"max|dp|={tf.get('max_abs_dp')} identical={tf.get('encode_identical')} {r.get('error', '')} ({time.time() - t:.0f}s)", flush=True)
            out_path.write_text(json.dumps(out, indent=1))
    out["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    out_path.write_text(json.dumps(out, indent=1))
    print("wrote", out_path)


def merge(args):
    """Combine raw run files (and `python -m jevzip.portability_check --json` outputs) into one
    JSON and print the markdown matrix."""
    raws = [json.loads(Path(p).read_text()) for p in args.raw]
    fixture_checks = [r for r in raws if "results" in r]
    raws = [r for r in raws if "runs" in r]
    rows = []
    for raw in raws:
        for r in raw["runs"]:
            rows.append(dict(r, env=raw["tag"], torch=raw["platform"]["torch"]))
    merged = {"what": "jevzip decode portability matrix (tools/portability.py)", "environments": raws,
              "fixture_checks": fixture_checks}
    Path(args.out).write_text(json.dumps(merged, indent=1) + "\n")
    print("| case | env | variant | decode | first wrong byte (decode idx / offset) | CDF entries differ | rows differ "
          "| coded intervals differ (24 / 16 / 12 bits) | max abs dp |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if "skipped" in r or "error" in r:
            print(f"| {r['case']} | {r['env']} | {r['variant']} | {r.get('skipped') or r.get('error')} | | | | | |")
            continue
        d, t = r.get("decode", {}), r["teacher_forced"]
        fw = d.get("first_wrong")
        fws = f"{fw['decode_index']} / {fw['offset']}" if fw else "-"
        co = t["coarser"]
        print(f"| {r['case']} | {r['env']} | {r['variant']}{' (bf16)' if r['bf16'] else ''} | {d.get('outcome', 'n/a')} | {fws} | "
              f"{t['cdf_entries_differ_frac']:.2%} | {t['rows_differ_frac']:.2%} | {t['interval_mismatches']} / "
              f"{co['16']['interval_mismatches']} / {co['12']['interval_mismatches']} of {t['rows']} | {t['max_abs_dp']:.2e} |")
    for f in fixture_checks:
        ok = sum(r["ok"] for r in f["results"])
        modes = sorted({r["mode"] for r in f["results"]})
        print(f"\nfixtures (tiny models), torch {f['decoded_on']['torch']} on {f['decoded_on']['os']}: {ok}/{len(f['results'])} ok; modes {modes}")
    print("wrote", args.out, file=sys.stderr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True, help="environment label, names the raw output file")
    r.add_argument("--cases", default="k32-fp32,k128-fp32,k32-bf16,k128-bf16")
    r.add_argument("--variants", default="control,t1,t2,t8,cross")
    r.add_argument("--pkg-root", default=str(REPO), help="directory holding the jevzip package to test")
    r.add_argument("--weights", default=str(REPO / "weights/jevzip-lanes-11m-code.pt"))
    r.add_argument("--ref-dir", default=str(REPO / "runs/ref"))
    r.add_argument("--ref-cache", required=True, help="directory for the encoder's cached probabilities (~130 MB per case)")
    r.add_argument("--tf-calls", type=int, default=0, help="compare only the first N forward calls (0 = all)")
    r.add_argument("--mps-calls", type=int, default=0, help="the same for mps runs")
    r.add_argument("--mps-stop-after", type=int, default=None, help="stop an mps decode N symbols after its first wrong byte")
    r.add_argument("--gpu-budget", type=float, default=45.0, help="seconds of mps model calls allowed in total")
    r.add_argument("--no-decode", action="store_true")
    r.add_argument("--out", default=None)
    m = sub.add_parser("merge")
    m.add_argument("raw", nargs="+")
    m.add_argument("--out", default=str(REPO / "results/portability/portability.json"))
    args = ap.parse_args(argv)
    (run if args.cmd == "run" else merge)(args)


if __name__ == "__main__":
    main()
