"""Decode the committed .jz fixtures and report whether this machine reproduces them.

    python -m jevzip.portability_check            # table + platform info; exit 0 if all decode, else 1
    python -m jevzip.portability_check --json out.json
    python -m jevzip.portability_check --make     # (maintainers) rebuild jevzip/testdata here

jevzip decodes only if the decoder computes bit-identical probabilities to the encoder's,
so a stream is tied to the torch build, CPU kernels and thread count that made it. The
fixtures in jevzip/testdata are tiny random models (exported with JevZip.export, so they
load like real weights) and short files coded with them on one machine (manifest.json
records which); CI runs this on other OSes and CPUs as an informational job. Per fixture
it reports ok, or the failure mode: the codec's own check that caught it (rANS stream
exhausted / trailing data / CRC), a wrong output nothing caught, or another exception. It
also re-encodes the original with the header's settings and reports whether the bytes
match the fixture, which is the same question asked from the encoder's side.

The models are random (weights scaled up so the distributions are peaked rather than near
uniform), d_model 32 and 2 layers: much smaller matrices than real weights, so they take
different kernel paths; a pass here is necessary, not sufficient, for real weights to
decode (results/portability has the full-size measurement).
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import codec
from .model import Config, JevZip

TESTDATA = Path(__file__).resolve().parent / "testdata"
MODELS = {  # name -> config of the tiny random model (seed = position in this dict)
    "tiny-bitrev": dict(ctx=32, block=64, d_model=32, n_layers=2, n_heads=2, ff_mult=2, first=4, order="bitrev"),
    "tiny-lanes": dict(ctx=32, block=64, d_model=32, n_layers=2, n_heads=2, ff_mult=2, first=4, order="lanes"),
}
WEIGHT_SCALE = 8.0  # matrices x8: peaked distributions, like a trained model's, not flat ones
ORIGINALS = {"a.bin": (0, 2048), "b.bin": (2048, 1000)}  # slices of SOURCE: 32 full blocks / 15 full + 1 short
SOURCE = Path(__file__).resolve().parent / "schedule.py"  # jevzip's own source (MIT), as of the fixtures' commit
CASES = [  # (model, original, rounds, threads)
    *((m, "a.bin", k, t) for m in MODELS for k in (8, 32) for t in (1, 4)),
    ("tiny-bitrev", "b.bin", 8, 1),
    ("tiny-lanes", "b.bin", 32, 4),
]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def platform_info() -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "os": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability()}


def failure_mode(msg: str) -> str:
    for key, mode in (("CRC", "CRC mismatch"), ("trailing", "rANS end state"), ("exhausted", "rANS stream exhausted"),
                      ("different model", "model fingerprint"), ("not a jevzip", "bad header")):
        if key in msg:
            return mode
    return "exception"


def make(force=False):
    """Build the fixtures in jevzip/testdata (fp32 only; the header records threads)."""
    if (TESTDATA / "manifest.json").exists() and not force:
        sys.exit("jevzip/testdata/manifest.json exists: pass --force to rebuild (old fixtures then stop meaning anything)")
    TESTDATA.mkdir(exist_ok=True)
    src = SOURCE.read_bytes()
    man = {"what": "jevzip decode-portability fixtures (python -m jevzip.portability_check)",
           "made": time.strftime("%Y-%m-%d %H:%M:%S %z"), "platform": platform_info(), "weight_scale": WEIGHT_SCALE,
           "models": {}, "originals": {}, "fixtures": []}
    models = {}
    for seed, (name, cfg) in enumerate(MODELS.items()):
        torch.manual_seed(seed)
        m = JevZip(Config(**cfg))
        with torch.no_grad():
            for v in m.state_dict().values():
                if v.dim() == 2:
                    v.mul_(WEIGHT_SCALE)
        path = TESTDATA / f"{name}.pt"
        torch.save(m.export(), path)
        models[name] = JevZip.load(path)  # code with the bf16-rounded weights a user would load
        man["models"][name] = {"file": path.name, "sha256": sha256(path.read_bytes()), "config": cfg, "seed": seed,
                               "fingerprint": codec.fingerprint(models[name]).hex()}
    for name, (start, n) in ORIGINALS.items():
        data = src[start : start + n]
        assert len(data) == n, f"{SOURCE} is too short for {name}"
        (TESTDATA / name).write_bytes(data)
        man["originals"][name] = {"sha256": sha256(data), "bytes": n, "source": f"{SOURCE.name}[{start}:{start + n}]"}
    for model, orig, rounds, threads in CASES:
        data = (TESTDATA / orig).read_bytes()
        blob = codec.compress(data, models[model], rounds=rounds, threads=threads, bf16=False, raw_fallback=False)
        assert codec.decompress(blob, models[model]) == data
        name = f"{model}-{Path(orig).stem}-k{rounds}-t{threads}.jz"
        (TESTDATA / name).write_bytes(blob)
        man["fixtures"].append({"file": name, "model": model, "original": orig, "rounds": rounds, "threads": threads,
                                "bf16": False, "bytes": len(blob), "sha256": sha256(blob)})
    (TESTDATA / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    total = sum(p.stat().st_size for p in TESTDATA.iterdir() if p.is_file())
    print(f"wrote {len(man['fixtures'])} fixtures to {TESTDATA} ({total / 1024:.0f} KiB)")


def check(fixture: dict, models: dict, originals: dict) -> dict:
    """Decode one fixture; also re-encode its original and compare with the fixture bytes."""
    blob = (TESTDATA / fixture["file"]).read_bytes()
    data = originals[fixture["original"]]
    model = models[fixture["model"]]
    r = {"fixture": fixture["file"], "ok": False}
    if sha256(blob) != fixture["sha256"]:
        return dict(r, mode="fixture file changed (sha256)")
    t = time.perf_counter()
    try:
        out = codec.decompress(blob, model)
        r.update(ok=out == data, mode="ok" if out == data else "wrong output, no error raised")
    except Exception as e:  # every failure mode is a result here
        r.update(mode=failure_mode(str(e)), error=f"{type(e).__name__}: {e}")
    h = dict(zip("magic version mode model_id n crc rounds streams threads bf16".split(), codec.HEADER.unpack_from(blob)))
    try:
        enc = codec.compress(data, model, rounds=h["rounds"], streams=h["streams"], threads=h["threads"], bf16=bool(h["bf16"]), raw_fallback=False)
        r["reencode_identical"] = enc == blob
        if enc != blob:
            n = min(len(enc), len(blob))
            r["reencode_first_diff_byte"] = next((i for i in range(n) if enc[i] != blob[i]), n)
    except Exception as e:
        r["reencode_identical"] = None
        r["reencode_error"] = f"{type(e).__name__}: {e}"
    r["seconds"] = round(time.perf_counter() - t, 3)
    return r


def run(json_out=None) -> int:
    man = json.loads((TESTDATA / "manifest.json").read_text())
    models, originals, bad = {}, {}, []
    for name, m in man["models"].items():
        path = TESTDATA / m["file"]
        if sha256(path.read_bytes()) != m["sha256"]:
            bad.append(f"{m['file']}: sha256 differs from manifest")
        models[name] = JevZip.load(path)
    for name, o in man["originals"].items():
        originals[name] = (TESTDATA / name).read_bytes()
        if sha256(originals[name]) != o["sha256"]:
            bad.append(f"{name}: sha256 differs from manifest")
    if bad:
        print("broken fixtures:", *bad, sep="\n  ")
        return 1
    here = platform_info()
    rows = [check(f, models, originals) for f in man["fixtures"]]
    enc = man["platform"]
    print(f"encoded on: torch {enc['torch']}, {enc['os']} ({enc['machine']}, {enc['cpu_capability']})")
    print(f"decoding on: torch {here['torch']}, {here['os']} ({here['machine']}, {here['processor'] or '?'}, {here['cpu_capability']}), numpy {here['numpy']}")
    w = max(len(r["fixture"]) for r in rows)
    print(f"\n{'fixture':{w}s}  decode  {'failure mode':24s}  re-encode identical")
    for r in rows:
        print(f"{r['fixture']:{w}s}  {'ok' if r['ok'] else 'FAIL':6s}  {'' if r['ok'] else r['mode']:24s}  {r.get('reencode_identical')}")
    n_ok = sum(r["ok"] for r in rows)
    print(f"\n{n_ok}/{len(rows)} fixtures decode on this machine")
    if json_out:
        Path(json_out).write_text(json.dumps({"encoded_on": enc, "decoded_on": here, "results": rows}, indent=1) + "\n")
    return 0 if n_ok == len(rows) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", help="also write the results here")
    ap.add_argument("--make", action="store_true", help="rebuild the fixtures on this machine (maintainers)")
    ap.add_argument("--force", action="store_true", help="with --make: overwrite existing fixtures")
    args = ap.parse_args(argv)
    if args.make:
        make(args.force)
        return 0
    return run(args.json)


if __name__ == "__main__":
    sys.exit(main())
