"""Where jevzip's real sizes exceed the model's cross-entropy (results/gap/gap.md).

    nice -n 10 .venv/bin/python results/gap/gap.py run      # CPU, threads 4, fp32: ~2 h; appends to runs.json
    .venv/bin/python results/gap/gap.py report             # runs.json -> gap.json + gap.md

run: codelength() (jevzip/codelength.py) with --encode, which codes exactly
what compress() codes (checked byte-for-byte against runs/ref), so the real
size, the quantized and float codelengths and the oracle-context codelength all
come from the same forward calls; plus the pre-refactor evalk (64 random
512-byte windows, git HEAD's make_batch/loss_bits) in bf16 and fp32.
"""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from jevzip.codec import CHUNK, default_streams  # noqa: E402
from jevzip.codelength import codelength, cross_entropy  # noqa: E402
from jevzip.model import JevZip  # noqa: E402
from jevzip.train import loss_bits, make_batch  # noqa: E402

OUT = Path(__file__).parent
WEIGHTS = REPO / "weights" / "jevzip-lanes-11m-code.pt"
FILES = {"networkx": REPO / "data/code/networkx.py.txt", "pydoc_topics": REPO / "data/code/pydoc_topics.txt"}
THREADS = 4
OLD_DEFAULT = lambda nb: max(1, nb // 8)  # noqa: E731  pre-refactor compress() default
# (file, K, streams or None = new default, oracle): most important first
JOBS = [("networkx", 32, None, True), ("networkx", 128, None, True), ("networkx", 32, "old", False), ("networkx", 128, "old", False),
        ("networkx", 32, 2048, False), ("networkx", 32, 64, False), ("networkx", 32, 16, False), ("networkx", 128, 2048, False),
        ("pydoc_topics", 32, None, True), ("pydoc_topics", 128, None, True), ("networkx", 32, 1, False), ("networkx", 32, "ce", False),
        ("pydoc_topics", 32, 113, False), ("pydoc_topics", 128, 113, False)]  # 113 = default_streams(1015) after its last change; "None" ran with 126


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def old_evalk(model, data, k, bf16, windows=64):
    """The pre-refactor evalk estimate, verbatim: mean of per-16-window-batch losses."""
    bits = []
    with torch.no_grad():
        for i in range(0, windows, 16):
            batch = make_batch(data, model.cfg, min(16, windows - i), np.random.default_rng(i), p_noctx=0, p_trunc=0, rounds=k)
            bits.append(loss_bits(model, batch, "cpu", amp=bf16).item())
    return float(np.mean(bits))


def run():
    torch.set_num_threads(THREADS)
    model = JevZip.load(WEIGHTS)
    path = OUT / "runs.json"
    db = json.loads(path.read_text()) if path.exists() else {"runs": [], "evalk_old": []}
    done = {(r["file"], r["K"], r["job"]) for r in db["runs"]}
    db["meta"] = {"weights": str(WEIGHTS.relative_to(REPO)), "weights_sha256": sha(WEIGHTS), "threads": THREADS, "bf16": False,
                  "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "machine": platform.platform(),
                  "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                  "files": {k: {"path": str(v.relative_to(REPO)), "sha256": sha(v), "n": v.stat().st_size} for k, v in FILES.items()},
                  "command": "nice -n 10 .venv/bin/python results/gap/gap.py run"}

    def save():
        path.write_text(json.dumps(db, indent=1) + "\n")

    if not db["evalk_old"]:
        data = np.memmap(FILES["networkx"], dtype=np.uint8, mode="r")
        for k in (32, 128):
            for bf16 in (True, False):
                t = time.time()
                db["evalk_old"].append({"file": "networkx", "K": k, "bf16": bf16, "windows": 64, "bpb": old_evalk(model, data, k, bf16),
                                        "seconds": round(time.time() - t, 1)})
                print(db["evalk_old"][-1], flush=True)
        save()
    for name, k, streams, oracle in JOBS:
        job = str(streams)
        if (name, k, job) in done:
            continue
        data = FILES[name].read_bytes()
        nb = -(-len(data) // model.cfg.block)
        t = time.time()
        if streams == "ce":
            cl = cross_entropy(data, model, k, THREADS)
        else:
            s = {None: default_streams(nb), "old": OLD_DEFAULT(nb)}.get(streams, streams)
            cl = codelength(data, model, k, s, THREADS, oracle=oracle, encode=True)
        B = model.cfg.block
        perblock = lambda a: np.add.reduceat(np.nan_to_num(a), np.arange(0, len(a), B)).tolist()  # noqa: E731
        rec = {"file": name, "K": k, "job": job, "n": cl.n, "blocks": nb, "streams": cl.streams, "depth": cl.depth, "calls": cl.calls,
               "calls_per_round_batch": -(-cl.streams // CHUNK), "header": cl.header, "body": cl.body, "size": cl.size,
               "bits": float(np.sum(cl.bits)), "bits_float": float(np.sum(cl.bits_float)), "seconds": round(time.time() - t, 1),
               "start_blocks": len(cl.starts) if streams != "ce" else 1}
        if oracle:
            rec |= {"oracle_bits": float(np.sum(cl.oracle_bits)), "oracle_bits_float": float(np.sum(cl.oracle_bits_float)),
                    "oracle_block_bits_float": perblock(cl.oracle_bits_float), "block_bits_float": perblock(cl.bits_float)}
        if streams == "ce":
            rec["block_bits_float"] = perblock(cl.bits_float)
        db["runs"].append(rec)
        print({k2: v for k2, v in rec.items() if "block_bits" not in k2}, flush=True)
        save()


def report():
    db = json.loads((OUT / "runs.json").read_text())
    runs = {(r["file"], r["K"], r["job"]): r for r in db["runs"]}
    out = {"meta": db["meta"], "decomposition": [], "streams_sweep": [], "evalk": [], "bootstrap": []}
    md = []
    for (name, k, job), r in sorted(runs.items()):
        if job == "ce":
            continue
        n = r["n"]
        base = runs.get((name, k, "None"))
        if base is None or "oracle_bits_float" not in base:
            continue
        oracle = base["oracle_bits_float"]
        nb = r["blocks"]
        label = " + ".join(lbl for s, lbl in ((default_streams(nb), "new default"), (OLD_DEFAULT(nb), "old default (blocks//8)")) if r["streams"] == s)
        if label:
            parts = {"header_and_rans_bytes": r["size"] - r["bits"] / 8, "quantization_bytes": (r["bits"] - r["bits_float"]) / 8,
                     "chain_start_bytes": (r["bits_float"] - oracle) / 8}
            out["decomposition"].append({"file": name, "K": k, "policy": label,
                                         "streams": r["streams"], "real_bytes": r["size"], "real_bpb": 8 * r["size"] / n,
                                         "oracle_ce_bytes": oracle / 8, "oracle_ce_bpb": oracle / n,
                                         "gap_pct": 100 * (8 * r["size"] / oracle - 1),
                                         "parts_bytes": parts, "parts_pct_of_oracle": {p: 100 * v * 8 / oracle for p, v in parts.items()}})
        out["streams_sweep"].append({"file": name, "K": k, "streams": r["streams"], "depth": r["depth"], "calls": r["calls"], "start_blocks": r["start_blocks"],
                                     "real_bytes": r["size"], "real_bpb": 8 * r["size"] / n, "over_oracle_pct": 100 * (8 * r["size"] / oracle - 1),
                                     "seconds_provisional": r["seconds"]})
    for e in db["evalk_old"]:
        base = runs.get((e["file"], e["K"], "None"))
        if base:
            o = base["oracle_bits_float"] / base["n"]
            out["evalk"].append(e | {"oracle_ce_bpb": o, "evalk_vs_oracle_pct": 100 * (e["bpb"] / o - 1)})
    for e in out["evalk"]:  # the observed gap: old evalk (bf16) -> real size at the old default, in bits/byte, summing exactly
        if not e["bf16"]:
            continue
        fp32 = next(x["bpb"] for x in out["evalk"] if x["K"] == e["K"] and not x["bf16"])
        r, base = runs[(e["file"], e["K"], "old")], runs[(e["file"], e["K"], "None")]
        n = r["n"]
        steps = {"evalk bf16 -> fp32": fp32 - e["bpb"], "64-window sample -> full file (oracle CE)": base["oracle_bits_float"] / n - fp32,
                 "chain starts (256 chains)": (r["bits_float"] - base["oracle_bits_float"]) / n, "quantization": (r["bits"] - r["bits_float"]) / n,
                 "header + rANS flush": (8 * r["size"] - r["bits"]) / n}
        out.setdefault("observed_gap", []).append({"file": e["file"], "K": e["K"], "evalk_bf16_bpb": e["bpb"], "real_bpb": 8 * r["size"] / n,
                                                   "gap_pct": 100 * (8 * r["size"] / n / e["bpb"] - 1), "steps_bpb": steps,
                                                   "steps_pct_of_evalk": {k2: 100 * v / e["bpb"] for k2, v in steps.items()}})
    rng = np.random.default_rng(0)
    for (name, k, job), r in sorted(runs.items()):
        if job != "None" or "oracle_block_bits_float" not in r:
            continue
        lens = np.minimum(512, r["n"] - 512 * np.arange(r["blocks"]))
        blk = np.array(r["oracle_block_bits_float"]) / lens  # bits/byte of each block
        est = np.array([blk[rng.choice(len(blk), 64, replace=False)].mean() for _ in range(10000)])
        out["bootstrap"].append({"file": name, "K": k, "windows": 64, "blocks": len(blk), "full_file_bpb": float(blk.mean()),
                                 "mean": float(est.mean()), "std": float(est.std()), "std_pct": float(100 * est.std() / blk.mean()),
                                 "p2_5": float(np.percentile(est, 2.5)), "p97_5": float(np.percentile(est, 97.5))})
    ce = runs.get(("networkx", 32, "ce"))
    if ce and ("networkx", 32, "None") in runs:
        out["cross_entropy_check"] = {"K": 32, "cross_entropy_bits": ce["bits_float"], "codelength_oracle_bits": runs[("networkx", 32, "None")]["oracle_bits_float"],
                                      "rel_diff": ce["bits_float"] / runs[("networkx", 32, "None")]["oracle_bits_float"] - 1, "calls": ce["calls"]}
    (OUT / "gap.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    {"run": run, "report": report}[sys.argv[1]]()
