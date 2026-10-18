"""Baseline compressors that jevzip is measured against, one registry for bench.py and the results.

    python -m jevzip.baselines FILE [--corpus TRAIN.bin] [--only NAME ...] [--group fast|all] [--json OUT]
    python -m jevzip.baselines --report OUT.md PART.json [PART.json ...]   # merge runs into one table

Every row is a measured size with a verified round trip (decompress and compare), wall times,
the exact command and the tool version. Three kinds of rows:

  standalone  a self-contained file: python-native gzip/bzip2/xz/zstd/brotli, the zstd and brotli
              CLIs at their strongest settings, 7-Zip PPMd, paq8px, cmix.
  dictionary  the decoder also needs a reference built from the training corpus (a zstd dictionary,
              the whole corpus for zstd --patch-from, a brotli raw dictionary). Its size is recorded
              as ref_bytes: it plays the part jevzip's weights play in total-size accounting.
  marginal    |C(P + x)| - |C(P)| with P a fixed tail of the training corpus: the size a coder
              primed on P adds for x. Equivalent to a preset-dictionary coder up to a few bytes;
              not a standalone file, so not round-trip verified (verified = null).

These are the apples-to-apples rows: jevzip also learns from the training corpus and does not
count its weights. CLI tools that are not installed are skipped with the reason; the strong
ones come from tools/build_competitors.sh (paq8px and cmix under downloads/, 7zz from Homebrew).
Times are wall seconds on whatever else the machine was doing: sizes are what is comparable.
"""
import argparse
import bz2
import gzip
import hashlib
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parent.parent
DL = REPO / "downloads"
PAQ8PX = DL / "paq8px-build" / "paq8px"
PAQ8PX_LEVEL = "-8L"  # level 8 (2.4 GB) + LSTM: -12L codes 100 KB of code to the same bytes in 7.6 GB
PAQ8PX_PRIMED_LEVEL = "-8"  # the primed rows code 16-21 MB: -8L would take hours per run (~2 KB/s)
CMIX = DL / "cmix-build" / "cmix"
CMIX_DICT = DL / "cmix-build" / "dictionary" / "english.dic"
DICT_SAMPLE = 32 << 20  # bytes of corpus cut into 4 KB samples for zstd --train
PRIME = {"xz": 64 << 20, "ppmd": 64 << 20, "paq8px": 16 << 20}  # |P| for the marginal rows
PPMD = "-m0=PPMd:mem=1024m:o=32"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _which(tool) -> str | None:
    return shutil.which(tool) or (str(tool) if os.access(tool, os.X_OK) and Path(tool).is_file() else None)


def _run(argv, cwd=None, stdin=None, stdout=None) -> float:
    """Run a command (raise with its stderr on failure); return wall seconds."""
    t = time.time()
    r = subprocess.run([str(a) for a in argv], cwd=cwd, stdin=stdin, stdout=stdout or subprocess.DEVNULL, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError(f"{' '.join(map(str, argv))}: exit {r.returncode}: {r.stderr.decode(errors='replace')[-2000:]}")
    return time.time() - t


def _version(argv, pattern=r".+") -> str:
    try:
        r = subprocess.run([str(a) for a in argv], capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return "?"
    m = re.search(pattern, (r.stdout + r.stderr).decode(errors="replace"))
    return m.group(0).strip() if m else "?"


def _build_version(binary: Path) -> str:
    f = binary.parent / "VERSION"
    return f.read_text().strip() if f.exists() else "?"


class Ctx:
    """What a run shares: the training corpus, a work dir, built dictionaries, cached |C(P)|."""

    def __init__(self, corpus=None, workdir=None, cache=None):
        self.corpus = Path(corpus).resolve() if corpus else None
        self.work = Path(workdir or tempfile.mkdtemp(prefix="jevzip-baselines-"))
        self.cache_path = Path(cache) if cache else None
        self.cache = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        self._built = {}

    def tail(self, n: int) -> Path:
        """The last n bytes of the corpus, as a file (P for the marginal rows)."""
        size = self.corpus.stat().st_size
        n = min(n, size)
        path = self.work / f"tail-{n}.bin"
        if not path.exists():
            with open(self.corpus, "rb") as f:
                f.seek(size - n)
                path.write_bytes(f.read())
        return path

    def sample(self) -> Path:
        """DICT_SAMPLE bytes of the corpus in 256 evenly spaced runs: a fixed, deterministic sample."""
        path = self.work / "dict-sample.bin"
        if not path.exists():
            size = self.corpus.stat().st_size
            run = min(DICT_SAMPLE, size) // 256
            with open(self.corpus, "rb") as f, open(path, "wb") as out:
                for i in range(256):
                    f.seek((size - run) * i // 255)
                    out.write(f.read(run))
        return path

    def once(self, key, build):
        if key not in self._built:
            self._built[key] = build()
        return self._built[key]

    def prefix_size(self, key: str, compute: Callable[[], int]) -> int:
        """|C(P)|, cached on disk across runs (it is the expensive half of every marginal row)."""
        if key not in self.cache:
            self.cache[key] = compute()
            if self.cache_path:  # re-read first: parallel runs share the file
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                disk = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}
                self.cache = disk | self.cache
                self.cache_path.write_text(json.dumps(self.cache, indent=1))
        return self.cache[key]


@dataclass
class Baseline:
    name: str
    kind: str  # standalone | dictionary | marginal
    group: str  # fast (seconds per MB) | all (adds the slow context-mixing coders)
    run: Callable  # (data, ctx) -> dict of measured fields
    skip: Callable = lambda ctx: None  # (ctx) -> reason to skip, or None


REGISTRY: list[Baseline] = []


def register(name, kind="standalone", group="fast", skip=None):
    def deco(fn):
        REGISTRY.append(Baseline(name, kind, group, fn, skip or (lambda ctx: None)))
        return fn

    return deco


# --- python-native


def _py(data, comp, decomp):
    t = time.time()
    blob = comp(data)
    enc = time.time() - t
    t = time.time()
    ok = decomp(blob) == data
    return dict(bytes=len(blob), verified=ok, enc_s=enc, dec_s=time.time() - t)


def _needs_module(mod):
    def skip(ctx):
        try:
            __import__(mod)
        except ImportError:
            return f"python module {mod} not installed"

    return skip


@register("gzip -9")
def _gzip(data, ctx):
    r = _py(data, lambda d: gzip.compress(d, 9, mtime=0), gzip.decompress)
    return r | dict(command="gzip.compress(data, 9, mtime=0)", version=f"zlib {zlib.ZLIB_RUNTIME_VERSION}")


@register("bzip2 -9")
def _bzip2(data, ctx):
    return _py(data, lambda d: bz2.compress(d, 9), bz2.decompress) | dict(command="bz2.compress(data, 9)", version=f"python {sys.version.split()[0]} bz2")


@register("xz -9e")
def _xz(data, ctx):
    r = _py(data, lambda d: lzma.compress(d, preset=9 | lzma.PRESET_EXTREME), lzma.decompress)
    return r | dict(command="lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)", version=f"python {sys.version.split()[0]} lzma")


@register("zstd -19", skip=_needs_module("zstandard"))
def _zstd(data, ctx):
    import zstandard

    r = _py(data, zstandard.ZstdCompressor(level=19).compress, zstandard.ZstdDecompressor().decompress)
    return r | dict(command="zstandard.ZstdCompressor(level=19).compress(data)", version=f"zstandard {zstandard.__version__} (libzstd {zstandard.ZSTD_VERSION})")


@register("brotli -q11", skip=_needs_module("brotli"))
def _brotli(data, ctx):
    import brotli

    r = _py(data, lambda d: brotli.compress(d, quality=11, lgwin=24), brotli.decompress)
    return r | dict(command="brotli.compress(data, quality=11, lgwin=24)", version=f"brotli {brotli.__version__}")


# --- CLI tools: data goes through files in the work dir


def _cli(data, ctx, enc, dec, out="x.c"):
    """enc/dec: argv lists using the names x (input), OUT (compressed), y (decoded), run in ctx.work."""
    d = Path(tempfile.mkdtemp(dir=ctx.work))
    (d / "x").write_bytes(data)
    try:
        enc_s = _run(enc, cwd=d)
        size = (d / out).stat().st_size
        dec_s = _run(dec, cwd=d)
        ok = (d / "y").read_bytes() == data
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return dict(bytes=size, verified=ok, enc_s=enc_s, dec_s=dec_s, command=" ".join(map(str, enc)), decode_command=" ".join(map(str, dec)))


def _needs(tool, why=""):
    return lambda ctx: None if _which(tool) else f"{tool} not found{why}"


def _needs_corpus(*more):
    def skip(ctx):
        if ctx.corpus is None:
            return "no training corpus given (--corpus / --train-corpus)"
        for m in more:
            if reason := m(ctx):
                return reason

    return skip


ZSTD_V = lambda: _version(["zstd", "--version"], r"v[\d.]+")  # noqa: E731
BROTLI_V = lambda: _version(["brotli", "--version"])  # noqa: E731
P7_V = lambda: _version(["7zz", "i"], r"7-Zip[^\n]*")  # noqa: E731
PAQ_V = lambda: _build_version(PAQ8PX)  # noqa: E731
CMIX_V = lambda: _build_version(CMIX)  # noqa: E731
NO_7ZZ = " (brew install sevenzip; tools/build_competitors.sh)"
NO_BUILD = " (run tools/build_competitors.sh)"


@register("zstd --ultra -22 --long=27", skip=_needs("zstd"))
def _zstd22(data, ctx):
    r = _cli(data, ctx, ["zstd", "-q", "--ultra", "-22", "--long=27", "x", "-o", "x.c"], ["zstd", "-q", "-d", "--long=27", "x.c", "-o", "y"])
    return r | dict(version=f"zstd {ZSTD_V()}")


@register("brotli -q 11 --large_window=30", skip=_needs("brotli"))
def _brotli30(data, ctx):
    r = _cli(data, ctx, ["brotli", "-q", "11", "--large_window=30", "x", "-o", "x.c"], ["brotli", "-d", "--large_window=30", "x.c", "-o", "y"])
    return r | dict(version=BROTLI_V(), note="large-window brotli: a bitstream standard brotli decoders reject")


@register("7zz PPMd o=32 mem=1g", group="all", skip=_needs("7zz", NO_7ZZ))
def _ppmd(data, ctx):
    r = _cli(data, ctx, ["7zz", "a", "-bso0", "-bsp0", "-t7z", PPMD, "x.7z", "x"], ["sh", "-c", "7zz e -bso0 -bsp0 -so x.7z > y"], out="x.7z")
    return r | dict(version=P7_V(), note="size includes the 7z container (~100 bytes of headers)")


@register(f"paq8px {PAQ8PX_LEVEL}", group="all", skip=_needs(PAQ8PX, NO_BUILD))
def _paq8px(data, ctx):
    r = _cli(data, ctx, [PAQ8PX, PAQ8PX_LEVEL, "x", "x.c"], [PAQ8PX, "-d", "x.c", "y"])
    return r | dict(version=PAQ_V(), note="level 8 (2.4 GB) + LSTM; built for this CPU: decodes with the same binary")


@register("cmix", group="all", skip=_needs(CMIX, NO_BUILD))
def _cmix(data, ctx):
    r = _cli(data, ctx, [CMIX, "-c", CMIX_DICT, "x", "x.c"], [CMIX, "-d", CMIX_DICT, "x.c", "y"])
    return r | dict(version=CMIX_V(), note=f"default mode with its english.dic ({CMIX_DICT.stat().st_size} B, not counted); ~19 GB RAM")


# --- dictionary rows: the decoder holds a reference built from the corpus


def zstd_dict(ctx, size: int) -> Path:
    """zstd --train on the fixed corpus sample cut into 4 KB samples."""

    def build():
        out = ctx.work / f"zstd-{size}.dict"
        _run(["zstd", "-q", "--train", "-B4096", f"--maxdict={size}", ctx.sample(), "-o", out])
        return out

    return ctx.once(("zstd-dict", size), build)


def _zstd_dict_row(size, label):
    @register(f"zstd -19 + {label} dict", kind="dictionary", skip=_needs_corpus(_needs("zstd")))
    def row(data, ctx):
        d = zstd_dict(ctx, size)
        r = _cli(data, ctx, ["zstd", "-q", "-19", "-D", d, "x", "-o", "x.c"], ["zstd", "-q", "-d", "-D", d, "x.c", "-o", "y"])
        return r | dict(version=f"zstd {ZSTD_V()}", ref_bytes=d.stat().st_size,
                        ref=f"zstd --train -B4096 --maxdict={size} on a {DICT_SAMPLE >> 20} MiB sample of the corpus (256 evenly spaced runs)")


_zstd_dict_row(110 << 10, "110 KiB")
_zstd_dict_row(1 << 20, "1 MiB")


@register("zstd --ultra -22 --long=31 --patch-from=corpus", kind="dictionary", skip=_needs_corpus(_needs("zstd")))
def _zstd_patch(data, ctx):
    c, mem = ctx.corpus, f"--memory={max(2048, 2 * (ctx.corpus.stat().st_size >> 20) + 256)}MB"
    r = _cli(data, ctx, ["zstd", "-q", "--ultra", "-22", "--long=31", mem, f"--patch-from={c}", "x", "-o", "x.c"],
             ["zstd", "-q", "-d", "--long=31", mem, f"--patch-from={c}", "x.c", "-o", "y"])
    return r | dict(version=f"zstd {ZSTD_V()}", ref_bytes=c.stat().st_size, ref="the whole training corpus")


def brotli_dict(ctx, n: int) -> Path:
    return ctx.tail(n)


@register("brotli -q 11 -w 24 -D corpus tail", kind="dictionary", skip=_needs_corpus(_needs("brotli")))
def _brotli_dict(data, ctx):
    # brotli's raw dictionary must sit inside the window, behind the data: with -w 24 (16 MiB - 16)
    # the most that helps is 16 MiB minus the input, so the dictionary is the corpus tail that fits.
    n = max(1 << 20, (1 << 24) - 16 - len(data) - (1 << 16))
    d = brotli_dict(ctx, n)
    r = _cli(data, ctx, ["brotli", "-q", "11", "-w", "24", "-D", d, "x", "-o", "x.c"], ["brotli", "-d", "-D", d, "x.c", "-o", "y"])
    return r | dict(version=BROTLI_V(), ref_bytes=d.stat().st_size, ref=f"the last {d.stat().st_size} bytes of the corpus (what fits the 16 MiB window)")


# --- marginal rows: |C(P + x)| - |C(P)|


def marginal(compress_size: Callable[[bytes], int], prefix: bytes, data: bytes, prefix_size: int | None = None) -> int:
    """Bytes that coding `data` after `prefix` adds: |C(P + x)| - |C(P)|. prefix_size: a known |C(P)|."""
    if prefix_size is None:
        prefix_size = compress_size(prefix)
    return compress_size(prefix + data) - prefix_size


MARGINAL_NOTE = "marginal (equivalent to a preset-dictionary coder up to a few bytes; not a standalone file)"


def _marginal_row(ctx, data, tool, cmd, size_fn):
    p = ctx.tail(PRIME[tool])
    pb = p.read_bytes()
    t = time.time()
    ps = ctx.prefix_size(f"{cmd}|{sha256(pb)}", lambda: size_fn(pb))
    t1 = time.time()
    total = size_fn(pb + data)
    return dict(bytes=total - ps, verified=None, enc_s=time.time() - t1, prefix_s=t1 - t, command=f"|{cmd}(P + x)| - |{cmd}(P)|",
                ref_bytes=len(pb), ref=f"P = the last {len(pb)} bytes of the corpus", note=MARGINAL_NOTE, prefix_compressed=ps)


def xz_size(b: bytes) -> int:
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "dict_size": min(1536 << 20, max(1 << 26, 1 << (len(b) - 1).bit_length()))}]
    return len(lzma.compress(b, format=lzma.FORMAT_XZ, filters=filt))


@register("xz -9e primed", kind="marginal", skip=_needs_corpus())
def _xz_primed(data, ctx):
    r = _marginal_row(ctx, data, "xz", "xz -9e --lzma2=dict=2^ceil(log2|P+x|)", xz_size)
    return r | dict(version=f"python {sys.version.split()[0]} lzma")


def _file_size(ctx, b: bytes, enc, out) -> int:
    d = Path(tempfile.mkdtemp(dir=ctx.work))
    try:
        (d / "x").write_bytes(b)
        _run(enc, cwd=d)
        return (d / out).stat().st_size
    finally:
        shutil.rmtree(d, ignore_errors=True)


@register("7zz PPMd o=32 mem=1g primed", kind="marginal", group="all", skip=_needs_corpus(_needs("7zz", NO_7ZZ)))
def _ppmd_primed(data, ctx):
    enc = ["7zz", "a", "-bso0", "-bsp0", "-t7z", PPMD, "x.7z", "x"]
    r = _marginal_row(ctx, data, "ppmd", f"7zz {PPMD}", lambda b: _file_size(ctx, b, enc, "x.7z"))
    return r | dict(version=P7_V())


@register(f"paq8px {PAQ8PX_PRIMED_LEVEL} primed", kind="marginal", group="all", skip=_needs_corpus(_needs(PAQ8PX, NO_BUILD)))
def _paq8px_primed(data, ctx):
    enc = [PAQ8PX, PAQ8PX_PRIMED_LEVEL, "x", "x.c"]
    r = _marginal_row(ctx, data, "paq8px", f"paq8px {PAQ8PX_PRIMED_LEVEL}", lambda b: _file_size(ctx, b, enc, "x.c"))
    return r | dict(version=PAQ_V())


# --- running and reporting


def select(group="fast", only=None, kinds=None) -> list[Baseline]:
    """Baselines in registry order: `only` (names or regexes) wins over `group` (fast | all | none)."""
    if only:
        return [b for b in REGISTRY if any(o == b.name or re.fullmatch(o, b.name) for o in only)]
    if group == "none":
        return []
    return [b for b in REGISTRY if (group == "all" or b.group == "fast") and (kinds is None or b.kind in kinds)]


def run(data: bytes, baselines=None, ctx: Ctx | None = None, log=None) -> list[dict]:
    """One result dict per baseline; skipped ones carry `skipped` (the reason), failed ones `error`."""
    ctx = ctx or Ctx()
    rows = []
    for b in REGISTRY if baselines is None else baselines:
        row = dict(name=b.name, kind=b.kind)
        if reason := b.skip(ctx):
            row["skipped"] = reason
        else:
            try:
                row |= b.run(data, ctx)
                row["bits_per_byte"] = 8 * row["bytes"] / len(data)
                if row.get("verified") is False:
                    row["error"] = "round trip FAILED: decoded bytes differ"
            except Exception as e:  # a broken tool is a result to report, not a reason to lose the others
                row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if log:
            log(row)
    return rows


def describe(row) -> str:
    if "skipped" in row:
        return f"{row['name']}: skipped ({row['skipped']})"
    if "error" in row:
        return f"{row['name']}: ERROR {row['error']}"
    ref = f" +ref {row['ref_bytes']}" if row.get("ref_bytes") else ""
    return f"{row['name']}: {row['bytes']} B, {row['bits_per_byte']:.4f} bpb{ref}, enc {row['enc_s']:.1f}s dec {row.get('dec_s') or 0:.1f}s"


def environment() -> dict:
    import platform

    return dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.machine(),
                zstd=ZSTD_V(), brotli=BROTLI_V(), xz=_version(["xz", "--version"], r"xz \(XZ Utils\) [\d.]+"), sevenzip=P7_V() if _which("7zz") else None,
                paq8px=PAQ_V() if PAQ8PX.exists() else None, cmix=CMIX_V() if CMIX.exists() else None)


def markdown(results: list[dict]) -> str:
    """One table per input file: results are {file, size, sha256, corpus, rows} dicts."""
    out = []
    for res in results:
        n = res["size"]
        out.append(f"### {res['file']} ({n} bytes, sha256 {res['sha256'][:16]}...)\n")
        if res.get("corpus"):
            out.append(f"training corpus: {res['corpus']} ({res['corpus_size']} bytes)\n")
        out.append("| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |")
        out.append("|---|---|---:|---:|---:|---|---:|---:|")
        ok = sorted((r for r in res["rows"] if "bytes" in r and "error" not in r), key=lambda r: r["bytes"])
        for r in ok:
            v = {True: "yes", False: "NO", None: "n/a"}[r.get("verified")]
            dec = f"{r['dec_s']:.1f}" if r.get("dec_s") is not None else ""
            out.append(f"| {r['name']} | {r['kind']} | {r['bytes']} | {8 * r['bytes'] / n:.4f} | {r.get('ref_bytes') or ''} | {v} | {r['enc_s']:.1f} | {dec} |")
        for r in res["rows"]:
            if "skipped" in r or "error" in r:
                out.append(f"| {r['name']} | {r['kind']} | | | | {'skipped: ' + r['skipped'] if 'skipped' in r else 'ERROR: ' + r['error'][:200]} | | |")
        out.append("")
    return "\n".join(out)


def merge(parts: list[dict]) -> list[dict]:
    """Combine runs on the same file (by sha256): one row per name, a later row replacing an earlier
    one unless it is only a skip (so a fast run does not hide a long run's measurement)."""
    by = {}
    for p in parts:
        for res in p["results"]:
            cur = by.setdefault(res["sha256"], dict(res, rows={}))
            for r in res["rows"]:
                if "skipped" not in r or r["name"] not in cur["rows"]:
                    cur["rows"][r["name"]] = r
    return [dict(res, rows=list(res["rows"].values())) for res in by.values()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="*")
    ap.add_argument("--corpus", type=Path, help="training corpus: enables the dictionary and marginal rows")
    ap.add_argument("--group", choices=["fast", "all", "none"], default="all")
    ap.add_argument("--only", nargs="+", help="baseline names or regexes (overrides --group)")
    ap.add_argument("--json", type=Path, help="write the results here")
    ap.add_argument("--cache", type=Path, default=REPO / "results" / "baselines" / "prefix-cache.json", help="|C(P)| cache for the marginal rows")
    ap.add_argument("--report", type=Path, help="merge the given JSON files (positional) into this markdown file")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    if args.list:
        for b in REGISTRY:
            print(f"{b.name:50s} {b.kind:10s} {b.group}")
        return
    if args.report:
        parts = [json.loads(p.read_text()) for p in args.files]
        results = merge(parts)
        args.report.write_text(markdown(results))
        if args.json:  # the merged results, with every run's exact command and environment
            runs = [dict(file=str(f), command=p["command"], started=p.get("started"), finished=p.get("finished"), environment=p["environment"])
                    for f, p in zip(args.files, parts)]
            args.json.write_text(json.dumps(dict(runs=runs, results=results), indent=1))
        print(args.report.read_text())
        return
    ctx = Ctx(args.corpus, cache=args.cache)
    chosen = select(args.group, args.only)
    out = dict(command=" ".join([sys.executable, "-m", "jevzip.baselines", *(argv if argv is not None else sys.argv[1:])]),
               started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), environment=environment(), results=[])
    for path in args.files:
        data = path.read_bytes()
        res = dict(file=str(path), size=len(data), sha256=sha256(data), rows=[])
        if args.corpus:
            res |= dict(corpus=str(args.corpus), corpus_size=args.corpus.stat().st_size, corpus_sha256=_file_sha(args.corpus))
        out["results"].append(res)
        print(f"### {path} ({len(data)} bytes)", flush=True)

        def log(row, res=res):
            res["rows"].append(row)
            print("  " + describe(row), flush=True)
            if args.json:  # rewrite after every row: a long run keeps what it has measured
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps(out, indent=1))

        run(data, chosen, ctx, log)
    out["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    shutil.rmtree(ctx.work, ignore_errors=True)


def _file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
