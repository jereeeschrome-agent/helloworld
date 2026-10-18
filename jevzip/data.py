"""Datasets: build, download and verify every byte file jevzip trains or benchmarks on.

    python -m jevzip.data code [--test-only]  # Python-source corpus from the installed packages -> data/code
    python -m jevzip.data enwik8              # standard 90M/5M/5M byte split -> data/enwik8
    python -m jevzip.data text8               # standard 90M/5M/5M character split -> data/text8
    python -m jevzip.data silesia             # the 12 Silesia corpus files -> data/silesia
    python -m jevzip.data ood                 # JSON, logs, C++ headers (out-of-domain) -> data/ood
    python -m jevzip.data verify [PREFIX ...] [--downloads] [--record]

datasets.json, next to this file, is the manifest: every file's size, SHA-256,
source URL, license and split, plus the suites the benchmarks run. Downloads
go to downloads/<name>/ and are checked against it before use. Outputs are
checked before they are written, and an existing output that differs is never
replaced without --force, so a corpus that a training run reads can't change
under it.

The code corpus depends on the packages installed on the build machine, so it
can't be rebuilt bit-identically elsewhere. Its two test files are published
as the release asset jevzip-code-testset.tar.xz (python -m jevzip.fetch).
This module imports only the standard library, so it can run on downloaded,
untrusted data as `python -I jevzip/data.py ...`.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA, DOWNLOADS = REPO / "data", REPO / "downloads"
MANIFEST = Path(__file__).resolve().parent / "datasets.json"

# code corpus: whole packages are held out so near-duplicate files can't leak across splits.
# mpmath is validation; networkx and pydoc_data (mostly English prose) are the benchmark sets.
VAL_PACKAGES = {"mpmath"}
TEST_PACKAGES = {"networkx": "networkx.py.txt", "pydoc_data": "pydoc_topics.txt"}
TEST_BYTES = 1 << 20
SEPARATOR = b"\n\n"
SPLIT = {"val": 5_000_000, "test": 5_000_000}  # enwik8, text8: the standard LM split keeps the last 10M bytes back
HEADER_DIRS = ("c10/core", "c10/util")  # PyTorch's own hand-written C++, not the vendored third-party headers


def load_manifest(path=MANIFEST) -> dict:
    return json.loads(Path(path).read_text())


def digest(data: bytes, algo="sha256") -> str:
    return hashlib.new(algo, data).hexdigest()


def file_digest(path, algo="sha256") -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, algo).hexdigest()


def git_sha1(data: bytes) -> str:
    """Git's blob id, so a file fetched from a pinned commit can be checked against that commit's tree."""
    return digest(b"blob %d\0" % len(data) + data, "sha1")


def mismatch(entry: dict, size: int, sha: str) -> str | None:
    """How a file disagrees with its manifest entry, or None. Fields the entry lacks are not checked."""
    if entry.get("size") is not None and size != entry["size"]:
        return f"size {size}, manifest says {entry['size']}"
    if entry.get("sha256") and sha != entry["sha256"]:
        return f"sha256 {sha}, manifest says {entry['sha256']}"
    return None


def split(data: bytes, val=SPLIT["val"], test=SPLIT["test"]) -> dict[str, bytes]:
    """The standard enwik8/text8 split: the last `test` bytes, the `val` bytes before them, and the rest as train."""
    if val < 0 or test < 0 or val + test >= len(data):
        raise ValueError(f"can't split {len(data)} bytes into val {val} and test {test}")
    n = len(data) - val - test
    return {"train": data[:n], "val": data[n : n + val], "test": data[n + val :]}


def put(path: Path, data: bytes, entry: dict | None = None, force=False) -> str:
    """Write `data` to `path` atomically, after checking it against its manifest entry; return what happened.

    A mismatch is an error for a reproducible entry (its inputs were verified, so it is a bug) and a warning
    for one built from this machine's files. An identical existing file is left alone; a different one is
    replaced only with force."""
    sha = digest(data)
    if entry and (bad := mismatch(entry, len(data), sha)):
        if entry.get("reproducible", True):
            raise ValueError(f"{path}: built {bad}; not written")
        print(f"warning: {path}: {bad} (it depends on the build machine)", file=sys.stderr)
    existed = path.exists()
    if existed:
        if path.stat().st_size == len(data) and file_digest(path) == sha:
            return "unchanged"
        if not force:
            raise FileExistsError(f"{path} exists with different contents; not replaced (use --force)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return "replaced" if existed else "wrote"


def download(key: str, manifest: dict, root=DOWNLOADS) -> Path:
    """root/key, fetched from the entry's URL if absent, checked against its size, SHA-256 and git blob id.

    A bad download is deleted before it lands. A bad existing file is reported, never deleted or replaced."""
    entry, path = manifest["downloads"][key], Path(root) / key
    fresh = not path.exists()
    if fresh:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
        print(f"downloading {entry['url']} -> {path}", file=sys.stderr)
    try:
        if fresh:
            limit = entry.get("size") or float("inf")
            with os.fdopen(fd, "wb") as f, urllib.request.urlopen(entry["url"], timeout=60) as r:
                got = 0
                while got <= limit and (chunk := r.read(1 << 20)):  # stop once past the promised size
                    f.write(chunk)
                    got += len(chunk)
        src = Path(tmp) if fresh else path
        size, sha = src.stat().st_size, file_digest(src)
        bad = mismatch(entry, size, sha)
        if not bad and entry.get("git_sha1") and git_sha1(src.read_bytes()) != entry["git_sha1"]:
            bad = f"git blob id differs from {entry['git_sha1']}"
        if bad:
            raise OSError(f"{path}: {bad}; " + ("download deleted" if fresh else "delete it to download it again"))
        if fresh:
            os.chmod(tmp, 0o644)
            os.replace(tmp, path)
    finally:
        if fresh:
            Path(tmp).unlink(missing_ok=True)
    if not entry.get("sha256"):
        print(f"{key}: {size} bytes, sha256 {sha}: not in the manifest yet (verify --downloads --record)", file=sys.stderr)
    return path


def read_member(zip_path: Path, name: str, entry: dict | None = None) -> bytes:
    """One member of a zip archive, checked against its manifest entry (size, md5, sha256: whichever it has)."""
    entry = entry or {}
    with zipfile.ZipFile(zip_path) as z:
        if entry.get("size") is not None and z.getinfo(name).file_size != entry["size"]:
            raise ValueError(f"{zip_path}:{name}: {z.getinfo(name).file_size} bytes, manifest says {entry['size']}")
        data = z.read(name)  # read into memory, never extracted by the archive's own paths
    bad = mismatch(entry, len(data), digest(data))
    if not bad and entry.get("md5") and digest(data, "md5") != entry["md5"]:
        bad = f"md5 {digest(data, 'md5')}, manifest says {entry['md5']}"
    if bad:
        raise ValueError(f"{zip_path}:{name}: {bad}")
    return data


def write_pieces(out_dir: Path, blob_path: Path, files: list[tuple[str, bytes]], blob: bytes, source="", force=False):
    """Write each file as it appears in `blob` (SEPARATOR-joined, maybe cut short) under out_dir, plus index.json.

    The index lists the files in blob order with their offset, size (bytes in the blob), full size and
    SHA-256, so per-file results can be computed and the blob checked against them (verify does both)."""
    listed, offset = [], 0
    for name, data in files:
        if offset >= len(blob):
            break
        piece = blob[offset : offset + len(data)]
        put(out_dir / name, piece, force=force)
        listed.append({"path": name, "offset": offset, "size": len(piece), "full_size": len(data), "sha256": digest(piece)})
        offset += len(data) + len(SEPARATOR)
    index = {"blob": os.path.relpath(blob_path, out_dir), "separator": SEPARATOR.decode(), "source": source, "files": listed}
    put(out_dir / "index.json", (json.dumps(index, indent=1) + "\n").encode(), force=force)
    return listed


def check_index(path: Path) -> list[str]:
    """Problems with an index.json: each listed file on disk, as a slice of the blob, and the blob fully covered."""
    index = json.loads(path.read_text())
    blob, sep = (path.parent / index["blob"]).read_bytes(), index["separator"].encode()
    problems, pos = [], 0
    for f in index["files"]:
        piece = blob[f["offset"] : f["offset"] + f["size"]]
        if f["offset"] != pos or (pos and blob[pos - len(sep) : pos] != sep):
            problems.append(f"{f['path']}: not at offset {pos} after a separator")
        if len(piece) != f["size"] or digest(piece) != f["sha256"]:
            problems.append(f"{f['path']}: its slice of {index['blob']} differs")
        p = path.parent / f["path"]
        if not p.is_file():
            problems.append(f"{f['path']}: missing")
        elif p.stat().st_size != f["size"] or file_digest(p) != f["sha256"]:
            problems.append(f"{f['path']}: differs")
        pos = f["offset"] + f["size"] + len(sep)
    end = pos - len(sep) if index["files"] else 0
    if not (0 <= len(blob) - end < len(sep) and blob[end:] == sep[: len(blob) - end]):
        problems.append(f"the listed files don't cover {index['blob']} ({end} of {len(blob)} bytes)")
    return problems


def verify(prefixes=(), data=DATA, downloads=None, manifest=None) -> list[tuple[str, str]]:
    """[(key, "ok" or the problem)] for every manifest file under `prefixes` (all if empty).

    An index entry also checks the files it lists. Given a `downloads` dir, download entries are checked too."""
    m = manifest or load_manifest()
    sections = [("files", Path(data))] + ([("downloads", Path(downloads))] if downloads else [])
    report = []
    for section, root in sections:
        for key, entry in m[section].items():
            if prefixes and not key.startswith(tuple(prefixes)):
                continue
            path = root / key
            if not path.is_file():
                report.append((key, "missing"))
                continue
            bad = mismatch(entry, path.stat().st_size, file_digest(path))
            if not bad and entry.get("index"):
                bad = "; ".join(check_index(path)) or None
            report.append((key, bad or "ok"))
    return report


def record(manifest_path=MANIFEST, data=DATA, downloads=DOWNLOADS) -> list[str]:
    """Fill in the size and SHA-256 that manifest entries (and zip members) lack, from the local files.

    Recorded values are never changed: a file that disagrees with one is verify's business."""
    m, filled = load_manifest(manifest_path), []
    for section, root in (("files", Path(data)), ("downloads", Path(downloads))):
        for key, entry in m[section].items():
            path = root / key
            if not path.is_file():
                continue
            if entry.get("size") is None or not entry.get("sha256"):
                if entry.get("size") is None:
                    entry["size"] = path.stat().st_size
                entry["sha256"] = entry.get("sha256") or file_digest(path)
                filled.append(key)
            for name, member in entry.get("members", {}).items():
                if member.get("size") is None or not member.get("sha256"):
                    blob = read_member(path, name, member)
                    member["size"], member["sha256"] = len(blob), digest(blob)
                    filled.append(f"{key}:{name}")
    Path(manifest_path).write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    return filled


def suite(name: str, data=DATA, manifest=None) -> dict[str, list[Path]]:
    """A benchmark suite from the manifest ("code", "standard", "ood"): group -> file paths."""
    m = manifest or load_manifest()
    return {group: [Path(data) / key for key in keys] for group, keys in m["suites"][name].items()}


# ---- builders ----


def package_roots() -> list[Path]:
    paths = {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"], *sys.path}
    # only interpreter/library dirs, never the cwd (which would pull in this repo)
    return sorted({Path(p).resolve() for p in paths if p and "python3" in p and Path(p).is_dir()})


def iter_sources():
    """Yield (top-level package, path relative to its root, path) for every .py file, each file once."""
    seen = set()
    for root in package_roots():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in ("__pycache__",) and not d.endswith((".dist-info", ".egg-info")))
            rel = Path(dirpath).relative_to(root).parts
            if rel and rel[0] in ("site-packages", "dist-packages"):
                continue  # visited as their own root
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                path = (Path(dirpath) / name).resolve()
                if path in seen:
                    continue
                seen.add(path)
                yield (rel[0] if rel else name[:-3]), Path(*rel, name).as_posix(), path


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def build_code(out: Path, test_only=False, force=False):
    """The Python-source corpus: train.bin, val.bin, the two 1 MiB test files, and networkx-files/ (per-file)."""
    m = load_manifest()["files"]
    splits = {"train": [], "val": [], **{k: [] for k in TEST_PACKAGES}}
    for pkg, rel, path in iter_sources():
        key = pkg if pkg in TEST_PACKAGES else "val" if pkg in VAL_PACKAGES else "train"
        if test_only and key not in TEST_PACKAGES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        splits[key].append((rel, data))
    for key, files in splits.items():
        if test_only and key not in TEST_PACKAGES:
            continue
        blob = SEPARATOR.join(data for _, data in files)
        name = TEST_PACKAGES.get(key, f"{key}.bin")
        if key in TEST_PACKAGES:
            blob = blob[:TEST_BYTES]
        status = put(out / name, blob, m.get(f"code/{name}"), force)
        print(f"{name}: {len(files)} files, {len(blob) / 1e6:.2f} MB, {status}")
        if key == "networkx":
            listed = write_pieces(out / "networkx-files", out / name, files, blob, f"networkx {_version('networkx')}", force)
            print(f"networkx-files/: {len(listed)} files ({sum(f['size'] < f['full_size'] for f in listed)} cut short) + index.json")


def build_split(name: str, out: Path, force=False, downloads=DOWNLOADS):
    """enwik8 or text8: the archive's single member, split into train/val/test.bin."""
    m = load_manifest()
    key = f"{name}/{name}.zip"
    data = read_member(download(key, m, downloads), name, m["downloads"][key]["members"][name])
    for part, blob in split(data).items():
        print(f"{name}/{part}.bin: {len(blob)} bytes, {put(out / f'{part}.bin', blob, m['files'].get(f'{name}/{part}.bin'), force)}")


def build_copies(group: str, out: Path, force=False, downloads=DOWNLOADS):
    """Every manifest file under `group`/ that is a download, or one member of a zip download, as is."""
    m, srcs = load_manifest(), {}
    for key, entry in m["files"].items():
        if not key.startswith(group + "/") or "from" not in entry:
            continue
        src = srcs.get(entry["from"]) or srcs.setdefault(entry["from"], download(entry["from"], m, downloads))
        if "member" in entry:
            data = read_member(src, entry["member"], m["downloads"][entry["from"]]["members"].get(entry["member"]))
        else:
            data = src.read_bytes()
        print(f"{key}: {len(data)} bytes, {put(out / key[len(group) + 1 :], data, entry, force)}")


def torch_include() -> Path:
    """The include dir of the installed torch wheel, found without importing torch."""
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("torch is not installed here; pass --torch-include DIR")
    return Path(spec.origin).parent / "include"


def header_files(include: Path, budget=TEST_BYTES) -> list[tuple[str, bytes]]:
    """The longest prefix of the sorted .h files under HEADER_DIRS whose SEPARATOR-joined size fits in `budget`."""
    names = sorted(p.relative_to(include).as_posix() for d in HEADER_DIRS for p in (include / d).rglob("*.h") if p.is_file())
    files, total = [], -len(SEPARATOR)
    for name in names:
        data = (include / name).read_bytes()
        total += len(SEPARATOR) + len(data)
        if total > budget:
            break
        files.append((name, data))
    return files


def build_headers(out: Path, include: Path | None = None, force=False):
    """ood/c-headers: ~1 MiB of whole C++ headers from torch's include dir, joined, plus the per-file copies."""
    include = include or torch_include()
    files = header_files(include)
    blob = SEPARATOR.join(data for _, data in files)
    entry = load_manifest()["files"].get("ood/c-headers/torch-c10.txt")
    print(f"ood/c-headers/torch-c10.txt: {len(files)} headers, {len(blob)} bytes, {put(out / 'torch-c10.txt', blob, entry, force)}")
    write_pieces(out / "torch-c10-files", out / "torch-c10.txt", files, blob, f"torch {_version('torch')} include/", force)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevzip.data", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    builds = {"code": "Python-source corpus from the installed packages", "enwik8": "enwik8, 90M/5M/5M bytes",
              "text8": "text8, 90M/5M/5M characters", "silesia": "the Silesia corpus", "ood": "out-of-domain: JSON, logs, C++ headers"}
    for name, help in builds.items():
        p = sub.add_parser(name, help=help)
        p.add_argument("--out", type=Path, default=DATA / name, help=f"output dir (default: {DATA / name})")
        p.add_argument("--force", action="store_true", help="replace existing outputs that differ")
        p.add_argument("--downloads", type=Path, default=DOWNLOADS, help=f"download dir (default: {DOWNLOADS})")
        if name == "code":
            p.add_argument("--test-only", action="store_true", help="only the held-out test files and networkx-files/")
        if name == "ood":
            p.add_argument("--torch-include", type=Path, help="torch's include dir (default: the installed torch's)")
    p = sub.add_parser("verify", help="check files against the manifest")
    p.add_argument("prefixes", nargs="*", help="only manifest keys starting with these (e.g. enwik8 ood/json)")
    p.add_argument("--data", type=Path, default=DATA, help=f"data dir (default: {DATA})")
    p.add_argument("--downloads", type=Path, nargs="?", const=DOWNLOADS, help=f"also check downloads (default dir: {DOWNLOADS})")
    p.add_argument("--record", action="store_true", help="first fill in sizes and SHA-256s the manifest lacks")
    args = ap.parse_args(argv)
    try:
        if args.cmd == "verify":
            if args.record:
                for key in record(data=args.data, downloads=args.downloads or DOWNLOADS):
                    print(f"recorded {key}", file=sys.stderr)
            report = verify(args.prefixes, args.data, args.downloads)
            for key, status in report:
                print(f"{status:8} {key}" if status in ("ok", "missing") else f"BAD      {key}: {status}")
            bad = sum(s != "ok" for _, s in report)
            print(f"{len(report) - bad}/{len(report)} ok", file=sys.stderr)
            sys.exit(1 if bad or not report else 0)
        elif args.cmd == "code":
            build_code(args.out, args.test_only, args.force)
        elif args.cmd in ("enwik8", "text8"):
            build_split(args.cmd, args.out, args.force, args.downloads)
        elif args.cmd == "silesia":
            build_copies("silesia", args.out, args.force, args.downloads)
        else:
            build_copies("ood", args.out, args.force, args.downloads)
            build_headers(args.out / "c-headers", args.torch_include, args.force)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
