"""Weights and test data: a manifest of release assets, a download cache, and name -> path.

    python -m jevzip.fetch [NAME ...]       # download into the cache (default: DEFAULT_MODEL)
    python -m jevzip.fetch --all            # everything in the manifest
    python -m jevzip.fetch --list
    python -m jevzip.fetch --register weights/NEW.pt [--kind weights|data] [--description ...] [--trained-on ...] [--recipe ...]

Weights are not in git. They are GitHub release assets, listed in assets.json
with their size and SHA-256. A download streams to a temporary file, is
checked against both, and only then is moved into place, so a file in the
cache is always a complete, verified copy. A .tar.xz asset is also unpacked
into the cache, so its members resolve by the same relative path as under
<repo>/data/ (e.g. "code/networkx.py.txt").

Cache: $JEVZIP_HOME, default ~/.cache/jevzip. Only urllib: no extra dependencies.
"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # a checkout keeps weights/ and data/ here
MANIFEST = Path(__file__).resolve().parent / "assets.json"
RELEASE_URL = "https://github.com/jereeeschrome-agent/jevzip/releases/download/v0.2.0/"
DEFAULT_MODEL = "jevzip-lanes-11m-enwik8.pt"
ARCHIVES = (".tar.xz", ".tar.gz", ".tgz")


def cache_dir() -> Path:
    return Path(os.environ.get("JEVZIP_HOME") or Path.home() / ".cache" / "jevzip").expanduser()


def load_manifest(path=None) -> dict:
    path = Path(path or MANIFEST)
    return json.loads(path.read_text()) if path.exists() else {}


def resolve(name_or_path) -> Path:
    """An existing path -> itself; else <repo>/weights/NAME or <repo>/data/NAME; else the cache.

    Never downloads: a missing file raises FileNotFoundError that names the fetch command."""
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path
    name = str(name_or_path)
    for c in (REPO / "weights" / name, REPO / "data" / name, cache_dir() / name):
        if c.exists():
            return c
    manifest = load_manifest()
    asset = next((a for a, e in manifest.items() if name in (a, *e.get("contents", ()))), None)
    note = "" if asset else f" ({name} is not in {MANIFEST.name} yet; python -m jevzip.fetch --list shows what is)"
    raise FileNotFoundError(f"{name}: no such file, nor under {REPO / 'weights'}, {REPO / 'data'} or {cache_dir()}; "
                            f'run "python -m jevzip.fetch {asset or name}" to download it{note}')


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, manifest: dict | None = None, dest=None) -> Path:
    """Download one manifest asset into `dest` (default: the cache) unless a verified copy is there; return its path."""
    manifest = load_manifest() if manifest is None else manifest
    if name not in manifest:
        raise ValueError(f"{name} is not in the manifest (python -m jevzip.fetch --list)")
    entry = manifest[name]
    dest = Path(dest) if dest else cache_dir()
    path = dest / name
    if path.exists() and not (path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]):
        print(f"{name}: cached copy does not match the manifest, deleting it", file=sys.stderr)
        path.unlink()
    if path.exists():
        print(f"{name}: already verified in {dest}", file=sys.stderr)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        _download(entry, path)
    if name.endswith(ARCHIVES):
        _unpack(path, dest)
    return path


def _download(entry: dict, path: Path):
    """Stream entry["url"] to a temp file next to `path`; keep it only if size and SHA-256 match."""
    size, h, got, shown = entry["size"], hashlib.sha256(), 0, -1
    tty = sys.stderr.isatty()
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(entry["url"], timeout=60) as r:
            while got <= size and (chunk := r.read(1 << 20)):  # stop once past the promised size
                f.write(chunk)
                h.update(chunk)
                got += len(chunk)
                pct = 100 * got // max(size, 1)
                if tty or pct // 10 > shown // 10:  # a terminal sees every MB, a log every 10%
                    print(f"\r{path.name}: {got / 1e6:.1f} / {size / 1e6:.1f} MB", end="" if tty else "\n", file=sys.stderr, flush=True)
                    shown = pct
        if tty:
            print(file=sys.stderr)
        if got != size or h.hexdigest() != entry["sha256"]:
            raise OSError(f"{path.name}: download is {got} bytes with sha256 {h.hexdigest()}, manifest says {size} bytes "
                          f"with sha256 {entry['sha256']}; deleted it")
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _unpack(path: Path, dest: Path):
    """Unpack a verified archive into dest: plain files and directories with relative, non-escaping names only."""
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if not (m.isfile() or m.isdir()) or Path(m.name).is_absolute() or ".." in Path(m.name).parts:
                raise ValueError(f"{path.name}: refusing archive member {m.name!r}")
        tar.extractall(dest, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))


def register(path, kind=None, description=None, trained_on=None, recipe=None, license=None, manifest_path=None) -> dict:
    """Add or update the manifest entry for a local file; its URL is the release URL + basename.

    Fields not given are kept from an existing entry. Weights also record params, order and config."""
    path, manifest_path = Path(path), Path(manifest_path or MANIFEST)
    manifest = load_manifest(manifest_path)
    old = manifest.get(path.name, {})
    kind = kind or old.get("kind") or ("weights" if path.suffix == ".pt" else "data")
    entry = {**old, "kind": kind, "url": RELEASE_URL + path.name, "sha256": sha256(path), "size": path.stat().st_size,
             "description": description or old.get("description", ""), "license": license or old.get("license", "MIT" if kind == "weights" else "")}
    if kind == "weights":
        import torch  # only here: downloading and resolving work without importing torch

        blob = torch.load(path, map_location="cpu", weights_only=True)
        entry |= {"params": sum(v.numel() for v in blob["state"].values()), "order": blob["config"]["order"], "config": blob["config"],
                  "trained_on": trained_on or old.get("trained_on", ""), "recipe": recipe or old.get("recipe", "")}
    elif path.name.endswith(ARCHIVES):
        with tarfile.open(path) as tar:
            entry["contents"] = sorted(m.name for m in tar.getmembers() if m.isfile())
    manifest[path.name] = entry
    manifest_path.write_text(json.dumps(dict(sorted(manifest.items())), indent=2) + "\n")
    return entry


def _where(name: str) -> str:
    if (REPO / "weights" / name).exists() or (REPO / "data" / name).exists():
        return "repo"
    return "cache" if (cache_dir() / name).exists() else "-"


def _summary(e: dict) -> str:
    if e.get("kind") == "weights":
        return ", ".join(x for x in (f"{e['params'] / 1e6:.2f}M params" if "params" in e else "", e.get("order", "")) if x)
    return ", ".join(e.get("contents", ())) or e.get("description", "")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m jevzip.fetch", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help=f"assets to download (default: {DEFAULT_MODEL})")
    ap.add_argument("--all", action="store_true", help="download every asset in the manifest")
    ap.add_argument("--list", action="store_true", help="show the manifest and what is already local")
    ap.add_argument("--register", type=Path, metavar="PATH", help="add or update the manifest entry for a local file")
    ap.add_argument("--kind", choices=["weights", "data"], help="with --register (default: weights for .pt, else data)")
    ap.add_argument("--description")
    ap.add_argument("--trained-on")
    ap.add_argument("--recipe")
    ap.add_argument("--license", help="with --register (default: MIT for weights)")
    args = ap.parse_args(argv)
    try:
        if args.register:
            entry = register(args.register, args.kind, args.description, args.trained_on, args.recipe, args.license)
            print(f"{args.register.name}: {entry['size']} bytes, sha256 {entry['sha256']} -> {MANIFEST}", file=sys.stderr)
        elif args.list:
            for name, e in load_manifest().items():
                print(f"{name:36} {e.get('kind', '?'):8} {e.get('size', 0) / 1e6:6.1f} MB  {_where(name):5}  {_summary(e)}")
            print(f"cache: {cache_dir()}")
        else:
            manifest = load_manifest()
            for name in args.names or (list(manifest) if args.all else [DEFAULT_MODEL]):
                print(fetch(name, manifest))  # stdout: the path, for scripts
    except (OSError, ValueError, KeyError, tarfile.TarError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
