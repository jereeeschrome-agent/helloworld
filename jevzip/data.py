"""Build a byte corpus from the Python sources installed on this machine.

Whole packages are held out so near-duplicate files can't leak across splits:
`mpmath` is validation, `networkx` and `pydoc_data` (mostly English prose) are
the benchmark sets.
"""
import argparse
import os
import sys
import sysconfig
from pathlib import Path

VAL_PACKAGES = {"mpmath"}
TEST_PACKAGES = {"networkx": "networkx.py.txt", "pydoc_data": "pydoc_topics.txt"}
TEST_BYTES = 1 << 20
SEPARATOR = b"\n\n"


def package_roots() -> list[Path]:
    paths = {sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"], *sys.path}
    # only interpreter/library dirs, never the cwd (which would pull in this repo)
    return sorted({Path(p).resolve() for p in paths if p and "python3" in p and Path(p).is_dir()})


def iter_sources():
    """Yield (top-level package, path) for every .py file, each file once."""
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
                yield (rel[0] if rel else name[:-3]), path


def build(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": [], **{k: [] for k in TEST_PACKAGES}}
    for pkg, path in iter_sources():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        key = pkg if pkg in TEST_PACKAGES else "val" if pkg in VAL_PACKAGES else "train"
        splits[key].append(data)
    for key, chunks in splits.items():
        blob = SEPARATOR.join(chunks)
        if key in TEST_PACKAGES:
            blob = blob[:TEST_BYTES]
            name = TEST_PACKAGES[key]
        else:
            name = f"{key}.bin"
        (out / name).write_bytes(blob)
        print(f"{name}: {len(chunks)} files, {len(blob) / 1e6:.2f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"))
    build(ap.parse_args().out)
