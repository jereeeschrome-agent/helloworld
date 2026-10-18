"""Download cache tests, no network: assets are served from a temp dir (file:// and a local http.server).

    python -m unittest jevzip.test_fetch
"""
import contextlib
import functools
import hashlib
import http.server
import io
import json
import os
import tarfile
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from . import fetch
from .model import Config, JevZip


def quiet(fn):
    """Run a test with fetch's progress lines (stderr) swallowed."""

    @functools.wraps(fn)
    def run(*a, **kw):
        with contextlib.redirect_stderr(io.StringIO()):
            return fn(*a, **kw)

    return run


class FetchTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.repo, self.srv, self.cache = (self.root / d for d in ("repo", "srv", "cache"))
        for d in (self.repo / "weights", self.repo / "data", self.srv):
            d.mkdir(parents=True)
        for patch in (mock.patch.object(fetch, "REPO", self.repo), mock.patch.object(fetch, "MANIFEST", self.root / "assets.json"),
                      mock.patch.dict(os.environ, {"JEVZIP_HOME": str(self.cache)})):
            patch.start()
            self.addCleanup(patch.stop)

    def asset(self, name, data: bytes, url=None, **override) -> dict:
        """Put `data` on the 'server' and return a manifest with its entry."""
        (self.srv / name).write_bytes(data)
        entry = {"kind": "data", "url": url or (self.srv / name).as_uri(), "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        return {name: entry | override}

    def leftovers(self):
        return sorted(p.name for p in self.cache.iterdir()) if self.cache.exists() else []

    @quiet
    def test_download_verify_skip(self):
        manifest = self.asset("a.bin", os.urandom(3 << 20))
        path = fetch.fetch("a.bin", manifest)
        self.assertEqual(path, self.cache / "a.bin")
        self.assertEqual(path.read_bytes(), (self.srv / "a.bin").read_bytes())
        (self.srv / "a.bin").unlink()  # a valid cached copy must not be fetched again
        self.assertEqual(fetch.fetch("a.bin", manifest), path)
        self.assertEqual(self.leftovers(), ["a.bin"])

    @quiet
    def test_http_server(self):
        data, gets = os.urandom(100_000), []

        class Handler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                gets.append(self.path)
                super().do_GET()

            def log_message(self, *a):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(self.srv)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        manifest = self.asset("b.bin", data, url=f"http://127.0.0.1:{server.server_port}/b.bin")
        for _ in range(2):
            self.assertEqual(fetch.fetch("b.bin", manifest).read_bytes(), data)
        self.assertEqual(gets, ["/b.bin"])
        bad = self.asset("b.bin", data, url=f"http://127.0.0.1:{server.server_port}/missing.bin")
        with self.assertRaises(OSError):  # HTTP 404 is an OSError too
            fetch.fetch("b.bin", bad, dest=self.root / "other")

    @quiet
    def test_mismatch_rejected(self):
        data = os.urandom(5000)
        for bad in ({"sha256": "0" * 64}, {"size": 4999}, {"size": 5001}):
            with self.subTest(bad=bad), self.assertRaisesRegex(OSError, "deleted it"):
                fetch.fetch("c.bin", self.asset("c.bin", data, **bad))
            self.assertEqual(self.leftovers(), [])  # neither the file nor a .part temp file is left

    @quiet
    def test_corrupt_cache_replaced(self):
        manifest = self.asset("d.bin", b"good bytes")
        self.cache.mkdir()
        (self.cache / "d.bin").write_bytes(b"bad bytes!")  # same size, wrong hash
        self.assertEqual(fetch.fetch("d.bin", manifest).read_bytes(), b"good bytes")
        with self.assertRaises(ValueError):
            fetch.fetch("nope.bin", manifest)

    @quiet
    def test_archive_unpacks_and_resolves(self):
        tar = self.srv / "set.tar.xz"
        with tarfile.open(tar, "w:xz") as t:
            info = tarfile.TarInfo("code/x.txt")
            info.size = 5
            t.addfile(info, io.BytesIO(b"hello"))
        entry = fetch.register(tar, description="test set")
        self.assertEqual((entry["kind"], entry["contents"]), ("data", ["code/x.txt"]))
        manifest = fetch.load_manifest()
        manifest["set.tar.xz"]["url"] = tar.as_uri()
        fetch.MANIFEST.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(FileNotFoundError, r'python -m jevzip\.fetch set\.tar\.xz"'):
            fetch.resolve("code/x.txt")  # names the archive that holds it
        fetch.fetch("set.tar.xz")
        self.assertEqual(fetch.resolve("code/x.txt").read_bytes(), b"hello")

    @quiet
    def test_archive_rejects_escaping_member(self):
        tar = self.srv / "evil.tar.xz"
        with tarfile.open(tar, "w:xz") as t:
            info = tarfile.TarInfo("../escape.txt")
            info.size = 1
            t.addfile(info, io.BytesIO(b"x"))
        with self.assertRaisesRegex(ValueError, "refusing"):
            fetch.fetch("evil.tar.xz", self.asset("evil.tar.xz", tar.read_bytes()))
        self.assertFalse((self.root / "escape.txt").exists())

    def test_resolve_order(self):
        self.cache.mkdir()
        for d in (self.repo / "weights", self.repo / "data", self.cache):
            (d / "m.pt").write_bytes(d.name.encode())
        explicit = self.root / "explicit.pt"
        explicit.write_bytes(b"x")
        self.assertEqual(fetch.resolve(explicit), explicit)  # an existing path wins
        self.assertEqual(fetch.resolve(str(explicit)), explicit)
        self.assertEqual(fetch.resolve("m.pt"), self.repo / "weights" / "m.pt")
        (self.repo / "weights" / "m.pt").unlink()
        self.assertEqual(fetch.resolve("m.pt"), self.repo / "data" / "m.pt")
        (self.repo / "data" / "m.pt").unlink()
        self.assertEqual(fetch.resolve("m.pt"), self.cache / "m.pt")
        (self.cache / "m.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, r'python -m jevzip\.fetch m\.pt"'):
            fetch.resolve("m.pt")
        with self.assertRaisesRegex(FileNotFoundError, fetch.DEFAULT_MODEL):
            fetch.resolve(fetch.DEFAULT_MODEL)

    def test_register_weights_and_cli(self):
        torch.manual_seed(0)
        model = JevZip(Config(ctx=16, block=32, d_model=32, n_layers=2, n_heads=2, first=2, order="lanes"))
        pt = self.root / "tiny.pt"
        torch.save(model.export(), pt)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fetch.main(["--register", str(pt), "--description", "tiny", "--recipe", "none"])
            fetch.main(["--register", str(pt), "--trained-on", "nothing"])  # an update keeps the other fields
        e = fetch.load_manifest()["tiny.pt"]
        self.assertEqual((e["kind"], e["order"], e["description"], e["recipe"], e["trained_on"]), ("weights", "lanes", "tiny", "none", "nothing"))
        self.assertEqual(e["params"], sum(p.numel() for p in model.parameters()))
        self.assertEqual((e["size"], e["sha256"]), (pt.stat().st_size, fetch.sha256(pt)))
        self.assertEqual(e["url"], fetch.RELEASE_URL + "tiny.pt")
        manifest = fetch.load_manifest() | {"bare.pt": {"kind": "weights", "url": "", "sha256": "", "size": 0}}
        fetch.MANIFEST.write_text(json.dumps(manifest))  # an entry without params/order must not break --list
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fetch.main(["--list"])
        self.assertIn("bare.pt", out.getvalue())
        manifest["tiny.pt"]["url"] = pt.as_uri()
        fetch.MANIFEST.write_text(json.dumps(manifest))
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            fetch.main(["tiny.pt"])
        self.assertEqual(out.getvalue().strip(), str(self.cache / "tiny.pt"))  # stdout: the path only
        self.assertEqual(JevZip.load(fetch.resolve("tiny.pt")).cfg, model.cfg)
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            fetch.main(["missing.pt"])


if __name__ == "__main__":
    unittest.main()
