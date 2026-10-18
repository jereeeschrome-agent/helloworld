"""Dataset tooling on synthetic files: splits, manifest checks, downloads (file:// URLs), per-file indexes. No network."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from jevzip import data as D


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestSplit(unittest.TestCase):
    def test_standard_split(self):
        blob = bytes(range(256)) * 4
        s = D.split(blob, val=100, test=50)
        self.assertEqual([len(s[k]) for k in ("train", "val", "test")], [874, 100, 50])
        self.assertEqual(s["train"] + s["val"] + s["test"], blob)
        self.assertEqual(s["test"], blob[-50:])

    def test_default_is_90_5_5(self):
        self.assertEqual(D.SPLIT, {"val": 5_000_000, "test": 5_000_000})

    def test_too_small(self):
        with self.assertRaises(ValueError):
            D.split(b"x" * 10, val=5, test=5)


class TestDigests(unittest.TestCase):
    def test_git_blob_id(self):
        self.assertEqual(D.git_sha1(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")  # `git hash-object`

    def test_mismatch(self):
        sha = D.digest(b"abc")
        self.assertIsNone(D.mismatch({"size": 3, "sha256": sha}, 3, sha))
        self.assertIsNone(D.mismatch({}, 3, sha))  # nothing recorded, nothing checked
        self.assertIn("size", D.mismatch({"size": 4}, 3, sha))
        self.assertIn("sha256", D.mismatch({"sha256": "0" * 64}, 3, sha))


class TestPut(Tmp):
    def test_write_unchanged_refuse_force(self):
        p = self.dir / "a" / "x.bin"
        self.assertEqual(D.put(p, b"one"), "wrote")
        self.assertEqual(D.put(p, b"one"), "unchanged")
        with self.assertRaises(FileExistsError):
            D.put(p, b"two")
        self.assertEqual(p.read_bytes(), b"one")
        self.assertEqual(D.put(p, b"two", force=True), "replaced")
        self.assertEqual(p.read_bytes(), b"two")
        self.assertEqual(list(p.parent.glob(".*")), [])  # no temp files left

    def test_manifest_mismatch(self):
        p = self.dir / "x.bin"
        with self.assertRaises(ValueError):
            D.put(p, b"abc", {"sha256": D.digest(b"abd")})
        self.assertFalse(p.exists())
        D.put(p, b"abc", {"size": 4, "reproducible": False})  # machine-dependent: a warning, still written
        self.assertEqual(p.read_bytes(), b"abc")


class TestDownload(Tmp):
    def setUp(self):
        super().setUp()
        self.src = self.dir / "src"
        self.src.mkdir()
        (self.src / "f.txt").write_bytes(b"payload\n")
        self.root = self.dir / "downloads"

    def manifest(self, **entry):
        return {"downloads": {"x/f.txt": {"url": (self.src / "f.txt").as_uri(), **entry}}}

    def test_fetch_and_check(self):
        good = dict(size=8, sha256=D.digest(b"payload\n"), git_sha1=D.git_sha1(b"payload\n"))
        p = D.download("x/f.txt", self.manifest(**good), self.root)
        self.assertEqual(p.read_bytes(), b"payload\n")
        self.assertEqual(D.download("x/f.txt", self.manifest(**good), self.root), p)  # present: checked, not fetched

    def test_bad_download_is_deleted(self):
        for bad in ({"sha256": "0" * 64}, {"size": 7}, {"git_sha1": "0" * 40}):
            with self.assertRaises(OSError):
                D.download("x/f.txt", self.manifest(**bad), self.root)
            self.assertEqual(list((self.root / "x").iterdir()), [])

    def test_bad_existing_file_is_kept(self):
        (self.root / "x").mkdir(parents=True)
        (self.root / "x" / "f.txt").write_bytes(b"other\n")
        with self.assertRaises(OSError):
            D.download("x/f.txt", self.manifest(sha256=D.digest(b"payload\n")), self.root)
        self.assertEqual((self.root / "x" / "f.txt").read_bytes(), b"other\n")


class TestZip(Tmp):
    def test_read_member(self):
        z = self.dir / "a.zip"
        with zipfile.ZipFile(z, "w") as f:
            f.writestr("m", b"member data")
        self.assertEqual(D.read_member(z, "m", {"size": 11, "md5": D.digest(b"member data", "md5")}), b"member data")
        for bad in ({"size": 10}, {"md5": "0" * 32}, {"sha256": "0" * 64}):
            with self.assertRaises(ValueError):
                D.read_member(z, "m", bad)


class TestPieces(Tmp):
    def build(self, budget, sub="a"):
        files = [("p/a.py", b"aaaa"), ("p/b.py", b"bb"), ("q/c.py", b"cccccc")]
        blob = D.SEPARATOR.join(d for _, d in files)[:budget]
        (self.dir / sub).mkdir()
        (self.dir / sub / "blob.txt").write_bytes(blob)
        return D.write_pieces(self.dir / sub / "files", self.dir / sub / "blob.txt", files, blob)

    def test_whole(self):
        listed = self.build(1000)
        self.assertEqual([(f["path"], f["offset"], f["size"]) for f in listed], [("p/a.py", 0, 4), ("p/b.py", 6, 2), ("q/c.py", 10, 6)])
        self.assertEqual((self.dir / "a" / "files" / "q" / "c.py").read_bytes(), b"cccccc")
        self.assertEqual(D.check_index(self.dir / "a" / "files" / "index.json"), [])

    def test_cut_short(self):
        listed = self.build(13)  # the blob ends inside c.py
        self.assertEqual((listed[-1]["size"], listed[-1]["full_size"]), (3, 6))
        self.assertEqual(D.check_index(self.dir / "a" / "files" / "index.json"), [])
        listed = self.build(5, "b")  # ends inside the separator after a.py: b.py is not listed
        self.assertEqual([f["path"] for f in listed], ["p/a.py"])
        self.assertEqual(D.check_index(self.dir / "b" / "files" / "index.json"), [])

    def test_detects_damage(self):
        self.build(1000, "a")
        (self.dir / "a" / "files" / "p" / "b.py").write_bytes(b"BB")
        self.assertTrue(any("b.py" in p for p in D.check_index(self.dir / "a" / "files" / "index.json")))
        self.build(1000, "b")
        (self.dir / "b" / "blob.txt").write_bytes((self.dir / "b" / "blob.txt").read_bytes() + b"extra")
        self.assertTrue(any("cover" in p for p in D.check_index(self.dir / "b" / "files" / "index.json")))


class TestManifest(Tmp):
    def test_verify_and_record(self):
        data, dl = self.dir / "data", self.dir / "dl"
        (data / "s").mkdir(parents=True)
        (data / "s" / "a.bin").write_bytes(b"A" * 10)
        (data / "s" / "b.bin").write_bytes(b"B" * 10)
        dl.mkdir()
        with zipfile.ZipFile(dl / "z.zip", "w") as z:
            z.writestr("m", b"zipped")
        m = {"files": {"s/a.bin": {"size": 10, "sha256": D.digest(b"A" * 10)}, "s/b.bin": {"size": 10, "sha256": "0" * 64},
                       "s/c.bin": {}, "t/d.bin": {}},
             "downloads": {"z.zip": {"members": {"m": {}}}}, "suites": {"x": {"g": ["s/a.bin"]}}}
        report = dict(D.verify(data=data, manifest=m))
        self.assertEqual(report["s/a.bin"], "ok")
        self.assertIn("sha256", report["s/b.bin"])
        self.assertEqual(report["s/c.bin"], "missing")
        self.assertEqual([k for k, _ in D.verify(["t/"], data=data, manifest=m)], ["t/d.bin"])
        self.assertEqual(D.suite("x", data, m), {"g": [data / "s" / "a.bin"]})

        path = self.dir / "m.json"
        path.write_text(json.dumps(m))
        (data / "t").mkdir()
        (data / "t" / "d.bin").write_bytes(b"dd")
        filled = D.record(path, data, dl)
        self.assertEqual(sorted(filled), ["t/d.bin", "z.zip", "z.zip:m"])
        r = D.load_manifest(path)
        self.assertEqual(r["files"]["s/b.bin"]["sha256"], "0" * 64)  # recorded values are never changed
        self.assertEqual(r["files"]["t/d.bin"], {"size": 2, "sha256": D.digest(b"dd")})
        self.assertEqual(r["downloads"]["z.zip"]["members"]["m"]["sha256"], D.digest(b"zipped"))
        self.assertEqual(dict(D.verify(data=data, downloads=dl, manifest=r))["z.zip"], "ok")

    def test_shipped_manifest(self):
        """The real manifest is well-formed: every file recorded, every 'from' a download, every suite key a file."""
        m = D.load_manifest()
        for key, e in m["files"].items():
            self.assertTrue(e.get("sha256") and e.get("size") is not None and e.get("license") and e.get("split"), key)
            if "from" in e:
                self.assertIn(e["from"], m["downloads"], key)
                if "member" in e:
                    self.assertIn(e["member"], m["downloads"][e["from"]]["members"], key)
        for key, e in m["downloads"].items():
            self.assertTrue(e["url"].startswith("https://") and e.get("sha256") and e.get("license"), key)
        for groups in m["suites"].values():
            for keys in groups.values():
                for k in keys:
                    self.assertIn(k, m["files"])
        self.assertEqual(m["files"]["enwik8/test.bin"]["sha256"], "86f62c0adeb6c27763f308b8913bec743626421a4d2b4010f1e177a40eef7750")
        self.assertLessEqual(sum(e["size"] for k, e in m["downloads"].items() if k.startswith("ood-")), 10_000_000)


class TestHeaders(Tmp):
    def test_prefix_within_budget(self):
        for i, d in enumerate(D.HEADER_DIRS):
            (self.dir / d).mkdir(parents=True)
            (self.dir / d / f"h{i}.h").write_bytes(b"x" * 10)
            (self.dir / d / "skip.cpp").write_bytes(b"y")
        names = [n for n, _ in D.header_files(self.dir, budget=10)]
        self.assertEqual(len(names), 1)
        self.assertEqual(len(D.header_files(self.dir, budget=22)), 2)  # 10 + separator + 10
        self.assertTrue(all(n.endswith(".h") for n, _ in D.header_files(self.dir)))


if __name__ == "__main__":
    unittest.main()
