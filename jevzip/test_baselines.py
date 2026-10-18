"""Baseline registry tests on tiny inputs: python -m unittest jevzip.test_baselines"""
import lzma
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path

from . import baselines as bl

TEXT = b"".join(b"def f%d(x):\n    return x * %d + len('%s')\n" % (i, i * 7, b"ab" * (i % 5)) for i in range(400))


class BaselinesTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.ctx = bl.Ctx(workdir=self.dir / "work")
        self.ctx.work.mkdir()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_python_native_round_trip(self):
        rows = bl.run(TEXT, bl.select(only=["gzip -9", "bzip2 -9", "xz -9e", "zstd -19", "brotli -q11"]), self.ctx)
        self.assertEqual(len(rows), 5)
        for r in rows:
            if "skipped" in r:  # zstandard / brotli modules absent
                continue
            self.assertNotIn("error", r, r)
            self.assertTrue(r["verified"], r["name"])
            self.assertLess(r["bytes"], len(TEXT) // 4)
            self.assertAlmostEqual(r["bits_per_byte"], 8 * r["bytes"] / len(TEXT))
            self.assertTrue(r["command"] and r["version"])
        self.assertEqual(rows[0]["bytes"], len(__import__("gzip").compress(TEXT, 9, mtime=0)))

    def test_marginal(self):
        size = lambda b: len(zlib.compress(b, 9))  # noqa: E731
        p, x = TEXT[:8000], TEXT[4000:6000]
        m = bl.marginal(size, p, x)
        self.assertEqual(m, size(p + x) - size(p))
        self.assertLess(m, size(x) // 4)  # x repeats inside the prefix's window
        self.assertEqual(bl.marginal(size, p, b"xyz", prefix_size=100), size(p + b"xyz") - 100)

    def test_xz_primed_row_and_cache(self):
        corpus = self.dir / "corpus.bin"
        corpus.write_bytes(TEXT * 2)
        ctx = bl.Ctx(corpus, workdir=self.dir / "w2", cache=self.dir / "cache.json")
        ctx.work.mkdir()
        x = TEXT[:3000]
        (r,) = bl.run(x, bl.select(only=["xz -9e primed"]), ctx)
        self.assertNotIn("error", r, r)
        self.assertIsNone(r["verified"])
        self.assertEqual(r["ref_bytes"], len(TEXT) * 2)
        self.assertEqual(r["bytes"], bl.xz_size(TEXT * 2 + x) - bl.xz_size(TEXT * 2))
        self.assertLess(r["bytes"], len(lzma.compress(x)))
        self.assertEqual(list(bl.json.loads((self.dir / "cache.json").read_text()).values()), [bl.xz_size(TEXT * 2)])

    def test_skips_and_errors(self):
        (r,) = bl.run(TEXT, bl.select(only=["xz -9e primed"]), self.ctx)  # no corpus
        self.assertIn("skipped", r)
        broken = bl.Baseline("broken", "standalone", "fast", lambda data, ctx: 1 / 0)
        (r,) = bl.run(TEXT, [broken], self.ctx)
        self.assertIn("ZeroDivisionError", r["error"])

    def test_select(self):
        fast = bl.select("fast")
        self.assertTrue(all(b.group == "fast" for b in fast))
        self.assertGreater(len(bl.select("all")), len(fast))
        self.assertEqual(bl.select("none"), [])
        self.assertEqual([b.name for b in bl.select(only=["zstd -19 .*"])], ["zstd -19 + 110 KiB dict", "zstd -19 + 1 MiB dict"])

    def test_merge_and_markdown(self):
        res = dict(file="f", size=100, sha256="0" * 64)
        a = dict(results=[res | dict(rows=[dict(name="cmix", kind="standalone", bytes=10, verified=True, enc_s=1.0, dec_s=1.0)])])
        b = dict(results=[res | dict(rows=[dict(name="cmix", kind="standalone", skipped="absent"), dict(name="gzip -9", kind="standalone", bytes=40, verified=True, enc_s=0.0, dec_s=0.0)])])
        (m,) = bl.merge([a, b])
        self.assertEqual({r["name"]: r.get("bytes") for r in m["rows"]}, {"cmix": 10, "gzip -9": 40})
        md = bl.markdown([m])
        self.assertLess(md.index("| cmix"), md.index("| gzip -9"))  # sorted by size
        self.assertIn("0.8000", md)

    @unittest.skipUnless(shutil.which("zstd"), "zstd CLI not installed")
    def test_zstd_cli_and_patch_from(self):
        corpus = self.dir / "corpus.bin"
        corpus.write_bytes(TEXT)
        ctx = bl.Ctx(corpus, workdir=self.dir / "w3")
        ctx.work.mkdir()
        rows = bl.run(TEXT[:5000], bl.select(only=["zstd --ultra -22 --long=27", "zstd --ultra -22 --long=31 --patch-from=corpus"]), ctx)
        for r in rows:
            self.assertNotIn("error", r, r)
            self.assertTrue(r["verified"])
        self.assertLess(rows[1]["bytes"], rows[0]["bytes"])  # x is inside the reference
        self.assertEqual(rows[1]["ref_bytes"], len(TEXT))


if __name__ == "__main__":
    unittest.main()
