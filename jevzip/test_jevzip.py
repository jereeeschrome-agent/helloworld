"""Round-trip tests with a tiny random model: python -m unittest jevzip.test_jevzip"""
import os
import unittest

import numpy as np
import torch

from .codec import HEADER, MODE_RAW, compress, decompress, default_streams
from .model import Config, JevZip
from .rans import Decoder, Encoder, quantize
from .schedule import coding_order, filtered_order, reveal_plan, round_bounds


def tiny_model(seed=0, order="bitrev"):
    torch.manual_seed(seed)
    return JevZip(Config(ctx=16, block=32, d_model=32, n_layers=2, n_heads=2, first=2, order=order)).eval()


class ScheduleTest(unittest.TestCase):
    def test_order_is_permutation(self):
        for n in (1, 2, 7, 32, 100, 512):
            self.assertEqual(sorted(coding_order(n).tolist()), list(range(n)))
        self.assertEqual(coding_order(8).tolist(), [0, 4, 2, 6, 1, 5, 3, 7])
        self.assertEqual(filtered_order(8, 5).tolist(), [0, 4, 2, 1, 3])
        self.assertEqual(filtered_order(8, 5, "linear").tolist(), [0, 1, 2, 3, 4])

    def test_lanes(self):
        order, bounds = reveal_plan("lanes", 8, 8, 4)
        self.assertEqual(order.tolist(), [0, 4, 1, 5, 2, 6, 3, 7])
        self.assertEqual(bounds, [0, 2, 4, 6, 8])
        order, bounds = reveal_plan("lanes", 8, 6, 4)  # truncated block
        self.assertEqual(order.tolist(), [0, 4, 1, 5, 2, 3])
        self.assertEqual(bounds, [0, 2, 4, 5, 6])
        for n in (1, 7, 32):
            for k in (1, 3, 8, 32, 100):
                order, bounds = reveal_plan("lanes", 32, n, k)
                self.assertEqual(sorted(order.tolist()), list(range(n)))
                self.assertEqual((bounds[0], bounds[-1]), (0, n))
                self.assertTrue(all(x < y for x, y in zip(bounds, bounds[1:])))

    def test_bounds(self):
        for n in (1, 5, 32, 512):
            for k in (1, 2, 8, 16, 1000):
                b = round_bounds(n, k)
                self.assertEqual((b[0], b[-1]), (0, n))
                self.assertTrue(all(x < y for x, y in zip(b, b[1:])))
                self.assertEqual(len(b) - 1, min(k, n))


class RansTest(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.full(256, 0.05), size=2000)
        probs[0] = 0
        probs[0, 7] = 1  # near-deterministic symbol
        cdfs = quantize(probs)
        syms = [int(rng.choice(256, p=p / p.sum())) for p in probs]
        enc = Encoder()
        for s, c in zip(syms[::-1], cdfs[::-1]):
            enc.push(int(c[s]), int(c[s + 1] - c[s]))
        buf = enc.finish()
        dec = Decoder(buf)
        self.assertEqual([dec.pop(c) for c in cdfs], syms)
        self.assertTrue(dec.exhausted())
        with self.assertRaises(ValueError):
            short = Decoder(buf[: len(buf) // 2 // 4 * 4])
            for c in cdfs:
                short.pop(c)


class CodecTest(unittest.TestCase):
    model = tiny_model()

    def roundtrip(self, data, **kw):
        kw.setdefault("raw_fallback", False)  # a random model never beats raw; force the model path
        blob = compress(data, self.model, threads=1, **kw)
        self.assertEqual(decompress(blob, self.model), data)
        return blob

    def test_sizes(self):
        text = (b"def f(x):\n    return x + 1\n" * 40)
        for n in (0, 1, 15, 31, 32, 33, 64, 100, len(text)):
            self.roundtrip(text[:n])

    def test_schedules(self):
        data = os.urandom(50) + b"abc" * 100
        for rounds in (1, 3, 16, 32, 64):
            for streams in (1, 2, 5, 1000):
                self.roundtrip(data, rounds=rounds, streams=streams)

    def test_default_streams(self):
        """streams=None picks default_streams(blocks) and stores it; files cut any other way still decode."""
        data = b"for i in range(3):\n    print(i)\n" * 60  # 60 blocks
        blob = self.roundtrip(data)
        self.assertEqual(HEADER.unpack_from(blob)[7], default_streams(-(-len(data) // 32)))
        self.roundtrip(data, streams=60 // 8)  # the pre-refactor default

    def test_other_orders(self):
        for order in ("linear", "lanes"):
            self._roundtrip_order(order)

    def _roundtrip_order(self, order):
        model = tiny_model(order=order)
        data = b"import os\nimport sys\n" * 9
        for rounds in (4, 7, 32):
            blob = compress(data, model, rounds, threads=1, raw_fallback=False)
            self.assertEqual(decompress(blob, model), data)

    def test_fp32(self):
        self.roundtrip(b"hello world " * 30, bf16=False)

    def test_incompressible_falls_back_to_raw(self):
        blob = self.roundtrip(os.urandom(300), raw_fallback=True)
        self.assertEqual(HEADER.unpack_from(blob)[2], MODE_RAW)

    def test_corrupt_stream_rejected(self):
        blob = bytearray(compress(b"def g(y):\n    pass\n" * 20, self.model, threads=1, raw_fallback=False))
        blob[-5] ^= 0x40
        with self.assertRaises(ValueError):
            decompress(bytes(blob), self.model)

    def test_bad_args(self):
        with self.assertRaises(ValueError):
            compress(b"x" * 10, self.model, threads=256)
        self.roundtrip(b"abc" * 30, rounds=100000)  # clamped, still fits the header

    def test_wrong_model_rejected(self):
        blob = compress(b"some text " * 20, self.model, threads=1, raw_fallback=False)
        with self.assertRaises(ValueError):
            decompress(blob, tiny_model(seed=1))


if __name__ == "__main__":
    unittest.main()
