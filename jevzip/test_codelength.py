"""codelength vs real compress() sizes with a tiny random model: python -m unittest jevzip.test_codelength"""
import os
import unittest

import numpy as np

from .codec import CHUNK, HEADER, Plan, compress, default_streams, forward_calls
from .codelength import codelength, cross_entropy
from .test_jevzip import tiny_model


class CodelengthTest(unittest.TestCase):
    model = tiny_model(order="lanes")

    def test_matches_compress(self):
        """header + bits/8 is compress()'s size up to the rANS flush: 8 state bytes plus < 4 bytes of word rounding."""
        text = b"def f(x):\n    return x + 1\n" * 40 + os.urandom(64)
        for n in (1, 31, 32, 100, 333, len(text)):
            for rounds in (1, 5, 32):
                for streams in (None, 1, 3, 1000):
                    data = text[:n]
                    blob = compress(data, self.model, rounds, streams, threads=1, raw_fallback=False)
                    cl = codelength(data, self.model, rounds, streams, threads=1, encode=True, oracle=True)
                    self.assertEqual(cl.header + cl.body, len(blob))  # encode=True codes the same symbols
                    self.assertEqual(HEADER.unpack_from(blob)[7], cl.streams)
                    slack = len(blob) - cl.header - np.sum(cl.bits) / 8
                    self.assertTrue(0 <= slack <= 12, (n, rounds, streams, slack))
                    self.assertFalse(np.isnan(cl.bits).any())
                    # float vs quantized: quantization only adds bits (up to rounding of tiny probabilities)
                    self.assertLess(abs(np.sum(cl.bits) - np.sum(cl.bits_float)), 0.01 * np.sum(cl.bits_float) + 1)
                    # oracle context changes chain starts only, and never the first block
                    starts = np.zeros(n, bool)
                    for b in cl.starts:
                        starts[b * 32 : (b + 1) * 32] = b > 0
                    np.testing.assert_array_equal(cl.oracle_bits[~starts], cl.bits[~starts])
                    self.assertEqual(cl.calls, forward_calls(Plan(n, 32, cl.streams), self.model.cfg, cl.rounds))

    def test_oracle_independent_of_streams(self):
        data = b"import os\nimport sys\n" * 30
        ref = codelength(data, self.model, 8, 1, threads=1, oracle=True)
        for s in (2, 7, 1000):
            cl = codelength(data, self.model, 8, s, threads=1, oracle=True)
            np.testing.assert_allclose(cl.oracle_bits_float, ref.oracle_bits_float, rtol=1e-4, atol=1e-4)
        ce = cross_entropy(data, self.model, 8, threads=1)
        np.testing.assert_allclose(ce.bits_float, ref.oracle_bits_float, rtol=1e-4, atol=1e-4)
        self.assertEqual(ce.calls, 8 * 2)  # 20 blocks in one step: two batches (full blocks, then the short one)

    def test_blocks_subset(self):
        data = b"x = [1, 2, 3]\n" * 20
        full = codelength(data, self.model, 4, 1, threads=1)
        part = codelength(data, self.model, 4, 1, threads=1, blocks=[0, 3])
        self.assertEqual(np.isfinite(part.bits).sum(), 64)
        np.testing.assert_allclose(part.bits[:32], full.bits[:32], rtol=1e-4, atol=1e-4)


class DefaultStreamsTest(unittest.TestCase):
    def test_no_more_calls_than_blocks_over_8(self):
        """The default never makes more forward calls per round than the old blocks // 8, and never more chains."""
        for nb in list(range(1, 2100)) + [4999, 5000, 20000]:
            n = nb * 512 - 7  # a short final block, which gets a batch of its own
            new, old = Plan(n, 512, default_streams(nb)), Plan(n, 512, max(1, nb // 8))
            calls = lambda p: sum(1 for _ in p.steps())
            self.assertLessEqual(calls(new), calls(old), nb)
            self.assertLessEqual(new.streams, old.streams, nb)
            if nb > 8 * CHUNK:
                self.assertLessEqual(new.streams, CHUNK)
                self.assertLessEqual(calls(new), -(-nb // CHUNK) + 1)  # the fewest calls any cut allows, + the short block's own


if __name__ == "__main__":
    unittest.main()
