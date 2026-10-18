"""Model tests (CPU, tiny random models): python -m unittest jevzip.test_model"""
import unittest
from unittest import mock

import numpy as np
import torch

from .model import PAD, Config, JevZip, step_masks
from .schedule import reveal_plan

C, B = 16, 32


def tiny_model(order="lanes", seed=0):
    torch.manual_seed(seed)
    return JevZip(Config(ctx=C, block=B, d_model=32, n_layers=2, n_heads=2, first=2, order=order)).eval()


def training_logits(model, ctx, blk, order, bounds):
    """Logits of the training forward path for windows `ctx` (b, C) / `blk` (b, n), in reveal order."""
    b, n = blk.shape
    ctok = np.full((b, C + B), PAD, np.int64)
    cpos = np.zeros((b, C + B), np.int64)
    ctok[:, :C], cpos[:, :C] = ctx, np.arange(C)
    ctok[:, C : C + n], cpos[:, C : C + n] = blk[:, order], C + order
    group = np.searchsorted(bounds, np.arange(n), side="right") - 1
    qpos = np.broadcast_to(C + order, (b, n))
    qvis = np.broadcast_to(C + np.asarray(bounds)[group], (b, n))
    with torch.no_grad():
        return model(*(torch.from_numpy(np.ascontiguousarray(a)) for a in (ctok, cpos, qpos, qvis)))


def step_logits(model, ctx, blk, order, bounds):
    """Logits of the incremental path (KV cache, one step per round), as a decoder computes them."""
    b = len(blk)
    cache, out = None, []
    tok, pos = ctx, np.broadcast_to(np.arange(C), (b, C))
    for r in range(len(bounds) - 1):
        slots = order[bounds[r] : bounds[r + 1]]
        qpos = np.broadcast_to(C + slots, (b, len(slots)))
        cache, logits = model.step(cache, *(torch.from_numpy(np.ascontiguousarray(a)) for a in (tok, pos, qpos)))
        out.append(logits)
        tok, pos = blk[:, slots], qpos
    return torch.cat(out, 1)


class StepTest(unittest.TestCase):
    def test_step_matches_training_forward(self):
        """Same visibility on both paths: each slot sees the context plus earlier rounds' bytes."""
        rng = np.random.default_rng(0)
        for kind in ("lanes", "bitrev", "linear"):
            model = tiny_model(kind)
            for n in (B, 21, 1):
                blk = rng.integers(0, 256, (3, n))
                for has_ctx in (True, False):
                    ctx = rng.integers(0, 256, (3, C)) if has_ctx else np.full((3, C), PAD)
                    for k in (1, 4, 7, B):
                        order, bounds = reveal_plan(kind, B, n, k, model.cfg.first)
                        with self.subTest(order=kind, n=n, ctx=has_ctx, rounds=k):
                            ref = training_logits(model, ctx, blk, order, bounds)
                            got = step_logits(model, ctx, blk, order, bounds)
                            self.assertEqual(got.shape, ref.shape)
                            torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)

    def test_masks(self):
        """Cached masks equal the per-call construction the codec's reference outputs were made with."""
        for n, past, s in ((16, 0, 4), (4, 16, 4), (1, 40, 1), (3, 5, None)):
            causal, slots = step_masks(n, past, s, torch.device("cpu"))
            self.assertTrue(torch.equal(causal, torch.ones(n, past + n, dtype=torch.bool).tril(past)))
            if s is None:
                self.assertIsNone(slots)
                continue
            want = torch.cat([torch.ones(s, past + n, dtype=torch.bool), torch.eye(s, dtype=torch.bool)], -1)
            self.assertTrue(torch.equal(slots, want))
        self.assertIs(step_masks(4, 16, 4, torch.device("cpu"))[0], step_masks(4, 16, 4, torch.device("cpu"))[0])
        causal, slots = step_masks(2, 3, 2, torch.device("meta"))
        self.assertEqual((causal.device.type, slots.device.type), ("meta", "meta"))

    def test_step_stays_on_device(self):
        """The whole incremental path runs on the inputs' device: on 'meta' any op that mixes in a
        CPU tensor (as the old CPU-built masks did on MPS) raises. The head is skipped because
        autocast has no 'meta' backend."""
        model = tiny_model().to("meta")
        kw = dict(device="meta", dtype=torch.long)
        with mock.patch.object(JevZip, "score_options", lambda self, g: g):
            cache, g = model.step(None, torch.zeros(2, C, **kw), torch.arange(C, device="meta").expand(2, C), torch.full((2, 4), C, **kw))
            self.assertEqual(g.device.type, "meta")
            cache, g = model.step(cache, torch.zeros(2, 4, **kw), torch.full((2, 4), C, **kw), torch.full((2, 4), C + 1, **kw))
            self.assertEqual((g.device.type, tuple(g.shape)), ("meta", (2, 4, 32)))
            cache, g = model.step(cache, torch.zeros(2, 4, **kw), torch.full((2, 4), C + 1, **kw), None)
            self.assertIsNone(g)
        self.assertTrue(all(c.k.device.type == "meta" and c.n == C + 8 for c in cache))


if __name__ == "__main__":
    unittest.main()
