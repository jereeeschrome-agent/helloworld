"""Jev-style two-stream byte transformer.

Two streams share every weight:

* content stream - the bytes already known (previous-block context, then the
  block's revealed bytes in coarse-to-fine order) under causal attention, so it
  can be KV-cached and extended group by group;
* query stream - one "slot" token per byte still to be decided. A slot sees
  only the revealed content (plus itself), never other slots, so every slot in
  a group is scored in one parallel pass and its prediction does not depend on
  which other slots are being asked about.

The head scores the closed set of 256 byte values against tied option
embeddings - a typed decision per slot instead of free-form generation.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BYTES = 256
MASK = 256  # query-slot token
PAD = 257  # "nothing here" (missing context / past end of data)
VOCAB = 258


@dataclass
class Config:
    ctx: int = 256
    block: int = 512
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    first: int = 8  # size of the first decoding group
    order: str = "bitrev"  # reveal order, see schedule.filtered_order

    @property
    def seq(self) -> int:
        return self.ctx + self.block


def rotary(pos, head_dim):
    """Rotary tables for the original byte offsets `pos` (b, t) -> cos, sin of shape (b, 1, t, head_dim/2).

    Reveal order scrambles sequence order, so attention needs relative byte
    distance to find a slot's neighbours; RoPE hands it that directly."""
    inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=pos.device, dtype=torch.float32) / head_dim))
    ang = pos[:, None, :, None].float() * inv
    return ang.cos(), ang.sin()


def _rotate(x, rot):
    cos, sin = rot
    x1, x2 = x.float().chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).to(x.dtype)


@lru_cache(maxsize=2048)
def step_masks(n: int, past: int, slots: int | None, device: torch.device):
    """Attention masks of one incremental step: (content (n, past+n), slots (slots, past+n+slots) or None).

    New content row i sees cached keys and new keys up to past + i; a slot sees all
    content plus its own key, never other slots. Built on `device` (an op mixing
    devices fails on GPUs) and cached per shape: every layer of a step and every
    block batch with the same schedule reuse them, saving ~5 small kernel launches
    per layer. Treat them as read-only."""
    t = past + n
    causal = torch.ones(n, t, dtype=torch.bool, device=device).tril(past)
    if slots is None:
        return causal, None
    ones = torch.ones(slots, t, dtype=torch.bool, device=device)
    return causal, torch.cat([ones, torch.eye(slots, dtype=torch.bool, device=device)], -1)


class KVCache:
    """Preallocated keys/values for one layer: extending by a group copies only the new rows."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.k = self.v = None
        self.n = 0

    def extend(self, k, v):
        if self.k is None:
            b, h, _, hd = k.shape
            self.k = k.new_empty(b, h, self.capacity, hd)
            self.v = v.new_empty(b, h, self.capacity, hd)
        t = k.shape[2]
        self.k[:, :, self.n : self.n + t] = k
        self.v[:, :, self.n : self.n + t] = v
        self.n += t
        return self.k[:, :, : self.n], self.v[:, :, : self.n]

    def scratch(self, k, v):
        """Cache views followed by k, v written into the free tail, which the next extend overwrites.

        Slots' own keys go here, so slot attention needs no copy of the cache."""
        t = k.shape[2]
        assert self.n + t <= self.capacity
        self.k[:, :, self.n : self.n + t] = k
        self.v[:, :, self.n : self.n + t] = v
        return self.k[:, :, : self.n + t], self.v[:, :, : self.n + t]


class Layer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.GELU(), nn.Linear(cfg.ff_mult * d, d))

    def _qkv(self, x, rot):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        return _rotate(q, rot), _rotate(k, rot), v

    @staticmethod
    def _merge(x):
        b, h, t, hd = x.shape
        return x.transpose(1, 2).reshape(b, t, h * hd)

    @staticmethod
    def _slot_mask(allow, s):
        """Slots see the allowed content keys plus their own key, never other slots."""
        eye = torch.eye(s, dtype=torch.bool, device=allow.device)
        return torch.cat([allow, eye.expand(*allow.shape[:-2], s, s)], -1)

    def forward(self, h, g, allow, rh, rg):
        """Training path. h: (b, L, d) content in reveal order; g: (b, S, d) slots; allow: (b, 1, S, L);
        rh/rg: rotary tables for their byte offsets."""
        qh, kh, vh = self._qkv(h, rh)
        qg, kg, vg = self._qkv(g, rg)
        h = h + self.proj(self._merge(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)))
        mask = self._slot_mask(allow, qg.shape[2])
        g = g + self.proj(self._merge(F.scaled_dot_product_attention(qg, torch.cat([kh, kg], 2), torch.cat([vh, vg], 2), attn_mask=mask)))
        return h + self.ff(self.ln2(h)), g + self.ff(self.ln2(g))

    def step(self, cache, h, g, rh, rg):
        """Incremental path: append new content h to the KV cache, then score slots g against all of it."""
        n = h.shape[1]
        causal, slots = step_masks(n, cache.n, None if g is None else g.shape[1], h.device)
        qh, kh, vh = self._qkv(h, rh)
        kh, vh = cache.extend(kh, vh)
        h = h + self.proj(self._merge(F.scaled_dot_product_attention(qh, kh, vh, attn_mask=causal)))
        h = h + self.ff(self.ln2(h))
        if g is None:
            return h, None
        qg, kg, vg = self._qkv(g, rg)
        k_all, v_all = cache.scratch(kg, vg)
        g = g + self.proj(self._merge(F.scaled_dot_product_attention(qg, k_all, v_all, attn_mask=slots)))
        return h, g + self.ff(self.ln2(g))


class JevZip(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(cfg.seq, d)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.option_bias = nn.Parameter(torch.zeros(N_BYTES))
        self.apply(self._init)
        for layer in self.layers:  # scale residual branches
            nn.init.normal_(layer.proj.weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)
            nn.init.normal_(layer.ff[2].weight, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def score_options(self, g):
        """Score all 256 byte values for every slot at once (kept in fp32: logit precision is bits)."""
        with torch.autocast(g.device.type, enabled=False):
            return self.ln_f(g.float()) @ self.tok.weight[:N_BYTES].T + self.option_bias

    def _slot_embed(self, qpos):
        return self.tok.weight[MASK] + self.pos(qpos)

    def forward(self, ctok, cpos, qpos, qvis):
        """ctok/cpos: (b, L) content tokens + positions in reveal order.
        qpos: (b, S) slot positions; qvis: (b, S) how many content tokens each slot may see."""
        h = self.tok(ctok) + self.pos(cpos)
        g = self._slot_embed(qpos)
        allow = (torch.arange(ctok.shape[1], device=ctok.device) < qvis[..., None])[:, None]
        hd = self.cfg.d_model // self.cfg.n_heads
        rh, rg = rotary(cpos, hd), rotary(qpos, hd)
        for layer in self.layers:
            h, g = layer(h, g, allow, rh, rg)
        return self.score_options(g)

    @torch.no_grad()
    def step(self, cache, ctok, cpos, qpos):
        """Reveal content (ctok at cpos), then score slots at qpos (or None). Returns (cache, logits);
        pass cache=None on the first call of a block batch."""
        h = self.tok(ctok) + self.pos(cpos)
        g = None if qpos is None else self._slot_embed(qpos)
        hd = self.cfg.d_model // self.cfg.n_heads
        rh, rg = rotary(cpos, hd), None if qpos is None else rotary(qpos, hd)
        if cache is None:
            cache = [KVCache(self.cfg.seq) for _ in self.layers]
        for layer, c in zip(self.layers, cache):
            h, g = layer.step(c, h, g, rh, rg)
        return cache, None if g is None else self.score_options(g)

    def export(self) -> dict:
        return {"config": asdict(self.cfg), "state": {k: v.detach().to("cpu", torch.bfloat16) for k, v in self.state_dict().items()}}

    @classmethod
    def load(cls, path) -> "JevZip":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(Config(**blob["config"]))
        model.load_state_dict({k: v.float() for k, v in blob["state"].items()})
        return model.eval()
