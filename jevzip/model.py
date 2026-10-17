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

    @property
    def seq(self) -> int:
        return self.ctx + self.block


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

    def _qkv(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        return q, k, v

    @staticmethod
    def _merge(x):
        b, h, t, hd = x.shape
        return x.transpose(1, 2).reshape(b, t, h * hd)

    def _slots(self, qg, kg, vg, kh, vh, allow=None):
        """Slot attention: revealed content keys (optionally masked) plus the slot's own key, never other slots."""
        s = qg.shape[2]
        eye = torch.eye(s, dtype=torch.bool, device=qg.device)
        if allow is None:
            mask = torch.cat([torch.ones(s, kh.shape[2], dtype=torch.bool, device=qg.device), eye], -1)
        else:
            mask = torch.cat([allow, eye.expand(allow.shape[0], 1, s, s)], -1)
        return F.scaled_dot_product_attention(qg, torch.cat([kh, kg], 2), torch.cat([vh, vg], 2), attn_mask=mask)

    def forward(self, h, g, allow):
        """Training path. h: (b, L, d) content in reveal order; g: (b, S, d) slots; allow: (b, 1, S, L)."""
        qh, kh, vh = self._qkv(h)
        qg, kg, vg = self._qkv(g)
        h = h + self.proj(self._merge(F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)))
        g = g + self.proj(self._merge(self._slots(qg, kg, vg, kh, vh, allow)))
        return h + self.ff(self.ln2(h)), g + self.ff(self.ln2(g))

    def step(self, cache, h, g):
        """Incremental path: append new content h to the KV cache, then score slots g against all of it."""
        b, n, _ = h.shape
        qh, kh, vh = self._qkv(h)
        if cache is not None:
            kh = torch.cat([cache[0], kh], 2)
            vh = torch.cat([cache[1], vh], 2)
        past = kh.shape[2] - n
        allow = torch.ones(n, past + n, dtype=torch.bool).tril(past)
        h = h + self.proj(self._merge(F.scaled_dot_product_attention(qh, kh, vh, attn_mask=allow)))
        h = h + self.ff(self.ln2(h))
        if g is None:
            return (kh, vh), h, None
        qg, kg, vg = self._qkv(g)
        g = g + self.proj(self._merge(self._slots(qg, kg, vg, kh, vh)))
        return (kh, vh), h, g + self.ff(self.ln2(g))


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
        for layer in self.layers:
            h, g = layer(h, g, allow)
        return self.score_options(g)

    @torch.no_grad()
    def step(self, cache, ctok, cpos, qpos):
        """Reveal content (ctok at cpos), then score slots at qpos (or None). Returns (cache, logits)."""
        h = self.tok(ctok) + self.pos(cpos)
        g = None if qpos is None else self._slot_embed(qpos)
        new = []
        for i, layer in enumerate(self.layers):
            c, h, g = layer.step(None if cache is None else cache[i], h, g)
            new.append(c)
        return new, None if g is None else self.score_options(g)

    def export(self) -> dict:
        return {"config": asdict(self.cfg), "state": {k: v.to(torch.bfloat16) for k, v in self.state_dict().items()}}

    @classmethod
    def load(cls, path) -> "JevZip":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(Config(**blob["config"]))
        model.load_state_dict({k: v.float() for k, v in blob["state"].items()})
        return model.eval()
