"""jevzip container format + compress/decompress.

The data is cut into blocks of `cfg.block` bytes and the blocks are dealt out to
`streams` contiguous chains. A block sees the last `cfg.ctx` bytes of the
previous block in its chain (the first block of a chain sees PAD). Chains are
independent, so at each step the current block of every chain can be decoded
together, in `rounds` parallel passes: the dependency depth is
rounds * ceil(blocks / streams), not one step per byte. On CPU each step's
blocks run in batches of at most CHUNK (a short final block gets its own
batch), so the forward calls made are rounds * sum over steps of
ceil(active / CHUNK).

The decoder must reproduce the encoder's probabilities bit for bit, so both
sides run the exact same sequence of forward calls (same batches, same thread
count). A model fingerprint and a CRC of the output guard against mismatches.
"""
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .model import PAD, JevZip
from .rans import Decoder, Encoder, quantize
from .schedule import filtered_order, round_bounds

MAGIC = b"JEVZ"
VERSION = 1
MODE_RAW, MODE_MODEL = 0, 1
HEADER = struct.Struct("<4sBB8sQIHIBB")  # magic ver mode model_id n crc rounds streams threads bf16
CHUNK = 128  # blocks per forward call (part of the format: changes batching)


def fingerprint(model: JevZip) -> bytes:
    h = hashlib.sha256(json.dumps(asdict(model.cfg), sort_keys=True).encode())  # order/schedule live in the config
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().to(torch.float32).numpy().tobytes())
    return h.digest()[:8]


@dataclass
class Plan:
    """Which block runs at which step: identical on both sides."""

    n: int
    block: int
    streams: int

    def __post_init__(self):
        nb = -(-self.n // self.block)
        self.streams = max(1, min(self.streams, nb))
        edges = np.linspace(0, nb, self.streams + 1).round().astype(int)
        self.chains = [range(a, b) for a, b in zip(edges[:-1], edges[1:])]
        self.depth = max(len(c) for c in self.chains)

    def steps(self):
        """Yield (block index, has context) batches, full-length blocks first."""
        for j in range(self.depth):
            active = [(c[j], j > 0) for c in self.chains if j < len(c)]
            full = [a for a in active if self.length(a[0]) == self.block]
            part = [a for a in active if self.length(a[0]) != self.block]
            for group in (full, part):
                for i in range(0, len(group), CHUNK):
                    yield group[i : i + CHUNK]

    def length(self, b):
        return min(self.block, self.n - b * self.block)


def _tensor(a):
    return torch.from_numpy(np.array(a, dtype=np.int64))  # always a fresh, writable copy


def _run_batch(model, buf, plan: Plan, batch, rounds, use_bf16, emit):
    """Code one batch of equal-length blocks. `buf` holds every byte known so far.

    For each round, emit(cdfs, positions) is called with the quantized CDFs of the
    round's slots and their absolute byte positions (block-major, slot order);
    it must make those bytes available in `buf` before returning (encoder: no-op,
    decoder: decode them).
    """
    cfg = model.cfg
    C, B = cfg.ctx, cfg.block
    n = plan.length(batch[0][0])
    order = filtered_order(B, n, cfg.order)
    bounds = round_bounds(n, rounds, cfg.first)
    starts = np.array([b * B for b, _ in batch])
    ctx = np.full((len(batch), C), PAD, np.int64)
    for i, (b, has_ctx) in enumerate(batch):
        if has_ctx:
            ctx[i] = buf[b * B - C : b * B]
    cache = None
    tok, pos = ctx, np.broadcast_to(np.arange(C), ctx.shape)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
        for r in range(len(bounds) - 1):
            slots = order[bounds[r] : bounds[r + 1]]
            qpos = np.broadcast_to(C + slots, (len(batch), len(slots)))
            cache, logits = model.step(cache, _tensor(tok), _tensor(pos), _tensor(qpos))
            probs = torch.softmax(logits.float(), -1).reshape(-1, logits.shape[-1]).numpy()
            where = (starts[:, None] + slots[None, :]).reshape(-1)
            emit(quantize(probs), where)
            tok = buf[where].reshape(len(batch), len(slots)).astype(np.int64)
            pos = qpos


def _header(mode, model_id, n, crc, rounds, streams, threads, bf16):
    return HEADER.pack(MAGIC, VERSION, mode, model_id, n, crc, rounds, streams, threads, bf16)


def compress(data: bytes, model: JevZip, rounds: int = 16, streams: int | None = None, threads: int = 4, bf16: bool = True, raw_fallback: bool = True) -> bytes:
    if not 1 <= threads <= 255:
        raise ValueError("threads must be in 1..255")
    rounds = max(1, min(rounds, model.cfg.block))  # round_bounds clamps to the block length anyway
    n = len(data)
    crc = zlib.crc32(data)
    model_id = fingerprint(model)
    if n == 0:
        return _header(MODE_RAW, model_id, 0, crc, 0, 0, 0, 0)
    torch.set_num_threads(threads)
    nb = -(-n // model.cfg.block)
    plan = Plan(n, model.cfg.block, streams or max(1, nb // 8))
    buf = np.frombuffer(data, np.uint8)
    syms = []

    def emit(cdfs, where):
        s = buf[where].astype(np.int64)
        rows = np.arange(len(s))
        syms.append(np.stack([cdfs[rows, s], cdfs[rows, s + 1] - cdfs[rows, s]], 1))

    for batch in plan.steps():
        _run_batch(model, buf, plan, batch, rounds, bf16, emit)
    enc = Encoder()
    for start, freq in np.concatenate(syms)[::-1].tolist():
        enc.push(start, freq)
    body = enc.finish()
    head = _header(MODE_MODEL, model_id, n, crc, rounds, plan.streams, threads, int(bf16))
    if raw_fallback and len(body) >= n:
        return _header(MODE_RAW, model_id, n, crc, 0, 0, 0, 0) + data
    return head + body


def decompress(blob: bytes, model: JevZip) -> bytes:
    magic, version, mode, model_id, n, crc, rounds, streams, threads, bf16 = HEADER.unpack_from(blob)
    if magic != MAGIC or version != VERSION:
        raise ValueError("not a jevzip v1 stream")
    body = blob[HEADER.size :]
    if mode == MODE_RAW:
        out = bytes(body[:n])
    else:
        if model_id != fingerprint(model):
            raise ValueError("stream was made with a different model")
        torch.set_num_threads(threads)
        plan = Plan(n, model.cfg.block, streams)
        buf = np.zeros(n, np.uint8)
        dec = Decoder(body)

        def emit(cdfs, where):
            for i, p in enumerate(where.tolist()):
                buf[p] = dec.pop(cdfs[i])

        for batch in plan.steps():
            _run_batch(model, buf, plan, batch, rounds, bool(bf16), emit)
        if not dec.exhausted():
            raise ValueError("trailing rANS data: decoder diverged or stream is corrupt")
        out = buf.tobytes()
    if zlib.crc32(out) != crc:
        raise ValueError("CRC mismatch: decoder diverged (different torch build / CPU kernels?)")
    return out
