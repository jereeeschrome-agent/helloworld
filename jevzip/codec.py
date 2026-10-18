"""jevzip container format + compress/decompress.

The data is cut into blocks of `cfg.block` bytes and the blocks are dealt out to
`streams` contiguous chains. A block sees the last `cfg.ctx` bytes of the
previous block in its chain (the first block of a chain sees PAD). Chains are
independent, so at each step the current block of every chain can be decoded
together, in `rounds` parallel passes: the dependency depth is
rounds * ceil(blocks / streams), not one step per byte. On CPU each step's
blocks run in batches of at most CHUNK (a short final block gets its own
batch), so the forward calls made are rounds * sum over steps of
ceil(active / CHUNK) (forward_calls). Chain starts are coded without context,
so the default (default_streams) is the fewest chains that make no more calls
than blocks // 8 chains did.

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
from .schedule import reveal_plan

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


def default_streams(blocks: int) -> int:
    """compress()'s chain count when none is given: the fewest chains that make
    no more forward calls than the pre-refactor default (blocks // 8).

    Every chain start is a block coded without context (on networkx.py.txt one
    costs ~13 bytes more than with context), so fewer chains are smaller, but
    on CPU a round makes one call per step (more while a step has over CHUNK
    blocks). blocks // 8 gave about 8 steps; any cut needs at least
    ceil(blocks / CHUNK) calls per round. So: allow D = the larger of the two
    step counts and cut the fewest chains that finish in D steps. Those are at
    most CHUNK chains, every step one call. 2048 blocks: 128 chains instead of
    256 (16 calls per round either way); 1015: 113 instead of 126. The header
    stores the count, so files made with any other count still decode.
    """
    old = max(1, blocks // 8)
    steps = max(-(-blocks // old), -(-blocks // CHUNK))
    return -(-blocks // steps)


def forward_calls(plan: Plan, cfg, rounds: int) -> int:
    """Sequential model.step calls that coding `plan` takes (the same on both sides): rounds per batch, summed."""
    return sum(len(reveal_plan(cfg.order, cfg.block, plan.length(b[0][0]), rounds, cfg.first)[1]) - 1 for b in plan.steps())


def _tensor(a, device):
    return torch.from_numpy(np.array(a, dtype=np.int64)).to(device)  # always a fresh, writable copy


def _run_batch(model, buf, plan: Plan, batch, rounds, use_bf16, emit):
    """Code one batch of equal-length blocks. `buf` holds every byte known so far.

    A block whose `has context` flag is set sees the cfg.ctx bytes before it in
    `buf`, else PAD (the codec sets it for all but chain starts; codelength's
    oracle mode sets it for every block after the first). For each round,
    emit(probs, positions) is called with the round's float32 probabilities
    (one row per slot, block-major in slot order; the codec quantize()s them)
    and their absolute byte positions; it must make those bytes available in
    `buf` before returning (encoder: no-op, decoder: decode them). Runs on the
    model's device: the codec keeps it on the CPU.
    """
    cfg = model.cfg
    C, B = cfg.ctx, cfg.block
    dev = model.option_bias.device
    n = plan.length(batch[0][0])
    order, bounds = reveal_plan(cfg.order, B, n, rounds, cfg.first)
    starts = np.array([b * B for b, _ in batch])
    ctx = np.full((len(batch), C), PAD, np.int64)
    for i, (b, has_ctx) in enumerate(batch):
        if has_ctx:
            ctx[i] = buf[b * B - C : b * B]
    cache = None
    tok, pos = ctx, np.broadcast_to(np.arange(C), ctx.shape)
    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=use_bf16):
        for r in range(len(bounds) - 1):
            slots = order[bounds[r] : bounds[r + 1]]
            qpos = np.broadcast_to(C + slots, (len(batch), len(slots)))
            cache, logits = model.step(cache, _tensor(tok, dev), _tensor(pos, dev), _tensor(qpos, dev))
            probs = torch.softmax(logits.float(), -1).reshape(-1, logits.shape[-1]).cpu().numpy()
            where = (starts[:, None] + slots[None, :]).reshape(-1)
            emit(probs, where)
            tok = buf[where].reshape(len(batch), len(slots)).astype(np.int64)
            pos = qpos


def _header(mode, model_id, n, crc, rounds, streams, threads, bf16):
    return HEADER.pack(MAGIC, VERSION, mode, model_id, n, crc, rounds, streams, threads, bf16)


def _symbols(cdfs, s):
    """(start, freq) of symbols s under their rows' CDFs."""
    rows = np.arange(len(s))
    return np.stack([cdfs[rows, s], cdfs[rows, s + 1] - cdfs[rows, s]], 1)


def _encode(syms) -> bytes:
    """rANS body for (start, freq) row arrays given in decode order."""
    enc = Encoder()
    for start, freq in np.concatenate(syms)[::-1].tolist():
        enc.push(start, freq)
    return enc.finish()


def compress(data: bytes, model: JevZip, rounds: int = 16, streams: int | None = None, threads: int = 4, bf16: bool = False, raw_fallback: bool = True) -> bytes:
    """Model-coded stream (or raw, if that is not smaller). streams=None: default_streams.
    bf16 autocast is opt-in: on CPUs without fast bf16 (Apple silicon) it is ~9x slower."""
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
    plan = Plan(n, model.cfg.block, streams or default_streams(nb))
    buf = np.frombuffer(data, np.uint8)
    syms = []

    def emit(probs, where):
        syms.append(_symbols(quantize(probs), buf[where].astype(np.int64)))

    for batch in plan.steps():
        _run_batch(model, buf, plan, batch, rounds, bf16, emit)
    body = _encode(syms)
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

        def emit(probs, where):
            cdfs = quantize(probs)
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
