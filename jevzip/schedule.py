"""Decoding schedules: a reveal order plus its cut into K parallel groups.

* "bitrev": coarse-to-fine bit-reversal order (first bytes spread across the
  block, later bytes have known neighbours on both sides), cut into K groups
  whose revealed count grows geometrically.
* "lanes": the block is split into ceil(block/K) contiguous lanes, each
  decoded left to right, all lanes advancing one byte per round. Every byte
  keeps its immediate left neighbour; K = block is plain autoregression.
* "linear": left to right, cut like "bitrev" (with K = block, the AR control).

Every byte in a group is predicted in parallel from the bytes revealed by
earlier groups.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def coding_order(n: int) -> np.ndarray:
    """Bit-reversal permutation of range(n) (n need not be a power of two)."""
    bits = max(1, (n - 1).bit_length())
    idx = np.arange(1 << bits, dtype=np.int64)
    rev = np.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    order = rev[rev < n]
    order.setflags(write=False)
    return order


def filtered_order(block: int, n: int, kind: str = "bitrev") -> np.ndarray:
    """Reveal order for a block truncated to its first n positions.

    "bitrev" is jevzip's coarse-to-fine order; "linear" is plain left-to-right
    (with one byte per round that is an ordinary autoregressive model, kept as
    a control)."""
    if kind == "linear":
        return np.arange(n, dtype=np.int64)
    if kind != "bitrev":  # "lanes" depends on K: use reveal_plan
        raise ValueError(f"unknown reveal order {kind!r}")
    order = coding_order(block)
    return order if n == block else order[order < n]


def round_bounds(n: int, rounds: int, first: int = 8) -> list[int]:
    """Cumulative group boundaries [0, m_1, ..., m_K = n] in the permuted order.

    Group r covers permuted indices [m_r, m_{r+1}) and is predicted from the
    m_r bytes already revealed. The revealed count m_r grows geometrically from
    `first` to n (so each group is a fixed fraction of what is already known),
    and every group holds at least one byte.
    """
    rounds = max(1, min(rounds, n))
    first = max(1, min(first, n))
    bounds = [0]
    for r in range(1, rounds + 1):
        if r == rounds:
            v = n
        else:
            v = round(first * (n / first) ** ((r - 1) / (rounds - 1)))
        v = min(max(v, bounds[-1] + 1), n - (rounds - r))
        bounds.append(v)
    return bounds


def reveal_plan(kind: str, block: int, n: int, rounds: int, first: int = 8) -> tuple[np.ndarray, list[int]]:
    """(order, bounds) for a block whose first n bytes exist: group r is order[bounds[r]:bounds[r+1]]."""
    if kind == "lanes":
        span = max(1, min(rounds, block))  # bytes per lane = rounds
        lanes = -(-block // span)
        pos = (np.arange(span)[:, None] + span * np.arange(lanes)[None, :]).reshape(-1)
        rnd = np.repeat(np.arange(span), lanes)
        keep = pos < n
        order, rnd = pos[keep], rnd[keep]
        return order, np.unique(np.searchsorted(rnd, np.arange(span + 1))).tolist()
    return filtered_order(block, n, kind), round_bounds(n, rounds, first)
