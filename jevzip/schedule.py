"""Coarse-to-fine decoding schedules.

A block of n bytes is revealed in a fixed permuted order (bit-reversal, so the
first bytes coded are spread evenly across the block and every later byte has
known neighbours on both sides). The permuted order is cut into K groups whose
sizes grow geometrically; every byte in a group is predicted in parallel from
the bytes revealed by earlier groups.
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


def filtered_order(block: int, n: int) -> np.ndarray:
    """Order for a block truncated to its first n positions."""
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
