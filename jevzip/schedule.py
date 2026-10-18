"""Decoding schedules: a reveal order plus its cut into K parallel groups.

* "lanes": the block is split into ceil(block/K) contiguous lanes, each
  decoded left to right, all lanes advancing one byte per round. Every byte
  except the first of each lane after lane 0 has its immediate left neighbour
  revealed before it (lane 0 starts after the previous-block context; the other
  ceil(block/K) - 1 lane starts are predicted in round 0 without one).
  K = block is plain autoregression.
* "linear": left to right, cut into K groups whose revealed count grows
  geometrically (with K = block, the AR control).

Every byte in a group is predicted in parallel from the bytes revealed by
earlier groups.
"""
import numpy as np


def filtered_order(block: int, n: int, kind: str = "linear") -> np.ndarray:
    """Reveal order for a block truncated to its first n positions ("linear" only; "lanes" depends on K: use reveal_plan)."""
    if kind != "linear":
        raise ValueError(f"unknown reveal order {kind!r}")
    return np.arange(n, dtype=np.int64)


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
