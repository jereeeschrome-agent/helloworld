"""Range asymmetric numeral systems (rANS) coder: 64-bit state, 32-bit words.

Same construction as ryg's rans64. Symbols are pushed in reverse decode order;
the decoder pops them forward.
"""
import numpy as np

PROB_BITS = 24
PROB_SCALE = 1 << PROB_BITS
RANS_L = 1 << 31
_MASK32 = (1 << 32) - 1


def quantize(probs: np.ndarray) -> np.ndarray:
    """(n, 256) float probabilities -> (n, 257) int64 CDFs summing to PROB_SCALE, every freq >= 1."""
    probs = np.asarray(probs, np.float64)
    k = probs.shape[-1]
    freq = np.floor(probs / probs.sum(-1, keepdims=True) * (PROB_SCALE - k)).astype(np.int64) + 1
    rows = np.arange(len(freq))
    freq[rows, freq.argmax(-1)] += PROB_SCALE - freq.sum(-1)
    cdf = np.zeros((len(freq), k + 1), np.int64)
    np.cumsum(freq, -1, out=cdf[:, 1:])
    return cdf


class Encoder:
    def __init__(self):
        self.x = RANS_L
        self.words: list[int] = []

    def push(self, start: int, freq: int):
        x = self.x
        if x >= ((RANS_L >> PROB_BITS) << 32) * freq:
            self.words.append(x & _MASK32)
            x >>= 32
        self.x = ((x // freq) << PROB_BITS) + (x % freq) + start

    def finish(self) -> bytes:
        words = self.words + [self.x & _MASK32, self.x >> 32]
        return np.asarray(words[::-1], dtype="<u4").tobytes()


class Decoder:
    def __init__(self, buf: bytes):
        self.words = np.frombuffer(buf, dtype="<u4").tolist()
        self.x = (self.words[0] << 32) | self.words[1]
        self.i = 2

    def pop(self, cdf: np.ndarray) -> int:
        """Decode one symbol given its (257,) CDF."""
        x = self.x
        slot = x & (PROB_SCALE - 1)
        s = int(np.searchsorted(cdf, slot, side="right")) - 1
        start = int(cdf[s])
        x = (int(cdf[s + 1]) - start) * (x >> PROB_BITS) + slot - start
        if x < RANS_L:
            x = (x << 32) | self.words[self.i]
            self.i += 1
        self.x = x
        return s
