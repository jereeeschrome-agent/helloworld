# jevzip

A lossless byte compressor built on a Jev-style transformer. Jev is TypeSafe AI's
non-autoregressive model that scores a closed set of options in parallel. jevzip
decodes a block in **K parallel rounds** instead of one model step per byte.

```
python -m jevzip c  file.py  file.jz  --rounds 16
python -m jevzip d  file.jz  file.py
```

## Idea

A neural compressor needs a probability for each byte that the decoder can
reproduce exactly. Autoregressive (AR) models give one byte per step, so
decoding takes N sequential steps. jevzip asks the model to decide many bytes
at once:

1. **Coarse-to-fine reveal order.** Each 512-byte block is revealed in
   bit-reversal order (0, 256, 128, 384, 64, ...). This order is cut into K
   groups that grow geometrically: 8 bytes, then roughly ×1.5 each round.
   Early groups are sparse "anchors". Later groups fill gaps that have known
   bytes on **both** sides.
2. **Two streams, shared weights** (XLNet-style):
   - *content stream*: known bytes (256 bytes of previous-block context, then
     the revealed bytes in reveal order) under causal attention, so it is
     KV-cached and extended one group at a time;
   - *slot stream*: one query token per undecided byte. A slot attends to the
     revealed content and to itself, **never to other slots**. Every slot in a
     group is therefore scored in one pass, and a slot's prediction doesn't
     depend on which other slots are being asked about. That makes the
     decoder's probabilities reproducible.
3. **Option scoring.** Each slot is scored against the 256 byte values using
   tied option embeddings. It's a typed choice from a closed label set, which
   is the Jev idea: "score all allowed options in parallel" instead of
   generating text.
4. **rANS** (64-bit state, 24-bit probabilities) turns the scores into bits.
   The encoder runs the decoder's exact sequence of forward calls, so both
   sides see bit-identical probabilities.

Blocks are dealt into `streams` independent chains. Each block sees its
predecessor in the chain as context, and every chain decodes in lockstep. The
sequential depth is `K × ceil(blocks / streams)` model passes, not N. With
`streams = blocks` the whole file decodes in K passes, but no block has
context.

Training sees every byte at the visibility it will have at decode time, in one
forward pass. The schedule K is sampled per window (log-uniform over 2..512),
so a single model serves any `--rounds` value.

## Results

RESULTS_PLACEHOLDER

## Caveats

- **Not new from scratch.** Each piece has prior art: two-stream attention
  (XLNet), any-order AR models (σ-GPT), fixed-schedule parallel context models
  from learned image compression (checkerboard context, M2T), and neural
  compressors (NNCP, "Language Modeling Is Compression", ts_zip). As far as I
  know, the new part is putting them together for byte-level lossless coding:
  a KV-cached two-stream any-order model, a bit-reversal schedule with
  geometric groups, and lockstep chains. Treat that claim with suspicion.
- **The model isn't counted in the sizes.** That is the usual convention for
  pretrained compressors (like brotli's built-in dictionary), but the weights
  are ~10 MB. You need many MB of similar data before this pays off.
- **Bitwise reproducibility.** Decoding requires the same weights, the same
  torch build, the same CPU kernels and the same thread count, all of which
  are stored in the header. A CRC catches divergence, but a file compressed on
  one machine isn't guaranteed to decode on a different CPU. This is the usual
  neural-compression problem (NNCP writes its own deterministic kernels).
- **Domain.** The model was trained for about 2 CPU-hours on Python source
  installed on the build box. It does well on Python and badly on anything
  else. Incompressible input falls back to raw storage.

## Reproduce

```
pip install -r jevzip/requirements.txt
python -m jevzip.data --out data            # corpus from installed site-packages
python -m jevzip.train --data data --minutes 110
python -m jevzip.bench data/networkx.py.txt data/pydoc_topics.txt --verify
python -m unittest jevzip.test_jevzip       # round-trip tests (tiny random model)
```

Splits are whole packages, so near-duplicate files can't leak across them:
`mpmath` is validation, and `networkx` (code) and `pydoc_data` (mostly English
prose) are the benchmarks. Neither is in the training set.
