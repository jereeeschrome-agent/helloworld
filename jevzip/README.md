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
   groups. The number of revealed bytes grows geometrically, from 8 up to
   512, by (512/8)^(1/(K-1)) per round (×1.32 at K=16). So each group is a
   fixed fraction of what's already known.
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
dependency depth is `K × ceil(blocks / streams)` rounds, not N bytes. With
`streams = blocks` it is just K, but then no block has context. This CPU
build runs each lockstep step's blocks in batches of at most 128, and gives a
short last block its own batch, so the forward calls actually made are
`K × Σ_steps ceil(active / 128)`. The bench reports that number. A wider
batch would reach the dependency depth, but it would change the format,
because batching is part of what the decoder must replay.

Attention uses RoPE on the original byte offsets. The reveal order scrambles
sequence order, and a slot mostly needs "what's 1, 2, 4 bytes to my left and
right". With learned absolute positions only, the model learned that far more
slowly (see Results).

Training sees every byte at the visibility it will have at decode time, in one
forward pass. The schedule K is sampled per window, log-uniform over 8..128.
The content stream doesn't depend on K, and a slot's prediction depends only
on its position and how much is revealed. So one model runs with any
`--rounds`, but it is only *trained* for 8 ≤ K ≤ 128. Below that (e.g. K=4),
slots see sparser contexts than training ever produced, and compression
suffers.

## Results

All numbers were measured on the build box: a 4-core Xeon with AMX, torch
2.14 on CPU, bf16 inference, 4 threads. Every jevzip row is a real compressed
file, decompressed and checked bit-exact. The model has 5.0M parameters and
was trained for 120 minutes: 3.4k steps of 16 windows, about 28M supervised
bytes. Default streams = blocks/8, so each chain is 8 blocks long.

**networkx**: held-out Python, 1.05 MB

| method | bytes | bits/byte | forward calls | enc KB/s | dec KB/s |
|---|---:|---:|---:|---:|---:|
| gzip -9 | 247078 | 1.885 | | | |
| bzip2 -9 | 204202 | 1.558 | | | |
| xz -9e | 199944 | 1.525 | | | |
| zstd -19 | 203658 | 1.554 | | | |
| brotli -q11 | 196333 | 1.498 | | | |
| jevzip K=8 | 339194 | 2.588 | 128 | 12.7 | 12.6 |
| jevzip K=16 | 316278 | 2.413 | 256 | 10.1 | 11.0 |
| jevzip K=32 | 308714 | 2.355 | 512 | 7.4 | 7.9 |
| jevzip K=64 | 307474 | 2.346 | 1024 | 5.2 | 5.7 |

**pydoc_data topics**: held-out, mostly English prose, 0.58 MB

| method | bytes | bits/byte | forward calls | enc KB/s | dec KB/s |
|---|---:|---:|---:|---:|---:|
| gzip -9 | 162425 | 2.247 | | | |
| bzip2 -9 | 116373 | 1.610 | | | |
| xz -9e | 111508 | 1.542 | | | |
| zstd -19 | 113580 | 1.571 | | | |
| brotli -q11 | 108258 | 1.497 | | | |
| jevzip K=8 | 193490 | 2.676 | 144 | 12.9 | 12.9 |
| jevzip K=16 | 182034 | 2.518 | 288 | 10.4 | 10.4 |
| jevzip K=32 | 178390 | 2.468 | 576 | 7.6 | 7.2 |
| jevzip K=64 | 178054 | 2.463 | 1152 | 4.7 | 5.0 |

**It does not beat xz, or anything else.** At this training budget it is
~1.55× larger than xz/brotli and ~100× slower. It is worse than every
baseline on the list, gzip included.

**Fewer rounds are nearly free.** These are model cross-entropy figures on
sampled networkx windows; the K=512 row is one byte per round, i.e. fully
sequential any-order decoding.

| K | 8 | 16 | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| bits/byte | 2.60 | 2.43 | 2.36 | 2.36 | 2.35 | 2.34 |

K=32 decodes in 16× fewer sequential rounds than K=512 and costs about 1%
more bits. K=16 costs about 4%. The prose test set shows the same shape:
2.41 bpb at K=512 and 2.43 at K=32.

**Where the bits go: the anchors.** Per-level cost at K=16 on networkx, in
bits per byte. Each row gives the share of the block in that level, the gap
to the nearest known neighbour, and the cost:

| reveal level | share of block | nearest known neighbour | cost |
|---|---:|---:|---:|
| stride ≥ 4 (anchors) | 25% | ≥ 4 bytes away | ~4.4 |
| stride 2 | 25% | 2 bytes away | ~3.0 |
| stride 1 | 50% | adjacent on both sides | ~1.0 |

The anchors are only 25% of the block but account for about half of all
bits. Filling a byte with a gap of four or more unknown bytes on each side is
hard for a small model; a left-to-right model never faces that task. With a
perfect model the chain rule makes every order cost the same, so this is a
learnability cost, not an information-theoretic one.

**Cross-block context barely matters, so full parallelism is cheap.** At
K=16, "no ctx" means every block is independent: the whole file's
dependency depth is 16 rounds.

| file | streams | dependency depth | bits/byte | enc / dec KB/s |
|---|---|---:|---:|---:|
| networkx | 2048 (no ctx) | 16 | 2.444 | 10.9 / 10.1 |
| networkx | 256 (default, 8-block chains) | 128 | 2.413 | 10.1 / 11.0 |
| networkx | 32 (64-block chains) | 1024 | 2.409 | 10.7 / 10.4 |
| pydoc | 1130 (no ctx) | 16 | 2.548 | 10.4 / 10.1 |
| pydoc | 141 (default, 8–9-block chains) | 144 | 2.518 | 10.4 / 10.4 |
| pydoc | 32 | 576 | 2.515 | 10.0 / 9.7 |

Dropping the 256 bytes of previous-block context costs about 1.3%. On this
CPU, speed is set by total compute, not by the number of sequential calls.
The low dependency depth would pay off on hardware that can run every block
at once.

**RoPE mattered most.** The same recipe with learned absolute positions only
reached 4.52 / 4.10 val bpb at steps 500 / 1000. With RoPE it reached
3.60 / 3.31.

AR_CONTROL_PLACEHOLDER

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
  torch build, the same CPU kernels and the same thread count. The header
  records the weights fingerprint, the thread count and the bf16 flag, so a
  wrong model is rejected up front. A different torch build or CPU kernel is
  not recorded: it is caught only after decoding, by the rANS end-state check
  plus the CRC. A file compressed on one machine isn't guaranteed to decode on
  a different CPU. This is the usual neural-compression problem (NNCP writes
  its own deterministic kernels).
- **Domain.** The model was trained for 2 hours on 4 CPU cores, on Python
  source installed on the build box. Expect it to do worse on anything that
  isn't Python or English. Incompressible input falls back to raw storage.

## Reproduce

```
pip install -r jevzip/requirements.txt
python -m jevzip.data --out data            # corpus from installed site-packages
python -m jevzip.train --data data --minutes 120
python -m jevzip.bench data/networkx.py.txt data/pydoc_topics.txt --verify
python -m unittest jevzip.test_jevzip       # round-trip tests (tiny random model)
```

Splits are whole packages, so near-duplicate files can't leak across them:
`mpmath` is validation, and `networkx` (code) and `pydoc_data` (mostly English
prose) are the benchmarks. Neither is in the training set.
