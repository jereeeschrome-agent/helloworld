# jevzip

A lossless byte compressor built on a Jev-style transformer. Jev is TypeSafe AI's
non-autoregressive model that scores a closed set of options in parallel.
jevzip decodes each 512-byte block in **K parallel rounds**: every round scores
all 256 byte values for many undecided bytes in one forward pass, instead of
one model step per byte.

```
python -m jevzip c  file.py  file.jz             # lanes model, K=128
python -m jevzip c  file.py  file.jz --rounds 32 # 4x fewer rounds: faster, larger
python -m jevzip d  file.jz  file.py
```

## Results

Every model below has the same 5.0M-parameter architecture and was trained
for 120 minutes on a 4-core CPU over a Python-source corpus. They differ only
in the **reveal order**, i.e. which bytes each round decides. All sizes are
real files from verified round-trips on held-out data: networkx (Python, 1.05
MB) and pydoc_data topics (mostly English prose, 0.58 MB). Figures are
bits per byte.

| method | rounds per block | networkx | pydoc | decode KB/s |
|---|---:|---:|---:|---:|
| gzip -9 | | 1.885 | 2.247 | |
| bzip2 -9 | | 1.558 | 1.610 | |
| zstd -19 | | 1.554 | 1.571 | |
| xz -9e | | 1.525 | 1.542 | |
| brotli -q11 | | 1.498 | 1.497 | |
| jevzip, linear (plain AR control) | 512 | **1.453** | 1.601 | 2.1 |
| jevzip, lanes | 512 | **1.485** | 1.628 | 2.2 |
| jevzip, lanes | 128 | 1.546 | 1.683 | 4.4 |
| jevzip, lanes | 64 | 1.613 | 1.745 | 6.9 |
| jevzip, lanes | 32 | 1.735 | 1.856 | 10.0 |
| jevzip, lanes | 16 | 1.956 | 2.068 | 12.8 |
| jevzip, bitrev | 32 | 2.355 | 2.468 | 10.3 |
| jevzip, bitrev | 16 | 2.413 | 2.518 | 11.8 |

What the table shows:

- **On held-out Python code, the sequential models beat xz -9e and brotli
  -q11.** The AR control is 4.7% smaller than xz and 3.0% smaller than brotli.
  The weights (~10 MB) are not counted; see Caveats.
- **On prose they don't.** The corpus is code, and the AR control is 4% behind
  xz there.
- **Parallel rounds trade size for depth.** Lanes at K=128 is within 1.4% of
  xz with 4× fewer sequential rounds than AR. K=32 cuts rounds 16× and is
  ~19% larger than AR.
- **The reveal order matters far more than the number of rounds.** The
  coarse-to-fine bitrev order was the original design. It loses badly at every
  K, because a small model struggles with far-apart "anchor" bytes; details
  below.

Decode speeds are from the 4-core CPU, which is compute-bound: going from 16
to 512 rounds costs only ~6× in speed here. On a GPU, where the per-round
batch runs in parallel, round count decides latency much more directly.

### A bigger lanes model: 11M parameters, 8 hours on an M4 Pro

The recipe was the same lanes order, with d_model 384, 6 layers, 6 heads,
512 bytes of context, and fp32 training on the Apple GPU (MPS) for 480 minutes:
16.3k steps of 32 windows, about 267M supervised bytes. Validation fell to
1.562 bpb at K=32, against 1.961 for the 5M model, and was flat over the last
~1,500 steps. Every row below is a verified round-trip, run on the M4 Pro's
CPU in fp32 with 8 threads.

networkx: the same 1.05 MB file as above, so the baselines are identical.

| method | rounds | bits/byte | vs xz -9e | vs brotli -q11 | decode KB/s |
|---|---:|---:|---:|---:|---:|
| xz -9e | | 1.525 | | | |
| brotli -q11 | | 1.498 | | | |
| jevzip lanes, 11M | 32 | **1.413** | −7.4% | −5.7% | 15.3 |
| jevzip lanes, 11M | 128 | **1.237** | −18.9% | −17.4% | 5.3 |

pydoc: that machine's Python ships a different 0.52 MB topics file, so these
are compared with their own baselines.

| method | rounds | bits/byte | vs xz -9e | vs brotli -q11 | decode KB/s |
|---|---:|---:|---:|---:|---:|
| xz -9e | | 1.582 | | | |
| brotli -q11 | | 1.530 | | | |
| jevzip lanes, 11M | 32 | **1.501** | −5.2% | −1.9% | 14.8 |
| jevzip lanes, 11M | 128 | **1.340** | −15.3% | −12.4% | 5.2 |

Model cross-entropy on networkx windows (`evalk`), by rounds K:

| K | 16 | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| bits/byte | 1.56 | 1.37 | 1.26 | 1.19 | 1.13 |

What changed with scale:

- **More compute closed the gap on prose.** The model now beats xz and brotli
  on prose too, even at K=32 with 16× fewer rounds than AR.
- **The 22 MB weights are still not counted.**
- **Real files come out ~3.5% larger than `evalk`'s estimates.** That's
  first-block-of-chain context loss plus coding overhead.

## How it works

A neural compressor needs a probability for each byte that the decoder can
reproduce exactly. jevzip builds those probabilities from four pieces.

1. **A reveal order and its cut into K groups** (`schedule.py`). Every byte in
   a group is predicted in parallel from the bytes revealed by earlier groups.
   - `lanes` (default): the block is split into 512/K contiguous lanes, each
     decoded left to right, and all lanes advance one byte per round. Bytes
     keep their left neighbour, except the first byte of each lane after lane
     0. K=512 is plain autoregression.
   - `bitrev`: coarse-to-fine bit-reversal order (0, 256, 128, 384, ...), cut
     so that the revealed count grows geometrically. Early rounds reveal sparse
     anchors; late rounds fill gaps with known bytes on both sides.
   - `linear`: left to right. With K=512 this is the AR control.
2. **Two streams with shared weights**, XLNet-style (`model.py`):
   - the *content stream* holds the known bytes (256 bytes of previous-block
     context, then the revealed bytes in reveal order) under causal attention.
     It is KV-cached and extended one group per round;
   - the *slot stream* has one query per undecided byte. A slot attends to the
     revealed content and to itself, never to other slots. So a whole group is
     scored in one pass, and a slot's prediction doesn't depend on which other
     slots are asked, which keeps the decoder reproducible.

   Attention uses RoPE on the original byte offsets, so "1, 2, 4 bytes to my
   left" is directly visible however the order scrambles the sequence.
3. **Option scoring.** Each slot is scored against tied embeddings of the 256
   byte values: a typed choice from a closed label set, the Jev idea.
4. **rANS coding** (`rans.py`, 64-bit state, 24-bit probabilities). The encoder
   replays the decoder's exact sequence of forward calls, so both sides see
   bit-identical probabilities. The header carries a model+config fingerprint,
   the thread count, the bf16 flag and a CRC. The decoder also checks the rANS
   end state.

Blocks are dealt into `streams` independent chains that decode in lockstep, and
each block uses the end of its chain predecessor as context. The dependency
depth is `K × ceil(blocks / streams)`. With `streams = blocks` that is K for the
whole file, at ~1.3% cost (see below). On CPU, each step runs in batches of at
most 128 blocks, and the bench counts the resulting forward calls.

Training (`train.py`) supervises every byte of a window in one forward pass,
at exactly the visibility it will have at decode time. K is drawn per window,
log-uniform over a range: 8..512 for lanes, 8..128 for bitrev. One model then
serves any `--rounds` in that range.

## Why the order matters

**Learning speed at equal compute.** These are validation bpb on held-out
`mpmath`, each model at its own operating K:

| step | 500 | 1000 | 2000 | 3000 | final |
|---|---:|---:|---:|---:|---:|
| linear / AR (K=512) | 3.10 | 2.12 | 1.81 | 1.69 | 1.687 |
| lanes (K=32) | 3.32 | 2.41 | 2.14 | 2.01 | 1.961 |
| bitrev (K=16) | 3.60 | 3.31 | 3.03 | 2.83 | 2.804 |
| bitrev, learned absolute positions instead of RoPE | 4.52 | 4.10 | | | |

**Where bitrev's bits go.** These are per-level costs at K=16 on networkx, in
bits per byte:

| reveal level | share of block | nearest known neighbour | cost |
|---|---:|---:|---:|
| stride ≥ 4 (anchors) | 25% | ≥ 4 bytes away | ~4.4 |
| stride 2 | 25% | 2 bytes away | ~3.0 |
| stride 1 | 50% | adjacent on both sides | ~1.0 |

The anchors are a quarter of the block and about half of all bits. With a
perfect model the chain rule makes every order cost the same, so this is a
learnability gap, not an information-theoretic one. But a small model can't
close it, which is why the bitrev design loses.

**Where lanes' bits go.** At K=16 (32 lanes of 16 bytes), the first byte of
each lane costs ~4.5 bits. That falls to ~2.2 by the fourth byte and to ~1.1–1.5
deep in the lane. Longer lanes (larger K) spread the lane-start cost over more
bytes.

**Bits/byte vs K** (model cross-entropy on sampled networkx windows,
`python -m jevzip.evalk`):

| K | 8 | 16 | 32 | 64 | 128 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| lanes | 2.32 | 1.93 | 1.71 | 1.59 | 1.52 | 1.47 |
| bitrev | 2.60 | 2.43 | 2.36 | 2.36 | 2.35 | 2.34 |
| linear / AR | | | | | | 1.43 |

**Context vs parallelism** (bitrev, K=16; "no ctx" means every block is
independent):

| file | streams | dependency depth | bits/byte |
|---|---|---:|---:|
| networkx | 2048 (no ctx) | 16 | 2.444 |
| networkx | 256 (default) | 128 | 2.413 |
| networkx | 32 | 1024 | 2.409 |
| pydoc | 1130 (no ctx) | 16 | 2.548 |
| pydoc | 141 (default) | 144 | 2.518 |

## Weights

| file | order | trained K | notes |
|---|---|---|---|
| `weights/jevzip-lanes.pt` | lanes | 8..512 | default; one model from K=8 to plain AR |
| `weights/jevzip-ar.pt` | linear | 512 | AR control; use `--rounds 512` |
| `weights/jevzip-bitrev.pt` | bitrev | 8..128 | the original coarse-to-fine design |

## Caveats

- **Weights aren't counted in the sizes.** That's the usual convention for
  pretrained compressors (like brotli's built-in dictionary), but each model
  file is ~10 MB. On a single 1 MB file, the model costs far more than it
  saves. The win only matters when one model serves lots of similar data.
- **Not new from scratch.** Each piece has prior art:
  - two-stream attention (XLNet);
  - any-order AR models (σ-GPT);
  - fixed-schedule parallel context models from learned image compression
    (checkerboard, M2T);
  - neural compressors (NNCP, "Language Modeling Is Compression", ts_zip).

  As far as I know, the combination is new: KV-cached two-stream decoding,
  lane / bit-reversal schedules served by one model, and lockstep chains.
  Treat that claim with suspicion.
- **Bitwise reproducibility.** Decoding needs the same weights, torch build,
  CPU kernels and thread count. The header rejects a wrong model or thread
  count up front. A different torch build or CPU is caught only after
  decoding, by the end-state check and the CRC. Files aren't portable across
  CPUs; that's the usual neural-compression problem.
- **Domain.** The models were trained on Python source installed on the build
  box. Expect worse results on other data. Incompressible input falls back to
  raw storage.

## Reproduce

```
pip install -r jevzip/requirements.txt
python -m jevzip.data --out data            # corpus from installed site-packages
python -m jevzip.train --data data --order lanes --kmin 8 --kmax 512 --val-rounds 32 --minutes 120 --out jevzip/weights/jevzip-lanes.pt
python -m jevzip.train --data data --order linear --kmin 512 --kmax 512 --val-rounds 512 --minutes 120 --out jevzip/weights/jevzip-ar.pt
python -m jevzip.train --data data --order bitrev --minutes 120 --out jevzip/weights/jevzip-bitrev.pt
python -m jevzip.bench data/networkx.py.txt data/pydoc_topics.txt --rounds 32 64 128 512 --verify
python -m jevzip.evalk data/networkx.py.txt --weights jevzip/weights/jevzip-lanes.pt
python -m unittest jevzip.test_jevzip       # round-trip tests (tiny random models)
```

Splits are whole packages, so near-duplicate files can't leak across them:
`mpmath` is validation, and `networkx` and `pydoc_data` are the benchmarks.
None of them is in the training set.

### Bigger models on a Mac (Apple-silicon GPU) or CUDA box

Training takes `--device mps` or `--device cuda` plus model-size flags.
Compression always runs on the CPU, so weights trained on a GPU work anywhere.
A compressed file decodes only on the same CPU and torch build that encoded it.

```
python3 -m venv .venv && . .venv/bin/activate
pip install torch numpy networkx mpmath sympy pandas   # corpus + the held-out test packages
python -m jevzip.data --out data
python -m jevzip.train --data data --device mps --order lanes --kmin 8 --kmax 512 --val-rounds 32 \
    --d-model 384 --n-layers 6 --n-heads 6 --ctx 512 --bs 32 --lr 1e-3 --warmup 500 --minutes 480 --no-amp \
    --out jevzip/weights/jevzip-lanes-big.pt
python -m jevzip.bench data/networkx.py.txt --weights jevzip/weights/jevzip-lanes-big.pt --rounds 32 128 --verify --fp32
```

On an M4 Pro (64 GB), fp32 was about 10% faster than bf16. The masked slot
attention makes MPS build full attention matrices, so 26M- and 86M-parameter
models were too slow for an 8-hour budget. A fused or block-sparse slot
attention is the obvious next step for scaling.
