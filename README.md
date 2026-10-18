# jevzip

[![tests](https://github.com/jereeeschrome-agent/jevzip/actions/workflows/tests.yml/badge.svg)](https://github.com/jereeeschrome-agent/jevzip/actions/workflows/tests.yml)
[License: MIT](LICENSE)

**Research toy.** A `.jz` file decodes only with the same torch build, CPU kernels and fp32/bf16
setting that encoded it ([portability](results/portability/portability.md)). Don't archive
anything with it.

jevzip is a lossless byte compressor built on an 11M-parameter transformer. It decodes each
512-byte block in **K parallel rounds** instead of one model step per byte (K=512 is plain
autoregression).

**Headline.** On enwik8's standard test split (the last 5 MB, model trained on the first 90 MB),
jevzip-11M at K=128 (4× fewer sequential rounds than AR) codes **1.332 bits/byte**. That is 39%
smaller than `xz -9e` and 29% smaller than xz primed with 64 MB of the training text. It is also
2–3% **behind** cmix and primed paq8px, and 68% behind a 0.5B-parameter LLM. Its 22 MB of weights
only pay off once one model serves **more than ~210 MB** of similar text.

| enwik8 test split (5 MB) | bits/byte | extra bytes the decoder needs |
|---|---:|---:|
| gzip -9 | 2.818 | 0 |
| zstd --ultra -22 --long=27 | 2.231 | 0 |
| xz -9e | 2.171 | 0 |
| brotli -q11 | 2.150 | 0 |
| zstd --ultra -22 --long=31, whole training set as `--patch-from` dictionary | 1.896 | 90 MB |
| xz -9e primed with 64 MB of training text (marginal size) | 1.876 | 64 MB |
| **jevzip-11M, K=32** (16× fewer rounds than AR) | **1.488** | 22 MB |
| paq8px -8L | 1.413 | 0 |
| **jevzip-11M, K=128** (4× fewer rounds) | **1.332** | 22 MB |
| cmix (2026-10, 194af9c) | 1.306 | 0 |
| paq8px -8 primed with 16 MB of training text (marginal size) | 1.289 | 16 MB |
| **jevzip-11M, K=512** (AR order, same model) | **1.287** | 22 MB |
| jevzip-11M **AR control** (same size and steps, K=512) | 1.249 | 22 MB |
| RWKV-4-169M, LLM codelength, state carried (ts_zip's model family) | 1.097 | 339 MB |
| Qwen2.5-0.5B, LLM codelength, 2048-token window | 0.794 | 989 MB |

All jevzip rows are real files from verified round trips (fp32, CPU, 6 threads). LLM rows are
ideal codelengths (an arithmetic coder adds < 0.01%); both LLMs very likely saw Wikipedia during
pretraining. Sources and full tables: [baselines](results/baselines/tables.md),
[LLM](results/llm/summary.md), [published SOTA](results/published/sota.md).

## Reproduce the headline

```bash
pip install -e ".[bench]"
python -m jevzip.fetch jevzip-lanes-11m-enwik8.pt && python -m jevzip.data enwik8
python -m jevzip.bench data/enwik8/test.bin --rounds 32 128 512 --verify
```

Weights are GitHub release assets (`v0.2.0`), checked against SHA-256 pins in
[`jevzip/assets.json`](jevzip/assets.json). The data command downloads enwik8 and checks it against
[`jevzip/datasets.json`](jevzip/datasets.json). Unit tests:
`python -m unittest discover -s jevzip -p "test_*.py" -t .`

## How it compares to the state of the art

- **Context mixing wins at this size.** cmix and paq8px use no pretrained model and match or beat
  jevzip-11M at K≥128. On held-out Python code they win by more: networkx is 0.876 bpb with cmix
  and 0.920 with paq8px -8L, against 1.225 for jevzip's code model at K=128.
- **LLMs win by a lot**, at 15–45× the model bytes. Qwen2.5-0.5B codes the same 5 MB at 0.794 bpb.
- **The whole-file benchmarks are another protocol.** LTCB and the Hutter Prize compress all of
  enwik8/enwik9 and count the decompressor. On whole enwik8 (100 MB) cmix v21 gets 1.170 bpb and
  nncp v3.2 1.193. The 2026 leaders are small transformers pretrained offline on enwik9, with their
  weights inside the archive: about 0.77 bpb on enwik9 including everything
  ([published numbers](results/published/sota.md)). We didn't run enwik9.

## Total size: compressed bytes + model

The model is 22.3 MB (bf16). Extrapolating each method's bits/byte linearly from the enwik8 test
split, total bytes (compressed data + model or dictionary) are:

| data size | xz -9e | zstd + 90 MB dictionary | paq8px -8L | cmix | jevzip K=128 + 22.3 MB |
|---:|---:|---:|---:|---:|---:|
| 1 MB | 0.27 MB | 90.2 MB | 0.18 MB | 0.16 MB | 22.5 MB |
| 100 MB | 27.1 MB | 113.7 MB | 17.7 MB | 16.3 MB | 39.0 MB |
| 1 GB | 271 MB | 327 MB | 177 MB | 163 MB | 189 MB |
| 10 GB | 2.71 GB | 2.46 GB | 1.77 GB | 1.63 GB | 1.69 GB |

jevzip K=128 crosses xz at **~213 MB** and paq8px -8L at **~2.2 GB**. It never crosses cmix,
which already codes fewer bits per byte. At K=512 (1.287) the crossing with cmix is ~9.6 GB.
Against zstd with the training set as its dictionary, jevzip is smaller at every size, because
that dictionary (90 MB) is larger than the model.

## Is the parallelism worth it? GPU decode latency

The pitch is fewer sequential rounds. Decode wall-clock on the M4 Pro GPU (MPS, fp32) for 64 KB
(128 blocks in one batch, so the dependency depth is K), including the per-round copy of the
probabilities to the host and the CPU entropy decoder
([latency](results/latency/latency.md)):

| K | sequential rounds | decode ms | vs AR (K=512) | bits/byte (this 64 KB slice, no cross-block context) |
|---:|---:|---:|---:|---:|
| 16 | 16 | 2,008 | 6.9× faster | 1.672 |
| 32 | 32 | 2,430 | 5.7× faster | 1.473 |
| 128 | 128 | 5,229 | 2.6× faster | 1.287 |
| 512 (AR, KV cache) | 512 | 13,784 | 1.0× | 1.225 |

32× fewer rounds buys 7× lower latency, not 32×. A round costs at least ~6 ms on MPS, because it
launches ~520 small kernels and waits for its probabilities, and from 8 blocks up the arithmetic
grows too. The GPU also decodes bit-exactly against a GPU encode (all K, all runs). On the CPU the
same file takes 7.0 s at K=16 and 28.2 s at K=512 (4×).

## The AR control: what parallel rounds cost

Same 11M architecture, same data, same number of optimizer steps (15,725), real round trips on the
enwik8 test split:

| model | K | bits/byte | vs AR control |
|---|---:|---:|---:|
| AR control (linear order, trained only at K=512) | 512 | 1.249 | |
| lanes (trained for K=8..512) | 512 | 1.287 | +3.0% |
| lanes | 128 | 1.332 | +6.6% |
| lanes | 32 | 1.488 | +19.1% |

Serving every K from one model costs 3% even at K=512. 4× fewer rounds costs 6.6%, and 16× fewer
costs 19%.

## Scaling and variance

Validation bpb (enwik8, K=32, sampled windows) after a fixed 24 minutes of MPS training each:

| params | 0.96M | 2.92M | 5.07M (3 seeds) | 11.1M | 25.9M |
|---|---:|---:|---:|---:|---:|
| steps | 2,500 | 1,120 | 1,215–1,378 | 742 | 413 |
| val bpb | 2.478 | 2.579 | 2.349 / 2.407 / 2.411 | 2.499 | 2.759 |

At 24 minutes every size is far from converged: the 11M model reaches 1.525 after 8 h. These
points mostly show that small models win at tiny budgets, and that the seed-to-seed spread is
about ±1.3%. **This is not a scaling law.** The only long runs are two 11M models at about 8 h,
and a real curve needs several sizes at several budgets, which this GPU budget didn't allow.

Out-of-domain data and per-file distributions are prepared ([datasets](results/datasets/README.md)):
text8, Silesia, JSON, logs, C++ headers, Polish text and binaries. They have baseline compressor
numbers but **no jevzip runs yet**.

## Coding overhead (real file vs model cross-entropy)

The old ~3.5% gap between file sizes and `evalk` was not coding overhead
([gap analysis](results/gap/gap.md)). On networkx at K=32 it broke down as:

- 1.5%: `evalk` sampled 64 windows, with a 2–3% standard deviation;
- 1.9%: the first block of every chain has no previous-block context;
- 0.02%: header and rANS overhead; quantization was negligible.

`evalk` is now exact over the whole file. The default `streams` is now `min(blocks, 128)`, which
keeps every CPU forward call full and halves the chain-start cost. Files now land within ~1% of
full-context cross-entropy, and within 0.5% only with 2× more forward calls.

## How it works

1. **Lanes schedule** (`schedule.py`): each block is split into 512/K contiguous lanes, each
   decoded left to right, all advancing one byte per round. Only lane starts lose their left
   neighbour.
2. **Two-stream transformer with shared weights** (`model.py`, XLNet-style): a KV-cached content
   stream holds the known bytes; one slot query per undecided byte attends to the content and to
   itself only, so a whole round is scored in one pass. RoPE on byte offsets. Each slot scores the
   256 byte values against tied embeddings.
3. **rANS** (`rans.py`, 24-bit probabilities). The encoder replays the decoder's exact forward
   calls. The header carries a model fingerprint, the thread count, the bf16 flag and a CRC.
4. **Chains**: blocks are dealt into `streams` chains that decode in lockstep, and each block sees
   its predecessor's last 512 bytes.

Training (`train.py`) supervises every byte of a window in one pass at its decode-time visibility,
with K drawn log-uniform from 8..512, so one model serves every K.

## Weights

| file | params | trained on | recipe |
|---|---:|---|---|
| `jevzip-lanes-11m-enwik8.pt` (default) | 11.1M | enwik8 first 90 MB | lanes, K=8..512, 15,725 steps × 32 windows, M4 Pro MPS, ~8 h |
| `jevzip-ar-11m-enwik8.pt` | 11.1M | enwik8 first 90 MB | linear order, K=512, same steps |
| `jevzip-lanes-11m-code.pt` | 11.1M | Python sources (site-packages) | lanes, K=8..512, 16.3k steps, ~8 h |

The code model's corpus depends on the packages installed on the build machine, so it can't be
rebuilt bit-identically elsewhere. Its held-out test files ship as the `jevzip-code-testset.tar.xz`
release asset.

## Caveats

- **Prior art.** Two-stream attention (XLNet), any-order AR (σ-GPT), fixed-schedule parallel
  context models (checkerboard, M2T), and neural compressors (NNCP, ts_zip, "Language Modeling Is
  Compression") all exist. The combination here may be new; treat that claim with suspicion.
- **Speed.** CPU encode and decode run at 2–10 KB/s.
- **Portability.** Decoding fails within the first 4 bytes after a torch upgrade (2.14.1 → 2.8.0),
  on MPS against a CPU encode, or across fp32/bf16. Thread count doesn't matter on this Mac.
  Coarser probability quantization doesn't fix it; integer-only inference would. Across machines,
  fixtures encoded on this M4 Pro (torch 2.14.1) decode 4/5 on GitHub's macOS arm64 runner with the
  same torch and 0/5 on its Linux x86_64 runner. CI reruns that check on every push and reports it
  as a warning.

## Site

`index.html` is a static page about the project (`npm install && npm run dev`); `pages.yml`
deploys it from `main`.
