# Where real sizes exceed the model's cross-entropy

Model `weights/jevzip-lanes-11m-code.pt` (sha256 548d8879…), CPU, fp32, threads 4, torch 2.14.1.
Files: `data/code/networkx.py.txt` (1,048,576 B, 2048 blocks) and `data/code/pydoc_topics.txt` (519,592 B, 1015 blocks).
Everything comes from `jevzip/codelength.py` with `encode=True`, which codes exactly what `compress()` codes.
`compress()` itself reproduces all `runs/ref/*.jz` byte for byte (fp32 K=32 and K=128, bf16 K=32).
Raw numbers are in `runs.json`, derived ones in `gap.json`. Reproduce with `nice -n 10 .venv/bin/python results/gap/gap.py run`, then `... gap.py report`.
"Oracle CE" means every block sees its true preceding 512 bytes. No decoder can have that, so it is the floor.

## The observed ~3.5% gap (networkx, old default of 256 chains)

The table walks from the old `evalk` estimate to the real size. Values are in bits/byte, and the steps add up exactly.

| step | K=32 | K=128 |
|---|---:|---:|
| old evalk (64 windows, bf16) | 1.36596 | 1.18675 |
| bf16 → fp32 in evalk | −0.00005 (−0.00%) | −0.00008 (−0.01%) |
| 64-window sample → full file (oracle CE) | +0.02062 (+1.51%) | +0.02460 (+2.07%) |
| chain-start blocks (256 chains, no context) | +0.02582 (+1.89%) | +0.02520 (+2.12%) |
| quantization (2^24 CDF) | −0.00003 (−0.00%) | −0.00004 (−0.00%) |
| header (34 B) + rANS flush | +0.00030 (+0.02%) | +0.00032 (+0.03%) |
| **real size** | **1.41261** (185,154 B) | **1.23674** (162,102 B) |
| total | +3.42% | +4.21% |

- **Sampling explains about half the gap.** It is noise, not bias. Bootstrapping 64 blocks from the per-block oracle costs gives a std of 2.9% (K=32) and 3.3% (K=128) around the full-file value. The old estimate sat 0.5σ and 0.6σ low. On a heterogeneous 1 MB file, a 64-window sample cannot resolve 0.5%.
- **Chain starts are the other half.** By default, 1 block in 8 was coded with PAD as its context. Each extra chain start costs about 13.6 bytes at K=32.
- **bf16, quantization and rANS are negligible.** Quantization is slightly *negative*: the argmax symbol receives the rounding remainder, and it is usually the actual byte.

## The coding overhead itself, at each default (real size vs oracle CE)

| file | K | policy | chains | real B | real bpb | oracle bpb | gap | hdr+rANS | quant | chain starts |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| networkx | 32 | old (blocks//8) | 256 | 185,154 | 1.4126 | 1.3865 | 1.88% | 0.022% | −0.003% | 1.86% |
| networkx | 32 | **new** | 128 | 183,502 | 1.4000 | 1.3865 | **0.97%** | 0.022% | −0.002% | 0.95% |
| networkx | 128 | old | 256 | 162,102 | 1.2367 | 1.2113 | 2.10% | 0.026% | −0.003% | 2.08% |
| networkx | 128 | **new** | 128 | 160,494 | 1.2245 | 1.2113 | **1.09%** | 0.025% | −0.002% | 1.07% |
| pydoc_topics | 32 | old | 126 | 97,474 | 1.5008 | 1.4803 | 1.39% | 0.043% | −0.017% | 1.36% |
| pydoc_topics | 32 | **new** | 113 | 97,454 | 1.5005 | 1.4803 | **1.37%** | 0.040% | −0.017% | 1.34% |
| pydoc_topics | 128 | old | 126 | 87,062 | 1.3405 | 1.3209 | 1.48% | 0.046% | −0.023% | 1.46% |
| pydoc_topics | 128 | **new** | 113 | 87,014 | 1.3397 | 1.3209 | **1.43%** | 0.046% | −0.024% | 1.40% |

Each "gap" column is the sum of the three columns to its right.

## Size vs streams vs forward calls (networkx)

| K | chains | steps | forward calls | real B | real bpb | over oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1 | 2048 | 65,536 | 181,770 | 1.3868 | 0.02% |
| 32 | 16 | 128 | 4,096 | 181,954 | 1.3882 | 0.12% |
| 32 | 64 | 32 | 1,024 | 182,630 | 1.3934 | 0.49% |
| 32 | **128 (new default)** | 16 | **512** | 183,502 | 1.4000 | 0.97% |
| 32 | 256 (old default) | 8 | 512 | 185,154 | 1.4126 | 1.88% |
| 32 | 2048 (no context) | 1 | 512 | 209,854 | 1.6011 | 15.5% |
| 128 | 128 (new default) | 16 | 2,048 | 160,494 | 1.2245 | 1.09% |
| 128 | 256 (old default) | 8 | 2,048 | 162,102 | 1.2367 | 2.10% |
| 128 | 2048 (no context) | 1 | 2,048 | 186,166 | 1.4203 | 17.3% |

Forward calls are the sequential `model.step` calls made by each of `compress` and `decompress`: rounds × batches, with at most CHUNK=128 blocks per batch. Once a file has more than CHUNK blocks, every policy with at least CHUNK chains makes the same number of calls (512 at K=32). The loss falls roughly in proportion to the number of chains. Wall-clock times are in `gap.json`. They are provisional because the machine was shared.

## Fixes

1. **`jevzip/codelength.py`** is a new tool. It gives the exact codelength of `compress()`'s schedule: the same Plan, batches, reveal plan and quantize, sharing `_run_batch` with the codec. It reports quantized bits, float bits, `--oracle-context`, `--encode` (the exact `compress()` size), `--per-byte` .npy files, `--streams`/`--rounds` sweeps and `--json`. `cross_entropy()` gives every block its true context in one pass. Its result was bit-identical to the oracle and to `streams=1` here (1,453,875.22 bits at K=32), at 512 calls instead of 65,536.
2. **`jevzip/evalk.py`** is now a thin wrapper over `cross_entropy()`. It computes the exact full-file estimate, fp32 by default, with `--bf16` as an opt-in. The old flags still work. `--windows N` now samples N blocks for a quick estimate on big files and defaults to the whole file. I kept the module rather than removing it because the README and users call `python -m jevzip.evalk`.
3. **The default streams for `compress()`** is now `default_streams(blocks)`: the fewest chains that make no more forward calls than the old `blocks // 8`. The allowed number of steps is D = max(the old step count, ceil(blocks / CHUNK)), and the chain count is ceil(blocks / D). That is never more than CHUNK chains, and every step is one call. A unit test checks that it is never more calls and never more chains for every block count from 1 to 2099 (plus 5k and 20k). The header stores the chain count, so old files still decode. The reference files (128 blocks, 16 chains) are unchanged.

## Where < 0.5% is not reachable, and why

- **Without more forward calls, < 0.5% is not reachable for these files.** The loss is about 13.6 bytes per chain start. With ≤ 0.5% as the budget, networkx (2048 blocks, ~1.39 bpb) can afford about 64 chains. That is 32 steps, so 2× the calls of the old default (row `64`: 0.49%).
  - The fewest calls any cut allows is ceil(2048/128) = 16 per round, which forces ≥ 128 chains and therefore ~1%.
  - pydoc_topics (1015 blocks) keeps about 1 chain in 9, because its old call budget is 9 steps (+1 call for the short last block). That gives ~1.4%.
- **Small files are worse.** Below 8 × CHUNK blocks, the old default's ~8 steps per round is the call budget, so about 1/8 of the blocks start chains. A 64 KB file has 16 of 128 blocks.
- **Large files improve.** Above 1024 blocks, the chain count stays ≤ 128 while the block count grows, so the chain-start share falls as 128 / blocks. Estimated from the per-start cost here (not measured): about 0.5% at ~4k blocks (2 MB), ~0.2% for the 5 MB enwik8 test split, ~0.01% at 100 MB.
- **To get < 0.5% on 1 MB files now, pass `--streams 64`.** That gives 0.49% for 2× the calls; `--streams 16` gives 0.12% for 8×. A future format could also give chain starts a cheap context, for example a short prefix coded first. Neither option is the default, because the brief caps forward calls.
- **The remaining non-chain overhead is ≤ 0.05% everywhere:** a 34-byte header plus about 6–7 bytes of rANS flush (this has to be the case: every encode had 0–12 bytes of slack; the networkx-64k refs had 6.3–6.9 B).
