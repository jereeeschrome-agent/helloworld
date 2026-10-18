# Baseline compressors

Sizes of standard and strong context-mixing compressors on the jevzip test files, plus the
"pretrained" variants that, like jevzip, learn from the training corpus. Machine: Mac mini M4 Pro,
macOS 27, shared with a GPU training job, so **times are provisional; sizes are what compare**.
Every row comes from `jevzip/baselines.py`; the exact command, tool version and input SHA-256 of
every row are in `networkx.json`, `pydoc_topics.json`, `enwik8-test.json` (and the raw per-job
files in `parts/`). How the jobs were run: `logs/run_cmix.sh`, `logs/run_paq8px.sh`, and

    python -m jevzip.baselines data/code/networkx.py.txt data/code/pydoc_topics.txt --group fast --corpus data/code/train.bin --json results/baselines/parts/fast-code.json
    python -m jevzip.baselines data/enwik8/test.bin --group fast --corpus data/enwik8/train.bin --json results/baselines/parts/fast-enwik8-test.json
    results/baselines/logs/report.sh      # merges parts/ into the JSON files and this page

## Kinds of rows

- **standalone**: a self-contained file, decoded and compared (verified = yes).
- **dictionary**: the decoder also needs a reference built from the training corpus; its size is
  in *ref bytes* and plays the part of jevzip's weights in total-size accounting.
  zstd dictionaries: `zstd --train -B4096 --maxdict=N` on a fixed 32 MiB sample (256 evenly spaced
  runs) of the corpus. `--patch-from`: the whole corpus as the reference, `--long=31` and
  `--memory` on both sides. brotli: the CLI's raw dictionary (`-D`) works, but the dictionary must
  fit the 16 MiB window together with the input, so it is the corpus tail that fits.
- **marginal**: |C(P + x)| - |C(P)|, P = the last 64 MiB of the corpus for xz (LZMA2 dictionary
  >= |P| + |x|) and the last 16 MiB for paq8px. Marginal (equivalent to a preset-dictionary coder
  up to a few bytes; not a standalone file), so not round-trip verified.

## Tools and settings

- python-native: gzip -9, bzip2 -9, xz preset 9|EXTREME, zstandard level 19, brotli q11 lgwin 24.
- CLIs: zstd 1.5.7 (`--ultra -22 --long=27`), brotli 1.2.0 (`--large_window=30` exists in this
  build; it changes nothing at these sizes: +2 bytes).
- paq8px v217 (commit c84f576), built by `tools/build_competitors.sh` (cmake Release, Apple clang).
  Standalone rows use **-8L** (level 8, 2.4 GB, + LSTM). On 100 KB of networkx: -8 14512 B in 16 s,
  -8L 14415 B in 52 s, -12L 14415 B in 72 s and 7.6 GB, so -12 buys nothing at these sizes.
  Primed rows use **-8** (no LSTM): coding 16-21 MiB at -8L (~2 KB/s) would take hours per run.
- cmix v21 (commit 194af9c), built with Apple clang `-Ofast -mcpu=native` (its makefile wants
  clang++-17 `-march=native`). Default mode with its `english.dic` (465 KB, not counted, as is usual).
  It needs ~19 GB RAM and spends ~4.5 min pretraining on that dictionary before every run, compress
  or decompress; it then codes ~1 KB/s here. Measured: 30 KB in 274 s, 100 KB in 279 s, 1 MiB in ~21 min.
  Projected for enwik8 test (5 MB): ~1.5 h each way, under the 4 h limit, so it was run.
- **7-Zip PPMd (7zz): not measured.** `brew install sevenzip` failed: the Homebrew prefix is owned
  by another macOS user and is not writable by this one. Its rows are recorded as skipped; rerun
  `tools/build_competitors.sh` and the `--only '7zz.*'` jobs once 7zz is installed.

## Observations

- zstd -19 with the 110 KiB dictionary is *worse* than zstd -19 without one on these large inputs
  (enwik8 test: 1512313 vs 1394418 B; networkx: 208599 vs 203658 B), round trip verified. Changing
  the window (`--zstd=wlog=23`) did not change it. A dictionary trained on 4 KB samples is meant for
  small inputs; the 1 MiB one helps a little (1365284 B on enwik8 test).
- The best "pretrained" fast coders gain only ~6-14% over their standalone versions: corpus
  knowledge that an LZ coder can use is mostly long repeats.
- jevzip smoke test (not a measurement row): `python -m jevzip.bench <copy of runs/ref/networkx-64k.bin>
  --weights weights/jevzip-lanes-11m-code.pt --rounds 32 --verify --threads 4` (fp32) gave 10334 B
  for the first 64 KiB of networkx, the same size as `runs/ref/networkx-64k-k32-fp32-t4.jz`, and
  decoded correctly; brotli -q11 gave 13612 B on the same 64 KiB.

Published reference numbers (LTCB, NNCP, ts_zip, ...) are in `results/published/sota.md`.
