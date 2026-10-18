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

_Generated 2026-10-18 16:43 EEST by results/baselines/logs/report.sh. Runs not finished at that time (their rows are missing or partial): none._

### data/enwik8/test.bin (5000000 bytes, sha256 86f62c0adeb6c277...)

training corpus: data/enwik8/train.bin (90000000 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 805295 | 1.2885 | 16777216 | n/a | 1715.1 |  |
| cmix | standalone | 815915 | 1.3055 |  | yes | 2621.1 | 2435.9 |
| paq8px -8L | standalone | 882938 | 1.4127 |  | yes | 2373.2 | 1513.5 |
| xz -9e primed | marginal | 1172432 | 1.8759 | 67108864 | n/a | 96.6 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 1185225 | 1.8964 | 90000000 | yes | 236.7 | 0.0 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 1294732 | 2.0716 | 11711664 | yes | 12.9 | 0.0 |
| brotli -q11 | standalone | 1343830 | 2.1501 |  | yes | 9.3 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 1343832 | 2.1501 |  | yes | 9.7 | 0.0 |
| xz -9e | standalone | 1356736 | 2.1708 |  | yes | 3.5 | 0.1 |
| zstd -19 + 1 MiB dict | dictionary | 1365284 | 2.1845 | 1048576 | yes | 4.0 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 1394409 | 2.2311 |  | yes | 3.5 | 0.0 |
| zstd -19 | standalone | 1394418 | 2.2311 |  | yes | 3.1 | 0.0 |
| bzip2 -9 | standalone | 1408792 | 2.2541 |  | yes | 0.3 | 0.3 |
| zstd -19 + 110 KiB dict | dictionary | 1512313 | 2.4197 | 112640 | yes | 1.1 | 0.0 |
| gzip -9 | standalone | 1761271 | 2.8180 |  | yes | 0.2 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |

### data/code/networkx.py.txt (1048576 bytes, sha256 97ac83b301c24bde...)

training corpus: data/code/train.bin (147802575 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 114525 | 0.8738 | 16777216 | n/a | 1475.1 |  |
| cmix | standalone | 114855 | 0.8763 |  | yes | 1234.6 | 1191.9 |
| paq8px -8L | standalone | 120654 | 0.9205 |  | yes | 495.9 | 489.5 |
| xz -9e primed | marginal | 187528 | 1.4307 | 67108864 | n/a | 52.9 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 190267 | 1.4516 | 147802575 | yes | 218.3 | 0.1 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 194907 | 1.4870 | 15663088 | yes | 2.3 | 0.0 |
| zstd -19 + 1 MiB dict | dictionary | 196156 | 1.4966 | 1048576 | yes | 0.8 | 0.0 |
| brotli -q11 | standalone | 196333 | 1.4979 |  | yes | 1.4 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 196335 | 1.4979 |  | yes | 1.4 | 0.0 |
| xz -9e | standalone | 199944 | 1.5255 |  | yes | 0.4 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 203613 | 1.5534 |  | yes | 0.4 | 0.0 |
| zstd -19 | standalone | 203658 | 1.5538 |  | yes | 0.4 | 0.0 |
| bzip2 -9 | standalone | 204202 | 1.5579 |  | yes | 0.1 | 0.1 |
| zstd -19 + 110 KiB dict | dictionary | 208599 | 1.5915 | 112640 | yes | 0.2 | 0.0 |
| gzip -9 | standalone | 247078 | 1.8851 |  | yes | 0.1 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |

### data/code/pydoc_topics.txt (519592 bytes, sha256 5b5a37bce813f6c1...)

training corpus: data/code/train.bin (147802575 bytes)

| method | kind | bytes | bits/byte | ref bytes | verified | enc s | dec s |
|---|---|---:|---:|---:|---|---:|---:|
| paq8px -8 primed | marginal | 58025 | 0.8934 | 16777216 | n/a | 1264.3 |  |
| cmix | standalone | 62315 | 0.9594 |  | yes | 497.0 | 583.1 |
| paq8px -8L | standalone | 64533 | 0.9936 |  | yes | 235.8 | 239.0 |
| xz -9e primed | marginal | 92100 | 1.4180 | 67108864 | n/a | 51.9 |  |
| zstd --ultra -22 --long=31 --patch-from=corpus | dictionary | 93440 | 1.4387 | 147802575 | yes | 101.4 | 0.1 |
| brotli -q 11 -w 24 -D corpus tail | dictionary | 97960 | 1.5083 | 16192072 | yes | 1.4 | 0.0 |
| brotli -q11 | standalone | 99367 | 1.5299 |  | yes | 0.6 | 0.0 |
| brotli -q 11 --large_window=30 | standalone | 99369 | 1.5300 |  | yes | 0.6 | 0.0 |
| zstd -19 + 1 MiB dict | dictionary | 99881 | 1.5378 | 1048576 | yes | 0.4 | 0.0 |
| xz -9e | standalone | 102768 | 1.5823 |  | yes | 0.2 | 0.0 |
| zstd -19 + 110 KiB dict | dictionary | 104206 | 1.6044 | 112640 | yes | 0.1 | 0.0 |
| zstd --ultra -22 --long=27 | standalone | 104599 | 1.6105 |  | yes | 0.2 | 0.0 |
| zstd -19 | standalone | 104634 | 1.6110 |  | yes | 0.1 | 0.0 |
| bzip2 -9 | standalone | 105606 | 1.6260 |  | yes | 0.0 | 0.0 |
| gzip -9 | standalone | 146277 | 2.2522 |  | yes | 0.0 | 0.0 |
| 7zz PPMd o=32 mem=1g | standalone | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
| 7zz PPMd o=32 mem=1g primed | marginal | | | | skipped: 7zz not found (brew install sevenzip; tools/build_competitors.sh) | | |
