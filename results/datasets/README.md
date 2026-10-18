# Datasets

Every byte file jevzip trains or benchmarks on, with sizes, SHA-256s, sources and licenses, is listed in
[`jevzip/datasets.json`](../../jevzip/datasets.json). `python -m jevzip.data <name>` builds a dataset and
`python -m jevzip.data verify [--downloads]` checks it. Machine-readable summary: [`datasets.json`](datasets.json).
Verified on 2026-10-18: 56/56 files and downloads OK.

| Dataset | Files | Size | Source | License | Use |
|---|---|---|---|---|---|
| code (train / val) | `data/code/train.bin`, `val.bin` | 147.8 MB / 1.9 MB | built here from CPython 3.12.13 stdlib + .venv site-packages; val = mpmath | per package (BSD/MIT/Apache/PSF) | training the code models |
| code (test) | `data/code/networkx.py.txt`, `pydoc_topics.txt` (+ `networkx-files/`, 77 files) | 1.05 MB + 0.52 MB | networkx 3.7 (first 1 MiB), CPython pydoc_data | BSD-3-Clause, PSF-2.0 | in-domain benchmark; release asset `jevzip-code-testset.tar.xz` |
| enwik8 | `data/enwik8/{train,val,test}.bin` | 90 M / 5 M / 5 M bytes | mattmahoney.net/dc/enwik8.zip (md5 a1fa5ffd... checked) | Wikipedia, CC BY-SA 3.0 / GFDL | standard benchmark (test.bin), training the enwik8 models |
| text8 | `data/text8/{train,val,test}.bin` | 90 M / 5 M / 5 M chars | mattmahoney.net/dc/text8.zip | Wikipedia, CC BY-SA 3.0 / GFDL | standard benchmark (test.bin) |
| Silesia | `data/silesia/` (12 files) | 211.9 MB | sun.aei.polsl.pl/~sdeor/corpus/silesia.zip | none stated; mixed copyrighted sources | standard compression benchmark |
| OOD: JSON | `data/ood/json/` (8 files) | 6.28 MB | simdjson `jsonexamples/` at v0.9.7 (33d926bd) | Apache-2.0 repo; third-party data, status unclear | out-of-domain |
| OOD: logs | `data/ood/logs/` (6 × `*_2k.log`) | 1.42 MB | logpai/loghub at dd61d095 | research/academic use only: **not redistributable** | out-of-domain |
| OOD: C++ headers | `data/ood/c-headers/torch-c10.txt` (156 headers, listed in `torch-c10-files/index.json`) | 1.03 MB | torch 2.14.1 wheel, `include/c10/{core,util}` | BSD-3-Clause (PyTorch) | out-of-domain |
| OOD: binaries | references to Silesia `mozilla`, `ooffice`, `sao`, `x-ray`, `mr` | 83.1 MB | Silesia | as Silesia | out-of-domain |
| OOD: non-English | reference to Silesia `reymont` (Polish) | 6.6 MB | Silesia | as Silesia | out-of-domain |

Suites (the `suites` section of the manifest): `code`, `standard` (enwik8, text8, Silesia) and `ood`
(json, logs, c-headers, binary, non-english). The OOD binary and Polish entries point at the Silesia
files rather than copying them.

## Notes

- **The code corpus is machine-specific.** It is built from the packages installed on the build machine,
  so it can't be rebuilt bit-identically elsewhere (its manifest entries say `"reproducible": false`).
  Rebuilding today gives identical `val.bin` and test files, but `train.bin` grows to 226.4 MB because
  transformers, tokenizers, matplotlib and others were installed after the corpus was built. The recorded
  147.8 MB `train.bin` is what the code models were trained on; `data.py` refuses to replace it without `--force`.
- **enwik8** re-derived into a temp dir from `downloads/enwik8/enwik8.zip` reproduces the three recorded
  SHA-256s exactly; `data/enwik8/` was not written.
- **simdjson** moved most of `jsonexamples/` to a separate repo after v0.9.x, so the files are pinned to
  v0.9.7. JSON and log downloads are checked against their git blob ids at the pinned commit as well as their SHA-256s.
- **text8.zip and silesia.zip** have no published checksums: their SHA-256s were recorded on first download.
- OOD downloads total 7.7 MB (limit 10 MB). The loghub logs are unsanitized production logs (IP addresses, host names).
