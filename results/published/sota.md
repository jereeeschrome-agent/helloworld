# Published reference numbers (state of the art and baselines)

Collected 2026-10-18 by reading each source page; machine-readable copy: `sota.json` (built and checked by `build_sota.py`). Research only: nothing here was run on this machine.

**Conventions.** bpb = 8 x bytes / original size, recomputed here (arithmetic for every number is in `sota.json`). enwik8 = 100,000,000 bytes, enwik9 = 1,000,000,000 bytes, Silesia = 211,938,580 bytes. "+dec" means the size includes the decompressor (and, for model-based entries, whatever weights it ships). **verified** = the number was read from the primary source on the access date and `build_sota.py` found it again on the quoted row of the live page. LTCB itself warns: "I have not verified results submitted by others", so the *tested by* column matters.

## 1. Large Text Compression Benchmark (LTCB)

Source: <https://mattmahoney.net/dc/text.html> (page "Last update: Oct. 14, 2026"). LTCB ranks by enwik9 compressed size + zipped decompressor; the enwik8 column is informational. "SFX" = self-extracting archive (decompressor inside the archive, LTCB lists 0).

| LTCB rank | Program | Options | enwik8 bytes | enwik8 bpb | enwik9 bytes | Decompressor | enwik9 total (+dec) | bpb (enwik9 total) | Tested by | Verified |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | ex-girlfriend 1.0 |  |  |  | 91,778,237 | SFX | 91,778,237 | 0.7342 | Ryan Landay (author), Threadripper 3970X | True |
| 2 | cmix-neif 1.0 |  |  |  | 91,913,343 | SFX | 91,913,343 | 0.7353 | Marcus Woelk (author), GCP c2d-highcpu-8 | True |
| 3 | makimono 1.0 |  |  |  | 95,793,872 | SFX | 95,793,872 | 0.7664 | Aleksandr Sklyarenko (author), EPYC 9755 VM | True |
| 4 | cmix-lex-transformer lexth11c |  |  |  | 95,836,613 | SFX | 95,836,613 | 0.7667 | Neel Patel (author), Modal 2 vCPU Xeon | True |
| 5 | zmix 1.0 |  |  |  | 96,096,261 | SFX | 96,096,261 | 0.7688 | James Byrne (author), Ryzen 9 5900X | True |
| 6 | rata-cmix 1.0.0 |  |  |  | 96,849,689 | SFX | 96,849,689 | 0.7748 | Dharmesh Patel (author), GCP n4d-highmem-2 | True |
| 7 | fx2-cmix-t-ft8 Oct 10 2026 |  |  |  | 96,945,005 | SFX | 96,945,005 | 0.7756 | Jesus Mejuto Gonzalez (author), Ryzen 9 5900X | True |
| 8 | fx2-cmix-transformer 24 Jul 2026 |  |  |  | 96,996,198 | SFX | 96,996,198 | 0.7760 | James Bowery, Ryzen 9 5900X | True |
| 9 | fractal-zip hutter-2026-09 |  | 12,303,845 | 0.9843 | 97,617,144 | SFX | 97,617,144 | 0.7809 | Scott Brooks (author), Core i7-11850H | True |
| 10 | tufazip v2 |  |  |  | 100,958,324 | 4,430,029 sd | 105,388,353 | 0.8431 | Tommy He (author), DGX B200 | True |
| 11 | altxs 1.0.0 |  |  |  | 93,434,410 | 13,490,401 xd | 106,924,811 | 0.8554 | Conrad Lippert-Zajaczkowski (author), EPYC 9V84 + GH100 | True |
| 12 | nncp v3.2 (2023-10-21) |  | 14,915,298 | 1.1932 | 106,632,363 | 628,955 xd | 107,261,318 | 0.8581 | Fabrice Bellard (author), Xeon E3-1230 v6 + RTX 3090 | True |
| 15 | cmix v21 | -t | 14,623,723 | 1.1699 | 107,963,380 | 281,387 sd | 108,244,767 | 0.8660 | Byron Knoll (author), Core i7-7700K | True |
| 26 | paq8px v206fix1 | -12L | 15,849,084 | 1.2679 | 124,696,410 | 402,949 s | 125,099,359 | 1.0008 | Sportman, Core i9 12900KS | True |
| - | paq8px v206fix1 | -12LT | 15,799,749 | 1.2640 | 124,619,348 | 512,427 sd | 125,131,775 | 1.0011 | Darek, Core i9-8950HK | True |
| 27 | durilca'kingsize | -m13000 -o41 -t2 | 16,209,219 | 1.2967 | 127,376,595 | 406,700 xd | 127,783,295 | 1.0223 | Zeelin, 2x Xeon Gold 6342 + Wine | True |
| 41 | ppmonstr J | -m1700 -o16 | 19,055,092 | 1.5244 | 157,007,383 | 42,019 x | 157,049,402 | 1.2564 | Matt Mahoney (maintainer) | True |
| 64 | lstm-compress v3 |  | 20,318,653 | 1.6255 | 173,874,407 | 144,567 s | 174,018,974 | 1.3922 | Byron Knoll (author), Core i7-7700K | True |
| 73 | 7-Zip PPMd (var. H based) 4.46a | -m0=ppmd:mem=1630m:o=10 -sfx7xCon.sfx | 21,197,559 | 1.6958 | 178,965,454 | SFX | 178,965,454 | 1.4317 | Matt Mahoney (note 23 gives the options) | True |
| - | 7-Zip PPMd (var. H based) 9.12b | -m0=ppmd:mem=2048m:o=10 | 21,060,863 | 1.6849 | 177,187,967 |  |  |  | Michal Hajicek, AMD Phenom II 965 | True |
| 82 | ppmd J1 (var. J) | -m256 -o10 -r1 | 21,388,296 | 1.7111 | 183,964,915 | 11,099 s | 183,976,014 | 1.4718 | Matt Mahoney (maintainer) | True |
| 99 | WinRAR PPM mode (PPMd var. I based) 3.60b3 | -mc7:128t+ -sfxWinCon.sfx | 22,713,569 | 1.8171 | 198,454,545 | SFX | 198,454,545 | 1.5876 | Matt Mahoney (maintainer) | True |
| 98 | xz 5.2.1 | --lzma2=preset=9e,dict=1GiB,lc=4,pb=0 | 24,703,772 | 1.9763 | 197,331,816 | 36,752 xd | 197,368,568 | 1.5789 | Andreas M. Nilsson, Core i7 (macOS) | True |
| - | xz 5.0.1 | -9 -e | 24,831,648 | 1.9865 |  |  |  |  | Matt Mahoney, Pentium T3200 | True |
| - | xz 5.0.1 | -9 | 24,865,244 | 1.9892 |  |  |  |  | Matt Mahoney, Pentium T3200 | True |
| - | 7-Zip LZMA 4.42 | -mx=9 -sfx7zCon.sfx | 24,996,113 | 1.9997 | 213,490,979 | SFX | 213,490,979 | 1.7079 | Matt Mahoney (maintainer) | True |
| 118 | zstd 0.6.0 | -22 --ultra | 25,405,601 | 2.0324 | 215,674,670 | 69,687 s | 215,744,357 | 1.7260 | Yann Collet (author), Core i7-3930K | True |
| 131 | brotli 18-Feb-2016 | -q 11 -w 24 | 25,764,698 | 2.0612 | 223,597,884 | 542,385 s | 224,140,269 | 1.7931 | Matt Mahoney, Core i7 M620 (Linux) | True |
| 145 | bzip2 1.0.2 | -9 | 29,008,736 | 2.3207 | 253,977,839 | 30,036 x | 254,007,875 | 2.0321 | Matt Mahoney (maintainer) | True |
| 180 | gzip 1.3.5 | -9 | 36,445,248 | 2.9156 | 322,591,995 | 38,801 x | 322,630,796 | 2.5810 | Matt Mahoney (maintainer) | True |

Rank "-" = row from a program's detail section, not the ranking table. Worked example: cmix v21 enwik8 = 8 x 14,623,723 / 100,000,000 = 1.1699 bpb; enwik9 total = 8 x 108,244,767 / 10^9 = 0.8660 bpb.

What the 2026 top of the table is: ranks 1-9 are Hutter-prize-style **self-extracting archives built around a small transformer trained offline on enwik9** (fx2-cmix-transformer: 6M parameters, 4-bit weights, ~2.9 MB, "pre-trained on enwik9 on 8 RTX 5090 GPUs for 26 hours"), mixed with cmix/paq context models and built for the Hutter limits (one CPU core, about 10 GB RAM; LTCB lists 31-67 h per direction on the testers' machines). The weights travel inside the archive, so they *are* counted. Their separate compressor executable (which also holds the weights) is not counted by LTCB but is counted by the Hutter prize:

| Program | archive9 (SFX) | compressor S1 | Hutter-style S1+S2 | bpb (S1+S2) | how S1+S2 was obtained |
|---|---:|---:|---:|---:|---|
| ex-girlfriend 1.0 | 91,778,237 | 3,995,122 | 95,773,359 | 0.7662 | computed here: 91,778,237 + 3,995,122 = 95,773,359 |
| cmix-neif 1.0 | 91,913,343 | 4,000,708 | 95,914,051 | 0.7673 | as printed on LTCB |
| makimono 1.0 | 95,793,872 | 3,984,978 | 99,788,850 | 0.7983 | as printed on LTCB |
| cmix-lex-transformer lexth11c | 95,836,613 | 3,477,137 | 99,313,750 | 0.7945 | as printed on LTCB |
| zmix 1.0 | 96,096,261 | 3,216,163 | 99,312,424 | 0.7945 | as printed on LTCB |
| rata-cmix 1.0.0 | 96,849,689 | 3,369,432 | 100,219,121 | 0.8018 | computed here: 96,849,689 + 3,369,432 = 100,219,121 |
| fx2-cmix-t-ft8 Oct 10 2026 | 96,945,005 | 3,422,266 | 100,367,271 | 0.8029 | as printed on LTCB |
| fx2-cmix-transformer 24 Jul 2026 | 96,996,198 | 3,428,474 | 100,424,672 | 0.8034 | as printed on LTCB |
| fractal-zip hutter-2026-09 | 97,617,144 | 3,836,817 | 101,453,961 | 0.8116 | as printed on LTCB |

GPU-bound entries: **altxs 1.0.0** needs a CUDA GPU (GH100, ~80 GB GPU memory, 330 h to decompress; its 13.5 MB decompressor holds the weights) and **tufazip v2** ran on 2x B200; **nncp v3.2** needs a GPU for acceptable speed but trains online, so it has no pretrained weights to count. Best results without any offline-trained model: cmix v21 (enwik8 14,623,723 = 1.1699 bpb, the smallest enwik8 on LTCB apart from fractal-zip, whose enwik9-trained weights have seen enwik8) and nncp v3.2 (enwik9 total 107,261,318 = 0.8581 bpb).

Baseline notes: LTCB has no plain PPMd var. I row; WinRAR's PPM mode is described there as "ppmd var. I, an earlier version of ppmd J". 7-Zip PPMd is described as equivalent to "ppmd var H (with minor changes)". The strongest zstd on LTCB is 0.6.0 (2016) `-22 --ultra`; **no `--long` or zstd >= 1.0 result is published there** (none found elsewhere either: measure locally). gzip is 1.3.5 -9 and brotli is the Feb-2016 build.

## 2. Bellard: nncp and ts_zip

| Page | Program | Dataset | Bytes | bpb | Counts model / decompressor? | Model | GPU | Verified |
|---|---|---|---:|---:|---|---|---|---|
| <https://bellard.org/nncp/> | nncp | enwik8 | 14,915,298 | 1.1932 | no (online-trained; LTCB adds the 628,955-byte program) | online transformer | yes for speed (RTX 4090 in readme) | True |
| <https://bellard.org/nncp/> | nncp | enwik9 | 106,632,363 | 0.8531 (+dec 107,261,318 = 0.8581) | no (online-trained; LTCB adds the 628,955-byte program) | online transformer | yes for speed (RTX 4090 in readme) | True |
| <https://bellard.org/ts_zip/> | ts_zip | enwik8 | 13,825,741 | 1.1061 | no: the 169M-parameter model is not counted | RWKV 169M v4, 8-bit weights, BF16 eval | yes for reasonable speed (up to 1 MB/s on RTX 4090) | True |
| <https://bellard.org/ts_zip/> | ts_zip | enwik9 | 135,443,237 | 1.0835 | no: the 169M-parameter model is not counted | RWKV 169M v4, 8-bit weights, BF16 eval | yes for reasonable speed (up to 1 MB/s on RTX 4090) | True |

ts_zip arithmetic: 8 x 13,825,741 / 10^8 = 1.1061 bpb (page: 1.106); 8 x 135,443,237 / 10^9 = 1.0835 (page: 1.084). nncp readme commands: `./nncp --cuda --profile enwik8 --preprocess 4096,512 c enwik8 out.bin` (14,915,298 bytes, 13.2 h) and `--profile enwik9 --preprocess 16384,512` (106,632,363 bytes, 2.8 days); test system Ryzen 9 3900 + RTX 4090, CPU 1 GB, GPU 6.6 GB. ts_zip says it is experimental, with no backward compatibility between versions, and that its output does not depend on the GPU/CPU model or thread count (deterministic evaluation), which jevzip's codec does not guarantee (decode needs the same torch build, kernels, threads and bf16 flag as encode).

## 3. Hutter Prize (enwik9, total = compressor + self-extracting archive)

Source: <http://prize.hutter1.net/>. Rules on the page: S := S1 + S2 (compressor comp.exe + archive.exe that rebuilds enwik9 with no input), "Must run in ≲50 hours using a single CPU core and <10GB RAM and <100GB HDD" on the committee's test machine, so GPU models are out.

| Date | Holder | Program | Total S (bytes) | bpb | Status | Verified |
|---|---|---|---:|---:|---|---|
| 24 Jul 2026 | Vladimir Ivanov | fx2-cmix-T (fx2-cmix-transformer) | 100,424,672 | 0.8034 | current record; award 7.46% = 37,300 EUR | True |
| 19 Jul 2026 | David Freelan | cmix-obias | 108,521,870 | 0.8682 | previous record | True |
| 26 Jun 2026 | Ibrahim Marcouch & Kaido Orav | cmix-lex | 109,671,639 | 0.8774 | previous record | True |
| 3 Sep 2024 | Kaido Orav & Byron Knoll | fx2-cmix | 110,793,128 | 0.8863 | previous record | True |

Record arithmetic: 8 x 100,424,672 / 10^9 = 0.8034 bpb (= LTCB fx2-cmix-transformer archive 96,996,198 + compressor 3,428,474). Next award needs total < 99,420,425. Pending Hutter-style claims below that line on LTCB: cmix-neif 1.0 (95,914,051), zmix 1.0 (99,312,424), cmix-lex-transformer lexth11c (99,313,750); not verified by the committee yet.

## 4. Deletang et al., "Language Modeling Is Compression" (ICLR 2024)

Source: <https://arxiv.org/abs/2309.10668> (v2, Table 1), code constants <https://github.com/google-deepmind/language_modeling_is_compression>. Protocol: enwik9 cut into 2048-byte chunks, each compressed with **no context from earlier chunks** (constants.py: 488,281 chunks x 2048 = 999,999,488 bytes). Raw rate = compressed/raw; adjusted rate adds 2 bytes per parameter (float16). LLMs get 2048 ASCII characters per chunk (bytes >127 have their top bit zeroed and the bit appended to the output) and only top-100 renormalized token probabilities. The small Transformers are byte-level and trained on enwik8, so enwik8 is in-distribution and the rest of enwik9 partly not.

| Model / compressor | Params | Raw rate | bpb (raw) | Adjusted rate | bpb (adjusted) | Verified |
|---|---:|---:|---:|---:|---:|---|
| Chinchilla 70B | 70B | 8.3% | 0.664 | 14008.3% | 1120.664 | True |
| Chinchilla 7B | 7B | 10.2% | 0.816 | 1410.2% | 112.816 | True |
| Chinchilla 1B | 1B | 11.3% | 0.904 | 211.3% | 16.904 | True |
| Llama 2 (7B) | 7B | 8.9% | 0.712 | 1408.9% | 112.712 | True |
| Transformer 3.2M (trained on enwik8) | 3.2M | 17.0% | 1.360 | 17.7% | 1.416 | True |
| Transformer 800K (trained on enwik8) | 800K | 21.7% | 1.736 | 21.9% | 1.752 | True |
| Transformer 200K (trained on enwik8) | 200K | 30.9% | 2.472 | 30.9% | 2.472 | True |
| gzip, 2048-byte chunks |  | 48.1% | 3.848 | 48.1% | 3.848 | True |
| LZMA2, 2048-byte chunks |  | 50.0% | 4.000 | 50.0% | 4.000 | True |
| gzip, unchunked |  | 32.3% | 2.584 | 32.3% | 2.584 | True |
| LZMA2, unchunked |  | 23.0% | 1.840 | 23.0% | 1.840 | True |

Arithmetic: bpb = 8 x rate; e.g. Chinchilla 70B 8 x 0.083 = 0.664 bpb raw, 8 x 140.083 = 1120.66 bpb adjusted. Rates are printed to 0.1%, so bpb is +-0.004. These chunked numbers are **not comparable to whole-file enwik9 results** (gzip goes from 32.3% unchunked to 48.1% chunked).

## 5. Language-model bpc on the standard enwik8 / text8 test split

Split: 90M / 5M / 5M characters; test = last 5,000,000 bytes (Al-Rfou et al.: "we split the data into 90M, 5M and 5M for training, dev and test respectively"). These are model cross-entropies, not compressed files: no arithmetic-coder overhead, no model size, and Transformer-XL-style models carry memory across segments. LM papers also list "cmix v13 1.23" for enwik8; that matches cmix v13's LTCB result on the **whole** 100 MB file learned from scratch (8 x 15,323,969 / 10^8 = 1.2259), so it is most likely not a test-split number.

| Benchmark | Model | Params | Test bpc | Source | Verified |
|---|---|---|---:|---|---|
| enwik8-test | Transformer-XL 24L | 277M | 0.99 | [Dai et al. 2019, Table 2](https://arxiv.org/abs/1901.02860) | True |
| enwik8-test | Transformer-XL 18L | 88M | 1.03 | [Dai et al. 2019, Table 2](https://arxiv.org/abs/1901.02860) | True |
| enwik8-test | Transformer-XL 12L | 41M | 1.06 | [Dai et al. 2019, Table 2](https://arxiv.org/abs/1901.02860) | True |
| enwik8-test | Al-Rfou et al. T64 (64L, ctx 512) | 235M train / 219M inference | 1.06 | [Al-Rfou et al. 2018 (AAAI 2019), Table 3](https://arxiv.org/abs/1808.04444) | True |
| enwik8-test | Al-Rfou et al. T12 (12L, ctx 512) | 44M train / 41M inference | 1.11 | [Al-Rfou et al. 2018, Table 3](https://arxiv.org/abs/1808.04444) | True |
| enwik8-test | SHA-RNN (4-layer SHA-LSTM, attention head per layer) | 54M | 1.068 | [Merity 2019, Table 1 (valid 1.096)](https://arxiv.org/abs/1911.11423) | True |
| enwik8-test | SHA-RNN (4-layer SHA-LSTM, single attention head) | 52M | 1.076 | [Merity 2019, Table 1 (valid 1.100)](https://arxiv.org/abs/1911.11423) | True |
| enwik8-test | Compressive Transformer 24L | not stated in paper (277M per Longformer Table 3, secondary) | 0.97 | [Rae et al. 2019 (ICLR 2020), Table 4](https://arxiv.org/abs/1911.05507) | True |
| enwik8-test | Adaptive-Span 24L | 209M | 0.98 | [Sukhbaatar et al. 2019, Table 2 (dev 1.00)](https://arxiv.org/abs/1905.07799) | True |
| enwik8-test | Adaptive-Span 12L (small) | 39M | 1.02 | [Sukhbaatar et al. 2019, Table 2 (dev 1.04)](https://arxiv.org/abs/1905.07799) | True |
| enwik8-test | Longformer (small) | 41M | 1.00 | [Beltagy et al. 2020, Table 2 (dev 1.02)](https://arxiv.org/abs/2004.05150) | True |
| enwik8-test | Longformer (large) | 102M | 0.99 | [Beltagy et al. 2020, Table 3](https://arxiv.org/abs/2004.05150) | True |
| enwik8-test | mLSTM + dynamic evaluation (Krause et al. 2017) | 46M | 1.08 | [secondary: quoted in Al-Rfou et al. Table 3; primary paper not checked](https://arxiv.org/abs/1808.04444) | False |
| enwik8-test | GPT-2 1542M, zero-shot (trained on WebText) | 1542M | 0.93 | [Radford et al. 2019, Table 3; zero-shot with invertible de-tokenizers; split not restated in the table](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | True |
| enwik8-test | GPT-2 117M, zero-shot (trained on WebText) | 117M | 1.16 | [Radford et al. 2019, Table 3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | True |
| text8-test | Transformer-XL 24L | 277M | 1.08 | [Dai et al. 2019, Table 3](https://arxiv.org/abs/1901.02860) | True |
| text8-test | Al-Rfou et al. T64 | 235M train / 219M inference | 1.13 | [Al-Rfou et al. 2018, Table 1](https://arxiv.org/abs/1808.04444) | True |
| text8-test | Al-Rfou et al. T12 | 44M train / 41M inference | 1.18 | [Al-Rfou et al. 2018, Table 1](https://arxiv.org/abs/1808.04444) | True |
| text8-test | Adaptive-Span 24L | 209M | 1.07 | [Sukhbaatar et al. 2019, Table 1 (dev 1.01)](https://arxiv.org/abs/1905.07799) | True |
| text8-test | Adaptive-Span 12L (small) | 38M | 1.11 | [Sukhbaatar et al. 2019, Table 1 (dev 1.05)](https://arxiv.org/abs/1905.07799) | True |
| text8-test | Longformer (small) | 41M | 1.10 | [Beltagy et al. 2020, Table 2 (dev 1.04)](https://arxiv.org/abs/2004.05150) | True |
| text8-test | GPT-2 1542M, zero-shot | 1542M | 0.98 | [Radford et al. 2019, Table 3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | True |

Small-model reference points (< 50M parameters) on enwik8 test: Longformer small 41M 1.00, Adaptive-Span 39M 1.02, Transformer-XL 12L 41M 1.06, Al-Rfou T12 44M 1.11; SHA-RNN is just above at 52-54M (1.068-1.076). GPT-2 is zero-shot (trained on WebText, evaluated with invertible de-tokenizers).

## 6. Silesia corpus

Source: <https://mattmahoney.net/dc/silesia.html> ("Programs listed are the latest version available as of Oct. 10, 2026"; open-source programs only; files compressed individually; data only, no decompressor), plus the cmix page <http://www.byronknoll.com/cmix.html> for exact per-file bytes.

| Program | Version / options | Total bytes | bpb | Verified |
|---|---|---:|---:|---|
| paq8px | v217 -12L | 27,682,006 | 1.0449 | True |
| paq8px | v206 -12TL | 28,241,197 | 1.0660 | True |
| cmix (after precomp v0.4.7 -cn) | v21 | 28,261,094 | 1.0668 | True |
| ppmd | -m256 -o16 -r1 (version not printed) | 47,021,495 | 1.7749 | True |
| 7-Zip (LZMA) | -mx=9 | 48,792,760 | 1.8418 | True |
| brotli | 18-Feb-2016 -q 11 -w 24 | 50,297,179 | 1.8986 | True |
| zstd | 0.6.0 -22 | 53,321,756 | 2.0127 | True |
| zstd | "5.1.0" -21 (version string as printed) | 53,862,053 | 2.0331 | True |
| bzip2 | (no options) | 54,506,769 | 2.0575 | True |
| gzip | -9 | 67,631,990 | 2.5529 | True |
| cmix (exact per-file bytes, byronknoll.com) | not printed (page lists v21 downloads; its enwik8/enwik9 rows equal v21) | 28,261,094 | 1.0668 | True |

Page inconsistency, cmix (after precomp v0.4.7 -cn) v21: per-file KB columns sum to 28,344 KB, inconsistent with the total 28,261,094 bytes (allowed (28,249, 28,261]): a typo in a per-file column on the page; dickens is printed 1892 but cmix's own page gives 1,802,071 bytes (1802 KB). The total is unaffected.

Arithmetic: 8 x 27,682,006 / 211,938,580 = 1.0449 bpb (paq8px_v217 -12L, current #1). xz is not listed on the Silesia page (7-Zip -mx=9, i.e. LZMA, is the closest). The zstd row printed as "5.1.0" is reproduced verbatim; GitHub's newest zstd release is v1.5.7 (2025-02-19) and there is no 5.x, so it is probably v1.5.1 (unconfirmed). GitHub's newest paq8px tag is v217, the version leading this table (LTCB's newest paq8px is v206fix1). Per-file sizes on the benchmark page are KB (truncated); exact cmix per-file bytes are in `sota.json` (`per_file_bytes`).

## 7. Reading jevzip numbers against these

- **enwik8 test split (last 5M bytes)** is the only protocol-matched comparison for a model trained on `data/enwik8/train.bin` (first 90M bytes): it lines up with section 5, except that jevzip reports real coded bytes (coder and framing overhead included) with a 512-byte context and no cross-segment memory. Nearest published sizes: Al-Rfou T12 (44M, context 512) 1.11, Transformer-XL 12L (41M) 1.06, Adaptive-Span 12L (39M) 1.02, Longformer small (41M) 1.00. jevzip-lanes-11m has 11.14M parameters, about 4x fewer.
- **Whole-file enwik8 / enwik9** with that model is not comparable to LTCB, Hutter, cmix or nncp: 90% of enwik8 is its training data (the same caveat as fractal-zip above), and its weights are not counted, while every LTCB "SFX" entry counts them.
- If the weights were counted (parameter count rounded to 11.14M): x 4 bytes (fp32) = about 44,560,000 bytes, adding 8 x 44,560,000 / 10^8 = 3.565 bpb on enwik8 and 0.356 bpb on enwik9; at 2 bytes/parameter (Deletang et al.'s convention) 1.782 and 0.178. For scale, fx2-cmix-transformer ships its 6M parameters in about 2.9 MB (4-bit).
- The closest published analogue (pretrained LM + arithmetic coding, model not counted) is ts_zip: 1.1061 bpb on enwik8, 1.0835 on enwik9, with a 169M RWKV trained on general text rather than on enwik.
- The Deletang et al. numbers only apply if jevzip codes independent 2048-byte chunks. No published numbers exist for the Python-source test sets in `data/code/`: baselines there (xz, zstd, brotli, cmix, paq8px) have to be measured locally.

## 8. Not found / unverified

- zstd `--ultra -22 --long` on enwik8/enwik9: no published number found (LTCB stops at zstd 0.6.0). Measure locally.
- PPMd var. I as a standalone program: not on LTCB or Silesia; only WinRAR's var.-I-based PPM mode (enwik9 198,454,545 SFX).
- nncp v3.2 parameter count: not stated (LTCB says v3 is ~199M; v3.2 uses "a larger transformer model").
- Compressive Transformer parameter count: not stated in the paper; 277M comes from Longformer Table 3 (secondary).
- GPT-2 rows were read from the PDF (pdftotext -layout, page 5), not re-checked by the script.
- `lm-enwik8-test-mlstm-dynamic-evaluation-krause-et-al-2017`: verified = False (secondary: quoted in Al-Rfou et al. Table 3; primary paper not checked)

## Sources (accessed 2026-10-18)

- ltcb: <https://mattmahoney.net/dc/text.html>, sha256 of fetched body `14e052254b2f65837074d0dd05194ff544753c354c1f5b2313751d3872d1193a`
- silesia: <https://mattmahoney.net/dc/silesia.html>, sha256 of fetched body `656fe01018de0becbe147e888d868c2ef23af5ffa904b70329e34ccff7e8c4fe`
- nncp: <https://bellard.org/nncp/>, sha256 of fetched body `3899a95dfc0eab9bde833511fc937b1ad4de477c11d1c8d9b98d3b199f6074f4`
- nncp_readme: <https://bellard.org/nncp/readme.txt>, sha256 of fetched body `46b9b60fcce10960c66126a2265f805b0591f405f83f9f24a63ce624239bcb96`
- ts_zip: <https://bellard.org/ts_zip/>, sha256 of fetched body `5ea477c795cc3a0f724048e744ad8bddfd9b5375da4b3c33555bdc05d72e762c`
- hutter: <http://prize.hutter1.net/>, sha256 of fetched body `a6a3dc8fc6344c3a12af67294912e821b725d74c7b1eeb05efe278126eecde7b`
- cmix: <http://www.byronknoll.com/cmix.html>, sha256 of fetched body `da40d0957604a4682d1a9fc5c6835a7a0979f36316a3e0d9f2f68832945fc29c`
- deletang: <https://arxiv.org/abs/2309.10668> (checked via https://arxiv.org/html/2309.10668v2), sha256 of fetched body `b1811ff19558cb15dceaef9075b0f18dc79c1c175be7f458200c58ea61c92feb`
- deletang_code: <https://github.com/google-deepmind/language_modeling_is_compression> (checked via https://raw.githubusercontent.com/google-deepmind/language_modeling_is_compression/main/constants.py), sha256 of fetched body `fde2c90013609b93f10747128d28d1a25530a53f9e054522bdf7ef43681348ae`
- txl: <https://arxiv.org/abs/1901.02860> (checked via https://ar5iv.labs.arxiv.org/html/1901.02860), sha256 of fetched body `5cb390d0f2d56e594473db6d26e776699521957d80a514fd4ee4b3606180f21e`
- alrfou: <https://arxiv.org/abs/1808.04444> (checked via https://ar5iv.labs.arxiv.org/html/1808.04444), sha256 of fetched body `c92ed43486910ce55c4f3aab245e6667f25d032078933e22922ed6a9f764916f`
- sharnn: <https://arxiv.org/abs/1911.11423> (checked via https://ar5iv.labs.arxiv.org/html/1911.11423), sha256 of fetched body `83b4d9c739db20daa15c75f4f12f6099065512d84c33d61803f2022d5f9ad6ab`
- compressive: <https://arxiv.org/abs/1911.05507> (checked via https://ar5iv.labs.arxiv.org/html/1911.05507), sha256 of fetched body `98c0487eeed47c70a9284411caa9ea5287cd00986acb2f008108acc3ac2ac5a3`
- adaptive: <https://arxiv.org/abs/1905.07799> (checked via https://ar5iv.labs.arxiv.org/html/1905.07799), sha256 of fetched body `12476353ee674d88832bacaf1706adf140d622b27b814482e9ea40e2bdf0b5b5`
- longformer: <https://arxiv.org/abs/2004.05150> (checked via https://ar5iv.labs.arxiv.org/html/2004.05150), sha256 of fetched body `62d1b216f72b85353344a148a079427187755969e112d01c276a5a6fa71f9918`
- zstd_releases: <https://github.com/facebook/zstd/releases> (checked via https://api.github.com/repos/facebook/zstd/releases?per_page=6), sha256 of fetched body `1cbbd08410d0c2069ac3d2d85d3b47e25fc59d2cbe2890bf0712df6f607a0f96`
- paq8px_tags: <https://github.com/hxim/paq8px/tags> (checked via https://api.github.com/repos/hxim/paq8px/tags?per_page=5), sha256 of fetched body `33669566455eea1d4afd23dd3f2fe550b200ff3737ae701447b199734a016a73`
- gpt2: <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>, PDF sha256 `d9d852e2894556e73f53cb22b7c605a9643d6f0b19bf604b429ed6192fa24f4e`
