# Decode portability of jevzip streams

What decodes a `.jz` made on this machine, and what does not. Inputs: `runs/ref/networkx-64k.bin`
(64 KiB of networkx source) and its reference streams, made with the pre-refactor codec (git HEAD
fdfd4b0), `weights/jevzip-lanes-11m-code.pt`, torch 2.14.1, CPU, threads 4, default streams,
K = 32 and 128, fp32 and bf16. Machine: Mac mini M4 Pro, macOS 27.2 arm64, Python 3.12.13,
numpy 2.5.4. Full numbers, commands, platform info and input SHA-256s: `portability.json`
(merged from `raw/*.json`).

Every run uses the real `codec.compress` / `codec.decompress` with hooks (`tools/portability.py`):

* **decode**: decompress under the variant: ok, or which check caught the failure and the first
  wrong byte (index in decode order / file offset).
* **teacher forcing**: compress the original under the variant and compare each forward call with
  the encoder's own probabilities (cached from a control run that reproduced the reference `.jz`
  byte for byte): quantized CDF entries that differ, rows with any difference, *coded intervals*
  that differ (the decoder stays in sync exactly while the coded symbol's `[start, start+freq)`
  matches, so one differing interval breaks the stream), and max |dp| of the float probabilities.
  The same interval count is given for CDFs quantized to 16 and 12 bits instead of 24.

## Matrix

| case | env | variant | decode | first wrong byte (decode idx / offset) | CDF entries differ | rows differ | coded intervals differ (24 / 16 / 12 bits) | max abs dp |
|---|---|---|---|---|---|---|---|---|
| k32-fp32 | torch 2.14.1 | control | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-fp32 | torch 2.14.1 | threads 1 | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-fp32 | torch 2.14.1 | threads 2 | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-fp32 | torch 2.14.1 | threads 8 | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-fp32 | torch 2.14.1 | cross (decoded as bf16) | rANS stream exhausted | 1 / 32 | 49.99% | 96.00% | 62715 / 50323 / 38390 of 65536 | 4.8e-02 |
| k128-fp32 | torch 2.14.1 | control | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k128-fp32 | torch 2.14.1 | threads 1 / 2 / 8 | ok (all three) | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k128-fp32 | torch 2.14.1 | cross (decoded as bf16) | rANS stream exhausted | 1 / 128 | 47.48% | 95.58% | 62455 / 48813 / 36764 of 65536 | 5.3e-02 |
| k32-bf16 | torch 2.14.1 | control | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-bf16 | torch 2.14.1 | threads 1 | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k32-bf16 | torch 2.14.1 | cross (decoded as fp32) | rANS stream exhausted | 1 / 32 | 49.99% | 96.00% | 62715 / 50323 / 38390 of 65536 | 4.8e-02 |
| k128-bf16 | torch 2.14.1 | control | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |
| k128-bf16 | torch 2.14.1 | cross (decoded as fp32) | rANS stream exhausted | 1 / 128 | 47.48% | 95.58% | 62455 / 48813 / 36764 of 65536 | 5.3e-02 |
| k32-fp32 | torch 2.8.0 (2nd venv) | control | rANS stream exhausted | 4 / 128 | 11.30% | 61.27% | 37574 / 1126 / 52 of 65536 | 5.2e-06 |
| k32-fp32 | torch 2.8.0 | threads 1 | rANS stream exhausted | 4 / 128 | 11.30% | 61.27% | 37574 / 1126 / 52 of 65536 | 5.2e-06 |
| k128-fp32 | torch 2.8.0 | control | rANS end state (trailing data) | 3 / 384 | 9.85% | 57.79% | 35958 / 1105 / 77 of 65536 | 4.9e-06 |
| k128-fp32 | torch 2.8.0 | threads 1 | rANS end state (trailing data) | 3 / 384 | 9.85% | 57.79% | 35958 / 1105 / 77 of 65536 | 4.9e-06 |
| k32-bf16 | torch 2.8.0 | control | rANS stream exhausted | 1 / 32 | 49.21% | 95.71% | 62560 / 49935 / 37711 of 65536 | 3.2e-02 |
| k32-fp32 | torch 2.14.1, MPS | model on mps (first 64 calls) | wrong from byte 4 (stopped) | 4 / 128 | 14.20% | 67.30% | 10569 / 529 / 32 of 16384 | 1.2e-05 |
| k32-bf16 | torch 2.14.1, MPS | model on mps, bf16 autocast (first 64 calls) | wrong from byte 1 (stopped) | 1 / 32 | 49.71% | 95.92% | 15673 / 12735 / 9703 of 16384 | 3.0e-02 |
| k32-fp32 | torch 2.14.1, refactored working tree | control | ok | - | 0 | 0 | 0 / 0 / 0 of 65536 | 0 |

Notes. Threads were overridden inside the harness (the header pins 4). bf16 x fp32 "cross" flips
the header's bf16 flag, which the normal path never does; the two cross rows of a K are the same
comparison seen from either side, hence equal numbers. MPS used 7.3 s of GPU in total (64 forward
calls teacher-forced per case, decode stopped 64 bytes after the first wrong byte; masks are
built on mps by running the step under `torch.device("mps")`). "rows differ" counts rows whose
24-bit CDF differs anywhere; in every non-control run 0% of rows were bit-identical as floats.
The torch 2.8.0 rows mean "same machine, same weights, older torch": nothing else changed. The
last row checks that the refactored codec (working tree at 13:45) still reproduces the
reference stream exactly. Code size of the K = 32 fp32 stream if the CDFs were quantized
coarser: 10,294 B at 24 bits, 10,321 B at 16, 10,926 B at 12 (+6.1%), 13,320 B at 10 (+29%).

**CI fixtures** (`jevzip/testdata`, tiny random models, `python -m jevzip.portability_check`):
torch 2.14.1 here: 10/10 decode, all re-encode byte-identical. torch 2.8.0, same Mac: 0/10
decode (all "rANS end state", none re-encode identically). So the fixtures do detect a
torch-version change even with d_model 32; the GitHub workflow runs them on ubuntu-latest
(x86_64) and macos-latest with whatever torch PyPI serves, which will be the first
cross-machine data point once pushed.

## Verdict (for the README)

jevzip's bitstream is a research artifact, not an interchange format: a `.jz` decodes only with
the same torch build on the same kind of CPU kernels, and the same fp32/bf16 setting, as the
encoder. On this Mac the thread count turned out not to matter (1, 2, 4 and 8 threads give
bit-identical probabilities for the 11M model, fp32 and bf16, K = 32 and 128), but changing only
the torch version (2.14.1 to 2.8.0) shifts every probability in its last bits (max |dp| 5e-6),
makes 58-61% of the quantized CDF rows differ, and decoding goes wrong at the 4th byte; running
the model on the GPU (MPS) does the same (max |dp| 1.2e-5), and bf16 versus fp32 is in effect a
different model (max |dp| 0.05). The decoder only notices at the end (rANS state or CRC), so a
mismatch costs a full decode. Coarser quantization does not fix this: rounding is a step function
and some probability always sits next to a step, so at 16 bits 1.7% of coded intervals still
differ, at 10 bits 14 of 65,536 still differ (any one breaks the stream) while the file grows
29%. A real fix makes the probabilities an exactly specified function of the input: integer or
fixed-point inference with a fixed reduction order (int8/int16 weights, int32 accumulation,
lookup-table exp/softmax, integer layer norm), as NNCP/ts_zip do with their own deterministic
LibNC kernels (ts_zip states its output does not depend on the CPU/GPU model or thread count).
Rough cost: a small C or Rust kernel set for ~6 ops (matmul, layer norm, rotary, attention
softmax, GELU, output softmax), weeks of work rather than days, probably 2-5x slower than
BLAS fp32 on CPU and no GPU path until ported, plus quantization-aware fine-tuning to keep the
size loss to ~1-3%. A cheaper interim step is to record the torch version and CPU architecture in
the header and refuse to decode on a mismatch with a clear message instead of a late CRC error.

## Reproduce

```
S=<scratch>; mkdir -p $S/headpkg && git archive HEAD jevzip/{__init__,codec,model,rans,schedule}.py | tar -x -C $S/headpkg
P=".venv/bin/python tools/portability.py run --pkg-root $S/headpkg --ref-cache $S/refcache"
$P --tag torch-2.14.1-fp32 --cases k32-fp32,k128-fp32 --variants control,t1,t2,t8,cross
$P --tag torch-2.14.1-bf16-k32 --cases k32-bf16 --variants control,cross,t1
$P --tag torch-2.14.1-bf16-k128 --cases k128-bf16 --variants control,cross
$P --tag torch-2.14.1-mps --cases k32-fp32,k32-bf16 --variants mps --gpu-budget 45 --mps-calls 64 --mps-stop-after 64
O="downloads/torch-old/venv/bin/python -I tools/portability.py run --pkg-root $S/headpkg --ref-cache $S/refcache"
$O --tag torch-2.8.0-fp32 --cases k32-fp32,k128-fp32 --variants control,t1
$O --tag torch-2.8.0-bf16-k32 --cases k32-bf16 --variants control
.venv/bin/python tools/portability.py run --ref-cache $S/refcache --tag worktree-control --cases k32-fp32 --variants control
.venv/bin/python -m jevzip.portability_check --json results/portability/raw/fixtures-torch-2.14.1.json
downloads/torch-old/venv/bin/python -I -c "import sys; sys.path.insert(0, '.'); from jevzip import portability_check as p; sys.exit(p.main(['--json', 'results/portability/raw/fixtures-torch-2.8.0.json']))"
.venv/bin/python tools/portability.py merge results/portability/raw/*.json
```

The reference cache (~130 MB per case) must be filled by the encoding environment (torch 2.14.1)
first. All CPU runs used `nice -n 10`; the ref cache and raw runs took ~70 min of wall time on
a loaded machine. The old venv (`downloads/torch-old/venv`) holds torch 2.8.0 + numpy 2.5.4 from
PyPI (macOS arm64 cp312 wheels).
