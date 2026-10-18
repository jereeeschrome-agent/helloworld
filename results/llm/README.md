# LLM-plus-arithmetic-coding baseline

"Language Modeling Is Compression" (Deletang et al., ICLR 2024) style baseline: the ideal codelength
sum(-log2 p(token)) of a file under a pretrained LM, which an arithmetic coder reaches within a few bytes per
coded sequence. All numbers here are labelled **LLM codelength (arithmetic coding adds < 0.01%)**: each record
carries `coder_overhead_est_rel` (31-bit frequency tables + 32 flush bits per sequence), at most 2.3e-5 for
whole-file runs. The LMIC 2048-byte chunk runs code ~2400 sequences, so their records carry the label
"arithmetic coding adds up to 1.40%" (a conservative 32 flush bits per chunk); rounding each chunk up to whole
bytes, as LMIC does, costs +0.18% (`codelength_bytes_whole_bytes_per_chunk`). Nothing
is actually encoded; a real compressor would also need bit-identical model evaluation on both sides.

Tool: `tools/llm_codelength.py` (run with `python -I`, `HF_HOME=downloads/hf`, offline, pinned revisions).
Bits are per ORIGINAL byte; the first token is charged (sequences start with `<|endoftext|>`, which both
models use as document separator). Tokenization is byte-exact: the NFC normalizer of both tokenizers is
removed and `decode(encode(text)) == text` is checked per file (all files here pass); non-UTF-8 files are
recorded as skipped. Model size for total-size accounting: RWKV-4-169M 169,342,464 params = 338.7 MB bf16;
Qwen2.5-0.5B 494,032,768 params = 988.1 MB bf16 (`total_with_model_bf16_bytes` in each record).

Context methods: `stream` (RWKV: whole file as one sequence, recurrent state carried, unbounded context),
`window` (W tokens, advance S, only new tokens scored), `chunks` (LMIC protocol: independent 2048-byte chunks
cut back to UTF-8 character starts, no cross-chunk context).

## Results so far (CPU, fp32, 4 threads; full table: `summary.md` / `summary.json`)

| file | model | method | bpb |
|---|---|---|---:|
| enwik8 test (5 MB) | RWKV-4-169M | stream | **1.0968** |
| enwik8 test | RWKV-4-169M | LMIC 2048-B chunks | 1.1188 |
| text8 test (5 MB) | RWKV-4-169M | stream | 1.2762 |
| text8 test | RWKV-4-169M | LMIC 2048-B chunks | 1.2763 |
| networkx.py.txt (1 MB) | RWKV-4-169M | stream | 0.8264 |
| pydoc_topics.txt (0.52 MB) | RWKV-4-169M | stream | 0.8690 |

Qwen2.5-0.5B full runs, RWKV sliding windows and all OOD files run on MPS via `runs/pending/40-llm.sh`
(not yet run when this was written). CPU validation slices (`validation/`): Qwen on the first 20 kB of enwik8
test 0.810 bpb (W1024/S512) vs 0.910 (LMIC chunks); first 20 kB of networkx 0.477 bpb (W2048/S1024).

**Sanity check vs ts_zip.** RWKV-4-169M (stream) gives 1.097 bpb on the enwik8 test split; ts_zip (same model
family, 8-bit weights, bf16, its own protocol) publishes 1.106 bpb on full enwik8 (`results/published/sota.json`).
Same ballpark (-0.8%); differences: 5 MB tail vs 100 MB whole file, fp32 vs quantized weights, ideal codelength
vs actual coded size. The LMIC protocol costs RWKV +2.0% on enwik8 (1.119 vs 1.097) but nothing on text8 (1.2763 vs 1.2762): on
lowercased, punctuation-free text8 (far from the Pile's style) the carried state gains nothing beyond 2 kB.

**Caveats.** Contamination: enwik8/text8 are Wikipedia (2006), and both models' pretraining data (the Pile for
RWKV-4: includes Wikipedia; Qwen2.5: undisclosed web-scale data) almost surely contains these articles or
near-copies; networkx 3.7 / pydoc_data / the OOD JSON and log files are public on GitHub and may be in Qwen's data.
These are codelengths of a 169M/494M-parameter model whose weights (339/988 MB) dwarf the files, so they are a
lower-bound-style reference for "how predictable is this text", not a competitive compressor size.

## Validation (`validation/validation.json`)

- `--selftest` (tiny random RWKV and Qwen2 models, CPU, ~3 s): chunk-parallel `wkv()` matches transformers'
  per-token RWKV loop (max err 2.8e-6); one window, one chunk and stream all equal a plain full-logits
  computation; padding/batching does not change window results; non-UTF-8 input is skipped.
- Real models, 2000-byte slices of enwik8 and networkx scored as one sequence by every method: bits agree to
  the printed 3 decimals (RWKV window = chunks = stream; Qwen window = chunks).
- 30 kB of enwik8, RWKV: stream 1.0511, window 1024/512 1.0456, LMIC chunks 1.1181 bpb. Windows of RWKV's
  training length are slightly better than an unbounded carried state here (0.5%).
- CPU throughput (machine shared with other agents' jobs, load ~12: provisional): RWKV 700-1600 tok/s
  (stream/chunks), Qwen ~350 scored tok/s with W/S=2 (~700 forward tok/s).

## Reproduce

    export HF_HOME=downloads/hf
    .venv/bin/python -I tools/llm_codelength.py --selftest
    .venv/bin/python -I tools/llm_codelength.py data/enwik8/test.bin --model rwkv-169m --method stream --json out.json
    bash runs/pending/40-llm.sh          # MPS, via the GPU queue
    .venv/bin/python -I tools/llm_codelength.py --summary results/llm

Exact commands, tool versions and input SHA-256s are inside every JSON in `runs/` and `validation/`;
CPU background drivers and logs are in `logs/`.
