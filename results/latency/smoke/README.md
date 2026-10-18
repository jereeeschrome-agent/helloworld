# Latency smoke tests (provisional, 2026-10-18 ~13:00)

Not the benchmark: the real run is `runs/pending/20-latency.sh` (GPU queue), which writes `results/latency/latency-{mps,cpu}.json` + `latency.md`.
These ran while the AR-control training job held the MPS device and other agents used the CPU, on tiny slices, 1 timed run: the
times are inflated and only show that the path works. Sizes (bits/byte) are exact.

- `latency-mps.json`: networkx 16K (32 blocks), K=16/128, both layouts. Model calls stay on MPS (the old CPU-built masks crashed
  there); probabilities were bit-identical across runs; a true same-device round trip (encode with MPS probabilities, decode on MPS)
  reproduced the input at K=16.
- MPS per-round probe (separate call, not in the JSON; K=512-style round = 1 new byte + 1 slot per block, 16 rounds): bare op + copy
  to host 0.79 ms; 1 block 16.7 ms per round with the probability copy, 9.2 ms when rounds are queued without waiting; 32 blocks 33.9 ms;
  128 blocks 48.3 ms. One round is 518 aten ops (~18 us each at 1 block), so small batches are bound by kernel launches plus the
  per-round sync, not arithmetic (contended GPU: to be re-measured by the queued run).
- `latency-cpu.json`: networkx 8K (16 blocks), K=16/512, both layouts, 4 threads (got ~1 core: shared machine); CPU round trip ok.
- `refcheck.json` / `refcheck.log`: compress() with the new model.py reproduces runs/ref fp32 K=32, fp32 K=128 and bf16 K=32
  byte-for-byte (bf16 K=128 not re-run).

GPU time used by these smoke tests: ~66 s (14 s probe + 52 s benchmark), slightly over the 60 s allowance.
