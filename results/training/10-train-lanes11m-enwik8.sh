#!/bin/bash
cd /Users/agent/helloworld-jevzip/runs/code
/Users/agent/helloworld-jevzip/.venv/bin/python -m jevzip.train --data /Users/agent/helloworld-jevzip/data/enwik8 --device mps --ctx 512 --bs 32 --lr 1e-3 --no-amp --val-batches 16 --order lanes --kmin 8 --kmax 512 --val-rounds 32 --d-model 384 --n-layers 6 --n-heads 6 --warmup 500 --minutes 480 --out /Users/agent/helloworld-jevzip/weights/jevzip-lanes-11m-enwik8.pt
