#!/bin/bash
cd /Users/agent/helloworld-jevzip/runs/code
# equal compute with the lanes run: same number of optimizer steps (wall-clock varied with machine load)
N=$(grep -E "^step [0-9]+ VAL" ../logs/10-train-lanes11m-enwik8.log | tail -1 | awk '{print $2}')
[ -n "$N" ] || { echo "no step count from job 10"; exit 1; }
echo "AR control: $N steps (matching job 10)"
/Users/agent/helloworld-jevzip/.venv/bin/python -m jevzip.train --data /Users/agent/helloworld-jevzip/data/enwik8 --device mps --ctx 512 --bs 32 --lr 1e-3 --no-amp --val-batches 16 --order linear --kmin 512 --kmax 512 --val-rounds 512 --d-model 384 --n-layers 6 --n-heads 6 --warmup 500 --steps $N --minutes 600 --out /Users/agent/helloworld-jevzip/weights/jevzip-ar-11m-enwik8.pt
