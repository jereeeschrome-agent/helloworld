#!/bin/bash
# RWKV-4-169M codelengths on CPU (4 threads, nice), run in the background by agent F.
cd /Users/agent/helloworld-jevzip
export HF_HOME=/Users/agent/helloworld-jevzip/downloads/hf
P="nice -n 10 .venv/bin/python -I tools/llm_codelength.py --model rwkv-169m --device cpu --threads 4"
F="data/enwik8/test.bin data/code/networkx.py.txt data/code/pydoc_topics.txt"
$P $F --method stream --json results/llm/runs/rwkv-169m-stream-cpu.json
$P data/enwik8/test.bin --method chunks --batch 8 --json results/llm/runs/rwkv-169m-lmic-chunks-cpu.json
$P $F --method window --window 1024 --stride 512 --batch 2 --json results/llm/runs/rwkv-169m-window-cpu.json
echo done
