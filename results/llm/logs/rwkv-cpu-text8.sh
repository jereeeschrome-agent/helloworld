#!/bin/bash
# RWKV-4-169M on text8 test (stream + LMIC chunks) on CPU (4 threads, nice), background run by agent F.
cd /Users/agent/helloworld-jevzip
export HF_HOME=/Users/agent/helloworld-jevzip/downloads/hf
P="nice -n 10 .venv/bin/python -I tools/llm_codelength.py --model rwkv-169m --device cpu --threads 4"
$P data/text8/test.bin --method stream --json results/llm/runs/rwkv-169m-stream-text8-cpu.json
$P data/text8/test.bin --method chunks --batch 8 --json results/llm/runs/rwkv-169m-lmic-chunks-text8-cpu.json
echo done
