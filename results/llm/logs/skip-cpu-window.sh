#!/bin/bash
# Cancels the RWKV window step of rwkv-cpu.sh (CPU contention); the MPS job runs it instead.
cd /Users/agent/helloworld-jevzip
pat="window""-cpu.json"
until pgrep -f "llm_codelength.py.*$pat" >/dev/null || ! pgrep -f rwkv-cpu.sh >/dev/null; do sleep 5; done
pkill -f "llm_codelength.py.*$pat" && echo "$(date) window step cancelled on CPU (machine contended): runs on MPS via runs/pending/40-llm.sh" >> results/llm/logs/rwkv-cpu.log
