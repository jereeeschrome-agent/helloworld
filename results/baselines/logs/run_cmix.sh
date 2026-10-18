#!/bin/sh
# cmix on the three test files, one at a time (~19 GB RAM each).
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
for f in networkx.py.txt pydoc_topics.txt; do
  nice -n 10 $PY -m jevzip.baselines data/code/$f --only cmix --json results/baselines/parts/cmix-${f%%.*}.json
done
nice -n 10 $PY -m jevzip.baselines data/enwik8/test.bin --only cmix --json results/baselines/parts/cmix-enwik8-test.json
