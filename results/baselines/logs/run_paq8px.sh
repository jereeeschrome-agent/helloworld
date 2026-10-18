#!/bin/sh
# paq8px: standalone -8L on the three test files, then the primed (-8, P = last 16 MiB) rows.
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
for f in networkx.py.txt pydoc_topics.txt; do
  nice -n 10 $PY -m jevzip.baselines data/code/$f --only 'paq8px -8L' --json results/baselines/parts/paq8px-${f%%.*}.json
done
nice -n 10 $PY -m jevzip.baselines data/enwik8/test.bin --only 'paq8px -8L' --json results/baselines/parts/paq8px-enwik8-test.json
nice -n 10 $PY -m jevzip.baselines data/code/networkx.py.txt data/code/pydoc_topics.txt --corpus data/code/train.bin --only 'paq8px -8 primed' --json results/baselines/parts/paq8px-primed-code.json
nice -n 10 $PY -m jevzip.baselines data/enwik8/test.bin --corpus data/enwik8/train.bin --only 'paq8px -8 primed' --json results/baselines/parts/paq8px-primed-enwik8-test.json
