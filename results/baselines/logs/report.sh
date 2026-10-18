#!/bin/sh
# Merge the per-job parts into results/baselines/{networkx,pydoc_topics,enwik8-test}.json and baselines.md.
# With --wait PID...: first wait for those background jobs to exit (the cmix and paq8px chains).
cd "$(dirname "$0")/../../.."
if [ "${1:-}" = --wait ]; then shift; for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 60; done; done; fi
PY=.venv/bin/python
P=results/baselines/parts
ex() { for f in "$@"; do [ -f "$f" ] && echo "$f"; done; }
$PY -m jevzip.baselines --report /tmp/jz-nx.md --json results/baselines/networkx.json $(ex $P/fast-code.json $P/ppmd-code.json $P/paq8px-networkx.json $P/paq8px-primed-code.json $P/cmix-networkx.json) > /dev/null
$PY -m jevzip.baselines --report /tmp/jz-pd.md --json results/baselines/pydoc_topics.json $(ex $P/fast-code.json $P/ppmd-code.json $P/paq8px-pydoc_topics.json $P/paq8px-primed-code.json $P/cmix-pydoc_topics.json) > /dev/null
$PY -m jevzip.baselines --report /tmp/jz-en.md --json results/baselines/enwik8-test.json $(ex $P/fast-enwik8-test.json $P/ppmd-enwik8-test.json $P/paq8px-enwik8-test.json $P/paq8px-primed-enwik8-test.json $P/cmix-enwik8-test.json) > /dev/null
$PY -m jevzip.baselines --report results/baselines/tables.md $(ex $P/fast-enwik8-test.json $P/ppmd-enwik8-test.json $P/paq8px-enwik8-test.json $P/paq8px-primed-enwik8-test.json $P/cmix-enwik8-test.json $P/fast-code.json $P/ppmd-code.json $P/paq8px-networkx.json $P/paq8px-pydoc_topics.json $P/paq8px-primed-code.json $P/cmix-networkx.json $P/cmix-pydoc_topics.json) > /dev/null
missing=""
for f in fast-code fast-enwik8-test ppmd-code ppmd-enwik8-test paq8px-networkx paq8px-pydoc_topics paq8px-enwik8-test paq8px-primed-code paq8px-primed-enwik8-test cmix-networkx cmix-pydoc_topics cmix-enwik8-test; do
  if [ ! -f $P/$f.json ] || ! grep -q '"finished"' $P/$f.json; then missing="$missing $f"; fi
done
{ cat results/baselines/notes.md; echo; echo "_Generated $(date '+%Y-%m-%d %H:%M %Z') by results/baselines/logs/report.sh. Runs not finished at that time (their rows are missing or partial):${missing:- none}._"; echo; cat results/baselines/tables.md; } > results/baselines/baselines.md
rm -f /tmp/jz-nx.md /tmp/jz-pd.md /tmp/jz-en.md
echo "report written $(date)"
