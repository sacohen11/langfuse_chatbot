#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ROOT="$PWD"
while [ ! -f "$ROOT/run_experiment.py" ] && [ "$ROOT" != "/" ]; do
  ROOT="$(dirname "$ROOT")"
done

if [ ! -f "$ROOT/run_experiment.py" ]; then
  echo "Could not find run_experiment.py. Run this from inside the repo."
  exit 1
fi

python "$ROOT/run_experiment.py" "$PWD/benchmark_config.json" > .autoresearch-last-output.txt
cat .autoresearch-last-output.txt
grep '^METRIC ' .autoresearch-last-output.txt > .autoresearch-last-metrics.txt || true
