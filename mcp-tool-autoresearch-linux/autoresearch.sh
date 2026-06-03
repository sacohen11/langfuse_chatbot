#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in ".venv/bin/python" "python3" "python"; do
    if [ -x "$candidate" ] || command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [ -z "$PYTHON_BIN" ]; then
  echo "Could not find a usable Python interpreter" >&2
  exit 127
fi

RESULT_FILE=".autoresearch-last-result.json"
OUTPUT_FILE=".autoresearch-last-output.txt"
METRICS_FILE=".autoresearch-last-metrics.txt"
rm -f "$RESULT_FILE" "$OUTPUT_FILE" "$METRICS_FILE"

bash checks.sh

if ! "$PYTHON_BIN" evaluate_tool_spec.py --spec-file tool_candidate.json --publish --result-file "$RESULT_FILE" > "$OUTPUT_FILE" 2>&1; then
  cat "$OUTPUT_FILE"
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

result = json.loads(Path(".autoresearch-last-result.json").read_text(encoding="utf-8"))
scores = result["dimension_scores"]
lines = [f"METRIC overall_score={float(result['overall_score']):.4f}"]
for name in ("invocation_accuracy", "input_schema_quality", "output_schema_quality", "description_clarity"):
    lines.append(f"METRIC {name}={float(scores[name]):.4f}")
lines.append(f"INFO prompt_version={result.get('prompt_attempt', {}).get('prompt_version')}")
lines.append(f"INFO dataset_run_url={result.get('dataset_run_url')}")
Path(".autoresearch-last-metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

cat "$METRICS_FILE"
