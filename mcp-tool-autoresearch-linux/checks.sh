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

"$PYTHON_BIN" - <<'PY'
import ast
import json
from pathlib import Path

for name in ["setup_langfuse.py", "evaluate_tool_spec.py", "save_best_tool_spec.py"]:
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)

spec = json.loads(Path("tool_candidate.json").read_text(encoding="utf-8"))
for field in ["name", "description", "input_schema", "output_schema"]:
    if field not in spec:
        raise SystemExit(f"Missing required tool spec field: {field}")
if spec["input_schema"].get("type") != "object":
    raise SystemExit("input_schema.type must be object")
if spec["output_schema"].get("type") != "object":
    raise SystemExit("output_schema.type must be object")
if not isinstance(spec["input_schema"].get("properties"), dict):
    raise SystemExit("input_schema.properties must be an object")
if not isinstance(spec["output_schema"].get("properties"), dict):
    raise SystemExit("output_schema.properties must be an object")
PY
