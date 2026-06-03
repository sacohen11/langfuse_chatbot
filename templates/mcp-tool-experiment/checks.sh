#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in ".venv/bin/python" "/mnt/c/Python313/python.exe" "C:/Python313/python.exe" "python3" "python"; do
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

for name in ["sync_tool_schema.py", "run_tool_experiment.py"]:
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)
json.loads(Path("mcp_tool_config.json").read_text(encoding="utf-8"))
json.loads(Path("tool_candidate.json").read_text(encoding="utf-8"))
PY

"$PYTHON_BIN" sync_tool_schema.py --validate-config
"$PYTHON_BIN" run_tool_experiment.py --validate-config
