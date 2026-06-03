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
from pathlib import Path

ast.parse(Path("run_experiment.py").read_text(encoding="utf-8"), filename="run_experiment.py")
PY

"$PYTHON_BIN" run_experiment.py --validate-config
test -s prompt_candidate.txt
