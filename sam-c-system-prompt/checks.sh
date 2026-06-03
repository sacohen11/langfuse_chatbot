#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  for candidate in \
    "C:/Python313/python.exe" \
    "/c/Python313/python.exe" \
    "/mnt/c/Python313/python.exe" \
    "python.exe" \
    "python"
  do
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

for name in ["setup_langfuse.py", "evaluate_prompt.py", "save_optimized_prompt.py", "save_best_prompt.py"]:
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)
PY
test -s prompt_candidate.txt
