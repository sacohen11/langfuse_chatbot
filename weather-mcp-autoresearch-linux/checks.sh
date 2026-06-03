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

for name in ["weather.py", "setup_langfuse.py", "evaluate_tool_spec.py", "save_best_tool_spec.py"]:
    ast.parse(Path(name).read_text(encoding="utf-8"), filename=name)

spec = json.loads(Path("weather_tool_candidate.json").read_text(encoding="utf-8"))
if "tools" not in spec or not isinstance(spec["tools"], list) or not spec["tools"]:
    raise SystemExit("Tool spec must include a non-empty tools array")
names = {tool.get("name") for tool in spec["tools"]}
if names != {"get-alerts", "get-forecast"}:
    raise SystemExit(f"Expected get-alerts and get-forecast tools, got {sorted(names)}")
for tool in spec["tools"]:
  for field in ["name", "description", "input_schema", "output_schema"]:
    if field not in tool:
        raise SystemExit(f"Missing required tool field: {field}")
  if tool["input_schema"].get("type") != "object":
      raise SystemExit(f"{tool['name']} input_schema.type must be object")
  if tool["output_schema"].get("type") != "object":
      raise SystemExit(f"{tool['name']} output_schema.type must be object")
  if not isinstance(tool["input_schema"].get("properties"), dict):
      raise SystemExit(f"{tool['name']} input_schema.properties must be an object")
  if not isinstance(tool["output_schema"].get("properties"), dict):
      raise SystemExit(f"{tool['name']} output_schema.properties must be an object")
PY
