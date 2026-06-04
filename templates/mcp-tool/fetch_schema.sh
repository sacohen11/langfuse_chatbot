#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

ROOT="$PWD"
while [ ! -f "$ROOT/fetch_schema.py" ] && [ "$ROOT" != "/" ]; do
  ROOT="$(dirname "$ROOT")"
done

if [ ! -f "$ROOT/fetch_schema.py" ]; then
  echo "Could not find fetch_schema.py. Run this from inside the repo."
  exit 1
fi

python "$ROOT/fetch_schema.py" "$PWD/config.json"
