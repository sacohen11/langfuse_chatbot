from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
ENV = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to MCP config.json")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config_path = Path(args.config).resolve()
    config = expand(json.loads(config_path.read_text(encoding="utf-8-sig")))
    server = config["server"]

    if server.get("schema_file"):
        schema = json.loads((config_path.parent / server["schema_file"]).read_text(encoding="utf-8-sig"))
    else:
        headers = {k: v for k, v in server.get("headers", {}).items() if v}
        request = urllib.request.Request(server["schema_url"], headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            schema = json.loads(response.read().decode("utf-8"))

    tool_spec = normalize(schema, server)
    output = config_path.parent / config["candidate_file"]
    output.write_text(json.dumps(tool_spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def normalize(schema: dict, server: dict) -> dict:
    if "server" in schema and "tools" in schema:
        return schema
    if "tools" in schema:
        return {
            "server": server_summary(server),
            "tools": schema["tools"],
        }
    if "name" in schema:
        return {
            "server": server_summary(server),
            "tools": [schema],
        }
    raise ValueError("Schema must be {server, tools}, {tools}, or a single tool object.")


def server_summary(server: dict) -> dict:
    return {
        "name": server["name"],
        "url": server.get("url", ""),
        "deployment": {
            "platform": "rancher/kubernetes",
            **server.get("rancher", {}),
        },
    }


def expand(value):
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, str):
        return ENV.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), value)
    return value


if __name__ == "__main__":
    main()
