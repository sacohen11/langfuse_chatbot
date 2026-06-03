from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync a hosted MCP tool schema JSON into tool_candidate.json.")
    parser.add_argument("--config", default="mcp_tool_config.json", help="MCP tool config JSON.")
    parser.add_argument("--fetch-schema", action="store_true", help="Fetch schema_url or schema_file and write candidate_tool_spec_file.")
    parser.add_argument("--check-health", action="store_true", help="Call server.health_url and fail if it is not reachable.")
    parser.add_argument("--validate-config", action="store_true", help="Validate config shape and exit.")
    args = parser.parse_args()

    load_dotenv(_repo_root() / ".env")
    load_dotenv(HERE / ".env")

    config_path = _resolve_path(args.config)
    config = _expand_env(_load_config(config_path))

    if args.validate_config:
        print(json.dumps({"ok": True, "server": config["server"]["name"]}, indent=2))
        return

    if args.check_health:
        health_url = config["server"].get("health_url")
        if not health_url:
            raise ValueError("server.health_url is required for --check-health")
        _fetch_url(health_url, config["server"])
        print(json.dumps({"ok": True, "health_url": health_url}, indent=2))

    if args.fetch_schema:
        schema = _load_schema(config)
        schema = _extract_schema_path(schema, config["server"].get("schema_json_path", ""))
        normalized = _normalize_schema(schema, config["server"])
        output_path = _resolve_path(config.get("candidate_tool_spec_file", "tool_candidate.json"))
        output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "tool_spec_file": str(output_path), "tool_count": len(_tools(normalized))}, indent=2))

    if not args.fetch_schema and not args.check_health:
        parser.error("Choose --fetch-schema, --check-health, or --validate-config")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required = ["participant_name", "experiment_name", "prompt_name", "dataset_name", "task_model", "judge_model", "server"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    server = config["server"]
    for key in ["name", "mcp_url"]:
        if not server.get(key):
            raise ValueError(f"server.{key} is required")
    if not server.get("schema_url") and not server.get("schema_file"):
        raise ValueError("server.schema_url or server.schema_file is required")
    return config


def _load_schema(config: dict[str, Any]) -> Any:
    server = config["server"]
    if server.get("schema_file"):
        schema_path = _resolve_path(server["schema_file"])
        return json.loads(schema_path.read_text(encoding="utf-8-sig"))
    return json.loads(_fetch_url(server["schema_url"], server).decode("utf-8"))


def _fetch_url(url: str, server: dict[str, Any]) -> bytes:
    headers = {key: value for key, value in server.get("headers", {}).items() if value}
    request = urllib.request.Request(url, headers=headers)
    context = None if server.get("tls_verify", True) else ssl._create_unverified_context()
    with urllib.request.urlopen(request, timeout=30, context=context) as response:
        return response.read()


def _extract_schema_path(schema: Any, path: str) -> Any:
    if not path:
        return schema
    current = schema
    for part in path.split("."):
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, list):
            current = current[int(part)]
        else:
            raise ValueError(f"Cannot descend into schema_json_path segment {part!r}")
    return current


def _normalize_schema(schema: Any, server: dict[str, Any]) -> dict[str, Any]:
    if isinstance(schema, dict) and "server" in schema and "tools" in schema:
        return schema
    if isinstance(schema, dict) and "tools" in schema:
        return {
            "server": {
                "name": server["name"],
                "description": schema.get("description", f"Hosted MCP server at {server['mcp_url']}"),
                "mcp_url": server["mcp_url"],
                "deployment": _deployment_metadata(server),
            },
            "tools": schema["tools"],
        }
    if isinstance(schema, dict) and {"name", "description"}.issubset(schema):
        return {
            "server": {
                "name": server["name"],
                "description": f"Hosted MCP server at {server['mcp_url']}",
                "mcp_url": server["mcp_url"],
                "deployment": _deployment_metadata(server),
            },
            "tools": [schema],
        }
    raise ValueError("Schema must contain either {server, tools}, {tools}, or a single tool object.")


def _deployment_metadata(server: dict[str, Any]) -> dict[str, Any]:
    keys = ["platform", "environment", "cluster", "namespace", "workload", "transport"]
    return {key: server.get(key, "") for key in keys if server.get(key)}


def _tools(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "tools" in spec:
        return list(spec["tools"])
    return [spec]


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2) or ""
            return os.getenv(name, default)
        return ENV_PATTERN.sub(replace, value)
    return value


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else HERE / candidate


def _repo_root() -> Path:
    current = HERE
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    return HERE.parent.parent


if __name__ == "__main__":
    main()
