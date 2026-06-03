from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")
DEFAULT_MODEL = "openai:gpt-4.1-mini"
DEFAULT_LANGFUSE_HOST = "https://cloud.langfuse.com"
DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant. Use tools when they improve correctness "
    "and keep track of the conversation."
)


def _coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    if lowered == "":
        return ""

    try:
        if "." in lowered:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _interpolate(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _interpolate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.getenv(name, default)

    return _coerce_scalar(_ENV_PATTERN.sub(replace, value))


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.0


@dataclass(frozen=True)
class LangfuseConfig:
    enabled: bool = True
    host: str = DEFAULT_LANGFUSE_HOST
    public_key: str = ""
    secret_key: str = ""

    @property
    def has_credentials(self) -> bool:
        return bool(self.public_key and self.secret_key)


@dataclass(frozen=True)
class AgentConfig:
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class AutoresearchConfig:
    workspace_dir: str = "autoresearch"
    candidate_count: int = 3


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    langfuse: LangfuseConfig = field(default_factory=LangfuseConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    autoresearch: AutoresearchConfig = field(default_factory=AutoresearchConfig)
    mcp_servers: dict[str, dict[str, Any]] = field(default_factory=dict)
    mcp_tool_name_prefix: bool = False

    @property
    def enabled_mcp_servers(self) -> dict[str, dict[str, Any]]:
        enabled: dict[str, dict[str, Any]] = {}
        for name, server in self.mcp_servers.items():
            if server.get("enabled", True):
                connection = dict(server)
                connection.pop("enabled", None)
                enabled[name] = connection
        return enabled


def load_config(path: str | Path = "config.example.yaml") -> AppConfig:
    load_dotenv()
    config_path = Path(path)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    data = _interpolate(raw)
    llm = data.get("llm", {})
    langfuse = data.get("langfuse", {})
    agent = data.get("agent", {})
    autoresearch = data.get("autoresearch", {})
    mcp = data.get("mcp", {})

    return AppConfig(
        llm=LLMConfig(
            model=str(llm.get("model", os.getenv("CHATBOT_MODEL", DEFAULT_MODEL))),
            temperature=float(llm.get("temperature", os.getenv("CHATBOT_TEMPERATURE", 0))),
        ),
        langfuse=LangfuseConfig(
            enabled=bool(langfuse.get("enabled", True)),
            host=str(
                langfuse.get(
                    "base_url",
                    langfuse.get("host", os.getenv("LANGFUSE_BASE_URL", os.getenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST))),
                )
            ),
            public_key=str(langfuse.get("public_key", os.getenv("LANGFUSE_PUBLIC_KEY", ""))),
            secret_key=str(langfuse.get("secret_key", os.getenv("LANGFUSE_SECRET_KEY", ""))),
        ),
        agent=AgentConfig(system_prompt=str(agent.get("system_prompt", DEFAULT_SYSTEM_PROMPT))),
        autoresearch=AutoresearchConfig(
            workspace_dir=str(autoresearch.get("workspace_dir", "autoresearch")),
            candidate_count=int(autoresearch.get("candidate_count", os.getenv("AUTORESEARCH_CANDIDATES", 3))),
        ),
        mcp_servers=dict(mcp.get("servers", {})),
        mcp_tool_name_prefix=bool(mcp.get("tool_name_prefix", False)),
    )
