from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from .config import AppConfig
from .langfuse_observability import (
    build_langchain_config,
    configure_langfuse,
    flush_langfuse,
    langfuse_observation,
    propagate_langfuse_attributes,
    sanitize_langfuse_metadata,
)
from .mcp_client import load_mcp_tools


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


@dataclass(frozen=True)
class ChatResponse:
    content: str
    session_id: str
    tool_calls: list[ToolCallRecord]
    raw_messages: list[Any] = field(default_factory=list)


class LangChainMCPChatbot:
    """Small stateful wrapper around a LangChain agent and MCP tools."""

    def __init__(
        self,
        *,
        agent: Any,
        langfuse_enabled: bool,
        system_prompt: str,
        model_name: str,
    ) -> None:
        self._agent = agent
        self._langfuse_enabled = langfuse_enabled
        self._system_prompt = system_prompt
        self._model_name = model_name
        self._sessions: dict[str, list[Any]] = {}

    @classmethod
    async def from_config(cls, config: AppConfig) -> "LangChainMCPChatbot":
        from langchain.agents import create_agent
        from langchain.chat_models import init_chat_model

        _, tools = await load_mcp_tools(
            config.enabled_mcp_servers,
            tool_name_prefix=config.mcp_tool_name_prefix,
        )
        langfuse_enabled = configure_langfuse(config.langfuse)
        model = init_chat_model(config.llm.model, temperature=config.llm.temperature)
        agent = create_agent(model, tools, system_prompt=config.agent.system_prompt)
        return cls(
            agent=agent,
            langfuse_enabled=langfuse_enabled,
            system_prompt=config.agent.system_prompt,
            model_name=config.llm.model,
        )

    async def achat(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatResponse:
        session = session_id or str(uuid4())
        trace_name = "chat-response"
        trace_tags = tags or ["chat"]
        trace_metadata = {
            "model": self._model_name,
            "session_id": session,
            **(metadata or {}),
        }
        messages = list(self._sessions.get(session, []))
        messages.append({"role": "user", "content": user_text})
        raw_messages: list[Any] = []
        content = ""
        tool_calls: list[ToolCallRecord] = []

        with langfuse_observation(
            enabled=self._langfuse_enabled,
            name=trace_name,
            input={"message": user_text},
            metadata=trace_metadata,
        ) as span:
            with propagate_langfuse_attributes(
                enabled=self._langfuse_enabled,
                trace_name=trace_name,
                session_id=session,
                user_id=user_id,
                tags=trace_tags,
                metadata=trace_metadata,
            ):
                result = await self._agent.ainvoke(
                    {"messages": messages},
                    config=build_langchain_config(
                        enabled=self._langfuse_enabled,
                        session_id=session,
                        user_id=user_id,
                        tags=trace_tags,
                        metadata=trace_metadata,
                        trace_name=trace_name,
                    ),
                )
            raw_messages = list(result.get("messages", []))
            content = _last_ai_content(raw_messages)
            tool_calls = _extract_tool_calls(raw_messages)
            if span is not None:
                span.update(
                    output={"content": content},
                    metadata=sanitize_langfuse_metadata(
                        {
                            **trace_metadata,
                            "tool_names": [call.name for call in tool_calls],
                            "tool_call_count": len(tool_calls),
                        }
                    ),
                )

        self._sessions[session] = raw_messages
        return ChatResponse(content=content, session_id=session, tool_calls=tool_calls, raw_messages=raw_messages)

    def flush_traces(self) -> None:
        flush_langfuse(self._langfuse_enabled)


def _last_ai_content(messages: list[Any]) -> str:
    for message in reversed(messages):
        message_type = getattr(message, "type", None) or getattr(message, "role", None)
        if message_type in {"ai", "assistant"}:
            content = getattr(message, "content", "")
            return _stringify_content(content)
    return ""


def _stringify_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def _extract_tool_calls(messages: list[Any]) -> list[ToolCallRecord]:
    records: list[ToolCallRecord] = []
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None) or []
        for call in tool_calls:
            if isinstance(call, dict):
                records.append(
                    ToolCallRecord(
                        name=str(call.get("name", "")),
                        args=dict(call.get("args") or {}),
                        id=call.get("id"),
                    )
                )
    return records
