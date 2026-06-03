from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langfuse_chatbot.chatbot import ChatResponse, LangChainMCPChatbot
from langfuse_chatbot.config import AppConfig, load_config
from langfuse_chatbot.evaluation.runner import evaluate_chatbot, load_scenarios


ChatbotFactory = Callable[[AppConfig], Awaitable[Any]]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class EvalRequest(BaseModel):
    scenarios_path: str = "evals/scenarios/tool_multiturn.json"


@dataclass(frozen=True)
class ChatEvent:
    timestamp: str
    session_id: str
    latency_ms: float
    input_chars: int
    output_chars: int
    tool_names: list[str]


@dataclass
class WebMetrics:
    chat_events: list[ChatEvent] = field(default_factory=list)
    last_eval: dict[str, Any] | None = None
    errors: list[dict[str, str]] = field(default_factory=list)

    def record_chat(self, event: ChatEvent) -> None:
        self.chat_events.append(event)
        self.chat_events = self.chat_events[-200:]

    def record_error(self, where: str, message: str) -> None:
        self.errors.append({"where": where, "message": message, "timestamp": _now_iso()})
        self.errors = self.errors[-50:]

    def summary(self, config: AppConfig) -> dict[str, Any]:
        sessions = {event.session_id for event in self.chat_events}
        latencies = [event.latency_ms for event in self.chat_events]
        tool_counts: dict[str, int] = {}
        for event in self.chat_events:
            for name in event.tool_names:
                tool_counts[name] = tool_counts.get(name, 0) + 1

        return {
            "chat": {
                "turns": len(self.chat_events),
                "sessions": len(sessions),
                "tool_calls": sum(tool_counts.values()),
                "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "recent_events": [event.__dict__ for event in reversed(self.chat_events[-12:])],
                "tool_counts": tool_counts,
            },
            "observability": {
                "langfuse_enabled": config.langfuse.enabled,
                "langfuse_configured": config.langfuse.has_credentials,
                "langfuse_host": config.langfuse.host,
                "mcp_servers": sorted(config.enabled_mcp_servers.keys()),
                "mcp_tool_name_prefix": config.mcp_tool_name_prefix,
            },
            "evaluation": self.last_eval,
            "errors": list(reversed(self.errors[-8:])),
        }


def create_app(
    config_path: str = "config.example.yaml",
    chatbot_factory: ChatbotFactory | None = None,
) -> FastAPI:
    config = load_config(config_path)
    metrics = WebMetrics()
    static_dir = Path(__file__).parent / "static"

    app = FastAPI(title="Langfuse Chatbot Console")
    app.mount("/assets", StaticFiles(directory=static_dir), name="assets")
    app.state.config = config
    app.state.chatbot = None
    app.state.metrics = metrics
    app.state.chatbot_factory = chatbot_factory or _default_chatbot_factory

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/metrics")
    async def get_metrics() -> dict[str, Any]:
        return metrics.summary(config)

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> dict[str, Any]:
        session_id = request.session_id or str(uuid4())
        started = time.perf_counter()
        try:
            chatbot = await _get_chatbot(app)
            response: ChatResponse = await chatbot.achat(
                request.message,
                session_id=session_id,
                user_id="web-user",
                tags=["web-chat"],
            )
        except Exception as exc:  # noqa: BLE001
            metrics.record_error("chat", str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        latency_ms = (time.perf_counter() - started) * 1000
        tool_names = [call.name for call in response.tool_calls]
        metrics.record_chat(
            ChatEvent(
                timestamp=_now_iso(),
                session_id=response.session_id,
                latency_ms=round(latency_ms, 1),
                input_chars=len(request.message),
                output_chars=len(response.content),
                tool_names=tool_names,
            )
        )
        return {
            "content": response.content,
            "session_id": response.session_id,
            "tool_calls": [{"name": call.name, "args": call.args, "id": call.id} for call in response.tool_calls],
            "latency_ms": round(latency_ms, 1),
        }

    @app.post("/api/evaluate")
    async def evaluate(request: EvalRequest) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            scenarios = load_scenarios(_resolve_workspace_path(request.scenarios_path))
            chatbot = await _get_chatbot(app)
            report = await evaluate_chatbot(chatbot, scenarios, user_id="web-eval-user")
        except Exception as exc:  # noqa: BLE001
            metrics.record_error("evaluation", str(exc))
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        payload = report.as_dict()
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        payload["scenario_count"] = len(scenarios)
        payload["timestamp"] = _now_iso()
        metrics.last_eval = payload
        return payload

    return app


async def _default_chatbot_factory(config: AppConfig) -> LangChainMCPChatbot:
    return await LangChainMCPChatbot.from_config(config)


async def _get_chatbot(app: FastAPI) -> Any:
    if app.state.chatbot is None:
        app.state.chatbot = await app.state.chatbot_factory(app.state.config)
    return app.state.chatbot


def _resolve_workspace_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
