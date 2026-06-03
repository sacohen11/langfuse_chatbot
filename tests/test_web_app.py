from pathlib import Path

from fastapi.testclient import TestClient

from langfuse_chatbot.chatbot import ChatResponse, ToolCallRecord
from langfuse_chatbot.config import AppConfig
from langfuse_chatbot.web.app import create_app


class FakeWebChatbot:
    def __init__(self) -> None:
        self.tool_calls: list[ToolCallRecord] = []

    async def achat(self, user_text: str, *, session_id: str | None = None, **kwargs: object) -> ChatResponse:
        if "weather" in user_text.lower():
            self.tool_calls.append(ToolCallRecord(name="get_weather", args={"city": "Boston"}))
            content = "Boston has light rain. Pack an umbrella."
        elif "umbrella" in user_text.lower():
            content = "Yes, pack an umbrella based on the earlier Boston weather."
        else:
            content = "Hello from the fake bot."
        return ChatResponse(content=content, session_id=session_id or "web-test", tool_calls=list(self.tool_calls))


async def fake_factory(config: AppConfig) -> FakeWebChatbot:
    return FakeWebChatbot()


def test_web_chat_records_metrics(tmp_path):
    config_path = _write_config(tmp_path)
    app = create_app(str(config_path), chatbot_factory=fake_factory)

    with TestClient(app) as client:
        response = client.post("/api/chat", json={"message": "Check weather", "session_id": "s1"})
        assert response.status_code == 200
        assert response.json()["tool_calls"][0]["name"] == "get_weather"

        metrics = client.get("/api/metrics").json()
        assert metrics["chat"]["turns"] == 1
        assert metrics["chat"]["tool_calls"] == 1


def test_web_evaluate_returns_report(tmp_path):
    config_path = _write_config(tmp_path)
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        """
[
  {
    "id": "weather_context",
    "description": "Use weather then remember it.",
    "turns": [
      {"user": "Check weather in Boston", "answer_contains": ["Boston"], "required_tools": ["get_weather"]},
      {"user": "Should I pack an umbrella?", "answer_contains": ["umbrella"], "forbidden_tools": ["get_weather"]}
    ]
  }
]
""",
        encoding="utf-8",
    )
    app = create_app(str(config_path), chatbot_factory=fake_factory)

    with TestClient(app) as client:
        response = client.post("/api/evaluate", json={"scenarios_path": str(scenarios_path)})
        assert response.status_code == 200
        payload = response.json()
        assert payload["overall_score"] == 1.0
        assert payload["scenario_count"] == 1


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
llm:
  model: openai:gpt-4.1-mini
langfuse:
  enabled: false
mcp:
  servers: {}
""",
        encoding="utf-8",
    )
    return config_path
