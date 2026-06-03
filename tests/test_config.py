from pathlib import Path

from langfuse_chatbot.config import load_config


def test_load_config_interpolates_env_and_filters_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "secret")
    path = tmp_path / "config.yaml"
    path.write_text(
        """
llm:
  model: openai:gpt-4.1-mini
  temperature: 0
mcp:
  servers:
    enabled_server:
      enabled: true
      transport: http
      url: https://example.test/mcp
      headers:
        Authorization: Bearer ${MY_TOKEN}
    disabled_server:
      enabled: false
      transport: stdio
      command: ignored
""",
        encoding="utf-8",
    )

    config = load_config(Path(path))

    assert config.llm.model == "openai:gpt-4.1-mini"
    assert config.enabled_mcp_servers == {
        "enabled_server": {
            "transport": "http",
            "url": "https://example.test/mcp",
            "headers": {"Authorization": "Bearer secret"},
        }
    }
