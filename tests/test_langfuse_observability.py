from langfuse_chatbot.langfuse_observability import build_langchain_config, sanitize_langfuse_metadata


def test_sanitize_langfuse_metadata_filters_secrets_and_stringifies_values():
    metadata = sanitize_langfuse_metadata(
        {
            "scenario_id": "math_memory",
            "tool-count": 2,
            "api_key": "should-not-leak",
            "notes": "x" * 250,
            "tags": ["chat", "eval"],
        }
    )

    assert metadata["scenario_id"] == "math_memory"
    assert metadata["tool_count"] == "2"
    assert metadata["tags"] == "chat,eval"
    assert len(metadata["notes"]) == 200
    assert "api_key" not in metadata


def test_build_langchain_config_sets_trace_name_session_user_and_tags():
    config = build_langchain_config(
        enabled=False,
        session_id="session-1",
        user_id="user-1",
        tags=["web-chat"],
        metadata={"scenario-id": "s1"},
        trace_name="chat-response",
    )

    assert config["run_name"] == "chat-response"
    assert config["tags"] == ["web-chat"]
    assert config["metadata"]["scenario_id"] == "s1"
    assert config["metadata"]["langfuse_session_id"] == "session-1"
    assert config["metadata"]["langfuse_user_id"] == "user-1"
    assert config["metadata"]["langfuse_tags"] == ["web-chat"]
