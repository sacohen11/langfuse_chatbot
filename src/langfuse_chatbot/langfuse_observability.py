from __future__ import annotations

from contextlib import contextmanager, nullcontext
import re
from typing import Any

from .config import LangfuseConfig


_SECRET_KEY_PATTERN = re.compile(r"(secret|token|password|authorization|api[_-]?key)", re.IGNORECASE)
_SAFE_METADATA_KEY_PATTERN = re.compile(r"[^a-zA-Z0-9_]")
_MAX_METADATA_VALUE_LENGTH = 200


def configure_langfuse(config: LangfuseConfig) -> bool:
    """Initialize Langfuse when credentials are available.

    Returns whether remote Langfuse publishing is active. Local tests and
    development should keep working without credentials.
    """

    if not config.enabled or not config.has_credentials:
        return False

    from langfuse import Langfuse

    Langfuse(
        public_key=config.public_key,
        secret_key=config.secret_key,
        base_url=config.host,
    )
    return True


def build_langchain_config(
    *,
    enabled: bool,
    session_id: str,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    trace_name: str = "chat-response",
) -> dict[str, Any]:
    run_metadata = sanitize_langfuse_metadata(metadata)
    run_metadata["langfuse_session_id"] = session_id
    if user_id:
        run_metadata["langfuse_user_id"] = user_id
    if tags:
        run_metadata["langfuse_tags"] = tags

    callbacks = []
    if enabled:
        from langfuse.langchain import CallbackHandler

        callbacks.append(CallbackHandler())

    return {
        "callbacks": callbacks,
        "metadata": run_metadata,
        "run_name": trace_name,
        "tags": tags or [],
    }


@contextmanager
def langfuse_observation(
    *,
    enabled: bool,
    name: str,
    input: Any | None = None,
    output: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    if not enabled:
        with nullcontext(None) as span:
            yield span
        return

    from langfuse import get_client

    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="span",
        name=name,
        input=input,
        output=output,
        metadata=sanitize_langfuse_metadata(metadata),
    ) as span:
        yield span


@contextmanager
def propagate_langfuse_attributes(
    *,
    enabled: bool,
    trace_name: str,
    session_id: str,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
):
    if not enabled:
        with nullcontext():
            yield
        return

    from langfuse import propagate_attributes

    with propagate_attributes(
        trace_name=trace_name,
        user_id=user_id,
        session_id=session_id,
        tags=tags or [],
        metadata=sanitize_langfuse_metadata(metadata),
    ):
        yield


def flush_langfuse(enabled: bool) -> None:
    if not enabled:
        return

    from langfuse import get_client

    get_client().flush()


def sanitize_langfuse_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    safe: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if value is None or _SECRET_KEY_PATTERN.search(str(key)):
            continue

        safe_key = _SAFE_METADATA_KEY_PATTERN.sub("_", str(key)).strip("_")
        if not safe_key:
            continue

        text = _stringify_metadata_value(value)
        if text:
            safe[safe_key[:80]] = text[:_MAX_METADATA_VALUE_LENGTH]
    return safe


def _stringify_metadata_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)
