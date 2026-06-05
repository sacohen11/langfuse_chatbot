from __future__ import annotations

import os
from typing import Any

import httpx


class LangGraphEvalClient:
    def __init__(
        self,
        url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        self.url = url or os.environ.get("LANGGRAPH_EVAL_URL", "http://localhost:8080/internal/eval/run")
        self.timeout = timeout if timeout is not None else float(os.environ.get("LANGGRAPH_EVAL_TIMEOUT", "120"))

    async def run_case(
        self,
        user_request: str,
        candidate_config: dict[str, Any],
        expected_output: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "user_request": user_request,
            "candidate_config": candidate_config,
            "expected_output": expected_output or {},
            "metadata": metadata or {},
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, json=payload)
            response.raise_for_status()
            data = response.json()

        if not isinstance(data, dict):
            raise ValueError("LangGraph eval endpoint must return a JSON object")
        return data
