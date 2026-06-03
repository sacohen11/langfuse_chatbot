from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Langfuse


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the best scored MCP tool spec attempt as the newest Langfuse prompt version.")
    parser.add_argument("--label", action="append", default=None, help="Label to apply. Defaults to autoresearch-best.")
    parser.add_argument("--also-production", action="store_true", help="Also apply the production label to the best version.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial/smoke-test attempts to be selected.")
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    config = json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))
    best = _best_attempt(
        HERE / config.get("attempt_log_file", "autoresearch.prompt_attempts.jsonl"),
        expected_item_count=int(config.get("expected_dataset_items", 0)),
        allow_partial=args.allow_partial,
    )

    labels = args.label or ["autoresearch-best"]
    if args.also_production and "production" not in labels:
        labels = [*labels, "production"]

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )
    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=best["tool_spec_text"],
        labels=labels,
        tags=["mcp-tool-spec", "autoresearch", "autoresearch-best"],
        config={
            "artifact_type": "mcp_weather_server_tool_spec",
            "selected_from_attempt_id": best["attempt_id"],
            "selected_from_prompt_version": best["prompt_version"],
            "selected_from_prompt_hash": best["prompt_hash"],
            "server_name": best["server_name"],
            "tool_names": best["tool_names"],
            "tool_spec": best["tool_spec"],
            "overall_score": best["overall_score"],
            "dimension_scores": best["dimension_scores"],
            "dataset_name": config["dataset_name"],
            "dataset_run_url": best.get("dataset_run_url"),
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
        },
        commit_message=f"Save best MCP tool spec attempt: overall_score={best['overall_score']:.4f}",
    )
    langfuse.flush()
    print(
        json.dumps(
            {
                "prompt_name": config["prompt_name"],
                "newest_best_version": getattr(prompt, "version", None),
                "selected_from_attempt_version": best["prompt_version"],
                "overall_score": best["overall_score"],
                "labels": labels,
            },
            indent=2,
        )
    )


def _best_attempt(path: Path, *, expected_item_count: int, allow_partial: bool) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No attempt log found at {path}. Run bash autoresearch.sh first.")

    attempts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if expected_item_count and not allow_partial:
        attempts = [
            attempt
            for attempt in attempts
            if int(attempt.get("item_count", 0)) >= expected_item_count and attempt.get("max_items") is None
        ]
    if not attempts:
        raise RuntimeError(f"No full-dataset attempts found in {path}. Run bash autoresearch.sh first.")

    return max(attempts, key=lambda item: float(item.get("overall_score", 0.0)))


if __name__ == "__main__":
    main()
