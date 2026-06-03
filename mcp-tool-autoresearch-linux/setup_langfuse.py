from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse


HERE = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(HERE / ".env")

    config = json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))
    spec = json.loads((HERE / config["seed_tool_spec_file"]).read_text(encoding="utf-8"))
    dataset_items = json.loads((HERE / config["dataset_items_file"]).read_text(encoding="utf-8"))

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )

    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=json.dumps(spec, indent=2, ensure_ascii=False),
        labels=["production", "baseline"],
        tags=["mcp-tool-spec", "autoresearch", "baseline"],
        config={
            "artifact_type": "mcp_tool_spec",
            "tool_name": spec["name"],
            "tool_spec": spec,
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
            "purpose": "Seed MCP tool spec for Pi autoresearch evaluation.",
        },
        commit_message="Seed MCP support ticket lookup tool spec.",
    )

    try:
        langfuse.create_dataset(
            name=config["dataset_name"],
            description="MCP tool-call scenarios for optimizing a support-ticket lookup tool specification.",
            metadata={
                "prompt_name": config["prompt_name"],
                "created_by": "mcp-tool-autoresearch-linux/setup_langfuse.py",
                "evaluation_style": "LLM-as-a-judge",
                "artifact_type": "mcp_tool_spec",
            },
        )
    except Exception as exc:  # noqa: BLE001
        if "already" not in str(exc).lower() and "exist" not in str(exc).lower():
            raise

    for item in dataset_items:
        langfuse.create_dataset_item(
            id=item["id"],
            dataset_name=config["dataset_name"],
            input=item["input"],
            expected_output=item["expected_output"],
            metadata=item.get("metadata", {}),
        )

    langfuse.flush()
    print(
        json.dumps(
            {
                "prompt_name": config["prompt_name"],
                "prompt_version": getattr(prompt, "version", None),
                "dataset_name": config["dataset_name"],
                "dataset_items": len(dataset_items),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

