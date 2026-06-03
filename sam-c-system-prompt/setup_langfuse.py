from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    load_dotenv(ROOT / ".env")
    load_dotenv(HERE / ".env")

    config = json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))
    prompt_text = (HERE / config["seed_prompt_file"]).read_text(encoding="utf-8").strip()
    dataset_items = json.loads((HERE / config["dataset_items_file"]).read_text(encoding="utf-8"))

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )

    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="chat",
        prompt=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": "{{customer_message}}"},
        ],
        labels=["production", "baseline"],
        tags=["sam-c-system-prompt", "autoresearch", "support-triage"],
        config={
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
            "purpose": "Seed system prompt for Pi autoresearch evaluation.",
        },
        commit_message="Seed Sam-C support triage system prompt.",
    )

    try:
        langfuse.create_dataset(
            name=config["dataset_name"],
            description="Five support-triage examples for optimizing the Sam-C system prompt.",
            metadata={
                "prompt_name": config["prompt_name"],
                "created_by": "sam-c-system-prompt/setup_langfuse.py",
                "evaluation_style": "LLM-as-a-judge",
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
