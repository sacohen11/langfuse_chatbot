from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the optimized Sam-C system prompt to Langfuse.")
    parser.add_argument("--prompt-file", default="prompt_candidate.txt", help="Optimized system prompt file.")
    parser.add_argument("--label", action="append", default=None, help="Label to apply. Defaults to autoresearch.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    load_dotenv(HERE / ".env")
    config = json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))
    prompt_text = (HERE / args.prompt_file).read_text(encoding="utf-8").strip()

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
        labels=args.label or ["autoresearch"],
        tags=["sam-c-system-prompt", "autoresearch", "support-triage"],
        config={
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
            "dataset_name": config["dataset_name"],
            "source": args.prompt_file,
            "save_mode": "current_prompt_file",
        },
        commit_message="Save Pi autoresearch optimized Sam-C system prompt.",
    )
    langfuse.flush()
    print(json.dumps({"prompt_name": config["prompt_name"], "version": getattr(prompt, "version", None)}, indent=2))


if __name__ == "__main__":
    main()
