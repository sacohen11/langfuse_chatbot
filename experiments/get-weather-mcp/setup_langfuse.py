import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


DATASET_ITEMS = [
    {
        "id": "get-weather-current-boston",
        "input": {"user_message": "What is the current weather in Boston, MA?"},
        "expected_output": {
            "should_call_tool": True,
            "tool_name": "get_weather",
            "arguments": {"location": "Boston, MA"},
            "answer_must_explain": ["temperature", "conditions", "location"],
        },
    },
    {
        "id": "get-weather-ambiguous-springfield",
        "input": {"user_message": "Is it raining in Springfield right now?"},
        "expected_output": {
            "should_call_tool": False,
            "reason": "location is ambiguous",
            "answer_must_explain": ["ask which Springfield"],
        },
    },
    {
        "id": "get-weather-forecast-seattle",
        "input": {
            "user_message": "Will I need a jacket in Seattle tomorrow morning?",
            "today": "2026-06-04",
        },
        "expected_output": {
            "should_call_tool": True,
            "tool_name": "get_weather",
            "arguments": {"location": "Seattle, WA", "time_window": "tomorrow morning"},
            "answer_must_explain": ["time window", "temperature", "precipitation or conditions"],
        },
    },
    {
        "id": "get-weather-unrelated",
        "input": {"user_message": "What is the capital of France?"},
        "expected_output": {
            "should_call_tool": False,
            "reason": "unrelated geography question",
        },
    },
    {
        "id": "get-weather-units",
        "input": {"user_message": "What's the temperature in Toronto in Celsius?"},
        "expected_output": {
            "should_call_tool": True,
            "tool_name": "get_weather",
            "arguments": {"location": "Toronto, ON", "units": "metric"},
            "answer_must_explain": ["Celsius", "conditions", "freshness"],
        },
    },
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def create_dataset_items(langfuse: Langfuse, dataset_name: str) -> None:
    for item in DATASET_ITEMS:
        try:
            langfuse.create_dataset_item(
                dataset_name=dataset_name,
                id=item["id"],
                input=item["input"],
                expected_output=item["expected_output"],
                metadata={"source": "get-weather-mcp setup"},
            )
        except Exception:
            pass


def main() -> None:
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")
    config = read_json(EXP_DIR / "benchmark_config.json")
    tool_spec = read_json(EXP_DIR / "tool.json")

    langfuse = Langfuse()

    try:
        langfuse.create_dataset(
            name=config["dataset_name"],
            description="Weather MCP tool-selection and schema-quality examples.",
            metadata={"experiment_name": config["experiment_name"], "kind": "mcp_tool"},
        )
    except Exception:
        pass

    create_dataset_items(langfuse, config["dataset_name"])

    langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=json.dumps(tool_spec, indent=2),
        labels=["production"],
        tags=["mcp_tool_schema", "weather", "autoresearch_seed"],
        config={
            "experiment_name": config["experiment_name"],
            "dataset_name": config["dataset_name"],
            "scores": config["scores"],
        },
        commit_message="Seed get_weather MCP tool spec for autoresearch.",
    )

    fetched = langfuse.get_prompt(config["prompt_name"], label="production", type="text")
    write_json(EXP_DIR / "tool.json", json.loads(str(fetched.prompt)))

    print(
        json.dumps(
            {
                "dataset_name": config["dataset_name"],
                "dataset_items": len(DATASET_ITEMS),
                "prompt_name": config["prompt_name"],
                "prompt_version": getattr(fetched, "version", None),
                "candidate_file": str(EXP_DIR / "tool.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
