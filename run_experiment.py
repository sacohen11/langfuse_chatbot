from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from langfuse.openai import OpenAI


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to config.json")
    parser.add_argument("--pull", action="store_true", help="Pull the Langfuse prompt into the candidate file.")
    parser.add_argument("--no-save", action="store_true", help="Do not save an attempt prompt version.")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    config_path = Path(args.config).resolve()
    workdir = config_path.parent
    config = read_json(config_path)
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )

    if args.pull:
        if config["kind"] != "prompt":
            raise SystemExit("--pull only applies to prompt experiments")
        prompt_text = pull_prompt(langfuse, config)
        (workdir / config["candidate_file"]).write_text(prompt_text + "\n", encoding="utf-8")
        print(f"Pulled {config['prompt_name']} into {config['candidate_file']}")
        return

    dataset = langfuse.get_dataset(config["dataset_name"])
    if not dataset.items:
        raise RuntimeError(f"Langfuse dataset has no items: {config['dataset_name']}")

    candidate_path = workdir / config["candidate_file"]
    candidate_text = candidate_path.read_text(encoding="utf-8").strip()
    openai = OpenAI()

    if config["kind"] == "prompt":
        task = lambda item: run_prompt_task(openai, config, candidate_text, item)
        artifact_type = "prompt"
    elif config["kind"] == "mcp_tool":
        tool_spec = json.loads(candidate_text)
        task = lambda item: run_mcp_task(openai, config, tool_spec, item)
        artifact_type = "mcp_tool_schema"
    else:
        raise ValueError(f"Unknown experiment kind: {config['kind']}")

    evaluators = [make_judge(openai, config, score) for score in config["scores"]]
    result = dataset.run_experiment(
        name=config["experiment_name"],
        run_name=f"{config['experiment_name']}-{now_id()}",
        description=f"Autoresearch run for {config['prompt_name']}",
        task=lambda *, item, **_: task(item),
        evaluators=evaluators,
        run_evaluators=[make_overall_evaluator([score["name"] for score in config["scores"]])],
        max_concurrency=1,
        metadata={
            "participant": config["participant"],
            "kind": config["kind"],
            "prompt_name": config["prompt_name"],
        },
    )

    scores = collect_scores(result, [score["name"] for score in config["scores"]])
    prompt_version = ""
    if not args.no_save:
        prompt_version = save_attempt(langfuse, config, artifact_type, candidate_text, scores)

    row = append_leaderboard(config, scores, prompt_version, getattr(result, "dataset_run_url", ""))
    langfuse.flush()

    output = {
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimensions"],
        "prompt_version": prompt_version,
        "leaderboard_row": row,
        "dataset_run_url": getattr(result, "dataset_run_url", ""),
    }
    print(json.dumps(output, indent=2))
    print(f"METRIC overall_score={scores['overall_score']:.4f}")
    for name, value in scores["dimensions"].items():
        print(f"METRIC {name}={value:.4f}")


def pull_prompt(langfuse: Langfuse, config: dict[str, Any]) -> str:
    prompt = langfuse.get_prompt(
        config["prompt_name"],
        type=config.get("prompt_type", "chat"),
        label=config.get("prompt_label", "production"),
        cache_ttl_seconds=0,
    )
    raw = prompt.prompt
    if isinstance(raw, str):
        return raw.strip()
    for message in raw:
        if message.get("role") == "system":
            return str(message.get("content", "")).strip()
    return json.dumps(raw, ensure_ascii=False)


def run_prompt_task(openai: OpenAI, config: dict[str, Any], system_prompt: str, item: Any) -> str:
    item_input = item.input if isinstance(item.input, dict) else {"input": item.input}
    user_text = render(config["user_template"], item_input)
    response = openai.chat.completions.create(
        model=config["task_model"],
        temperature=0,
        max_completion_tokens=config.get("max_completion_tokens", 300),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return response.choices[0].message.content or ""


def run_mcp_task(openai: OpenAI, config: dict[str, Any], tool_spec: dict[str, Any], item: Any) -> str:
    item_input = item.input if isinstance(item.input, dict) else {"user_request": item.input}
    response = openai.chat.completions.create(
        model=config["task_model"],
        temperature=0,
        max_completion_tokens=350,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Decide whether to call a hosted MCP tool. Return JSON only with "
                    "should_call_tool, tool_name, arguments, expected_result_usage, and rationale."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "server": config.get("server", {}),
                        "tool_spec": tool_spec,
                        "user_request": item_input["user_request"],
                        "available_context": item_input.get("available_context", {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    )
    return response.choices[0].message.content or "{}"


def make_judge(openai: OpenAI, config: dict[str, Any], score: dict[str, str]):
    def judge(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        payload = {
            "rubric": score["rubric"],
            "input": input,
            "output": output,
            "expected_output": expected_output,
            "instructions": "Return JSON with score from 0 to 1 and rationale.",
        }
        response = openai.chat.completions.create(
            model=config["judge_model"],
            temperature=0,
            max_completion_tokens=250,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict but fair evaluator."},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            value = max(0.0, min(1.0, float(data.get("score", 0))))
            comment = str(data.get("rationale", data.get("reasoning", "")))
        except Exception:
            value, comment = 0.0, f"Invalid judge JSON: {raw[:120]}"
        return Evaluation(name=score["name"], value=value, comment=comment)

    judge.__name__ = f"{score['name']}_judge"
    return judge


def make_overall_evaluator(score_names: list[str]):
    def overall(*, item_results: list[Any], **_: Any) -> Evaluation:
        values = []
        for item_result in item_results:
            for evaluation in getattr(item_result, "evaluations", []) or []:
                if evaluation.name in score_names:
                    values.append(float(evaluation.value))
        return Evaluation(name="overall_score", value=statistics.mean(values) if values else 0.0)

    return overall


def collect_scores(result: Any, score_names: list[str]) -> dict[str, Any]:
    values = {name: [] for name in score_names}
    for item_result in getattr(result, "item_results", []) or []:
        for evaluation in getattr(item_result, "evaluations", []) or []:
            if evaluation.name in values:
                values[evaluation.name].append(float(evaluation.value))
    dimensions = {name: statistics.mean(vals) if vals else 0.0 for name, vals in values.items()}
    return {"overall_score": statistics.mean(dimensions.values()) if dimensions else 0.0, "dimensions": dimensions}


def save_attempt(langfuse: Langfuse, config: dict[str, Any], artifact_type: str, text: str, scores: dict[str, Any]) -> str:
    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=text,
        labels=[],
        tags=["autoresearch", artifact_type],
        config={
            "participant": config["participant"],
            "experiment_name": config["experiment_name"],
            "artifact_type": artifact_type,
            "overall_score": scores["overall_score"],
            "dimension_scores": scores["dimensions"],
            "prompt_hash": sha(text),
        },
        commit_message=f"{config['experiment_name']} score={scores['overall_score']:.4f}",
    )
    return str(getattr(prompt, "version", ""))


def append_leaderboard(config: dict[str, Any], scores: dict[str, Any], prompt_version: str, run_url: str) -> dict[str, str]:
    path = ROOT / "leaderboard.csv"
    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "participant": config["participant"],
        "experiment_name": config["experiment_name"],
        "kind": config["kind"],
        "prompt_name": config["prompt_name"],
        "prompt_version": prompt_version,
        "overall_score": f"{scores['overall_score']:.6f}",
        "dataset_name": config["dataset_name"],
        "dataset_run_url": run_url or "",
        "scores_json": json.dumps(scores["dimensions"], sort_keys=True),
    }
    for name, value in scores["dimensions"].items():
        row[f"score:{name}"] = f"{value:.6f}"

    rows = []
    columns = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in columns:
            columns.append(key)
    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return row


def render(template: str, data: dict[str, Any]) -> str:
    out = template
    for key, value in data.items():
        out = out.replace("{{" + key + "}}", str(value))
    return out


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def now_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
