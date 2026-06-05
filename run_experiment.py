from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import os
import statistics
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from langfuse.openai import OpenAI

from evals.langgraph_mcp_e2e import run_langgraph_mcp_e2e


ROOT = Path(__file__).resolve().parent


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT / ".env", encoding="utf-8-sig")

    config_path = Path(args.config).resolve()
    workdir = config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    langfuse = make_langfuse()

    if args.pull:
        pull_prompt(langfuse, config, workdir)
        return

    if config["kind"] == "langgraph_mcp_e2e":
        asyncio.run(run_langgraph_mcp_e2e(config, workdir, langfuse, update_leaderboard))
        return

    dataset = langfuse.get_dataset(config["dataset_name"])
    if not dataset.items:
        raise RuntimeError(f"Langfuse dataset has no items: {config['dataset_name']}")

    candidate = (workdir / config["candidate_file"]).read_text(encoding="utf-8").strip()
    openai = OpenAI()
    task, artifact_type = make_task(openai, config, candidate)
    score_names = [score["name"] for score in config["scores"]]

    result = dataset.run_experiment(
        name=config["experiment_name"],
        run_name=f"{config['experiment_name']}-{stamp()}",
        description=f"Autoresearch run for {config['prompt_name']}",
        task=lambda *, item, **_: task(item),
        evaluators=[make_judge(openai, config, score) for score in config["scores"]],
        run_evaluators=[make_overall(score_names)],
        max_concurrency=1,
        metadata={
            "participant": config["participant"],
            "kind": config["kind"],
            "prompt_name": config["prompt_name"],
        },
    )

    scores = summarize_scores(result, score_names)
    version = "" if args.no_save else save_prompt_version(langfuse, config, artifact_type, candidate, scores)
    row = update_leaderboard(config, scores, version, getattr(result, "dataset_run_url", ""))
    langfuse.flush()
    print_metrics(scores, version, row, getattr(result, "dataset_run_url", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--pull", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def make_langfuse() -> Langfuse:
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST"),
    )


def pull_prompt(langfuse: Langfuse, config: dict, workdir: Path) -> None:
    prompt = langfuse.get_prompt(
        config["prompt_name"],
        type=config.get("prompt_type", "chat" if config["kind"] == "prompt" else "text"),
        label=config.get("prompt_label", "production"),
        cache_ttl_seconds=0,
    )
    prompt_text = extract_prompt_text(prompt.prompt)
    candidate_path = workdir / config["candidate_file"]
    if config["kind"] == "prompt":
        candidate_path.write_text(prompt_text + "\n", encoding="utf-8")
    elif config["kind"] in {"mcp_tool", "langgraph_mcp_e2e"}:
        candidate = build_tool_candidate(prompt_text, getattr(prompt, "config", None))
        candidate_path.write_text(json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        raise SystemExit(f"--pull does not support experiment kind: {config['kind']}")
    print(f"Pulled {config['prompt_name']} into {config['candidate_file']}")


def extract_prompt_text(raw_prompt) -> str:
    if isinstance(raw_prompt, str):
        return raw_prompt.strip()
    return str(
        next(
            (m.get("content", "") for m in raw_prompt if isinstance(m, dict) and m.get("role") == "system"),
            json.dumps(raw_prompt),
        )
    ).strip()


def build_tool_candidate(prompt_text: str, prompt_config) -> dict:
    config = prompt_config if isinstance(prompt_config, dict) else {}
    tool_config = config.get("tool_suite") or config.get("candidate_config") or pick_tool_config(config)
    if isinstance(tool_config, dict) and tool_config:
        return {"system_prompt": prompt_text, **tool_config}

    try:
        parsed = json.loads(prompt_text)
    except json.JSONDecodeError:
        return {"system_prompt": prompt_text}
    return parsed if isinstance(parsed, dict) else {"system_prompt": prompt_text}


def pick_tool_config(config: dict) -> dict:
    return {
        key: config[key]
        for key in ("server", "tool_choice", "tools", "tool_routing")
        if key in config
    }


def make_task(openai: OpenAI, config: dict, candidate: str):
    if config["kind"] == "prompt":
        def prompt_task(item):
            user_input = item.input if isinstance(item.input, dict) else {"input": item.input}
            return call_openai(openai, config["task_model"], [
                {"role": "system", "content": candidate},
                {"role": "user", "content": render(config["user_template"], user_input)},
            ], config.get("max_completion_tokens", 300))
        return prompt_task, "prompt"

    if config["kind"] == "mcp_tool":
        tool_spec = json.loads(candidate)
        system_prompt = tool_spec.get(
            "system_prompt",
            "Decide whether to call a hosted MCP tool. Return JSON only.",
        )
        tool_payload = {key: value for key, value in tool_spec.items() if key != "system_prompt"}

        def tool_task(item):
            data = item.input if isinstance(item.input, dict) else {"user_request": item.input}
            payload = {
                "server": config.get("server", {}),
                "tool_spec": tool_payload,
                "user_request": data["user_request"],
                "available_context": data.get("available_context", {}),
            }
            return call_openai(openai, config["task_model"], [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ], config.get("max_completion_tokens", 300), json_mode=True)

        return tool_task, "mcp_tool_schema"

    raise ValueError(f"Unknown experiment kind: {config['kind']}")


def make_judge(openai: OpenAI, config: dict, score: dict):
    def judge(*, input: dict, output: str, expected_output: dict, **_) -> Evaluation:
        payload = {
            "rubric": score["rubric"],
            "input": input,
            "output": output,
            "expected_output": expected_output,
            "instructions": "Return JSON with score from 0 to 1 and rationale.",
        }
        raw = call_openai(openai, config["judge_model"], [
            {"role": "system", "content": "You are a strict but fair evaluator."},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], 250, json_mode=True)

        try:
            data = json.loads(raw)
            value = max(0.0, min(1.0, float(data.get("score", 0))))
            comment = str(data.get("rationale", data.get("reasoning", "")))
        except Exception:
            value, comment = 0.0, f"Invalid judge JSON: {raw[:120]}"
        return Evaluation(name=score["name"], value=value, comment=comment)

    judge.__name__ = f"{score['name']}_judge"
    return judge


def make_overall(score_names: list[str]):
    def overall(*, item_results: list, **_) -> Evaluation:
        return Evaluation(name="overall_score", value=mean(evaluation_values(item_results, score_names)))

    return overall


def summarize_scores(result, score_names: list[str]) -> dict:
    dimensions = {}
    for name in score_names:
        dimensions[name] = mean(evaluation_values(getattr(result, "item_results", []) or [], [name]))
    return {"overall_score": mean(dimensions.values()), "dimensions": dimensions}


def evaluation_values(item_results, score_names: list[str]) -> list[float]:
    return [
        float(evaluation.value)
        for item in item_results
        for evaluation in (getattr(item, "evaluations", []) or [])
        if evaluation.name in score_names
    ]


def save_prompt_version(langfuse: Langfuse, config: dict, artifact_type: str, text: str, scores: dict) -> str:
    prompt_type = config.get("prompt_type", "text")
    prompt_body = text
    prompt_config = {
        "participant": config["participant"],
        "experiment_name": config["experiment_name"],
        "artifact_type": artifact_type,
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimensions"],
        "prompt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
    }
    if config["kind"] == "prompt" and prompt_type == "chat":
        prompt_body = [
            {"role": "system", "content": text},
            {"role": "user", "content": config.get("user_template", "{{input}}")},
        ]
    elif config["kind"] in {"mcp_tool", "langgraph_mcp_e2e"}:
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            candidate = {}
        if isinstance(candidate, dict) and "system_prompt" in candidate:
            prompt_body = str(candidate["system_prompt"])
            prompt_config = {
                **{key: value for key, value in candidate.items() if key != "system_prompt"},
                "_autoresearch": prompt_config,
            }

    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type=prompt_type,
        prompt=prompt_body,
        labels=[],
        tags=["autoresearch", artifact_type],
        config=prompt_config,
        commit_message=f"{config['experiment_name']} score={scores['overall_score']:.4f}",
    )
    return str(getattr(prompt, "version", ""))


def update_leaderboard(config: dict, scores: dict, version: str, run_url: str) -> dict:
    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "participant": config["participant"],
        "experiment_name": config["experiment_name"],
        "kind": config["kind"],
        "prompt_name": config.get("prompt_name", config.get("candidate_file", config["experiment_name"])),
        "prompt_version": version,
        "overall_score": f"{scores['overall_score']:.6f}",
        "dataset_name": config["dataset_name"],
        "dataset_run_url": run_url or "",
        "scores_json": json.dumps(scores["dimensions"], sort_keys=True),
    }
    row.update({f"score:{name}": f"{value:.6f}" for name, value in scores["dimensions"].items()})

    path = ROOT / "leaderboard.csv"
    rows, columns = [], list(row)
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(dict.fromkeys([*(reader.fieldnames or []), *row]))

    rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return row


def call_openai(openai: OpenAI, model: str, messages: list[dict], max_tokens: int, json_mode: bool = False) -> str:
    kwargs = {"model": model, "temperature": 0, "max_completion_tokens": max_tokens, "messages": messages}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = openai.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def print_metrics(scores: dict, version: str, row: dict, run_url: str) -> None:
    print(json.dumps({
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimensions"],
        "prompt_version": version,
        "leaderboard_row": row,
        "dataset_run_url": run_url,
    }, indent=2))
    print(f"METRIC overall_score={scores['overall_score']:.4f}")
    for name, value in scores["dimensions"].items():
        print(f"METRIC {name}={value:.4f}")


def render(template: str, data: dict) -> str:
    for key, value in data.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def mean(values) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


if __name__ == "__main__":
    main()
