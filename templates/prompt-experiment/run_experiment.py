from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import statistics
import subprocess
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from langfuse.openai import OpenAI


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a configurable Langfuse prompt experiment.")
    parser.add_argument("--config", default="experiment_config.json", help="Experiment config JSON file.")
    parser.add_argument("--prompt-file", default=None, help="Candidate system prompt file.")
    parser.add_argument("--publish", action="store_true", help="Publish a Langfuse dataset run.")
    parser.add_argument("--pull-prompt", action="store_true", help="Pull the configured Langfuse prompt into the candidate file and exit.")
    parser.add_argument("--validate-config", action="store_true", help="Validate config shape and exit without network calls.")
    parser.add_argument("--run-name", default=None, help="Optional exact Langfuse dataset run name.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit dataset items for smoke tests.")
    parser.add_argument("--result-file", default=None, help="Optional JSON file to write the result.")
    parser.add_argument("--no-save-prompt-version", action="store_true", help="Skip saving this attempt as a Langfuse prompt version.")
    parser.add_argument("--no-leaderboard", action="store_true", help="Skip updating the repo-level leaderboard.csv.")
    args = parser.parse_args()

    repo_root = _repo_root()
    load_dotenv(repo_root / ".env")
    load_dotenv(HERE / ".env")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = HERE / config_path
    config = _load_config(config_path)

    if args.validate_config:
        print(json.dumps({"ok": True, "scores": [score["name"] for score in config["scores"]]}, indent=2))
        return

    langfuse = _langfuse_client()

    prompt_path = Path(args.prompt_file or config.get("candidate_prompt_file", "prompt_candidate.txt"))
    if not prompt_path.is_absolute():
        prompt_path = HERE / prompt_path

    if args.pull_prompt:
        prompt_text = _fetch_prompt_text(langfuse, config)
        prompt_path.write_text(prompt_text + "\n", encoding="utf-8")
        print(json.dumps({"prompt_file": str(prompt_path), "prompt_name": config["prompt_name"]}, indent=2))
        return

    system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not system_prompt or system_prompt.startswith("Replace this file"):
        system_prompt = _fetch_prompt_text(langfuse, config)
        prompt_path.write_text(system_prompt + "\n", encoding="utf-8")

    dataset = langfuse.get_dataset(config["dataset_name"])
    items = dataset.items[: args.max_items] if args.max_items else dataset.items
    if not items:
        raise RuntimeError(f"Dataset {config['dataset_name']!r} has no items.")

    openai = OpenAI()

    def task(*, item, **_: Any) -> str:
        user_message = _render_template(config.get("user_template", "{{" + config["input_key"] + "}}"), item.input)
        response = openai.chat.completions.create(
            model=config["task_model"],
            temperature=float(config.get("temperature", 0)),
            max_completion_tokens=int(config.get("max_completion_tokens", 280)),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    evaluators = [_make_evaluator(openai, config, score_config) for score_config in config["scores"]]
    score_names = [score["name"] for score in config["scores"]]

    if args.publish:
        run_name = args.run_name or f"{config['experiment_name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        result = langfuse.run_experiment(
            name=config["experiment_name"],
            run_name=run_name,
            description=f"Prompt experiment for {config['prompt_name']}",
            data=items,
            task=task,
            evaluators=evaluators,
            run_evaluators=[_make_aggregate_evaluator(score_names)],
            max_concurrency=1,
            metadata={
                "participant_name": config["participant_name"],
                "prompt_name": config["prompt_name"],
                "task_model": config["task_model"],
                "judge_model": config["judge_model"],
                "config_file": config_path.name,
            },
        )
        scores = _scores_from_experiment_result(result, score_names)
        dataset_run_url = getattr(result, "dataset_run_url", None)
    else:
        scores = _run_locally(items, task, evaluators, score_names)
        dataset_run_url = None

    output = {
        "participant_name": config["participant_name"],
        "experiment_name": config["experiment_name"],
        "prompt_name": config["prompt_name"],
        "prompt_file": str(prompt_path),
        "dataset_name": config["dataset_name"],
        "item_count": len(items),
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimension_scores"],
        "dataset_run_url": dataset_run_url,
    }

    if not args.no_save_prompt_version:
        attempt = _save_attempt_prompt_version(
            langfuse=langfuse,
            config=config,
            system_prompt=system_prompt,
            prompt_path=prompt_path,
            scores=scores,
            dataset_run_url=dataset_run_url,
            item_count=len(items),
            max_items=args.max_items,
        )
        _append_attempt_log(config, attempt)
        output["prompt_attempt"] = attempt

    if not args.no_leaderboard:
        output["leaderboard_path"] = str(_update_leaderboard(repo_root, config, output))

    langfuse.flush()

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"METRIC overall_score={scores['overall_score']:.4f}")
    for name, value in scores["dimension_scores"].items():
        print(f"METRIC {name}={value:.4f}")

    if args.result_file:
        Path(args.result_file).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        example = path.with_name("experiment_config.example.json")
        raise FileNotFoundError(f"Missing {path.name}. Copy {example.name} to {path.name} and edit it.")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = ["participant_name", "experiment_name", "prompt_name", "dataset_name", "task_model", "judge_model", "input_key", "scores"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    if not isinstance(config["scores"], list) or not config["scores"]:
        raise ValueError("Config must include at least one score.")
    for score in config["scores"]:
        if not score.get("name") or not score.get("type"):
            raise ValueError("Each score must include name and type.")
        if score["type"] == "llm_judge" and not score.get("rubric"):
            raise ValueError(f"llm_judge score {score['name']!r} requires rubric.")
        if score["type"] == "managed_langfuse" and not score.get("evaluator_name"):
            raise ValueError(f"managed_langfuse score {score['name']!r} requires evaluator_name.")
    return config


def _langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )


def _fetch_prompt_text(langfuse: Langfuse, config: dict[str, Any]) -> str:
    prompt = langfuse.get_prompt(
        config["prompt_name"],
        type=config.get("prompt_type", "chat"),
        label=config.get("prompt_label"),
        version=config.get("prompt_version"),
        cache_ttl_seconds=0,
    )
    raw_prompt = getattr(prompt, "prompt", "")
    if isinstance(raw_prompt, str):
        return raw_prompt.strip()
    for message in raw_prompt:
        if message.get("role") == "system":
            return str(message.get("content", "")).strip()
    return json.dumps(raw_prompt, ensure_ascii=False)


def _make_evaluator(openai: OpenAI, config: dict[str, Any], score_config: dict[str, Any]):
    score_type = score_config["type"]
    if score_type == "llm_judge":
        return _make_llm_judge_evaluator(openai, config["judge_model"], score_config["name"], score_config["rubric"])
    if score_type == "managed_langfuse":
        evaluator = _get_managed_langfuse_evaluator(score_config["evaluator_name"])
        return _make_managed_langfuse_evaluator(openai, config["judge_model"], score_config, evaluator)
    raise ValueError(f"Unsupported score type: {score_type}")


def _make_llm_judge_evaluator(openai: OpenAI, judge_model: str, name: str, rubric: str):
    def evaluator(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        judge_prompt = {
            "rubric": rubric,
            "input": input,
            "assistant_output": output,
            "expected_output": expected_output,
            "instructions": "Return JSON with score from 0 to 1 and a short rationale. Be strict but fair.",
        }
        response = openai.chat.completions.create(
            model=judge_model,
            temperature=0,
            max_completion_tokens=220,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise evaluator for assistant outputs."},
                {"role": "user", "content": json.dumps(judge_prompt, ensure_ascii=False)},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return Evaluation(name=name, value=0.0, comment=f"Judge returned invalid JSON: {raw[:160]}")
        return Evaluation(name=name, value=_clamp_score(data.get("score", 0.0)), comment=str(data.get("rationale", data.get("reasoning", ""))))

    evaluator.__name__ = f"{name}_judge"
    return evaluator


def _make_managed_langfuse_evaluator(openai: OpenAI, judge_model: str, score_config: dict[str, Any], evaluator_config: dict[str, Any]):
    name = score_config["name"]
    prompt_template = evaluator_config["prompt"]
    variable_map = score_config.get("variable_map", {})
    evaluator_name = evaluator_config.get("name", score_config["evaluator_name"])
    evaluator_version = evaluator_config.get("version")

    def evaluator(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        values = {
            variable: _resolve_mapping(source, input=input, output=output, expected_output=expected_output)
            for variable, source in variable_map.items()
        }
        for variable in evaluator_config.get("variables", []):
            values.setdefault(variable, _default_variable_value(variable, input=input, output=output, expected_output=expected_output))
        prompt = prompt_template
        for variable, value in values.items():
            prompt = prompt.replace("{{" + variable + "}}", _stringify(value))
        response = openai.chat.completions.create(
            model=judge_model,
            temperature=0,
            max_completion_tokens=220,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "Run the provided Langfuse managed evaluator. Return JSON only with score and reasoning.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return Evaluation(name=name, value=0.0, comment=f"Evaluator returned invalid JSON: {raw[:160]}")
        reasoning = str(data.get("reasoning", data.get("rationale", "")))
        return Evaluation(name=name, value=_clamp_score(data.get("score", 0.0)), comment=f"Langfuse managed {evaluator_name} v{evaluator_version}: {reasoning}")

    evaluator.__name__ = f"{name}_managed_langfuse"
    return evaluator


def _get_managed_langfuse_evaluator(name: str) -> dict[str, Any]:
    base_url = os.environ["LANGFUSE_BASE_URL"].rstrip("/")
    credentials = f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/public/unstable/evaluators?limit=50",
        headers={"Authorization": f"Basic {base64.b64encode(credentials).decode('ascii')}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    for evaluator in payload.get("data", []):
        if evaluator.get("scope") == "managed" and evaluator.get("name") == name:
            return evaluator
    raise RuntimeError(f"Langfuse managed evaluator {name!r} was not found.")


def _make_aggregate_evaluator(score_names: list[str]):
    def aggregate(*, item_results: list[Any], **_: Any) -> Evaluation:
        values: list[float] = []
        for item_result in item_results:
            for evaluation in getattr(item_result, "evaluations", []) or []:
                if getattr(evaluation, "name", None) in score_names:
                    values.append(float(evaluation.value))
        return Evaluation(name="overall_score", value=statistics.mean(values) if values else 0.0)

    aggregate.__name__ = "overall_score_aggregate"
    return aggregate


def _scores_from_experiment_result(result: Any, score_names: list[str]) -> dict[str, Any]:
    values_by_name: dict[str, list[float]] = {name: [] for name in score_names}
    for item_result in getattr(result, "item_results", []) or []:
        for evaluation in getattr(item_result, "evaluations", []) or []:
            if getattr(evaluation, "name", None) in values_by_name:
                values_by_name[evaluation.name].append(float(evaluation.value))
    dimension_scores = {name: statistics.mean(values) if values else 0.0 for name, values in values_by_name.items()}
    return {"overall_score": statistics.mean(dimension_scores.values()) if dimension_scores else 0.0, "dimension_scores": dimension_scores}


def _run_locally(items: list[Any], task, evaluators: list[Any], score_names: list[str]) -> dict[str, Any]:
    values_by_name: dict[str, list[float]] = {name: [] for name in score_names}
    for item in items:
        output = task(item=item)
        for evaluator in evaluators:
            evaluation = evaluator(input=item.input, output=output, expected_output=item.expected_output, metadata=item.metadata)
            values_by_name[evaluation.name].append(float(evaluation.value))
    dimension_scores = {name: statistics.mean(values) if values else 0.0 for name, values in values_by_name.items()}
    return {"overall_score": statistics.mean(dimension_scores.values()) if dimension_scores else 0.0, "dimension_scores": dimension_scores}


def _save_attempt_prompt_version(
    *,
    langfuse: Langfuse,
    config: dict[str, Any],
    system_prompt: str,
    prompt_path: Path,
    scores: dict[str, Any],
    dataset_run_url: str | None,
    item_count: int,
    max_items: int | None,
) -> dict[str, Any]:
    attempt_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="chat",
        prompt=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": config.get("user_template", "{{" + config["input_key"] + "}}")},
        ],
        labels=[],
        tags=list(config.get("tags", [])),
        config={
            "participant_name": config["participant_name"],
            "experiment_name": config["experiment_name"],
            "attempt_id": attempt_id,
            "prompt_hash": prompt_hash,
            "overall_score": scores["overall_score"],
            "dimension_scores": scores["dimension_scores"],
            "item_count": item_count,
            "max_items": max_items,
            "dataset_name": config["dataset_name"],
            "dataset_run_url": dataset_run_url,
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
            "source": prompt_path.name,
        },
        commit_message=f"{config['experiment_name']} attempt {attempt_id}: overall_score={scores['overall_score']:.4f}",
    )
    return {
        "attempt_id": attempt_id,
        "participant_name": config["participant_name"],
        "experiment_name": config["experiment_name"],
        "prompt_name": config["prompt_name"],
        "prompt_version": getattr(prompt, "version", None),
        "prompt_hash": prompt_hash,
        "prompt_text": system_prompt,
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimension_scores"],
        "item_count": item_count,
        "max_items": max_items,
        "dataset_run_url": dataset_run_url,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _append_attempt_log(config: dict[str, Any], attempt: dict[str, Any]) -> None:
    path = HERE / config.get("attempt_log_file", "prompt_attempts.jsonl")
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(attempt, ensure_ascii=False) + "\n")


def _update_leaderboard(repo_root: Path, config: dict[str, Any], output: dict[str, Any]) -> Path:
    path = Path(config.get("leaderboard_path", "leaderboard.csv"))
    if not path.is_absolute():
        path = repo_root / path
    attempt = output.get("prompt_attempt", {})
    row = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "participant_name": config["participant_name"],
        "experiment_name": config["experiment_name"],
        "prompt_name": config["prompt_name"],
        "prompt_version": attempt.get("prompt_version", ""),
        "overall_score": f"{float(output['overall_score']):.6f}",
        "task_model": config["task_model"],
        "judge_model": config["judge_model"],
        "dataset_name": config["dataset_name"],
        "item_count": str(output["item_count"]),
        "dataset_run_url": output.get("dataset_run_url") or "",
        "prompt_hash": attempt.get("prompt_hash", ""),
        "scores_json": json.dumps(output["dimension_scores"], sort_keys=True),
    }
    for name, value in output["dimension_scores"].items():
        row[f"score:{name}"] = f"{float(value):.6f}"

    rows: list[dict[str, str]] = []
    columns: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            columns = list(reader.fieldnames or [])
            rows = [dict(existing) for existing in reader]
    for column in row:
        if column not in columns:
            columns.append(column)
    rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for existing in rows:
            writer.writerow(existing)
    return path


def _repo_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=HERE,
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip())
    except Exception:
        return HERE.parent.parent


def _render_template(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", _stringify(value))
    return rendered


def _resolve_mapping(source: str, *, input: dict[str, Any], output: str, expected_output: dict[str, Any]) -> Any:
    if source == "output":
        return output
    if source == "input":
        return input
    if source == "expected_output":
        return expected_output
    if source.startswith("input."):
        return _dig(input, source.removeprefix("input."))
    if source.startswith("expected_output."):
        return _dig(expected_output, source.removeprefix("expected_output."))
    return source


def _default_variable_value(variable: str, *, input: dict[str, Any], output: str, expected_output: dict[str, Any]) -> Any:
    if variable in {"generation", "answer", "output"}:
        return output
    if variable in {"query", "question", "input"}:
        return _first_string(input) or input
    if variable in {"ground_truth", "expected_output"}:
        return expected_output
    if variable == "context":
        return input.get("context") or expected_output.get("context") or ""
    return ""


def _dig(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return current


def _first_string(value: dict[str, Any]) -> str:
    for item in value.values():
        if isinstance(item, str):
            return item
    return ""


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _clamp_score(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    main()
