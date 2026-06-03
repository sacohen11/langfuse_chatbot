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
    parser = argparse.ArgumentParser(description="Evaluate a hosted MCP tool spec with Langfuse dataset runs.")
    parser.add_argument("--config", default="mcp_tool_config.json", help="MCP tool config JSON.")
    parser.add_argument("--spec-file", default=None, help="Candidate MCP tool spec JSON file.")
    parser.add_argument("--publish", action="store_true", help="Publish a Langfuse dataset run.")
    parser.add_argument("--validate-config", action="store_true", help="Validate config shape and exit.")
    parser.add_argument("--run-name", default=None, help="Optional exact Langfuse dataset run name.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit dataset items for smoke tests.")
    parser.add_argument("--result-file", default=None, help="Optional JSON file to write the result.")
    parser.add_argument("--no-save-prompt-version", action="store_true", help="Skip saving this attempt as a Langfuse prompt version.")
    parser.add_argument("--no-leaderboard", action="store_true", help="Skip updating the repo-level leaderboard.csv.")
    args = parser.parse_args()

    repo_root = _repo_root()
    load_dotenv(repo_root / ".env")
    load_dotenv(HERE / ".env")

    config_path = _resolve_path(args.config)
    config = _load_config(config_path)

    if args.validate_config:
        print(json.dumps({"ok": True, "scores": [score["name"] for score in config["scores"]]}, indent=2))
        return

    spec_path = _resolve_path(args.spec_file or config.get("candidate_tool_spec_file", "tool_candidate.json"))
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)
    tools = _tools(spec)
    if not tools:
        raise ValueError(f"{spec_path} does not contain any tools.")

    langfuse = _langfuse_client()
    dataset = langfuse.get_dataset(config["dataset_name"])
    items = dataset.items[: args.max_items] if args.max_items else dataset.items
    if not items:
        raise RuntimeError(f"Dataset {config['dataset_name']!r} has no items.")

    openai = OpenAI()

    def task(*, item, **_: Any) -> str:
        response = openai.chat.completions.create(
            model=config["task_model"],
            temperature=0,
            max_completion_tokens=350,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are testing hosted MCP tool specifications. Given a user request, decide whether "
                        "the agent should call one of these tools. Return JSON only with: should_call_tool "
                        "(boolean), tool_name (string or null), arguments (object), expected_result_usage "
                        "(string), and rationale (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "mcp_server": config["server"],
                            "tool_spec": spec,
                            "user_request": item.input["user_request"],
                            "available_context": item.input.get("available_context", {}),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        return response.choices[0].message.content or "{}"

    evaluators = [_make_evaluator(openai, config, score_config, spec) for score_config in config["scores"]]
    score_names = [score["name"] for score in config["scores"]]

    if args.publish:
        run_name = args.run_name or f"{config['experiment_name']}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        result = langfuse.run_experiment(
            name=config["experiment_name"],
            run_name=run_name,
            description=f"Hosted MCP tool spec evaluation for {config['server']['name']}",
            data=items,
            task=task,
            evaluators=evaluators,
            run_evaluators=[_make_aggregate_evaluator(score_names)],
            max_concurrency=1,
            metadata={
                "participant_name": config["participant_name"],
                "prompt_name": config["prompt_name"],
                "server_name": config["server"]["name"],
                "mcp_url": config["server"]["mcp_url"],
                "tool_names": [tool["name"] for tool in tools],
                "task_model": config["task_model"],
                "judge_model": config["judge_model"],
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
        "server_name": config["server"]["name"],
        "mcp_url": config["server"]["mcp_url"],
        "tool_names": [tool["name"] for tool in tools],
        "spec_file": str(spec_path),
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
            spec=spec,
            spec_text=spec_text,
            spec_path=spec_path,
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
        raise FileNotFoundError(f"Missing {path}")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    required = ["participant_name", "experiment_name", "prompt_name", "dataset_name", "task_model", "judge_model", "server", "scores"]
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")
    server = config["server"]
    for key in ["name", "mcp_url"]:
        if not server.get(key):
            raise ValueError(f"server.{key} is required")
    if not isinstance(config["scores"], list) or not config["scores"]:
        raise ValueError("Config must include at least one score.")
    for score in config["scores"]:
        if not score.get("name") or not score.get("type"):
            raise ValueError("Each score must include name and type.")
    return config


def _langfuse_client() -> Langfuse:
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )


def _make_evaluator(openai: OpenAI, config: dict[str, Any], score_config: dict[str, Any], spec: dict[str, Any]):
    if score_config["type"] == "llm_judge":
        return _make_llm_judge_evaluator(openai, config["judge_model"], score_config, spec)
    if score_config["type"] == "managed_langfuse":
        evaluator = _get_managed_langfuse_evaluator(score_config["evaluator_name"])
        return _make_managed_langfuse_evaluator(openai, config["judge_model"], score_config, evaluator)
    raise ValueError(f"Unsupported score type: {score_config['type']}")


def _make_llm_judge_evaluator(openai: OpenAI, judge_model: str, score_config: dict[str, Any], spec: dict[str, Any]):
    name = score_config["name"]
    rubric = score_config["rubric"]

    def evaluator(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        judge_payload = {
            "dimension": name,
            "rubric": rubric,
            "tool_spec": spec,
            "scenario": input,
            "model_tool_decision": output,
            "expected_output": expected_output,
            "instructions": "Return JSON with score from 0 to 1 and a short rationale. Be strict but fair.",
        }
        response = openai.chat.completions.create(
            model=judge_model,
            temperature=0,
            max_completion_tokens=260,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a precise evaluator for MCP tool descriptions and JSON schemas."},
                {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)},
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
    evaluator_name = evaluator_config.get("name", score_config["evaluator_name"])
    evaluator_version = evaluator_config.get("version")

    def evaluator(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        values = {
            variable: _resolve_mapping(source, input=input, output=output, expected_output=expected_output)
            for variable, source in score_config.get("variable_map", {}).items()
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
                {"role": "system", "content": "Run the provided Langfuse managed evaluator. Return JSON only with score and reasoning."},
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
    spec: dict[str, Any],
    spec_text: str,
    spec_path: Path,
    scores: dict[str, Any],
    dataset_run_url: str | None,
    item_count: int,
    max_items: int | None,
) -> dict[str, Any]:
    attempt_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    prompt_hash = hashlib.sha256(spec_text.encode("utf-8")).hexdigest()[:12]
    tools = _tools(spec)
    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=spec_text,
        labels=[],
        tags=["mcp-tool-spec", "hosted-mcp", "autoresearch-attempt"],
        config={
            "artifact_type": "hosted_mcp_tool_spec",
            "participant_name": config["participant_name"],
            "experiment_name": config["experiment_name"],
            "attempt_id": attempt_id,
            "prompt_hash": prompt_hash,
            "server": config["server"],
            "tool_names": [tool["name"] for tool in tools],
            "tool_spec": spec,
            "overall_score": scores["overall_score"],
            "dimension_scores": scores["dimension_scores"],
            "item_count": item_count,
            "max_items": max_items,
            "dataset_name": config["dataset_name"],
            "dataset_run_url": dataset_run_url,
            "task_model": config["task_model"],
            "judge_model": config["judge_model"],
            "source": spec_path.name,
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
        "server_name": config["server"]["name"],
        "mcp_url": config["server"]["mcp_url"],
        "tool_names": [tool["name"] for tool in tools],
        "tool_spec": spec,
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimension_scores"],
        "item_count": item_count,
        "max_items": max_items,
        "dataset_run_url": dataset_run_url,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _append_attempt_log(config: dict[str, Any], attempt: dict[str, Any]) -> None:
    path = HERE / config.get("attempt_log_file", "tool_attempts.jsonl")
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
        "artifact_type": "hosted_mcp_tool_spec",
        "prompt_name": config["prompt_name"],
        "prompt_version": attempt.get("prompt_version", ""),
        "overall_score": f"{float(output['overall_score']):.6f}",
        "task_model": config["task_model"],
        "judge_model": config["judge_model"],
        "dataset_name": config["dataset_name"],
        "item_count": str(output["item_count"]),
        "dataset_run_url": output.get("dataset_run_url") or "",
        "prompt_hash": attempt.get("prompt_hash", ""),
        "server_name": config["server"]["name"],
        "mcp_url": config["server"]["mcp_url"],
        "tool_names": ",".join(output.get("tool_names", [])),
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
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _tools(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "tools" in spec:
        return list(spec["tools"])
    return [spec]


def _resolve_mapping(source: str, *, input: dict[str, Any], output: str, expected_output: dict[str, Any]) -> Any:
    if source == "output":
        return output
    if source.startswith("input."):
        return _dig(input, source.removeprefix("input."))
    if source.startswith("expected_output."):
        return _dig(expected_output, source.removeprefix("expected_output."))
    return source


def _default_variable_value(variable: str, *, input: dict[str, Any], output: str, expected_output: dict[str, Any]) -> Any:
    if variable in {"generation", "answer", "output"}:
        return output
    if variable in {"query", "question", "input"}:
        return str(input.get("user_request", input))
    if variable in {"ground_truth", "expected_output"}:
        return expected_output
    return ""


def _dig(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part, "")
        else:
            return ""
    return current


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _clamp_score(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else HERE / candidate


def _repo_root() -> Path:
    try:
        result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, check=True, capture_output=True, text=True)
        return Path(result.stdout.strip())
    except Exception:
        return HERE.parent.parent


if __name__ == "__main__":
    main()
