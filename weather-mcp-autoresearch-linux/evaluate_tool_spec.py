from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import statistics
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langfuse import Evaluation, Langfuse
from langfuse.openai import OpenAI


HERE = Path(__file__).resolve().parent
JUDGE_DIMENSIONS = {
    "invocation_accuracy": "Does the model choose whether to call the right weather tool and provide correct, minimal arguments?",
    "input_schema_quality": "Do the input schemas make required fields, location limits, and argument boundaries clear?",
    "output_schema_quality": "Do the output schemas include the fields needed for useful downstream weather responses?",
    "description_clarity": "Are the tool descriptions specific, concise, and clear about when to use or not use each tool?",
}
MANAGED_CONCISENESS_EVALUATOR = "Conciseness"
CONCISENESS_DIMENSION = "conciseness"
SCORE_DIMENSIONS = tuple(JUDGE_DIMENSIONS) + (CONCISENESS_DIMENSION,)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an MCP tool spec with Langfuse dataset runs.")
    parser.add_argument("--spec-file", default=None, help="MCP tool spec JSON file to evaluate.")
    parser.add_argument("--publish", action="store_true", help="Publish a linked dataset run in Langfuse.")
    parser.add_argument("--run-name", default=None, help="Optional exact Langfuse dataset run name.")
    parser.add_argument("--max-items", type=int, default=None, help="Limit dataset items for smoke tests.")
    parser.add_argument("--result-file", default=None, help="Optional JSON file to write the evaluation result.")
    parser.add_argument(
        "--no-save-prompt-version",
        action="store_true",
        help="Skip creating a Langfuse prompt version for this evaluated attempt.",
    )
    args = parser.parse_args()

    load_dotenv(HERE / ".env")
    config = json.loads((HERE / "experiment_config.json").read_text(encoding="utf-8"))

    spec_path = Path(args.spec_file or HERE / config["candidate_tool_spec_file"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec_text = json.dumps(spec, indent=2, ensure_ascii=False)

    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
    )
    dataset = langfuse.get_dataset(config["dataset_name"])
    items = dataset.items[: args.max_items] if args.max_items else dataset.items
    if not items:
        raise RuntimeError(f"Dataset {config['dataset_name']!r} has no items. Run setup_langfuse.py first.")

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
                        "You are testing MCP weather tool specifications. Given a user request, decide whether the "
                        "agent should call one of these tools. Return JSON only with: should_call_tool (boolean), "
                        "tool_name (string or null), arguments (object), expected_result_usage (string), and "
                        "rationale (string)."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
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

    conciseness_evaluator = _get_managed_langfuse_evaluator(MANAGED_CONCISENESS_EVALUATOR)
    evaluators = [
        *[
            _make_judge_evaluator(openai, config["judge_model"], name, rubric, spec)
            for name, rubric in JUDGE_DIMENSIONS.items()
        ],
        _make_managed_conciseness_evaluator(openai, config["judge_model"], conciseness_evaluator),
    ]

    if args.publish:
        run_name = args.run_name or f"mcp-tool-spec-eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        result = langfuse.run_experiment(
            name="mcp-tool-spec-evaluation",
            run_name=run_name,
            description="LLM-as-a-judge evaluation for an MCP tool description and schemas.",
            data=items,
            task=task,
            evaluators=evaluators,
            run_evaluators=[_aggregate_evaluator],
            max_concurrency=1,
            metadata={
                "prompt_name": config["prompt_name"],
                "spec_file": spec_path.name,
                "server_name": spec["server"]["name"],
                "tool_names": [tool["name"] for tool in spec["tools"]],
                "task_model": config["task_model"],
                "judge_model": config["judge_model"],
            },
        )
        scores = _scores_from_experiment_result(result)
        dataset_run_url = getattr(result, "dataset_run_url", None)
    else:
        scores = _run_locally(items, task, evaluators)
        dataset_run_url = None

    langfuse.flush()
    output = {
        "spec_file": str(spec_path),
        "prompt_name": config["prompt_name"],
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

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"METRIC overall_score={scores['overall_score']:.4f}")
    for name, value in scores["dimension_scores"].items():
        print(f"METRIC {name}={value:.4f}")
    if args.result_file:
        Path(args.result_file).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")


def _make_judge_evaluator(openai: OpenAI, judge_model: str, name: str, rubric: str, spec: dict[str, Any]):
    def evaluator(*, input: dict[str, Any], output: str, expected_output: dict[str, Any], **_: Any) -> Evaluation:
        score, rationale = _judge(openai, judge_model, name, rubric, spec, input, output, expected_output)
        return Evaluation(name=name, value=score, comment=rationale)

    evaluator.__name__ = f"{name}_judge"
    return evaluator


def _make_managed_conciseness_evaluator(openai: OpenAI, judge_model: str, evaluator: dict[str, Any]):
    evaluator_prompt = evaluator["prompt"]
    evaluator_version = evaluator.get("version")

    def managed_conciseness(*, input: dict[str, Any], output: str, **_: Any) -> Evaluation:
        query = str(input.get("user_request", input))
        prompt = evaluator_prompt.replace("{{query}}", query).replace("{{generation}}", output)
        response = openai.chat.completions.create(
            model=judge_model,
            temperature=0,
            max_completion_tokens=220,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Run the provided Langfuse managed evaluator. Return JSON only with "
                        "`score` from 0 to 1 and `reasoning` as one concise sentence."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return Evaluation(name=CONCISENESS_DIMENSION, value=0.0, comment=f"Evaluator returned invalid JSON: {raw[:160]}")
        score = max(0.0, min(1.0, float(data.get("score", 0.0))))
        reasoning = str(data.get("reasoning", data.get("rationale", "")))
        return Evaluation(
            name=CONCISENESS_DIMENSION,
            value=score,
            comment=f"Langfuse managed {MANAGED_CONCISENESS_EVALUATOR} v{evaluator_version}: {reasoning}",
        )

    managed_conciseness.__name__ = "managed_conciseness_judge"
    return managed_conciseness


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


def _judge(
    openai: OpenAI,
    judge_model: str,
    dimension_name: str,
    rubric: str,
    spec: dict[str, Any],
    item_input: dict[str, Any],
    tool_decision_output: str,
    expected_output: dict[str, Any],
) -> tuple[float, str]:
    judge_payload = {
        "dimension": dimension_name,
        "rubric": rubric,
        "tool_spec": spec,
        "scenario": item_input,
        "model_tool_decision": tool_decision_output,
        "expected_output": expected_output,
        "instructions": "Return JSON with score from 0 to 1 and a short rationale. Be strict but fair.",
    }
    response = openai.chat.completions.create(
        model=judge_model,
        temperature=0,
        max_completion_tokens=260,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a precise evaluator for MCP tool descriptions and JSON schemas.",
            },
            {"role": "user", "content": json.dumps(judge_payload, ensure_ascii=False)},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return 0.0, f"Judge returned invalid JSON: {raw[:160]}"
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return score, str(data.get("rationale", ""))


def _aggregate_evaluator(*, item_results: list[Any], **_: Any) -> Evaluation:
    values: list[float] = []
    for item_result in item_results:
        for evaluation in getattr(item_result, "evaluations", []) or []:
            if getattr(evaluation, "name", None) in SCORE_DIMENSIONS:
                values.append(float(evaluation.value))
    return Evaluation(name="overall_score", value=statistics.mean(values) if values else 0.0)


def _scores_from_experiment_result(result: Any) -> dict[str, Any]:
    values_by_name: dict[str, list[float]] = {name: [] for name in SCORE_DIMENSIONS}
    for item_result in getattr(result, "item_results", []) or []:
        for evaluation in getattr(item_result, "evaluations", []) or []:
            if getattr(evaluation, "name", None) in values_by_name:
                values_by_name[evaluation.name].append(float(evaluation.value))

    dimension_scores = {
        name: statistics.mean(values) if values else 0.0
        for name, values in values_by_name.items()
    }
    overall = statistics.mean(dimension_scores.values()) if dimension_scores else 0.0
    return {"overall_score": overall, "dimension_scores": dimension_scores}


def _run_locally(items: list[Any], task, evaluators: list[Any]) -> dict[str, Any]:
    values_by_name: dict[str, list[float]] = {name: [] for name in SCORE_DIMENSIONS}
    for item in items:
        output = task(item=item)
        for evaluator in evaluators:
            evaluation = evaluator(input=item.input, output=output, expected_output=item.expected_output, metadata=item.metadata)
            values_by_name[evaluation.name].append(float(evaluation.value))
    dimension_scores = {
        name: statistics.mean(values) if values else 0.0
        for name, values in values_by_name.items()
    }
    return {
        "overall_score": statistics.mean(dimension_scores.values()) if dimension_scores else 0.0,
        "dimension_scores": dimension_scores,
    }


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
    prompt = langfuse.create_prompt(
        name=config["prompt_name"],
        type="text",
        prompt=spec_text,
        labels=[],
        tags=["mcp-tool-spec", "autoresearch", "autoresearch-attempt"],
        config={
            "artifact_type": "mcp_weather_server_tool_spec",
            "attempt_id": attempt_id,
            "prompt_hash": prompt_hash,
            "server_name": spec["server"]["name"],
            "tool_names": [tool["name"] for tool in spec["tools"]],
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
        commit_message=f"Autoresearch MCP weather tool spec attempt {attempt_id}: overall_score={scores['overall_score']:.4f}",
    )
    return {
        "attempt_id": attempt_id,
        "prompt_name": config["prompt_name"],
        "prompt_version": getattr(prompt, "version", None),
        "prompt_hash": prompt_hash,
        "server_name": spec["server"]["name"],
        "tool_names": [tool["name"] for tool in spec["tools"]],
        "tool_spec": spec,
        "tool_spec_text": spec_text,
        "overall_score": scores["overall_score"],
        "dimension_scores": scores["dimension_scores"],
        "item_count": item_count,
        "max_items": max_items,
        "dataset_run_url": dataset_run_url,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _append_attempt_log(config: dict[str, Any], attempt: dict[str, Any]) -> None:
    path = HERE / config.get("attempt_log_file", "autoresearch.prompt_attempts.jsonl")
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(attempt, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
