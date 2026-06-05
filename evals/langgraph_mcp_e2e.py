from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from langfuse import Langfuse

from clients.langgraph_eval_client import LangGraphEvalClient
from evals.scorers import aggregate_scores, apply_guardrails, score_e2e_case


async def run_langgraph_mcp_e2e(
    config: dict[str, Any],
    workdir: Path,
    langfuse: Langfuse,
    update_leaderboard: Callable[[dict, dict, str, str], dict],
) -> None:
    dataset = langfuse.get_dataset(config["dataset_name"])
    if not dataset.items:
        raise RuntimeError(f"Langfuse dataset has no items: {config['dataset_name']}")

    candidate_config = json.loads((workdir / config["candidate_file"]).read_text(encoding="utf-8-sig"))
    client = LangGraphEvalClient()
    case_results = []

    for item in dataset.items:
        item_input = item.input if isinstance(item.input, dict) else {"user_request": item.input}
        user_request = str(item_input.get("user_request") or item_input.get("input") or "")
        if not user_request:
            raise ValueError(f"Dataset item is missing user_request: {getattr(item, 'id', '')}")

        expected = item.expected_output if isinstance(item.expected_output, dict) else {}
        metadata = {
            "experiment_name": config["experiment_name"],
            "dataset_name": config["dataset_name"],
            "dataset_item_id": getattr(item, "id", None),
            "participant": config.get("participant"),
        }
        response = await client.run_case(
            user_request=user_request,
            candidate_config=candidate_config,
            expected_output=expected,
            metadata=metadata,
        )
        scored = score_e2e_case(response, expected, config.get("weights"))
        metrics = apply_guardrails(scored["metrics"], config.get("guardrails"))
        log_langgraph_scores(langfuse, response, metrics, scored["observations"], config, metadata)

        case_results.append(
            {
                "dataset_item_id": getattr(item, "id", None),
                "input": item_input,
                "expected_output": expected,
                "response": response,
                "scores": metrics,
                "observation_scores": scored["observations"],
            }
        )

    aggregate = aggregate_scores([case["scores"] for case in case_results], config.get("weights"))
    aggregate = apply_guardrails(aggregate, config.get("guardrails"))
    summary = {
        "experiment_name": config["experiment_name"],
        "kind": config["kind"],
        "dataset_name": config["dataset_name"],
        "candidate_file": config["candidate_file"],
        "item_count": len(case_results),
        "scores": aggregate,
        "cases": case_results,
    }
    (workdir / "latest_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    row = update_leaderboard(
        config,
        {"overall_score": aggregate.get("overall_score", 0.0), "dimensions": aggregate},
        "",
        "",
    )
    langfuse.flush()
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2))
    print(f"METRIC overall_score={aggregate.get('overall_score', 0.0):.4f}")
    for name, value in sorted(aggregate.items()):
        if name != "overall_score" and isinstance(value, (int, float)):
            print(f"METRIC {name}={float(value):.4f}")
    print(f"LEADERBOARD_ROW {json.dumps(row, sort_keys=True)}")


def log_langgraph_scores(
    langfuse: Langfuse,
    response: dict[str, Any],
    metrics: dict[str, float],
    observations: list[dict[str, Any]],
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    trace_id = response.get("trace_id")
    if not trace_id:
        return

    comment = f"{config['experiment_name']}; dataset_item_id={metadata.get('dataset_item_id')}"
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            langfuse.create_score(trace_id=trace_id, name=name, value=float(value), comment=comment)

    for observation in observations:
        observation_id = observation.get("observation_id")
        if not observation_id:
            continue
        tool_name = observation.get("model_tool_name")
        langfuse.create_score(
            trace_id=trace_id,
            observation_id=observation_id,
            name="obs_expected_tool_called",
            value=float(observation.get("called", 0.0)),
            comment=f"{comment}; expected_tool={tool_name}",
        )
        langfuse.create_score(
            trace_id=trace_id,
            observation_id=observation_id,
            name="obs_arg_quality",
            value=float(observation.get("arg_quality", 0.0)),
            comment=f"{comment}; expected_tool={tool_name}",
        )
