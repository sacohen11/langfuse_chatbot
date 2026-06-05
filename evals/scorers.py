from __future__ import annotations

import json
from typing import Any


DEFAULT_WEIGHTS = {
    "tool_recall": 0.3,
    "args_quality": 0.25,
    "tool_order_correct": 0.1,
    "mcp_execution_success": 0.2,
    "answer_contains_expected": 0.1,
    "no_forbidden_tool_calls": 0.05,
}


def score_e2e_case(
    result: dict[str, Any],
    expected_output: dict[str, Any] | None,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    expected = expected_output or {}
    calls = _expected_calls(expected)
    tool_events = _tool_events(result)
    selected_tools = _selected_tools(result, tool_events)

    metrics = {
        "tool_recall": _tool_recall(calls, tool_events, selected_tools),
        "args_quality": _args_quality(calls, tool_events, result),
        "tool_order_correct": _tool_order(calls, selected_tools, expected.get("tool_order", "any")),
        "mcp_execution_success": _execution_success(tool_events, expected_calls=calls),
        "answer_contains_expected": _answer_contains(result.get("final_answer", ""), expected.get("expected_answer_contains", [])),
        "no_forbidden_tool_calls": 1.0,
        "forbidden_tool_call_count": float(_forbidden_count(selected_tools, expected.get("forbidden_tools", []))),
    }
    metrics["no_forbidden_tool_calls"] = 1.0 if metrics["forbidden_tool_call_count"] == 0 else 0.0
    metrics["overall_score"] = _weighted(metrics, weights or DEFAULT_WEIGHTS)

    observations = _observation_scores(calls, tool_events, result)
    return {"metrics": metrics, "observations": observations}


def apply_guardrails(metrics: dict[str, float], guardrails: dict[str, Any] | None) -> dict[str, float]:
    checked = dict(metrics)
    rules = guardrails or {}
    min_success = rules.get("min_mcp_execution_success")
    if min_success is not None and checked.get("mcp_execution_success", 0.0) < float(min_success):
        checked["overall_score"] = 0.0

    max_forbidden = rules.get("max_forbidden_tool_call_count")
    if max_forbidden is not None and checked.get("forbidden_tool_call_count", 0.0) > float(max_forbidden):
        checked["overall_score"] = 0.0
    return checked


def aggregate_scores(case_scores: list[dict[str, float]], weights: dict[str, float] | None = None) -> dict[str, float]:
    if not case_scores:
        return {"overall_score": 0.0}

    names = sorted({name for score in case_scores for name in score})
    aggregate = {}
    for name in names:
        values = [float(score.get(name, 0.0)) for score in case_scores]
        aggregate[name] = sum(values) if name == "forbidden_tool_call_count" else sum(values) / len(values)
    aggregate["overall_score"] = _weighted(aggregate, weights or DEFAULT_WEIGHTS)
    return aggregate


def _expected_calls(expected: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(expected.get("expected_tool_calls"), list):
        return [
            {
                "model_tool_name": call.get("model_tool_name") or call.get("tool_name"),
                "arg_contains": call.get("arg_contains", {}),
                "required": call.get("required", True),
            }
            for call in expected["expected_tool_calls"]
        ]

    if expected.get("expected_tool"):
        return [
            {
                "model_tool_name": expected["expected_tool"],
                "arg_contains": expected.get("expected_arg_contains", {}),
                "required": True,
            }
        ]

    return []


def _tool_events(result: dict[str, Any]) -> list[dict[str, Any]]:
    events = [
        step for step in result.get("steps", [])
        if isinstance(step, dict) and step.get("kind", "tool") == "tool"
    ]
    if events:
        return events

    events = []
    for tool_name in result.get("selected_tools", []) or []:
        args_list = result.get("tool_args", {}).get(tool_name, [{}])
        if isinstance(args_list, dict):
            args_list = [args_list]
        for args in args_list or [{}]:
            events.append({"model_tool_name": tool_name, "args": args, "success": True, "error": None})
    return events


def _selected_tools(result: dict[str, Any], tool_events: list[dict[str, Any]]) -> list[str]:
    if result.get("selected_tools"):
        return [str(tool) for tool in result.get("selected_tools", [])]
    return [str(event.get("model_tool_name") or event.get("tool_name") or "") for event in tool_events]


def _tool_recall(calls: list[dict[str, Any]], events: list[dict[str, Any]], selected_tools: list[str]) -> float:
    required = [call for call in calls if call.get("required", True)]
    if not required:
        return 1.0
    names = selected_tools or [str(event.get("model_tool_name", "")) for event in events]
    found = sum(1 for call in required if call.get("model_tool_name") in names)
    return found / len(required)


def _args_quality(calls: list[dict[str, Any]], events: list[dict[str, Any]], result: dict[str, Any]) -> float:
    if not calls:
        return 1.0

    values = []
    for call in calls:
        tool_name = call.get("model_tool_name")
        matching = [event for event in events if event.get("model_tool_name") == tool_name]
        if not matching and tool_name in (result.get("tool_args") or {}):
            matching = [{"args": args} for args in result["tool_args"].get(tool_name, [])]
        values.append(_best_arg_match(matching, call.get("arg_contains", {})))
    return sum(values) / len(values)


def _best_arg_match(events: list[dict[str, Any]], expected_args: dict[str, Any]) -> float:
    if not events:
        return 0.0
    if not expected_args:
        return 1.0
    return max(_arg_match(event.get("args", {}), expected_args) for event in events)


def _arg_match(args: Any, expected_args: dict[str, Any]) -> float:
    args_text = _json_text(args)
    scores = []
    for key, expected in expected_args.items():
        if isinstance(args, dict) and key in args:
            haystack = _json_text(args[key])
        else:
            haystack = args_text
        scores.append(1.0 if str(expected).lower() in haystack else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def _tool_order(calls: list[dict[str, Any]], selected_tools: list[str], mode: str) -> float:
    expected = [call["model_tool_name"] for call in calls if call.get("required", True) and call.get("model_tool_name")]
    if not expected:
        return 1.0
    if mode == "strict":
        return 1.0 if selected_tools[: len(expected)] == expected else 0.0
    if mode == "subsequence":
        return 1.0 if _is_subsequence(expected, selected_tools) else 0.0
    return 1.0


def _execution_success(events: list[dict[str, Any]], expected_calls: list[dict[str, Any]]) -> float:
    if not events:
        return 0.0 if expected_calls else 1.0
    values = [1.0 if event.get("success", False) is True and not event.get("error") else 0.0 for event in events]
    return sum(values) / len(values)


def _answer_contains(answer: str, expected_strings: list[str]) -> float:
    if not expected_strings:
        return 1.0
    answer_l = str(answer).lower()
    values = [1.0 if str(text).lower() in answer_l else 0.0 for text in expected_strings]
    return sum(values) / len(values)


def _forbidden_count(selected_tools: list[str], forbidden_tools: list[str]) -> int:
    forbidden = set(forbidden_tools or [])
    return sum(1 for tool in selected_tools if tool in forbidden)


def _observation_scores(
    calls: list[dict[str, Any]],
    events: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    observations = []
    for index, call in enumerate(calls):
        tool_name = call.get("model_tool_name")
        matching = [event for event in events if event.get("model_tool_name") == tool_name]
        called = bool(matching)
        arg_quality = _best_arg_match(matching, call.get("arg_contains", {}))
        observation_id = next((event.get("observation_id") for event in matching if event.get("observation_id")), None)
        observations.append(
            {
                "expected_index": index,
                "model_tool_name": tool_name,
                "called": 1.0 if called else 0.0,
                "arg_quality": arg_quality,
                "observation_id": observation_id,
            }
        )
    return observations


def _weighted(metrics: dict[str, float], weights: dict[str, float]) -> float:
    total_weight = sum(float(value) for value in weights.values())
    if total_weight <= 0:
        return 0.0
    return sum(float(metrics.get(name, 0.0)) * float(weight) for name, weight in weights.items()) / total_weight


def _is_subsequence(expected: list[str], selected: list[str]) -> bool:
    cursor = 0
    for tool in selected:
        if cursor < len(expected) and tool == expected[cursor]:
            cursor += 1
    return cursor == len(expected)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False).lower()
    except TypeError:
        return str(value).lower()
