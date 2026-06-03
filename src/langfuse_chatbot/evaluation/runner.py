from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from .models import (
    AsyncChatbot,
    ConversationScenario,
    EvaluationReport,
    ScenarioResult,
    TurnExpectation,
    TurnResult,
)


def load_scenarios(path: str | Path) -> list[ConversationScenario]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios: list[ConversationScenario] = []
    for item in data:
        scenarios.append(
            ConversationScenario(
                id=item["id"],
                description=item.get("description", item["id"]),
                tags=list(item.get("tags", [])),
                turns=[
                    TurnExpectation(
                        user=turn["user"],
                        answer_contains=list(turn.get("answer_contains", [])),
                        required_tools=list(turn.get("required_tools", [])),
                        forbidden_tools=list(turn.get("forbidden_tools", [])),
                    )
                    for turn in item["turns"]
                ],
            )
        )
    return scenarios


async def evaluate_chatbot(
    chatbot: AsyncChatbot,
    scenarios: list[ConversationScenario],
    *,
    user_id: str = "eval-user",
) -> EvaluationReport:
    results = []
    for scenario in scenarios:
        results.append(await run_scenario(chatbot, scenario, user_id=user_id))
    return EvaluationReport(results)


async def run_scenario(
    chatbot: AsyncChatbot,
    scenario: ConversationScenario,
    *,
    user_id: str = "eval-user",
) -> ScenarioResult:
    session_id = f"eval-{scenario.id}-{uuid4()}"
    turn_results: list[TurnResult] = []

    previous_tool_count = 0
    for turn in scenario.turns:
        response = await chatbot.achat(
            turn.user,
            session_id=session_id,
            user_id=user_id,
            tags=["evaluation", *scenario.tags],
            metadata={"scenario_id": scenario.id},
        )
        new_tool_calls = response.tool_calls[previous_tool_count:]
        previous_tool_count = len(response.tool_calls)
        tool_names = [call.name for call in new_tool_calls]

        turn_results.append(
            TurnResult(
                user=turn.user,
                output=response.content,
                tool_names=tool_names,
                answer_contains_score=_contains_score(response.content, turn.answer_contains),
                required_tool_score=_required_tool_score(tool_names, turn.required_tools),
                forbidden_tool_score=_forbidden_tool_score(tool_names, turn.forbidden_tools),
            )
        )

    return ScenarioResult(
        scenario_id=scenario.id,
        description=scenario.description,
        turns=turn_results,
    )


def _contains_score(output: str, expected_fragments: list[str]) -> float:
    if not expected_fragments:
        return 1.0
    lowered = output.lower()
    hits = sum(1 for fragment in expected_fragments if fragment.lower() in lowered)
    return hits / len(expected_fragments)


def _required_tool_score(tool_names: list[str], required_tools: list[str]) -> float:
    if not required_tools:
        return 1.0
    hits = sum(1 for tool in required_tools if _tool_was_used(tool, tool_names))
    return hits / len(required_tools)


def _forbidden_tool_score(tool_names: list[str], forbidden_tools: list[str]) -> float:
    if not forbidden_tools:
        return 1.0
    return 0.0 if any(_tool_was_used(tool, tool_names) for tool in forbidden_tools) else 1.0


def _tool_was_used(expected: str, actual_tool_names: list[str]) -> bool:
    return any(name == expected or name.endswith(f"_{expected}") or name.endswith(f".{expected}") for name in actual_tool_names)
