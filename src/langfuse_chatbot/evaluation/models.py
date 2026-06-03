from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from langfuse_chatbot.chatbot import ChatResponse


@dataclass(frozen=True)
class TurnExpectation:
    user: str
    answer_contains: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConversationScenario:
    id: str
    description: str
    turns: list[TurnExpectation]
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TurnResult:
    user: str
    output: str
    tool_names: list[str]
    answer_contains_score: float
    required_tool_score: float
    forbidden_tool_score: float


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    description: str
    turns: list[TurnResult]

    @property
    def score(self) -> float:
        values: list[float] = []
        for turn in self.turns:
            values.extend(
                [
                    turn.answer_contains_score,
                    turn.required_tool_score,
                    turn.forbidden_tool_score,
                ]
            )
        return sum(values) / len(values) if values else 0.0

    @property
    def required_tool_coverage(self) -> float:
        values = [turn.required_tool_score for turn in self.turns]
        return sum(values) / len(values) if values else 1.0

    @property
    def answer_quality(self) -> float:
        values = [turn.answer_contains_score for turn in self.turns]
        return sum(values) / len(values) if values else 1.0

    @property
    def forbidden_tool_avoidance(self) -> float:
        values = [turn.forbidden_tool_score for turn in self.turns]
        return sum(values) / len(values) if values else 1.0


@dataclass(frozen=True)
class EvaluationReport:
    scenarios: list[ScenarioResult]

    @property
    def overall_score(self) -> float:
        return _average([scenario.score for scenario in self.scenarios])

    @property
    def required_tool_coverage(self) -> float:
        return _average([scenario.required_tool_coverage for scenario in self.scenarios])

    @property
    def answer_quality(self) -> float:
        return _average([scenario.answer_quality for scenario in self.scenarios])

    @property
    def forbidden_tool_avoidance(self) -> float:
        return _average([scenario.forbidden_tool_avoidance for scenario in self.scenarios])

    def as_dict(self) -> dict[str, object]:
        return {
            "overall_score": self.overall_score,
            "required_tool_coverage": self.required_tool_coverage,
            "answer_quality": self.answer_quality,
            "forbidden_tool_avoidance": self.forbidden_tool_avoidance,
            "scenarios": [
                {
                    "id": scenario.scenario_id,
                    "description": scenario.description,
                    "score": scenario.score,
                    "required_tool_coverage": scenario.required_tool_coverage,
                    "answer_quality": scenario.answer_quality,
                    "forbidden_tool_avoidance": scenario.forbidden_tool_avoidance,
                    "turns": [
                        {
                            "user": turn.user,
                            "output": turn.output,
                            "tool_names": turn.tool_names,
                            "answer_contains_score": turn.answer_contains_score,
                            "required_tool_score": turn.required_tool_score,
                            "forbidden_tool_score": turn.forbidden_tool_score,
                        }
                        for turn in scenario.turns
                    ],
                }
                for scenario in self.scenarios
            ],
        }


class AsyncChatbot(Protocol):
    async def achat(self, user_text: str, *, session_id: str | None = None, **kwargs: object) -> ChatResponse:
        ...


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
