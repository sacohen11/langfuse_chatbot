from __future__ import annotations

from typing import Callable, Awaitable

from langfuse_chatbot.chatbot import LangChainMCPChatbot
from langfuse_chatbot.config import AppConfig

from .models import ConversationScenario, ScenarioResult
from .runner import run_scenario


AgentFactory = Callable[[], Awaitable[LangChainMCPChatbot]]


async def publish_experiment(
    *,
    config: AppConfig,
    scenarios: list[ConversationScenario],
    agent_factory: AgentFactory,
    name: str = "agentic-chatbot-multiturn-tool-eval",
) -> object:
    """Run scenarios through Langfuse's experiment runner."""

    from langfuse import Evaluation, get_client

    chatbot = await agent_factory()
    langfuse = get_client()

    async def task(*, item: dict[str, object], **_: object) -> dict[str, object]:
        scenario = _scenario_from_item(item)
        result = await run_scenario(chatbot, scenario)
        return result_to_dict(result)

    def overall_score(*, output: dict[str, object], **_: object) -> Evaluation:
        return Evaluation(name="overall_score", value=float(output["score"]))

    def required_tool_coverage(*, output: dict[str, object], **_: object) -> Evaluation:
        return Evaluation(name="required_tool_coverage", value=float(output["required_tool_coverage"]))

    def answer_quality(*, output: dict[str, object], **_: object) -> Evaluation:
        return Evaluation(name="answer_quality", value=float(output["answer_quality"]))

    def forbidden_tool_avoidance(*, output: dict[str, object], **_: object) -> Evaluation:
        return Evaluation(name="forbidden_tool_avoidance", value=float(output["forbidden_tool_avoidance"]))

    return langfuse.run_experiment(
        name=name,
        description="Multi-turn agentic chatbot evaluation with MCP tool-call scoring.",
        data=[scenario_to_item(scenario) for scenario in scenarios],
        task=task,
        evaluators=[
            overall_score,
            required_tool_coverage,
            answer_quality,
            forbidden_tool_avoidance,
        ],
        max_concurrency=1,
        metadata={
            "model": config.llm.model,
            "mcp_servers": sorted(config.enabled_mcp_servers.keys()),
        },
    )


def scenario_to_item(scenario: ConversationScenario) -> dict[str, object]:
    return {
        "input": {
            "id": scenario.id,
            "description": scenario.description,
            "tags": scenario.tags,
            "turns": [
                {
                    "user": turn.user,
                    "answer_contains": turn.answer_contains,
                    "required_tools": turn.required_tools,
                    "forbidden_tools": turn.forbidden_tools,
                }
                for turn in scenario.turns
            ],
        },
        "expected_output": {
            "min_score": 0.85,
            "required_tools": sorted({tool for turn in scenario.turns for tool in turn.required_tools}),
        },
        "metadata": {"scenario_id": scenario.id, "tags": scenario.tags},
    }


def result_to_dict(result: ScenarioResult) -> dict[str, object]:
    return {
        "scenario_id": result.scenario_id,
        "description": result.description,
        "score": result.score,
        "required_tool_coverage": result.required_tool_coverage,
        "answer_quality": result.answer_quality,
        "forbidden_tool_avoidance": result.forbidden_tool_avoidance,
        "turns": [
            {
                "user": turn.user,
                "output": turn.output,
                "tool_names": turn.tool_names,
                "answer_contains_score": turn.answer_contains_score,
                "required_tool_score": turn.required_tool_score,
                "forbidden_tool_score": turn.forbidden_tool_score,
            }
            for turn in result.turns
        ],
    }


def _scenario_from_item(item: dict[str, object]) -> ConversationScenario:
    payload = item["input"] if "input" in item else item
    if not isinstance(payload, dict):
        raise TypeError("Langfuse experiment item input must be a scenario object.")

    from .models import TurnExpectation

    return ConversationScenario(
        id=str(payload["id"]),
        description=str(payload.get("description", payload["id"])),
        tags=list(payload.get("tags", [])),
        turns=[
            TurnExpectation(
                user=str(turn["user"]),
                answer_contains=list(turn.get("answer_contains", [])),
                required_tools=list(turn.get("required_tools", [])),
                forbidden_tools=list(turn.get("forbidden_tools", [])),
            )
            for turn in payload["turns"]
            if isinstance(turn, dict)
        ],
    )
