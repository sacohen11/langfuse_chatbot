from langfuse_chatbot.chatbot import ChatResponse, ToolCallRecord
from langfuse_chatbot.evaluation.models import ConversationScenario, TurnExpectation
from langfuse_chatbot.evaluation.runner import evaluate_chatbot


class FakeChatbot:
    def __init__(self) -> None:
        self._tool_calls: list[ToolCallRecord] = []

    async def achat(self, user_text: str, *, session_id: str | None = None, **kwargs: object) -> ChatResponse:
        if "multiply" in user_text:
            self._tool_calls.append(ToolCallRecord(name="calculator", args={"expression": "17*23"}))
            content = "17 multiplied by 23 is 391."
        elif "add 9" in user_text:
            self._tool_calls.append(ToolCallRecord(name="calculator", args={"expression": "391+9"}))
            content = "Adding 9 to the previous result gives 400."
        else:
            content = "I can answer from context."
        return ChatResponse(content=content, session_id=session_id or "test", tool_calls=list(self._tool_calls))


async def test_evaluate_chatbot_scores_multiturn_tool_use():
    scenario = ConversationScenario(
        id="math_memory",
        description="Use calculator and remember the result.",
        turns=[
            TurnExpectation(
                user="Use the calculator to multiply 17 by 23.",
                answer_contains=["391"],
                required_tools=["calculator"],
            ),
            TurnExpectation(
                user="Now add 9 to that previous result.",
                answer_contains=["400"],
                required_tools=["calculator"],
            ),
        ],
    )

    report = await evaluate_chatbot(FakeChatbot(), [scenario])

    assert report.overall_score == 1.0
    assert report.required_tool_coverage == 1.0
    assert report.answer_quality == 1.0


class NoToolChatbot:
    async def achat(self, user_text: str, *, session_id: str | None = None, **kwargs: object) -> ChatResponse:
        return ChatResponse(content="The answer is 391.", session_id=session_id or "test", tool_calls=[])


async def test_evaluate_chatbot_penalizes_missing_required_tools():
    scenario = ConversationScenario(
        id="missing_tool",
        description="Requires a calculator call.",
        turns=[
            TurnExpectation(
                user="Use the calculator to multiply 17 by 23.",
                answer_contains=["391"],
                required_tools=["calculator"],
            )
        ],
    )

    report = await evaluate_chatbot(NoToolChatbot(), [scenario])

    assert report.answer_quality == 1.0
    assert report.required_tool_coverage == 0.0
    assert report.overall_score < 1.0


class PrefixedToolChatbot:
    async def achat(self, user_text: str, *, session_id: str | None = None, **kwargs: object) -> ChatResponse:
        return ChatResponse(
            content="The answer is 391.",
            session_id=session_id or "test",
            tool_calls=[ToolCallRecord(name="math_calculator")],
        )


async def test_evaluate_chatbot_matches_prefixed_mcp_tool_names():
    scenario = ConversationScenario(
        id="prefixed_tool",
        description="MCP adapter may prefix tool names with the server.",
        turns=[
            TurnExpectation(
                user="Use the calculator to multiply 17 by 23.",
                answer_contains=["391"],
                required_tools=["calculator"],
            )
        ],
    )

    report = await evaluate_chatbot(PrefixedToolChatbot(), [scenario])

    assert report.required_tool_coverage == 1.0
