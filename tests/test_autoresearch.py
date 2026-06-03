from __future__ import annotations

from dataclasses import dataclass

from langfuse_chatbot.autoresearch import (
    PromptSnapshot,
    ResearchCandidate,
    ResearchResult,
    prepare_autoresearch,
    research_result_from_response,
    run_autoresearch,
    save_autoresearch_result,
    write_json,
)
from langfuse_chatbot.config import AppConfig, AutoresearchConfig, LangfuseConfig


@dataclass
class FakePrompt:
    name: str
    type: str
    prompt: str
    version: int
    config: dict
    labels: list[str]
    tags: list[str]


class FakeLangfuse:
    def __init__(self) -> None:
        self.created = []
        self.experiments = []
        self.flushed = False

    def get_prompt(self, name, **kwargs):
        assert kwargs["type"] == "text"
        assert kwargs["label"] == "production"
        assert kwargs["cache_ttl_seconds"] == 0
        return FakePrompt(
            name=name,
            type="text",
            prompt="Answer using {{tone}} detail.",
            version=4,
            config={"model": "gpt-4.1-mini"},
            labels=["production"],
            tags=["support"],
        )

    def run_experiment(self, **kwargs):
        self.experiments.append(kwargs)
        output = kwargs["task"](item=kwargs["data"][0])
        score = kwargs["evaluators"][0](output=output)
        return {"name": kwargs["name"], "score": score.value}

    def create_prompt(self, **kwargs):
        self.created.append(kwargs)
        return FakePrompt(
            name=kwargs["name"],
            type=kwargs["type"],
            prompt=kwargs["prompt"],
            version=5,
            config=kwargs["config"],
            labels=kwargs["labels"],
            tags=kwargs["tags"],
        )

    def flush(self):
        self.flushed = True


class FakeResearcher:
    def optimize(self, snapshot: PromptSnapshot, *, context: str, candidate_count: int) -> ResearchResult:
        assert snapshot.prompt == "Answer using {{tone}} detail."
        assert context == "make it sharper"
        assert candidate_count == 2
        return ResearchResult(
            candidates=[
                ResearchCandidate(prompt="Answer briefly using {{tone}} detail.", score=0.7),
                ResearchCandidate(prompt="Answer precisely using {{tone}} detail.", score=0.95, rationale="Clearer."),
            ],
            winner_index=1,
            report={"method": "fake"},
        )


def test_run_autoresearch_saves_winner_and_local_artifacts(tmp_path):
    config = AppConfig(
        langfuse=LangfuseConfig(public_key="pk", secret_key="sk"),
        autoresearch=AutoresearchConfig(workspace_dir=str(tmp_path), candidate_count=2),
    )
    langfuse = FakeLangfuse()

    run = run_autoresearch(
        config=config,
        prompt_name="support/triage",
        context="make it sharper",
        researcher=FakeResearcher(),
        langfuse_client=langfuse,
    )

    assert run.result.winner.prompt == "Answer precisely using {{tone}} detail."
    assert run.saved_prompt["version"] == 5
    assert langfuse.created[0]["prompt"] == "Answer precisely using {{tone}} detail."
    assert langfuse.created[0]["labels"] == ["autoresearch"]
    assert langfuse.created[0]["config"]["autoresearch"]["source_prompt_version"] == 4
    assert langfuse.experiments
    assert langfuse.flushed
    assert (run.run_dir / "source_prompt.json").exists()
    assert (run.run_dir / "optimized_prompt.json").exists()
    assert (run.run_dir / "saved_prompt.json").exists()


def test_prepare_and_save_folder_workflow(tmp_path):
    config = AppConfig(
        langfuse=LangfuseConfig(public_key="pk", secret_key="sk"),
        autoresearch=AutoresearchConfig(workspace_dir=str(tmp_path), candidate_count=2),
    )
    langfuse = FakeLangfuse()
    prep = prepare_autoresearch(
        config=config,
        prompt_name="support/triage",
        context="make it sharper",
        langfuse_client=langfuse,
    )

    assert (prep.run_dir / "task.json").exists()
    assert (prep.run_dir / "source_prompt.json").exists()
    assert (prep.run_dir / "prompt.txt").exists()

    write_json(
        prep.run_dir / "optimized_prompt.json",
        {
            "candidates": [
                {"prompt": "Candidate one", "score": 0.5},
                {"prompt": "Candidate two", "score": 0.9, "rationale": "best"},
            ]
        },
    )

    run = save_autoresearch_result(
        config=config,
        run_dir=prep.run_dir,
        langfuse_client=langfuse,
        publish_experiment=False,
    )

    assert run.result.winner.prompt == "Candidate two"
    assert langfuse.created[-1]["prompt"] == "Candidate two"
    assert langfuse.created[-1]["labels"] == ["autoresearch"]


def test_research_result_from_response_selects_highest_score():
    result = research_result_from_response(
        {
            "candidates": [
                {"prompt": "first", "score": 0.2},
                {"prompt": "second", "score": 0.8, "rationale": "better"},
            ],
            "report": {"notes": "ok"},
        },
        prompt_type="text",
    )

    assert result.winner_index == 1
    assert result.winner.prompt == "second"
    assert result.winner.rationale == "better"
    assert result.report == {"notes": "ok"}


def test_research_result_from_response_accepts_chat_string_candidate():
    result = research_result_from_response(
        {"optimized_prompt": "You are concise.", "score": 1},
        prompt_type="chat",
    )

    assert result.winner.prompt == [{"role": "system", "content": "You are concise."}]
