from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .config import AppConfig


PromptContent = str | list[dict[str, Any]]


@dataclass(frozen=True)
class PromptSnapshot:
    name: str
    type: str
    prompt: PromptContent
    version: int | None = None
    label: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchCandidate:
    prompt: PromptContent
    score: float
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchResult:
    candidates: list[ResearchCandidate]
    winner_index: int
    report: dict[str, Any] = field(default_factory=dict)

    @property
    def winner(self) -> ResearchCandidate:
        return self.candidates[self.winner_index]


@dataclass(frozen=True)
class AutoresearchRun:
    run_dir: Path
    source: PromptSnapshot
    result: ResearchResult
    saved_prompt: dict[str, Any]
    experiment_name: str


@dataclass(frozen=True)
class AutoresearchPrep:
    run_dir: Path
    source: PromptSnapshot
    task_path: Path


class PromptResearcher(Protocol):
    def optimize(
        self,
        snapshot: PromptSnapshot,
        *,
        context: str,
        candidate_count: int,
    ) -> ResearchResult:
        ...


def run_autoresearch(
    *,
    config: AppConfig,
    prompt_name: str,
    prompt_type: str = "text",
    label: str | None = "production",
    version: int | None = None,
    context: str = "",
    candidate_count: int | None = None,
    output_labels: list[str] | None = None,
    output_prompt_name: str | None = None,
    publish_experiment: bool = True,
    researcher: PromptResearcher | None = None,
    langfuse_client: Any | None = None,
) -> AutoresearchRun:
    """In-process convenience wrapper for callers that provide a researcher.

    The CLI uses the folder workflow instead: prepare a run directory, let the
    autoresearch harness produce result JSON, then save the result to Langfuse.
    """

    if researcher is None:
        raise RuntimeError("run_autoresearch requires a researcher. Use prepare_autoresearch and save_autoresearch_result for folder-based runs.")

    langfuse = langfuse_client or _build_langfuse_client(config)
    prep = prepare_autoresearch(
        config=config,
        prompt_name=prompt_name,
        prompt_type=prompt_type,
        label=label,
        version=version,
        context=context,
        candidate_count=candidate_count,
        langfuse_client=langfuse,
    )

    result = researcher.optimize(
        prep.source,
        context=context,
        candidate_count=candidate_count or config.autoresearch.candidate_count,
    )
    write_research_result(prep.run_dir, result)

    return save_autoresearch_result(
        config=config,
        run_dir=prep.run_dir,
        output_labels=output_labels,
        output_prompt_name=output_prompt_name,
        publish_experiment=publish_experiment,
        langfuse_client=langfuse,
    )


def prepare_autoresearch(
    *,
    config: AppConfig,
    prompt_name: str,
    prompt_type: str = "text",
    label: str | None = "production",
    version: int | None = None,
    context: str = "",
    candidate_count: int | None = None,
    langfuse_client: Any | None = None,
) -> AutoresearchPrep:
    if label and version is not None:
        raise ValueError("Specify either a prompt label or a prompt version, not both.")

    langfuse = langfuse_client or _build_langfuse_client(config)
    snapshot = fetch_prompt_snapshot(
        langfuse,
        prompt_name=prompt_name,
        prompt_type=prompt_type,
        label=label,
        version=version,
    )
    run_dir = create_run_dir(Path(config.autoresearch.workspace_dir), prompt_name)
    write_json(run_dir / "source_prompt.json", asdict(snapshot))
    write_prompt_file(run_dir, snapshot)

    task = {
        "task": "optimize_langfuse_prompt",
        "prompt_name": snapshot.name,
        "prompt_type": snapshot.type,
        "prompt_version": snapshot.version,
        "prompt_label": snapshot.label,
        "context": context,
        "candidate_count": candidate_count or config.autoresearch.candidate_count,
        "source_files": {
            "metadata": "source_prompt.json",
            "prompt": "prompt.chat.json" if snapshot.type == "chat" else "prompt.txt",
        },
        "output_contract": {
            "preferred_file": "optimized_prompt.json",
            "accepted_files": ["optimized_prompt.json", "autoresearch_result.json", "candidates.json"],
            "shape": {
                "candidates": [
                    {
                        "prompt": "string for text prompts, or chat message array for chat prompts",
                        "score": "number from 0 to 1",
                        "rationale": "short explanation",
                        "metadata": "optional object",
                    }
                ],
                "winner_index": "optional integer; highest score is used when omitted",
                "report": "optional object",
            },
        },
    }
    task_path = run_dir / "task.json"
    write_json(task_path, task)
    (run_dir / "README.md").write_text(
        "Run autoresearch in this directory, then write optimized_prompt.json.\n"
        "The save step will create a new Langfuse prompt version from the winning candidate.\n",
        encoding="utf-8",
    )

    return AutoresearchPrep(run_dir=run_dir, source=snapshot, task_path=task_path)


def save_autoresearch_result(
    *,
    config: AppConfig,
    run_dir: str | Path,
    output_labels: list[str] | None = None,
    output_prompt_name: str | None = None,
    publish_experiment: bool = True,
    result_path: str | Path | None = None,
    langfuse_client: Any | None = None,
) -> AutoresearchRun:
    langfuse = langfuse_client or _build_langfuse_client(config)
    run_path = Path(run_dir)
    snapshot = prompt_snapshot_from_dict(json.loads((run_path / "source_prompt.json").read_text(encoding="utf-8")))
    result = load_research_result(run_path, prompt_type=snapshot.type, result_path=Path(result_path) if result_path else None)
    if not result.candidates:
        raise ValueError("Autoresearch result did not include any prompt candidates.")

    write_research_result(run_path, result)

    experiment_name = f"autoresearch-{_slug(snapshot.name)}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    if publish_experiment:
        publish_candidate_experiment(
            langfuse,
            experiment_name=experiment_name,
            snapshot=snapshot,
            result=result,
        )

    labels = output_labels if output_labels is not None else ["autoresearch"]
    saved = langfuse.create_prompt(
        name=output_prompt_name or snapshot.name,
        type=snapshot.type,
        prompt=result.winner.prompt,
        labels=labels,
        tags=sorted({"autoresearch", *snapshot.tags}),
        config=_optimized_config(snapshot, result, experiment_name),
        commit_message=f"Autoresearch optimization from {snapshot.name} v{snapshot.version or 'unknown'}",
    )
    saved_summary = prompt_client_summary(saved)
    write_json(run_path / "saved_prompt.json", saved_summary)

    flush = getattr(langfuse, "flush", None)
    if callable(flush):
        flush()

    return AutoresearchRun(
        run_dir=run_path,
        source=snapshot,
        result=result,
        saved_prompt=saved_summary,
        experiment_name=experiment_name,
    )


def load_research_result(
    run_dir: Path,
    *,
    prompt_type: str,
    result_path: Path | None = None,
) -> ResearchResult:
    paths = [result_path] if result_path else [
        run_dir / "optimized_prompt.json",
        run_dir / "autoresearch_result.json",
        run_dir / "candidates.json",
    ]
    for path in paths:
        if path is None or not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            data = {"candidates": data}
        if not isinstance(data, dict):
            raise ValueError(f"Autoresearch result file must contain an object or candidate list: {path}")
        return research_result_from_response(data, prompt_type=prompt_type)
    raise FileNotFoundError(f"No autoresearch result file found in {run_dir}. Expected optimized_prompt.json, autoresearch_result.json, or candidates.json.")


def write_research_result(run_dir: Path, result: ResearchResult) -> None:
    write_json(run_dir / "candidates.json", [asdict(candidate) for candidate in result.candidates])
    write_json(run_dir / "report.json", result.report)
    write_json(run_dir / "optimized_prompt.json", asdict(result.winner))


def fetch_prompt_snapshot(
    langfuse: Any,
    *,
    prompt_name: str,
    prompt_type: str,
    label: str | None,
    version: int | None,
) -> PromptSnapshot:
    prompt = langfuse.get_prompt(
        prompt_name,
        type=prompt_type,
        label=label,
        version=version,
        cache_ttl_seconds=0,
    )
    config = getattr(prompt, "config", {}) or {}
    if not isinstance(config, dict):
        config = {"value": config}
    return PromptSnapshot(
        name=str(getattr(prompt, "name", prompt_name)),
        type=str(getattr(prompt, "type", prompt_type)),
        prompt=getattr(prompt, "prompt"),
        version=getattr(prompt, "version", version),
        label=label,
        config=config,
        labels=list(getattr(prompt, "labels", [])),
        tags=list(getattr(prompt, "tags", [])),
    )


def publish_candidate_experiment(
    langfuse: Any,
    *,
    experiment_name: str,
    snapshot: PromptSnapshot,
    result: ResearchResult,
) -> object:
    from langfuse import Evaluation

    def task(*, item: dict[str, Any], **_: object) -> dict[str, Any]:
        candidate = result.candidates[int(item["input"]["candidate_index"])]
        return {
            "prompt": candidate.prompt,
            "score": candidate.score,
            "rationale": candidate.rationale,
            "metadata": candidate.metadata,
            "selected": int(item["input"]["candidate_index"]) == result.winner_index,
        }

    def candidate_score(*, output: dict[str, Any], **_: object) -> Evaluation:
        return Evaluation(
            name="candidate_score",
            value=float(output["score"]),
            comment=str(output.get("rationale", "")),
        )

    return langfuse.run_experiment(
        name=experiment_name,
        description=f"Autoresearch prompt optimization for {snapshot.name}.",
        data=[
            {
                "input": {
                    "prompt_name": snapshot.name,
                    "source_version": snapshot.version,
                    "candidate_index": index,
                    "candidate_prompt": candidate.prompt,
                },
                "expected_output": {"goal": "Improve the source prompt while preserving Langfuse variables."},
                "metadata": {
                    "prompt_type": snapshot.type,
                    "source_labels": snapshot.labels,
                    "selected": index == result.winner_index,
                },
            }
            for index, candidate in enumerate(result.candidates)
        ],
        task=task,
        evaluators=[candidate_score],
        max_concurrency=1,
        metadata={
            "prompt_name": snapshot.name,
            "source_version": snapshot.version,
            "winner_index": result.winner_index,
            "report": result.report,
        },
    )


def research_result_from_response(data: dict[str, Any], *, prompt_type: str) -> ResearchResult:
    if "choices" in data:
        data = _from_openai_style_response(data)

    raw_candidates = data.get("candidates")
    if raw_candidates is None:
        raw_candidates = [
            {
                "prompt": data.get("optimized_prompt") or data.get("prompt") or data.get("text"),
                "score": data.get("score", 1.0),
                "rationale": data.get("rationale", ""),
                "metadata": data.get("metadata", {}),
            }
        ]

    candidates = [
        ResearchCandidate(
            prompt=_coerce_prompt_content(item.get("prompt") or item.get("optimized_prompt"), prompt_type=prompt_type),
            score=float(item.get("score", 0.0)),
            rationale=str(item.get("rationale", "")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in raw_candidates
        if isinstance(item, dict) and (item.get("prompt") is not None or item.get("optimized_prompt") is not None)
    ]
    if not candidates:
        raise ValueError("Autoresearch result did not include prompt candidates.")

    requested_winner = data.get("winner_index")
    winner_index = int(requested_winner) if requested_winner is not None else _best_candidate_index(candidates)
    if winner_index < 0 or winner_index >= len(candidates):
        winner_index = _best_candidate_index(candidates)

    return ResearchResult(
        candidates=candidates,
        winner_index=winner_index,
        report=dict(data.get("report", {})),
    )


def prompt_client_summary(prompt: Any) -> dict[str, Any]:
    return {
        "name": getattr(prompt, "name", None),
        "type": getattr(prompt, "type", None),
        "version": getattr(prompt, "version", None),
        "labels": list(getattr(prompt, "labels", [])),
        "tags": list(getattr(prompt, "tags", [])),
    }


def prompt_snapshot_from_dict(data: dict[str, Any]) -> PromptSnapshot:
    return PromptSnapshot(
        name=str(data["name"]),
        type=str(data.get("type", "text")),
        prompt=_coerce_prompt_content(data.get("prompt"), prompt_type=str(data.get("type", "text"))),
        version=data.get("version"),
        label=data.get("label"),
        config=dict(data.get("config", {})),
        labels=list(data.get("labels", [])),
        tags=list(data.get("tags", [])),
    )


def create_run_dir(base_dir: Path, prompt_name: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / f"{timestamp}-{_slug(prompt_name)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_prompt_file(run_dir: Path, snapshot: PromptSnapshot) -> None:
    if snapshot.type == "chat":
        write_json(run_dir / "prompt.chat.json", snapshot.prompt)
        return
    (run_dir / "prompt.txt").write_text(str(snapshot.prompt), encoding="utf-8")


def _build_langfuse_client(config: AppConfig) -> Any:
    if not config.langfuse.enabled or not config.langfuse.has_credentials:
        raise RuntimeError("Langfuse credentials are required for prompt autoresearch.")

    from langfuse import Langfuse

    return Langfuse(
        public_key=config.langfuse.public_key,
        secret_key=config.langfuse.secret_key,
        base_url=config.langfuse.host,
    )


def _optimized_config(snapshot: PromptSnapshot, result: ResearchResult, experiment_name: str) -> dict[str, Any]:
    config = dict(snapshot.config)
    config["autoresearch"] = {
        "source_prompt_name": snapshot.name,
        "source_prompt_version": snapshot.version,
        "source_labels": snapshot.labels,
        "experiment_name": experiment_name,
        "winner_index": result.winner_index,
        "winner_score": result.winner.score,
        "winner_rationale": result.winner.rationale,
        "report": result.report,
    }
    return config


def _from_openai_style_response(data: dict[str, Any]) -> dict[str, Any]:
    content = data["choices"][0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"optimized_prompt": content}


def _coerce_prompt_content(value: Any, *, prompt_type: str) -> PromptContent:
    if prompt_type == "chat":
        if isinstance(value, list):
            return [dict(message) for message in value if isinstance(message, dict)]
        if isinstance(value, str):
            return [{"role": "system", "content": value}]
        raise ValueError("Chat prompt candidates must be a message list or string.")

    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _best_candidate_index(candidates: list[ResearchCandidate]) -> int:
    return max(range(len(candidates)), key=lambda index: candidates[index].score)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()
    return slug or "prompt"
