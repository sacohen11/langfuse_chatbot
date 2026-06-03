from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .autoresearch import prepare_autoresearch, save_autoresearch_result
from .chatbot import LangChainMCPChatbot
from .config import load_config
from .evaluation.langfuse_experiment import publish_experiment
from .evaluation.runner import evaluate_chatbot, load_scenarios
from .langfuse_observability import flush_langfuse


def main() -> None:
    parser = argparse.ArgumentParser(description="LangChain MCP chatbot with Langfuse evaluation.")
    parser.add_argument("--config", default="config.example.yaml", help="Path to YAML config.")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("chat", help="Start an interactive chat session.")
    web_parser = subparsers.add_parser("web", help="Start the browser UI.")
    web_parser.add_argument("--host", default="127.0.0.1", help="Host for the web server.")
    web_parser.add_argument("--port", type=int, default=8000, help="Port for the web server.")

    eval_parser = subparsers.add_parser("eval", help="Run multi-turn tool-call evaluations.")
    eval_parser.add_argument("--scenarios", required=True, help="Path to scenario JSON.")
    eval_parser.add_argument("--publish-langfuse", action="store_true", help="Publish as a Langfuse experiment.")
    eval_parser.add_argument("--min-score", type=float, default=0.85, help="Exit non-zero below this score.")

    research_parser = subparsers.add_parser("autoresearch", help="Prepare and save Langfuse prompt autoresearch runs.")
    research_subparsers = research_parser.add_subparsers(dest="autoresearch_command", required=True)

    pull_parser = research_subparsers.add_parser("pull", help="Pull a Langfuse prompt into an autoresearch run folder.")
    pull_parser.add_argument("prompt_name", help="Langfuse prompt name, including folder path if any.")
    pull_parser.add_argument("--type", choices=["text", "chat"], default="text", help="Langfuse prompt type.")
    pull_parser.add_argument("--label", default="production", help="Prompt label to fetch.")
    pull_parser.add_argument("--version", type=int, default=None, help="Specific prompt version to fetch.")
    pull_parser.add_argument("--context", default="", help="Optimization goal or extra research instructions.")
    pull_parser.add_argument("--context-file", default=None, help="Read optimization context from a text file.")
    pull_parser.add_argument("--candidates", type=int, default=None, help="Number of candidates to request.")

    save_parser = research_subparsers.add_parser("save", help="Save an autoresearch result as a Langfuse prompt version.")
    save_parser.add_argument("run_dir", help="Autoresearch run directory created by `autoresearch pull`.")
    save_parser.add_argument("--result", default=None, help="Optional result JSON path. Defaults to files in the run directory.")
    save_parser.add_argument(
        "--output-name",
        default=None,
        help="Langfuse prompt name to save. Defaults to creating a new version of the source prompt.",
    )
    save_parser.add_argument(
        "--output-label",
        action="append",
        default=None,
        help="Label for the optimized prompt version. Repeat for multiple labels. Defaults to autoresearch.",
    )
    save_parser.add_argument(
        "--skip-experiment",
        action="store_true",
        help="Create the optimized prompt version without publishing candidate experiment rows.",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "chat":
        asyncio.run(_chat(config))
    elif args.command == "web":
        _web(args.config, args.host, args.port)
    elif args.command == "eval":
        asyncio.run(_eval(config, Path(args.scenarios), args.publish_langfuse, args.min_score))
    elif args.command == "autoresearch":
        _autoresearch(args, config)


async def _chat(config) -> None:
    chatbot = await LangChainMCPChatbot.from_config(config)
    session_id = "cli"
    print("Chat started. Type 'exit' or 'quit' to stop.")
    try:
        while True:
            user_text = input("> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                break
            response = await chatbot.achat(user_text, session_id=session_id, user_id="cli-user")
            print(response.content)
    finally:
        chatbot.flush_traces()


async def _eval(config, scenarios_path: Path, publish_langfuse: bool, min_score: float) -> None:
    scenarios = load_scenarios(scenarios_path)

    if publish_langfuse:
        result = await publish_experiment(
            config=config,
            scenarios=scenarios,
            agent_factory=lambda: LangChainMCPChatbot.from_config(config),
        )
        print(result.format())
        flush_langfuse(config.langfuse.enabled and config.langfuse.has_credentials)
        return

    chatbot = await LangChainMCPChatbot.from_config(config)
    try:
        report = await evaluate_chatbot(chatbot, scenarios)
        print(json.dumps(report.as_dict(), indent=2))
        if report.overall_score < min_score:
            raise SystemExit(f"Evaluation score {report.overall_score:.3f} below threshold {min_score:.3f}")
    finally:
        chatbot.flush_traces()


def _web(config_path: str, host: str, port: int) -> None:
    import uvicorn
    from .web.app import create_app

    uvicorn.run(create_app(config_path), host=host, port=port, reload=False)


def _autoresearch(args, config) -> None:
    if args.autoresearch_command == "pull":
        context = args.context
        if args.context_file:
            context = Path(args.context_file).read_text(encoding="utf-8")

        prep = prepare_autoresearch(
            config=config,
            prompt_name=args.prompt_name,
            prompt_type=args.type,
            label=None if args.version is not None else args.label,
            version=args.version,
            context=context,
            candidate_count=args.candidates,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(prep.run_dir),
                    "task_path": str(prep.task_path),
                    "source_prompt": {
                        "name": prep.source.name,
                        "type": prep.source.type,
                        "version": prep.source.version,
                        "label": prep.source.label,
                    },
                },
                indent=2,
            )
        )
        flush_langfuse(config.langfuse.enabled and config.langfuse.has_credentials)
        return

    if args.autoresearch_command == "save":
        run = save_autoresearch_result(
            config=config,
            run_dir=args.run_dir,
            result_path=args.result,
            output_labels=args.output_label,
            output_prompt_name=args.output_name,
            publish_experiment=not args.skip_experiment,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(run.run_dir),
                    "experiment_name": run.experiment_name,
                    "winner_score": run.result.winner.score,
                    "saved_prompt": run.saved_prompt,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
