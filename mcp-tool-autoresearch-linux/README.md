# MCP Tool Autoresearch for Linux

This folder is a standalone Linux-copyable Pi autoresearch experiment for
optimizing an MCP tool spec with Langfuse.

The candidate artifact is `tool_candidate.json`. It is stored in Langfuse Prompt
Management as a text prompt because Langfuse gives prompt versions, labels,
tags, and config metadata. The JSON prompt content is the MCP tool spec.

## Setup on Linux

```bash
cd mcp-tool-autoresearch-linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your Langfuse and OpenAI keys.

Initialize the experiment Git repo that Pi autoresearch expects:

```bash
bash bootstrap_git.sh
```

Create the seed prompt and dataset in Langfuse:

```bash
python setup_langfuse.py
```

Smoke test the benchmark:

```bash
bash checks.sh
bash autoresearch.sh
```

## Run with Pi

Install autoresearch if needed:

```bash
pi install npm:pi-autoresearch
```

Then start Pi in this folder:

```bash
pi --provider openai --model gpt-4.1-mini
```

Inside Pi, run:

```text
/autoresearch start
/autoresearch optimize tool_candidate.json for higher overall_score. Run bash autoresearch.sh as the benchmark. Only edit tool_candidate.json, autoresearch.md, or autoresearch.ideas.md. Do not edit evaluate_tool_spec.py, dataset_items.json, setup_langfuse.py, save_best_tool_spec.py, or experiment_config.json.
```

When finished:

```bash
python save_best_tool_spec.py --label autoresearch-best
```
