# Weather MCP Autoresearch for Linux

This folder is a standalone Linux-copyable Pi autoresearch experiment for a
classic MCP weather server.

It includes:
- `weather.py`: a Python MCP server with `get-alerts` and `get-forecast`,
  modeled after the official MCP weather quickstart.
- `weather_tool_candidate.json`: the MCP tool registration payload Pi optimizes.
- Langfuse setup/evaluation scripts that save every attempted tool spec as a
  versioned text prompt.

## Setup on Linux

```bash
cd weather-mcp-autoresearch-linux
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

## Run the MCP Server

The server runs over stdio:

```bash
python weather.py
```

In an MCP host, configure the command to the absolute path of your virtualenv
Python and pass the absolute path to `weather.py`.

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
/autoresearch optimize weather_tool_candidate.json for higher overall_score. Run bash autoresearch.sh as the benchmark. Only edit weather_tool_candidate.json, autoresearch.md, or autoresearch.ideas.md. Do not edit weather.py, evaluate_tool_spec.py, dataset_items.json, setup_langfuse.py, save_best_tool_spec.py, or experiment_config.json.
```

When finished:

```bash
python save_best_tool_spec.py --label autoresearch-best
```

