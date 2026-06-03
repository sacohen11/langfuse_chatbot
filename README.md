# Langfuse Chatbot

A small LangChain chatbot scaffold with:

- multi-turn conversations
- MCP tools from external servers or your own HTTPS MCP servers
- Langfuse tracing
- evaluation of agentic behavior across multi-turn tool-call scenarios
- deterministic tests that run without live model credentials

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

Set `OPENAI_API_KEY` for real model calls. Set `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL` when you want traces and experiment
results in Langfuse.

## Langfuse Tracing

The chatbot uses the current Langfuse Python SDK and LangChain callback handler.
Each chat turn creates a named `chat-response` root span with explicit user
message input and assistant output. LangChain model/tool observations are nested
under that span, while `session_id`, `user_id`, tags, and sanitized metadata are
propagated so traces are easy to filter in Langfuse. CLI and evaluation runs
flush queued events before exit.

## Configure MCP Servers

Edit `config.example.yaml` or create your own config file. HTTPS MCP servers are
configured with an `https://.../mcp` URL:

```yaml
mcp:
  servers:
    my_server:
      enabled: true
      transport: http
      url: https://my-server.example.com/mcp
      headers:
        Authorization: Bearer ${MY_MCP_TOKEN}
```

For private CAs or self-signed development certificates, configure trust at the
Python/runtime level before running the client, for example `SSL_CERT_FILE` or
`REQUESTS_CA_BUNDLE` pointing at your CA bundle.

## Chat

```powershell
langfuse-chatbot chat --config config.example.yaml
```

## Web UI

Start the local console:

```powershell
python -m langfuse_chatbot.cli --config config.example.yaml web --port 8000
```

Open `http://127.0.0.1:8000`.

The UI includes:

- a multi-turn chatbot playground
- session, latency, tool-call, and error metrics
- Langfuse configuration status and host link
- MCP server visibility
- one-click evaluation using `evals/scenarios/tool_multiturn.json`

## Evaluate

Run deterministic local scoring over multi-turn scenarios:

```powershell
langfuse-chatbot eval --config config.example.yaml --scenarios evals/scenarios/tool_multiturn.json
```

Publish an experiment to Langfuse:

```powershell
langfuse-chatbot eval --config config.example.yaml --scenarios evals/scenarios/tool_multiturn.json --publish-langfuse
```

The evaluation harness scores more than the final answer. It tracks whether the
agent used required tools, avoided forbidden tools, preserved context across
turns, and satisfied expected answer checks.

If you enable `mcp.tool_name_prefix`, LangChain MCP adapters namespace tool names
with the server name. The evaluator accepts both exact tool names and namespaced
suffixes, so `calculator` still matches `math_calculator`.

The package also exposes `load_mcp_resources(...)` and `load_mcp_prompt(...)` in
`langfuse_chatbot.mcp_client` for MCP servers that publish resources or prompt
templates in addition to tools.

## Prompt Autoresearch

Pull a prompt from Langfuse into an autoresearch run folder:

```powershell
langfuse-chatbot --config config.example.yaml autoresearch pull support/triage `
  --type chat `
  --label production `
  --context "Improve tool-use precision and keep all {{variables}} intact."
```

Run your autoresearch harness against the generated folder. The folder includes
`task.json`, `source_prompt.json`, and either `prompt.txt` or
`prompt.chat.json`. When the harness is done, write one of these files in the
same run folder:

- `optimized_prompt.json`
- `autoresearch_result.json`
- `candidates.json`

Then save the winning candidate as a new Langfuse prompt version:

```powershell
langfuse-chatbot --config config.example.yaml autoresearch save autoresearch\20260602-120000-support-triage
```

Configure the local workspace in `.env` or YAML:

```yaml
autoresearch:
  workspace_dir: autoresearch
  candidate_count: 3
```

Each run writes `source_prompt.json`, `candidates.json`, `report.json`,
`optimized_prompt.json`, and `saved_prompt.json` under `autoresearch/` as the
workflow progresses. By default the optimized version gets the `autoresearch`
label; repeat `--output-label` or use `--output-name` if you want a separate
prompt.

## Shared Prompt Experiment Template

Use `templates/prompt-experiment` when someone wants to bring their own Langfuse
prompt, dataset, and scoring rubric into an autoresearch loop.

```powershell
New-Item -ItemType Directory experiments -Force
Copy-Item -Recurse templates\prompt-experiment experiments\my-experiment
cd experiments\my-experiment
notepad experiment_config.json
python run_experiment.py --pull-prompt
bash autoresearch.sh
```

The template config asks for the participant name, experiment name, Langfuse
prompt name, dataset name, model choices, and score dimensions. Scores can be
custom LLM judge rubrics or managed Langfuse evaluators such as `Conciseness`.

Each successful run saves a new Langfuse prompt version and appends a row to the
repo-level `leaderboard.csv`, including the participant, prompt version,
overall score, per-score columns, and Langfuse dataset run URL.

## Test

```powershell
pytest
```

The tests use fake chatbots and do not call an LLM, MCP server, or Langfuse.
