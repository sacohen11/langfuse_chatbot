# Langfuse Autoresearch Starter

Small, bash-first templates for running Pi autoresearch against Langfuse prompts
and hosted MCP tool schemas.

## Bash On Windows

Use one of these:

- **Git Bash**: install [Git for Windows](https://git-scm.com/download/win), then open "Git Bash" in this repo. This is the easiest option for `bash autoresearch.sh`.
- **WSL**: install Ubuntu with `wsl --install`, then run the same files from a Linux shell.

This repo assumes Git Bash or WSL. On this machine, `bash` in PowerShell points
to WSL first. To force Git Bash from PowerShell, run:

```powershell
& "C:\Program Files\Git\bin\bash.exe" autoresearch.sh
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Git Bash on Windows
source .venv/bin/activate      # WSL/Linux
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with shared credentials:

```bash
OPENAI_API_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
LANGGRAPH_EVAL_URL=http://localhost:8080/internal/eval/run
LANGGRAPH_EVAL_TIMEOUT=120
```

## Prompt Experiment

```bash
mkdir -p experiments
cp -r templates/prompt experiments/my-prompt
cd experiments/my-prompt
vim config.json
python ../../run_experiment.py --pull config.json
bash autoresearch.sh
```

Then in Pi:

```text
/autoresearch optimize prompt.txt for higher overall_score. Run bash autoresearch.sh as the benchmark.
```

## Hosted MCP Tool Experiment

Use this for MCP servers deployed in Rancher/Kubernetes or anywhere else.

```bash
mkdir -p experiments
cp -r templates/mcp-tool experiments/my-tool
cd experiments/my-tool
vim config.json
bash fetch_schema.sh
bash autoresearch.sh
```

Then in Pi:

```text
/autoresearch optimize tool.json for higher overall_score. Run bash autoresearch.sh as the benchmark.
```

`templates/mcp-tool/config.json` contains:

- deployed MCP URL
- schema URL or local schema file
- Rancher/Kubernetes metadata: cluster, namespace, workload
- prompt name in Langfuse where attempts are saved
- dataset name in Langfuse
- scores/rubrics

## LangGraph MCP E2E Experiment

Use this when the real LangGraph app is deployed elsewhere and this repo should
act only as the external eval harness. First port-forward the service:

```bash
kubectl -n <namespace> port-forward svc/<langgraph-service-name> 8080:80
```

Then run the experiment locally:

```bash
mkdir -p experiments
cp -r templates/langgraph-mcp-e2e experiments/main-agent-e2e
cd experiments/main-agent-e2e
vim config.json
vim tool_suite.json
bash autoresearch.sh
```

Then in Pi:

```text
/autoresearch optimize tool_suite.json for higher overall_score. Run bash autoresearch.sh as the benchmark. You may edit only tool_suite.json.
```

The deployed LangGraph app should expose `POST /internal/eval/run`, accept the
candidate tool suite for one request, and return the Langfuse `trace_id` plus a
compact structured summary of the run. The full tool trace can stay in
Langfuse; this repo uses `trace_id` to attach scores to that trace and uses the
compact response fields for local scoring. This repo owns scoring, Langfuse
score logging, `latest_results.json`, and Pi-readable `METRIC` lines.

## Leaderboard

Every successful run appends to `leaderboard.csv` at the repo root.

The runner prints Pi-readable metric lines:

```text
METRIC overall_score=0.9123
METRIC task_success=0.9000
```

## Files That Matter

- `run_experiment.py`: one simple runner for prompt and MCP tool experiments.
- `fetch_schema.py`: pulls a hosted MCP JSON schema into `tool.json`.
- `templates/prompt`: copyable prompt experiment.
- `templates/mcp-tool`: copyable hosted MCP tool experiment.
- `templates/langgraph-mcp-e2e`: copyable deployed LangGraph E2E experiment.
- `leaderboard.csv`: shared leaderboard.
