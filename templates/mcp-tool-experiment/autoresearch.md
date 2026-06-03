# Autoresearch: Hosted MCP Tool Spec

## Objective
Optimize `tool_candidate.json`, the MCP tool/server schema used by a hosted MCP
deployment.

## Hosted Server Config
- Deployment details live in `mcp_tool_config.json`.
- `server.mcp_url` points at the deployed MCP endpoint.
- `server.schema_url` or `server.schema_file` points at the JSON schema to sync.
- Run `python sync_tool_schema.py --fetch-schema` before starting if the
  deployment schema changed.

## Metrics
- `overall_score`: mean of all configured score dimensions.
- Secondary metrics are configured in `mcp_tool_config.json`.

## How to Run

Use this benchmark only:

```bash
bash autoresearch.sh
```

The benchmark prints parseable lines:

```text
METRIC overall_score=<number>
METRIC <score_name>=<number>
```

## Files in Scope
- `tool_candidate.json`
- `autoresearch.md`
- `autoresearch.ideas.md`

## Off Limits
- Do not edit `run_tool_experiment.py`.
- Do not edit `sync_tool_schema.py`.
- Do not edit `mcp_tool_config.json` during a comparison loop unless changing deployments.
- Do not edit `.env` or print secrets.
- Do not change the dataset while comparing tool specs.

## Constraints
- Preserve the actual deployed tool names unless the deployment itself changed.
- Keep JSON Schema valid.
- Keep descriptions specific and concise.
- Use `bash autoresearch.sh` as the benchmark command.

## What's Been Tried
- Add notes here as experiments accumulate.
