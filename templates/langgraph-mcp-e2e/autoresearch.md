# LangGraph MCP E2E Autoresearch

Objective: improve `tool_suite.json` for a deployed LangGraph agent.

Editable file: `tool_suite.json` only.

Benchmark command:

```bash
bash autoresearch.sh
```

Primary metric: `overall_score`, higher is better.

Secondary metrics:

- `tool_recall`: expected tools were selected.
- `args_quality`: expected argument strings appeared in tool args.
- `tool_order_correct`: required order was respected.
- `mcp_execution_success`: called MCP tools succeeded.
- `answer_contains_expected`: final answer contains expected strings.
- `no_forbidden_tool_calls`: forbidden tools were avoided.

Important:

- Optimize only `tool_suite.json`.
- Keep JSON valid.
- Do not change `benchmark_config.json`, `config.json`, datasets, scorers, runtime code, or `autoresearch.sh`.
- The benchmark calls the deployed LangGraph eval endpoint over HTTP.
- Avoid regressions in guardrail metrics.
