# Get Weather MCP Autoresearch

Objective: improve `tool.json` for the `get_weather` MCP tool.

Editable file: `tool.json` only.

Benchmark command:

```bash
bash autoresearch.sh
```

Primary metric: `overall_score`, higher is better.

Secondary metrics:

- `tool_selection`: the model knows when to call `get_weather`.
- `parameter_quality`: the input schema is clear, minimal, and hard to misuse.
- `output_quality`: the output schema tells the model how to interpret weather data.
- `conciseness`: descriptions are compact without losing meaning.

Important:

- Optimize the tool description, input schema, and output schema.
- Keep the tool name `get_weather`.
- Do not change `benchmark_config.json`, `config.json`, the dataset, or the benchmark script.
- If a change to any file other than `tool.json` seems necessary, stop and run the benchmark instead.
- Do not cheat by tailoring the schema to exact dataset item wording only.
- Save attempted versions through the benchmark; it writes each candidate to Langfuse.
