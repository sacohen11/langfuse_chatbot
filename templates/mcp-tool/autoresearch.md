# MCP Tool Autoresearch

Objective: improve `tool.json` for the MCP tool.

Editable file: `tool.json` only.

Benchmark command:

```bash
bash autoresearch.sh
```

Primary metric: `overall_score`, higher is better.

Secondary metrics:

- `tool_selection`: the model knows when to call the tool.
- `parameter_quality`: the input schema is clear, minimal, and hard to misuse.
- `output_quality`: the output schema tells the model how to interpret tool results.
- `conciseness`: descriptions are compact without losing meaning.

Important:

- Optimize only the tool description, input schema, and output schema.
- Keep the tool name stable unless your experiment explicitly tests renaming.
- Do not change `benchmark_config.json`, `config.json`, the dataset, or the benchmark script.
- If a change to any file other than `tool.json` seems necessary, stop and run the benchmark instead.
- Do not cheat by tailoring the schema to exact dataset item wording only.
- The benchmark saves each attempted version to Langfuse.
