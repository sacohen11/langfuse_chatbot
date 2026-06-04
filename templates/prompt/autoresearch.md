# System Prompt Autoresearch

Objective: improve `prompt.txt` for the system prompt.

Editable file: `prompt.txt` only.

Benchmark command:

```bash
bash autoresearch.sh
```

Primary metric: `overall_score`, higher is better.

Secondary metrics:

- `task_success`: responses satisfy the user request.
- `tone`: responses are clear, helpful, and appropriate.
- `safety`: responses avoid unsafe, fabricated, or policy-violating content.
- `conciseness`: responses are compact without losing useful meaning.

Important:

- Optimize only the system prompt text in `prompt.txt`.
- Keep the prompt generally useful; do not overfit to exact dataset wording.
- Do not change `benchmark_config.json`, `config.json`, the dataset, or the benchmark script.
- If a change to any file other than `prompt.txt` seems necessary, stop and run the benchmark instead.
- The benchmark saves each attempted prompt version to Langfuse.
