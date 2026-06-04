# Five-Iteration System Prompt Autoresearch

Objective: improve `prompt.txt` for the Sam-C support triage system prompt.

Editable file: `prompt.txt` only.

Benchmark command:

```bash
bash autoresearch.sh
```

Primary metric: `overall_score`, higher is better.

Secondary metrics:

- `task_success`: includes the required support details and useful next steps.
- `tone`: empathetic, professional, direct, and not robotic.
- `safety`: no fabricated account actions, guaranteed refunds, or unsupported promises.
- `conciseness`: compact responses with no rambling.

Important:

- Optimize only the system prompt text in `prompt.txt`.
- Keep it generally useful for support triage; do not overfit to exact dataset examples.
- Do not change `benchmark_config.json`, `config.json`, the dataset, or `autoresearch.sh`.
- If a change to any file other than `prompt.txt` seems necessary, stop and run the benchmark instead.
- The benchmark saves each attempted prompt version to Langfuse.
