# Prompt Experiment Template

Use this folder as a starting point for a new prompt optimization experiment.

## Quick Start

1. Copy this folder to a new experiment folder, for example:

   ```powershell
   Copy-Item -Recurse templates\prompt-experiment experiments\my-prompt-experiment
   cd experiments\my-prompt-experiment
   Copy-Item experiment_config.example.json experiment_config.json
   ```

2. Edit `experiment_config.json`:

   - `participant_name`: who should appear on the leaderboard.
   - `experiment_name`: a readable name for this experiment.
   - `prompt_name`: the Langfuse prompt name to evaluate and save attempts to.
   - `dataset_name`: the Langfuse dataset to evaluate against.
   - `scores`: the score dimensions to include in `overall_score`.

3. Pull the current prompt text from Langfuse into `prompt_candidate.txt`:

   ```powershell
   python run_experiment.py --pull-prompt
   ```

4. Run one benchmark:

   ```powershell
   bash autoresearch.sh
   ```

5. Run Pi autoresearch in the folder:

   ```text
   /autoresearch optimize prompt_candidate.txt for higher overall_score. Run bash autoresearch.sh as the benchmark.
   ```

Each successful benchmark:

- saves the attempted prompt as a new Langfuse prompt version;
- writes a local `prompt_attempts.jsonl` entry;
- updates the repo-level `leaderboard.csv`;
- prints parseable `METRIC ...` lines for Pi.

## Score Types

Use `llm_judge` for custom rubric scores:

```json
{
  "name": "task_success",
  "type": "llm_judge",
  "rubric": "Does the assistant satisfy the expected behavior?"
}
```

Use `managed_langfuse` for managed Langfuse evaluators such as `Conciseness`:

```json
{
  "name": "conciseness",
  "type": "managed_langfuse",
  "evaluator_name": "Conciseness",
  "variable_map": {
    "query": "input.customer_message",
    "generation": "output"
  }
}
```

`overall_score` is the mean of all configured score dimensions.
