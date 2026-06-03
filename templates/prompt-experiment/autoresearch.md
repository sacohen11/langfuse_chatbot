# Autoresearch: Prompt Experiment Template

## Objective
Optimize `prompt_candidate.txt` for the primary metric `overall_score`.

## Metrics
- `overall_score`: mean of the configured score dimensions in `experiment_config.json`.
- Secondary metrics are whatever appears in `scores`.

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
- `prompt_candidate.txt`
- `autoresearch.md`
- `autoresearch.ideas.md`

## Off Limits
- Do not edit `run_experiment.py`.
- Do not edit `experiment_config.json` during an optimization loop unless you are intentionally changing the experiment.
- Do not edit `.env` or print secrets.
- Do not edit dataset definitions while comparing prompt candidates.

## Constraints
- Preserve all variables and placeholders expected by the prompt.
- Prefer focused prompt edits over broad rewrites.
- Use `bash autoresearch.sh` as the benchmark command.

## What's Been Tried
- Add notes here as runs accumulate.
