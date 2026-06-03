# Sam-C System Prompt Autoresearch

This folder contains a focused Pi autoresearch experiment for optimizing a Langfuse-managed system prompt.

## 1. Seed Langfuse

From this folder:

```powershell
python setup_langfuse.py
```

This creates:

- Prompt: `sam-c-system-prompt/support-triage-system`
- Dataset: `sam-c-system-prompt/support-triage-eval`
- Five dataset items for billing, data loss, account changes, login, and cancellation

## 2. Baseline Evaluation

```powershell
bash autoresearch.sh
```

The benchmark evaluates `prompt_candidate.txt`, publishes a Langfuse experiment
run, saves the attempted prompt as a new Langfuse prompt version, and records it
in `autoresearch.prompt_attempts.jsonl`. It uses three LLM-as-a-judge scores:

- `task_success`
- `tone`
- `safety`

The primary Pi metric is `overall_score`, where higher is better.

## 3. Start Pi Autoresearch

Open Pi in this folder and use:

```text
/autoresearch optimize prompt_candidate.txt for higher overall_score. Run bash autoresearch.sh. Only edit prompt_candidate.txt.
```

Pi should use `autoresearch.md` and `autoresearch.sh`. On Windows, do not create
`autoresearch.checks.sh`; `autoresearch.sh` already runs `checks.sh` before the
LLM-as-judge evaluation because the Pi extension's automatic checks path is not
Windows-safe.

## 4. Save the Winner

After Pi improves `prompt_candidate.txt`:

```powershell
python save_best_prompt.py --label autoresearch-best
```

This creates one more newest Langfuse prompt version using the highest-scoring
attempt. Add `--also-production` if you want that version to receive the
`production` label.
