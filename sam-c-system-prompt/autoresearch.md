# Autoresearch: Sam-C Support Triage System Prompt

## Objective
Optimize `prompt_candidate.txt`, the system prompt for a customer-support triage assistant.
The assistant should produce concise, empathetic, safe support replies for common customer
messages without overpromising refunds, data recovery, policy exceptions, or account actions.

## Metrics
- **Primary**: `overall_score` (unitless 0-1, higher is better) - mean of all judge scores.
- **Secondary**:
  - `task_success` - matches expected behavior and required substance.
  - `tone` - empathetic, calm, concise, and direct.
  - `safety` - avoids overpromising, inventing policy, or unsafe guarantees.
  - `conciseness` - Langfuse-managed Conciseness evaluator score for whether the response directly and succinctly answers the user message without unnecessary detail.

## How to Run
`bash autoresearch.sh`

The script first runs correctness checks equivalent to `bash checks.sh`, then runs
`evaluate_prompt.py` against the Langfuse dataset and outputs:

```text
METRIC overall_score=<number>
METRIC task_success=<number>
METRIC tone=<number>
METRIC safety=<number>
METRIC conciseness=<number>
```

## Files in Scope
- `prompt_candidate.txt` - the only file Pi should optimize.
- `autoresearch.md` - update the "What's Been Tried" section as experiments accumulate.
- `autoresearch.ideas.md` - optional backlog for future prompt-improvement ideas.
- `autoresearch.prompt_attempts.jsonl` - append-only log of every evaluated prompt version and score.
- `checks.sh` - standalone correctness checks, also run by `autoresearch.sh`.

## Off Limits
- Do not edit `dataset_items.json` during the optimization loop.
- Do not edit `evaluate_prompt.py` to make the metric easier.
- Do not edit `setup_langfuse.py` or `save_optimized_prompt.py` unless the API contract breaks.
- Do not edit `save_best_prompt.py` unless the Langfuse API contract breaks.
- Do not create `autoresearch.checks.sh` on Windows. The Pi extension currently invokes it with a malformed Windows path.
- Do not change `.env` or print secrets.

## Constraints
- Preserve the assistant's role as a customer-support reply writer.
- Keep the prompt reasonably short and operational; prefer precise behavioral rules over long prose.
- The optimized prompt must work with a user message injected as the only user turn.
- No new dependencies.
- Use `bash autoresearch.sh` as the only Pi benchmark command; it includes checks.

## Starting Hypotheses
- The seed prompt is too vague; it does not explicitly tell the assistant to ask for missing identifiers.
- The seed prompt does not warn against guaranteeing refunds, recovery, or account actions.
- A stronger prompt should use a repeatable response pattern: acknowledge, address, ask/escalate, set expectations.
- The prompt should mention topic-specific caution for billing, data loss, login, and cancellation.

## What's Been Tried
- Seed prompt: `You are Sam-C Support Assistant. Help customers with their message. Be friendly and concise.`

## Langfuse Prompt Versions
Every successful `bash autoresearch.sh` run creates a new Langfuse prompt version
for the attempted `prompt_candidate.txt`, tagged `autoresearch-attempt`, with the
judge scores in prompt config. Attempts are also logged in
`autoresearch.prompt_attempts.jsonl`.

After the loop, run:

`python save_best_prompt.py --label autoresearch-best`

This creates one more newest Langfuse prompt version from the best scored
attempt. Add `--also-production` only when you want that best version deployed as
the default production prompt.
