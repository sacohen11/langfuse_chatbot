# Autoresearch: MCP Tool Specification

## Objective
Optimize `tool_candidate.json`, an MCP tool specification. The target is the
tool name, description, input schema, and output schema for a support-ticket
lookup tool.

## What Pi Should Optimize
- `description`: when the tool should and should not be called.
- `input_schema`: JSON Schema properties, required fields, descriptions, and
  safety constraints for arguments.
- `output_schema`: fields the tool returns so an agent can interpret results and
  respond usefully.

## Primary Metric
- `overall_score`, higher is better.

## Secondary Metrics
- `invocation_accuracy`: the model chooses whether to call the tool correctly
  and supplies the right arguments.
- `input_schema_quality`: the schema makes required and forbidden arguments
  clear without encouraging sensitive or unrelated data.
- `output_schema_quality`: the output contract includes fields needed for
  downstream responses.
- `description_clarity`: the tool description is specific, concise, and
  distinguishes this tool from unrelated tools.

## How to Run
Use this benchmark only:

```bash
bash autoresearch.sh
```

The benchmark writes parseable lines:

```text
METRIC overall_score=<number>
METRIC invocation_accuracy=<number>
METRIC input_schema_quality=<number>
METRIC output_schema_quality=<number>
METRIC description_clarity=<number>
```

## Files in Scope
- `tool_candidate.json`: the only required optimization target.
- `autoresearch.md`: update notes as experiments accumulate.
- `autoresearch.ideas.md`: optional backlog for future ideas.

## Off Limits
- Do not edit `dataset_items.json`.
- Do not edit `evaluate_tool_spec.py`.
- Do not edit `setup_langfuse.py`.
- Do not edit `save_best_tool_spec.py`.
- Do not change `.env` or print secrets.

## Starting Hypotheses
- The seed description is too vague; it does not explain when not to call the
  tool.
- The input schema should forbid passwords, emails, customer ids, and other
  unnecessary data.
- The output schema should include `summary`, `last_updated_at`, `owner_team`,
  `assignee_name`, and `next_step`, not only `status`.
- A better description should explicitly require a ticket id and say to ask for
  the id if missing.

## Langfuse Versions
Each successful benchmark saves the attempted tool spec as a new Langfuse text
prompt version tagged `autoresearch-attempt`. The prompt config includes the
parsed JSON spec, dataset run URL, and scores.

After the loop, run:

```bash
python save_best_tool_spec.py --label autoresearch-best
```

Add `--also-production` only when you want that best tool spec promoted as the
production label in Langfuse.

