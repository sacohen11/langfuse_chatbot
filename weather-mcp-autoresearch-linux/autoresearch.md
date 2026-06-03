# Autoresearch: Weather MCP Tool Specifications

## Objective
Optimize `weather_tool_candidate.json`, the MCP tool registration payload for a
classic weather server based on the official MCP weather quickstart.

The example server in `weather.py` exposes:
- `get-alerts`: gets active National Weather Service alerts for a US state.
- `get-forecast`: gets a National Weather Service forecast for a US latitude and
  longitude.

## What Pi Should Optimize
- Tool descriptions: when each weather tool should and should not be called.
- Input schemas: JSON Schema properties, required fields, ranges, and argument
  descriptions.
- Output schemas: fields the tool returns so agents can interpret alert and
  forecast results.

## Primary Metric
- `overall_score`, higher is better. This is the mean of all judge scores.

## Secondary Metrics
- `invocation_accuracy`: model chooses the correct weather tool, or no tool,
  and supplies correct minimal arguments.
- `input_schema_quality`: schemas clearly specify state codes, latitude,
  longitude, US/NWS limits, and avoid unrelated arguments.
- `output_schema_quality`: output contracts include useful alert/forecast
  fields for downstream responses.
- `description_clarity`: descriptions are concise and distinguish alerts,
  forecasts, unsupported locations, and educational weather questions.
- `conciseness`: Langfuse-managed Conciseness evaluator score for whether the
  model tool-call decision directly and succinctly answers the request without
  unnecessary detail.

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
METRIC conciseness=<number>
```

## Files in Scope
- `weather_tool_candidate.json`: the only required optimization target.
- `autoresearch.md`: update notes as experiments accumulate.
- `autoresearch.ideas.md`: optional backlog for future ideas.

## Off Limits
- Do not edit `dataset_items.json`.
- Do not edit `evaluate_tool_spec.py`.
- Do not edit `setup_langfuse.py`.
- Do not edit `save_best_tool_spec.py`.
- Do not edit `weather.py` during the prompt/spec optimization loop.
- Do not change `.env` or print secrets.

## Starting Hypotheses
- `get-alerts` should explicitly require a two-letter US state code and should
  not be used for forecasts.
- `get-forecast` should explicitly require numeric latitude and longitude,
  explain valid ranges, and state that the NWS endpoint is for US locations.
- The descriptions should say not to call either tool for general weather
  education questions.
- Alert outputs should include event, area, severity, headline, description, and
  instructions.
- Forecast outputs should include period names, temperature/unit, wind, short
  forecast, and detailed forecast.

## Langfuse Versions
Each successful benchmark saves the attempted weather MCP tool spec as a new
Langfuse text prompt version tagged `autoresearch-attempt`. The prompt config
includes the parsed JSON spec, dataset run URL, and scores.

After the loop, run:

```bash
python save_best_tool_spec.py --label autoresearch-best
```

Add `--also-production` only when you want that best spec promoted with the
`production` label in Langfuse.
