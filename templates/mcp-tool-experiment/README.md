# MCP Tool Experiment Template

Use this template when the MCP server is already deployed, for example on a
Rancher-managed Kubernetes workload, and the experiment should evaluate the
tool descriptions plus input/output JSON schemas exposed by that deployment.

## Quick Start

1. Copy this template:

   ```powershell
   New-Item -ItemType Directory experiments -Force
   Copy-Item -Recurse templates\mcp-tool-experiment experiments\my-mcp-tool
   cd experiments\my-mcp-tool
   notepad mcp_tool_config.json
   ```

2. Edit `mcp_tool_config.json`:

   - `server.mcp_url`: the deployed MCP HTTP endpoint.
   - `server.schema_url`: the deployed JSON schema document for the server/tools.
   - `server.headers`: optional auth headers, using environment variables such as `${MCP_SERVER_TOKEN}`.
   - `prompt_name`: where attempted tool specs are saved in Langfuse prompt management.
   - `dataset_name`: the Langfuse dataset of tool-call scenarios.
   - `scores`: the score dimensions used for `overall_score`.

3. Pull the deployed schema JSON into `tool_candidate.json`:

   ```powershell
   python sync_tool_schema.py --fetch-schema
   ```

4. Run one benchmark:

   ```powershell
   bash autoresearch.sh
   ```

5. Run Pi:

   ```text
   /autoresearch optimize tool_candidate.json for higher overall_score. Run bash autoresearch.sh as the benchmark.
   ```

Each successful benchmark saves the attempted tool spec as a new Langfuse text
prompt version and appends a row to the repo-level `leaderboard.csv`.

## Expected Schema Shapes

The schema URL or file can return either a single tool:

```json
{
  "name": "lookup_customer",
  "description": "Look up a customer by id.",
  "input_schema": {},
  "output_schema": {}
}
```

or a server with multiple tools:

```json
{
  "server": {
    "name": "support-tools",
    "description": "Support operations MCP server."
  },
  "tools": [
    {
      "name": "lookup_customer",
      "description": "Look up a customer by id.",
      "input_schema": {},
      "output_schema": {}
    }
  ]
}
```

If your deployment exposes a different shape, set `schema_json_path` in
`mcp_tool_config.json` to the object containing the tool or server spec.
