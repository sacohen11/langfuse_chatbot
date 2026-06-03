from __future__ import annotations

from typing import Any


class NoMCPServersConfigured(RuntimeError):
    """Raised when code explicitly requires MCP servers but none are enabled."""


async def load_mcp_tools(
    server_configs: dict[str, dict[str, Any]],
    *,
    tool_name_prefix: bool = False,
) -> tuple[Any | None, list[Any]]:
    """Create a MultiServerMCPClient and load all tools from enabled servers."""

    if not server_configs:
        return None, []

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(server_configs, tool_name_prefix=tool_name_prefix)
    tools = await client.get_tools()
    return client, tools


async def load_mcp_resources(
    server_configs: dict[str, dict[str, Any]],
    *,
    server_name: str | None = None,
    uris: str | list[str] | None = None,
) -> list[Any]:
    """Load MCP resources as LangChain Blob objects."""

    if not server_configs:
        raise NoMCPServersConfigured("No MCP servers are enabled.")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(server_configs)
    return await client.get_resources(server_name, uris=uris)


async def load_mcp_prompt(
    server_configs: dict[str, dict[str, Any]],
    *,
    server_name: str,
    prompt_name: str,
    arguments: dict[str, Any] | None = None,
) -> list[Any]:
    """Load an MCP prompt as LangChain chat messages."""

    if not server_configs:
        raise NoMCPServersConfigured("No MCP servers are enabled.")

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(server_configs)
    return await client.get_prompt(server_name, prompt_name, arguments=arguments)
