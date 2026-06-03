"""Minimal FastMCP server you can put behind TLS.

Run with your preferred ASGI/TLS setup or reverse proxy. For local development,
terminate TLS at Caddy, nginx, or uvicorn/hypercorn and expose /mcp over HTTPS.
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("custom-tools")


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    allowed = set("0123456789+-*/(). ")
    if any(char not in allowed for char in expression):
        raise ValueError("Only arithmetic expressions are supported.")
    return str(eval(expression, {"__builtins__": {}}, {}))


@mcp.tool()
def get_weather(city: str) -> str:
    """Return example weather for a city."""
    return f"{city}: light rain, 52 F, carry an umbrella."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
