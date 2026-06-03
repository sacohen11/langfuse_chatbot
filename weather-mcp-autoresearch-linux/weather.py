from __future__ import annotations

from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP


NWS_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-mcp-autoresearch/1.0"

mcp = FastMCP("weather")


async def make_nws_request(url: str) -> dict[str, Any] | None:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/geo+json"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None


def format_alert(feature: dict[str, Any]) -> str:
    props = feature.get("properties", {})
    return "\n".join(
        [
            f"Event: {props.get('event', 'Unknown')}",
            f"Area: {props.get('areaDesc', 'Unknown')}",
            f"Severity: {props.get('severity', 'Unknown')}",
            f"Headline: {props.get('headline', 'No headline')}",
            f"Description: {props.get('description', 'No description')}",
            f"Instructions: {props.get('instruction', 'No specific instructions')}",
        ]
    )


@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get active National Weather Service alerts for a two-letter US state code."""
    state_code = state.upper()
    data = await make_nws_request(f"{NWS_API_BASE}/alerts?area={state_code}")
    if not data:
        return f"Failed to retrieve alerts for {state_code}."

    features = data.get("features", [])
    if not features:
        return f"No active alerts for {state_code}."

    alerts = [format_alert(feature) for feature in features[:20]]
    return f"Active alerts for {state_code}:\n\n" + "\n\n---\n\n".join(alerts)


@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get a National Weather Service forecast for a US location by latitude and longitude."""
    points_url = f"{NWS_API_BASE}/points/{latitude:.4f},{longitude:.4f}"
    points_data = await make_nws_request(points_url)
    if not points_data:
        return (
            f"Failed to retrieve grid point data for coordinates {latitude}, {longitude}. "
            "The National Weather Service API generally supports US locations only."
        )

    forecast_url = points_data.get("properties", {}).get("forecast")
    if not forecast_url:
        return "No forecast URL found for this location."

    forecast_data = await make_nws_request(forecast_url)
    if not forecast_data:
        return "Failed to retrieve forecast data."

    periods = forecast_data.get("properties", {}).get("periods", [])
    if not periods:
        return "No forecast periods available."

    formatted = []
    for period in periods[:5]:
        formatted.append(
            "\n".join(
                [
                    f"{period.get('name', 'Period')}:",
                    f"Temperature: {period.get('temperature')} {period.get('temperatureUnit', '')}",
                    f"Wind: {period.get('windSpeed', 'Unknown')} {period.get('windDirection', '')}",
                    f"Forecast: {period.get('detailedForecast', period.get('shortForecast', 'No forecast'))}",
                ]
            )
        )
    return "\n\n".join(formatted)


if __name__ == "__main__":
    mcp.run(transport="stdio")

