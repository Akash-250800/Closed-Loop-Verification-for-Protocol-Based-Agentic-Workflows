"""MCP reference server for the strict two-agent demonstration.

Three independent read-only references are exposed:
A: complete country/capital table
B: complete country/presidency index
C: dated presidential terms for a presidency reference

No tool contains or receives the expected final answer.
"""
from __future__ import annotations

from multiref_server import MultiRefData


def make_two_agent_feedback_server():
    from mcp.server.mcpserver import MCPServer

    data = MultiRefData()
    server = MCPServer(
        "Strict Two-Agent Multi-Reference MCP",
        instructions=(
            "Two specialised agents list independent read-only references. Their retrieved evidence is merged, "
            "then a third reference is queried using the merged key. Do not answer from model memory."
        ),
    )

    @server.tool()
    def list_country_reference() -> dict:
        """Reference A (countries.csv): list capital, country code and country name rows."""
        rows = [
            {
                "capital": row["capital"],
                "country_code": row["country_code"],
                "country_name": row["country_name"],
            }
            for row in data.countries
        ]
        return {"reference": "countries.csv", "rows": rows}

    @server.tool()
    def list_president_index_reference() -> dict:
        """Reference B (president_index.csv): list country/presidency/candidate-president mapping rows."""
        rows = [
            {
                "country_code": row["country_code"],
                "presidency_reference": row["presidency_reference"],
                "president_id": row["president_id"],
            }
            for row in data.index
        ]
        return {"reference": "president_index.csv", "rows": rows}

    @server.tool()
    def list_presidential_terms(presidency_reference: str) -> dict:
        """Reference C (president_terms.csv): list dated term rows for a retrieved presidency reference."""
        key = presidency_reference.strip()
        rows = [
            {
                "presidency_reference": row["presidency_reference"],
                "president_id": row["president_id"],
                "president_name": row["president_name"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
            }
            for row in data.terms
            if row["presidency_reference"] == key
        ]
        if not rows:
            raise ValueError(f"Presidency reference not found in Reference C: {presidency_reference}")
        return {"reference": "president_terms.csv", "presidency_reference": key, "rows": rows}

    return server


if __name__ == "__main__":
    make_two_agent_feedback_server().run()
