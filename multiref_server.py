"""MCP server for the genuine multi-reference benchmark.

The final answer is not stored in one dictionary. The agent must traverse three
independent CSV references:
  capital -> country_code -> presidency_reference -> term covering target date.
"""
from __future__ import annotations

import csv
import unicodedata
from datetime import date
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "multiref"


def _load_csv(name: str) -> list[dict[str, str]]:
    path = DATA_DIR / name
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return " ".join(value.casefold().replace(",", "").split())


class MultiRefData:
    """Pure read-only data layer, kept separate so the reference chain is testable without MCP."""

    def __init__(self):
        self.countries = _load_csv("countries.csv")
        self.index = _load_csv("president_index.csv")
        self.terms = _load_csv("president_terms.csv")

    def country_by_capital(self, capital: str) -> dict:
        wanted = _norm(capital)
        for row in self.countries:
            if _norm(row["capital"]) == wanted:
                return {
                    "capital": row["capital"],
                    "country_code": row["country_code"],
                    "country_name": row["country_name"],
                }
        raise ValueError(f"Capital not found in reference A: {capital}")

    def president_ids(self, country_code: str) -> dict:
        code = country_code.strip().upper()
        rows = [row for row in self.index if row["country_code"].upper() == code]
        if not rows:
            raise ValueError(f"Country code not found in reference B: {country_code}")
        references = {row["presidency_reference"] for row in rows}
        if len(references) != 1:
            raise ValueError(f"Reference B is inconsistent for country code: {country_code}")
        return {
            "country_code": code,
            "presidency_reference": next(iter(references)),
            "president_ids": [row["president_id"] for row in rows],
        }

    def president_by_date(self, presidency_reference: str, target_date: str) -> dict:
        try:
            target = date.fromisoformat(target_date)
        except ValueError as exc:
            raise ValueError("target_date must use YYYY-MM-DD") from exc
        reference = presidency_reference.strip()
        matches = []
        for row in self.terms:
            if row["presidency_reference"] != reference:
                continue
            start = date.fromisoformat(row["start_date"])
            end = date.fromisoformat(row["end_date"])
            if start <= target < end:
                matches.append(row)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one term match for {presidency_reference} on {target_date}; found {len(matches)}"
            )
        row = matches[0]
        return {
            "president_id": row["president_id"],
            "president_name": row["president_name"],
            "target_date": target_date,
            "term_start": row["start_date"],
            "term_end": row["end_date"],
        }


def make_multiref_server():
    from mcp.server.mcpserver import MCPServer

    data = MultiRefData()
    server = MCPServer(
        "Multi-Reference Historical MCP",
        instructions=(
            "Read-only frozen reference tools. Resolve evidence across separate CSV references. "
            "Do not guess identifiers that have not been retrieved."
        ),
    )

    @server.tool()
    def lookup_country_by_capital(capital: str) -> dict:
        """Reference A: resolve a capital name to its country code and country name."""
        return data.country_by_capital(capital)

    @server.tool()
    def lookup_president_ids(country_code: str) -> dict:
        """Reference B: list president identifiers and return the key for the term reference."""
        return data.president_ids(country_code)

    @server.tool()
    def lookup_president_by_date(presidency_reference: str, target_date: str) -> dict:
        """Reference C: find which term in a retrieved presidency reference contains target_date."""
        return data.president_by_date(presidency_reference, target_date)

    return server


if __name__ == "__main__":
    make_multiref_server().run()
