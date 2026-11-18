"""Pure merge and date-range selection logic for the strict two-agent demo."""
from __future__ import annotations

from datetime import date


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().replace(",", "").split())


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Date must use YYYY-MM-DD: {value!r}") from exc


def merge_agent_reference_lists(country_payload: dict, index_payload: dict, capital: str) -> dict:
    """Merge only the two MCP-retrieved lists; no hidden benchmark truth is used."""
    country_rows = country_payload.get("rows") if isinstance(country_payload, dict) else None
    index_rows = index_payload.get("rows") if isinstance(index_payload, dict) else None
    if not isinstance(country_rows, list) or not isinstance(index_rows, list):
        raise ValueError("Both agent payloads must contain rows lists")

    country_matches = [r for r in country_rows if _norm(r.get("capital", "")) == _norm(capital)]
    if len(country_matches) != 1:
        raise ValueError(f"Expected one country for capital {capital!r}; found {len(country_matches)}")
    country = country_matches[0]

    code = country["country_code"]
    president_rows = [r for r in index_rows if r.get("country_code") == code]
    if not president_rows:
        raise ValueError(f"No Reference B rows for retrieved country code {code!r}")
    refs = {r.get("presidency_reference") for r in president_rows}
    if len(refs) != 1:
        raise ValueError(f"Inconsistent presidency reference for retrieved code {code!r}")

    return {
        "capital": country["capital"],
        "country_code": code,
        "country_name": country["country_name"],
        "presidency_reference": next(iter(refs)),
        "candidate_president_ids": [r["president_id"] for r in president_rows],
        "sources_merged": [country_payload.get("reference"), index_payload.get("reference")],
    }


def select_president_for_interval(term_payload: dict, start_date: str, end_date: str) -> dict:
    """Select only from retrieved Reference C rows; no hidden answer/oracle is consulted."""
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must not be before start_date")
    rows = term_payload.get("rows") if isinstance(term_payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("Reference C payload must contain a rows list")

    containing = []
    for row in rows:
        term_start = _parse_date(row["start_date"])
        term_end = _parse_date(row["end_date"])
        # Stored term_end is exclusive; requested interval must lie wholly inside one term.
        if term_start <= start and end < term_end:
            containing.append(row)

    if len(containing) != 1:
        raise ValueError(
            f"Expected exactly one president whose term contains {start_date}..{end_date}; found {len(containing)}"
        )
    row = containing[0]
    return {
        "president_id": row["president_id"],
        "president_name": row["president_name"],
        "requested_start": start_date,
        "requested_end": end_date,
        "term_start": row["start_date"],
        "term_end": row["end_date"],
        "selection_rule": "requested interval lies wholly inside retrieved term",
    }
