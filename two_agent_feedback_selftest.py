"""Deterministic gate for list-merge, dependency and date-range logic. No Ollama/MCP required."""
from multiref_server import MultiRefData
from two_agent_feedback_logic import merge_agent_reference_lists, select_president_for_interval


def main():
    data = MultiRefData()
    country_payload = {
        "reference": "countries.csv",
        "rows": [
            {"capital": r["capital"], "country_code": r["country_code"], "country_name": r["country_name"]}
            for r in data.countries
        ],
    }
    index_payload = {
        "reference": "president_index.csv",
        "rows": [
            {
                "country_code": r["country_code"],
                "presidency_reference": r["presidency_reference"],
                "president_id": r["president_id"],
            }
            for r in data.index
        ],
    }
    cases = [
        ("Washington D.C.", "2011-01-01", "2012-12-31", "Barack Obama"),
        ("Paris", "2009-01-01", "2010-12-31", "Nicolas Sarkozy"),
        ("Mexico City", "2008-01-01", "2010-12-31", "Felipe Calderon"),
        ("Brasilia", "2012-01-01", "2014-12-31", "Dilma Rousseff"),
    ]
    for capital, start, end, expected in cases:
        merged = merge_agent_reference_lists(country_payload, index_payload, capital)
        c_rows = [
            {
                "presidency_reference": r["presidency_reference"],
                "president_id": r["president_id"],
                "president_name": r["president_name"],
                "start_date": r["start_date"],
                "end_date": r["end_date"],
            }
            for r in data.terms
            if r["presidency_reference"] == merged["presidency_reference"]
        ]
        answer = select_president_for_interval({"reference": "president_terms.csv", "rows": c_rows}, start, end)
        assert answer["president_name"] == expected, (capital, start, end, answer)
    print("STRICT TWO-AGENT SELF-TEST: PASS (4/4 list merges and dependent range resolutions)")


if __name__ == "__main__":
    main()
