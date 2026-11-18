"""Validate that the saved live trace covers the client's strict two-agent requirements."""
from __future__ import annotations

import json
from pathlib import Path

TRACE = Path("two_agent_feedback_trace.json")


def main():
    if not TRACE.exists():
        raise SystemExit("VALIDATION FAIL: two_agent_feedback_trace.json is missing")
    t = json.loads(TRACE.read_text(encoding="utf-8"))

    assert t.get("service_created_agents") == ["Agent A", "Agent B"]
    a, b, c = t["agent_a"], t["agent_b"], t["reference_c_retrieved"]
    merged = t["merged_agent_lists"]
    final = t["final_answer"]

    assert a["retrieved"].get("reference") == "countries.csv"
    assert b["retrieved"].get("reference") == "president_index.csv"
    assert c.get("reference") == "president_terms.csv"
    assert isinstance(a["retrieved"].get("rows"), list) and len(a["retrieved"]["rows"]) > 1
    assert isinstance(b["retrieved"].get("rows"), list) and len(b["retrieved"]["rows"]) > 1

    # A must contribute the merged country code.
    a_codes = {r.get("country_code") for r in a["retrieved"]["rows"]}
    assert merged.get("country_code") in a_codes
    # B must contribute the merged presidency key/candidate IDs for that retrieved country code.
    b_rows = [r for r in b["retrieved"]["rows"] if r.get("country_code") == merged.get("country_code")]
    assert b_rows
    assert merged.get("presidency_reference") in {r.get("presidency_reference") for r in b_rows}
    # C call must be impossible to form correctly until the merged A+B key exists.
    assert t["reference_c_call"]["args"].get("presidency_reference") == merged.get("presidency_reference")
    # Final answer must be selected from C rows; A and B do not contain president names.
    assert final.get("president_name") in {r.get("president_name") for r in c.get("rows", [])}
    assert all("president_name" not in r for r in a["retrieved"]["rows"])
    assert all("president_name" not in r for r in b["retrieved"]["rows"])
    # Explicit generated/retrieved/augmented explanation must be persisted.
    semantics = t.get("pipeline_semantics", {})
    assert semantics.get("generated") and semantics.get("retrieved") and semantics.get("augmented_through_mcp")

    print("STRICT TWO-AGENT TRACE VALIDATION: PASS")
    print("- MCP orchestration service created two agents")
    print("- Agent A and Agent B each listed a separate reference")
    print("- their lists were merged into grounded country/presidency evidence")
    print("- Reference C call depended on the merged key")
    print("- final answer came only from retrieved dated term rows")
    print("- generated / retrieved / MCP-augmented pipeline explanation was saved")


if __name__ == "__main__":
    main()
