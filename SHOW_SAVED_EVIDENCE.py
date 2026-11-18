from pathlib import Path
import csv, json, sys

ROOT = Path(__file__).resolve().parent

def show_multiref():
    p = ROOT / "results_multiref_final" / "trials.csv"
    rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    faulted = [
        r for r in rows
        if r.get("condition") == "closed_loop" and r.get("fault") != "none"
    ]
    completed = sum(int(float(r["task_completed"])) for r in faulted)

    ok = len(rows) == 36 and len(faulted) == 8 and completed == 8

    print("=" * 72)
    print("SCREENSHOT 6 - SAVED MULTI-REFERENCE EXPERIMENT EVIDENCE")
    print("=" * 72)
    print(f"Saved trial records: {len(rows)}/36")
    print(f"FINAL CHECK: closed-loop completed {completed}/{len(faulted)} faulted multi-reference tasks.")
    print(f"Evidence source: {p}")
    print("MULTI-REFERENCE SAVED RESULTS VALIDATION:", "PASS" if ok else "FAIL")
    print("=" * 72)
    return ok

def show_two_agent():
    p = ROOT / "two_agent_feedback_trace.json"
    d = json.loads(p.read_text(encoding="utf-8"))

    agents = d.get("service_created_agents", [])
    a = d["agent_a"]["retrieved"]
    b = d["agent_b"]["retrieved"]
    merged = d["merged_agent_lists"]
    c = d["reference_c_retrieved"]
    final = d["final_answer"]

    ok = (
        agents == ["Agent A", "Agent B"]
        and a.get("reference") == "countries.csv"
        and b.get("reference") == "president_index.csv"
        and c.get("reference") == "president_terms.csv"
        and final.get("president_name") == "Nicolas Sarkozy"
    )

    print()
    print("=" * 72)
    print("SCREENSHOT 7 - SAVED TWO-AGENT MCP EVIDENCE")
    print("=" * 72)
    print("MCP orchestration service created:", ", ".join(agents))
    print(f"Agent A listed: {a['reference']} ({len(a['rows'])} rows)")
    print(f"Agent B listed: {b['reference']} ({len(b['rows'])} rows)")
    print("Merged A+B evidence:", json.dumps(merged, ensure_ascii=False))
    print(f"Reference C retrieved: {c['reference']} ({len(c['rows'])} rows)")
    print("FINAL ANSWER:", final["president_name"])
    print("STRICT TWO-AGENT SAVED TRACE VALIDATION:", "PASS" if ok else "FAIL")
    print("=" * 72)
    return ok

if __name__ == "__main__":
    ok1 = show_multiref()
    input("\nPress ENTER after taking Screenshot 6 to show Screenshot 7...")
    ok2 = show_two_agent()
    sys.exit(0 if (ok1 and ok2) else 1)
