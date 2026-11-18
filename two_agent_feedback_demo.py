"""Live strict demo covering the client's full two-agent/multi-reference suggestion."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from two_agent_feedback_service import TwoAgentMCPOrchestrationService

TRACE_PATH = Path("two_agent_feedback_trace.json")
EXPLANATION_PATH = Path("two_agent_pipeline_evidence.txt")


async def run(capital: str, start_date: str, end_date: str, model: str):
    service = TwoAgentMCPOrchestrationService(model=model)
    trace = await service.resolve(capital, start_date, end_date)
    TRACE_PATH.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    semantics = trace["pipeline_semantics"]
    explanation = (
        "TWO-AGENT MCP PIPELINE EVIDENCE\n\n"
        f"Question: {trace['question']}\n\n"
        "GENERATED:\n- " + "\n- ".join(semantics["generated"]) + "\n\n"
        "RETRIEVED:\n- " + "\n- ".join(semantics["retrieved"]) + "\n\n"
        "AUGMENTED THROUGH MCP:\n" + semantics["augmented_through_mcp"] + "\n\n"
        f"FINAL ANSWER: {trace['final_answer']['president_name']}\n"
    )
    EXPLANATION_PATH.write_text(explanation, encoding="utf-8")

    print("\n=== STRICT TWO-AGENT MCP MULTI-REFERENCE DEMONSTRATION ===")
    print("MCP orchestration service created:", ", ".join(trace["service_created_agents"]))
    print("Question:", trace["question"])
    print("\nAgent A generated:", json.dumps(trace["agent_a"]["generated_call"], ensure_ascii=False))
    print("Agent A listed:", trace["agent_a"]["retrieved"]["reference"], f"({len(trace['agent_a']['retrieved']['rows'])} rows)")
    print("Agent B generated:", json.dumps(trace["agent_b"]["generated_call"], ensure_ascii=False))
    print("Agent B listed:", trace["agent_b"]["retrieved"]["reference"], f"({len(trace['agent_b']['retrieved']['rows'])} rows)")
    print("Merged A+B evidence:", json.dumps(trace["merged_agent_lists"], ensure_ascii=False))
    print("Reference C call:", json.dumps(trace["reference_c_call"], ensure_ascii=False))
    print("Reference C retrieved:", trace["reference_c_retrieved"]["reference"], f"({len(trace['reference_c_retrieved']['rows'])} rows)")
    print("\nFINAL ANSWER:", trace["final_answer"]["president_name"])
    print("Trace saved:", TRACE_PATH)
    print("Pipeline explanation saved:", EXPLANATION_PATH)
    return trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capital", default="Paris")
    parser.add_argument("--start-date", default="2009-01-01")
    parser.add_argument("--end-date", default="2010-12-31")
    parser.add_argument("--model", default="qwen2.5:1.5b")
    args = parser.parse_args()
    asyncio.run(run(args.capital, args.start_date, args.end_date, args.model))


if __name__ == "__main__":
    main()
