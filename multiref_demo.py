"""Run one live Ollama+MCP multi-reference chain and print a defence-friendly trace."""
import asyncio
import json
from pathlib import Path

from agents import new_agent
from benchmarks import multiref_tasks
from check_ollama import check
from experiment import run_trial


async def main():
    model = "qwen2.5:1.5b"
    check(model=model)
    task = multiref_tasks()[0]
    agent = new_agent("ollama", model)
    try:
        row = await run_trial(
            agent,
            task,
            condition="closed_loop",
            fault="none",
            feedback_style="schema_diff",
            retry_budget=2,
            rep=1,
        )
    finally:
        if hasattr(agent, "aclose"):
            await agent.aclose()

    print("\n=== MULTI-REFERENCE MCP DEMONSTRATION ===")
    print("Question:", task["description"])
    print("References: countries.csv -> president_index.csv -> president_terms.csv")
    print("\nGenerated/retrieved chain:")
    final_output = None
    for event in row["events"]:
        if event.get("stage") == "circuit_breaker":
            print("  CIRCUIT BREAKER:", event.get("reason"))
            continue
        if event.get("proposal"):
            print(f"  Step {event.get('step')} generated call: {event['proposal']}")
        if event.get("output") is not None:
            print(f"  Step {event.get('step')} MCP retrieved: {json.dumps(event['output'], ensure_ascii=False)}")
            final_output = event["output"]
        if event.get("error"):
            print(f"  Step {event.get('step')} verifier: {event['error'].get('code')} - {event['error'].get('message')}")
        if event.get("feedback_sent"):
            print(f"  Step {event.get('step')} feedback: {event['feedback_sent']}")

    if row["task_completed"] and isinstance(final_output, dict):
        print("\nFINAL ANSWER:", final_output.get("president_name", final_output))
    else:
        print("\nFINAL ANSWER: task did not complete")
    print("Task completed:", bool(row["task_completed"]))
    print("Retries:", row["retry_count"])
    print("Tokens:", row["total_tokens"])

    Path("multiref_demo_trace.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("Trace saved: multiref_demo_trace.json")


if __name__ == "__main__":
    asyncio.run(main())
