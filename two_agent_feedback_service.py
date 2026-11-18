"""MCP orchestration service that creates and coordinates exactly two LLM agents.

Agent A lists Reference A. The service identifies the requested country's retrieved code.
Agent B then lists Reference B; the service merges that list using Agent A's retrieved code.
The resulting presidency_reference is required to retrieve Reference C.
Finally the service selects one answer from the retrieved dated term rows.
"""
from __future__ import annotations

from agents import new_agent
from two_agent_feedback_logic import merge_agent_reference_lists, select_president_for_interval
from two_agent_feedback_server import make_two_agent_feedback_server
from verification import parse_proposal, unwrap_mcp_reply, validate


async def _agent_list_reference(agent, client, all_tools, *, tool_name: str, task: dict, instruction: str, label: str, history=()):
    catalog = {tool_name: all_tools[tool_name]}
    step = {"instruction": instruction, "mutating": False}
    raw, usage = await agent.generate(task, step, catalog, "", history)
    call, err = parse_proposal(raw)
    if err:
        raise RuntimeError(f"{label} generated invalid proposal: {err.message}; raw={raw!r}")
    err = validate(call, catalog, step, history=history)
    if err:
        raise RuntimeError(f"{label} proposal failed verification: {err.message}; call={call!r}")
    reply = await client.call_tool(call["tool"], call["args"])
    payload, err = unwrap_mcp_reply(reply)
    if err:
        raise RuntimeError(f"{label} MCP execution failed: {err.message}")
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise RuntimeError(f"{label} returned malformed list payload: {payload!r}")
    return {
        "label": label,
        "generated_call": call,
        "retrieved": payload,
        "tokens": None if usage is None else int(usage["input"]) + int(usage["output"]),
    }


class TwoAgentMCPOrchestrationService:
    """The MCP-based service itself instantiates the two specialised LLM agents."""

    def __init__(self, model: str = "qwen2.5:1.5b"):
        self.model = model

    async def resolve(self, capital: str, start_date: str, end_date: str) -> dict:
        from mcp.client import Client

        task = {
            "description": (
                f"Who was president of the country whose capital is {capital} throughout "
                f"the interval {start_date} to {end_date}? Use only MCP-retrieved references."
            )
        }

        # Strictly satisfy the requested setup: the orchestration service creates two agents.
        agent_a = new_agent("ollama", self.model)
        agent_b = new_agent("ollama", self.model)

        try:
            async with Client(make_two_agent_feedback_server()) as client:
                tools = {tool.name: tool for tool in (await client.list_tools()).tools}

                result_a = await _agent_list_reference(
                    agent_a,
                    client,
                    tools,
                    tool_name="list_country_reference",
                    task=task,
                    instruction=(
                        "Agent A: retrieve and list the complete contents of your assigned Reference A through MCP. "
                        "Do not answer the president question from memory."
                    ),
                    label="Agent A",
                )

                # Step dependency: we cannot identify the country code until A has actually returned Reference A.
                # Extract only the requested country from A now; B is executed after A, not in parallel.
                a_rows = result_a["retrieved"]["rows"]
                norm_capital = " ".join(capital.casefold().replace(",", "").split())
                country_matches = [
                    r for r in a_rows
                    if " ".join(str(r.get("capital", "")).casefold().replace(",", "").split()) == norm_capital
                ]
                if len(country_matches) != 1:
                    raise RuntimeError(f"Reference A did not uniquely resolve capital {capital!r}")
                retrieved_country_code = country_matches[0]["country_code"]

                result_b = await _agent_list_reference(
                    agent_b,
                    client,
                    tools,
                    tool_name="list_president_index_reference",
                    task=task,
                    instruction=(
                        "Agent B: retrieve and list the complete contents of your assigned Reference B through MCP. "
                        f"Agent A has already retrieved country_code={retrieved_country_code!r}; the service will merge "
                        "your returned list against that retrieved code. Do not invent a president."
                    ),
                    label="Agent B",
                    history=({"tool": result_a["generated_call"]["tool"], "output": result_a["retrieved"]},),
                )

                # The two retrieved lists are merged; this is where A's result constrains B's rows.
                merged_ab = merge_agent_reference_lists(result_a["retrieved"], result_b["retrieved"], capital)

                # Step dependency: Reference C cannot be constructed before the A+B merge produces this key.
                reference_c_call = {
                    "tool": "list_presidential_terms",
                    "args": {"presidency_reference": merged_ab["presidency_reference"]},
                }
                reply = await client.call_tool(reference_c_call["tool"], reference_c_call["args"])
                reference_c, err = unwrap_mcp_reply(reply)
                if err:
                    raise RuntimeError(f"Reference C retrieval failed: {err.message}")
                if not isinstance(reference_c, dict) or not isinstance(reference_c.get("rows"), list):
                    raise RuntimeError(f"Reference C returned malformed payload: {reference_c!r}")

                final = select_president_for_interval(reference_c, start_date, end_date)
        finally:
            await agent_a.aclose()
            await agent_b.aclose()

        merged_all = {
            **merged_ab,
            "requested_interval": {"start": start_date, "end": end_date},
            "sources": [
                result_a["retrieved"].get("reference"),
                result_b["retrieved"].get("reference"),
                reference_c.get("reference"),
            ],
        }

        return {
            "question": task["description"],
            "pipeline_semantics": {
                "generated": [
                    "Agent A's MCP tool call",
                    "Agent B's MCP tool call",
                    "Reference C call constructed from the merged retrieved presidency_reference",
                ],
                "retrieved": [
                    "Reference A country/capital rows from countries.csv",
                    "Reference B country/presidency rows from president_index.csv",
                    "Reference C dated term rows from president_terms.csv",
                ],
                "augmented_through_mcp": (
                    "MCP-retrieved A and B lists are merged to create grounded identifiers; the merged presidency_reference "
                    "augments the next MCP call to C; retrieved term dates are then filtered against the requested interval."
                ),
            },
            "service_created_agents": ["Agent A", "Agent B"],
            "agent_a": result_a,
            "agent_b": result_b,
            "merged_agent_lists": merged_ab,
            "reference_c_call": reference_c_call,
            "reference_c_retrieved": reference_c,
            "merged_evidence": merged_all,
            "final_answer": final,
        }
