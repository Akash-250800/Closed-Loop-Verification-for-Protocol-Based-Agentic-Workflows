FULL-FEEDBACK TWO-AGENT MCP DEMONSTRATION

This supplementary demo is deliberately separate from the already validated 36-trial
multi-reference GVEF experiment. It does not overwrite those results.

Pipeline
--------
1. The MCP orchestration service creates Agent A and Agent B.
2. Agent A generates a tool call and lists Reference A: countries.csv.
3. Only after A returns can the service identify the requested country's country_code.
4. Agent B then generates a tool call and lists independent Reference B: president_index.csv.
5. The two retrieved lists are merged using the country_code discovered from A.
6. The merge yields presidency_reference and candidate president IDs.
7. That retrieved/merged presidency_reference is required to construct the Reference C MCP call.
8. Reference C returns dated rows from president_terms.csv. It does NOT return a preselected answer.
9. The service checks the requested start/end date interval against those retrieved term rows and returns one suitable president.

No expected president and no expected_tool oracle is supplied to either live LLM agent.

What is generated
-----------------
Agent A MCP call; Agent B MCP call; Reference C call constructed from merged retrieved evidence.

What is retrieved
-----------------
Reference A country rows; Reference B presidency index rows; Reference C dated term rows.

How MCP augments the task
-------------------------
Identifiers retrieved from earlier MCP references are merged and used to construct the next call.
The final answer is selected only after the third MCP reference supplies dated term evidence.

Run one command
---------------
RUN_TWO_AGENT_FULL_FEEDBACK.bat

A successful run must end with BOTH:
STRICT TWO-AGENT TRACE VALIDATION: PASS
FULL-FEEDBACK TWO-AGENT DEMO PASSED

Evidence created
----------------
two_agent_feedback_trace.json
two_agent_pipeline_evidence.txt

Keep both files plus the final terminal output/screenshot.
