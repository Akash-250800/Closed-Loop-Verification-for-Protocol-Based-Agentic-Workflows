@echo off
setlocal
cd /d "%~dp0"

echo === FULL-FEEDBACK TWO-AGENT MCP DEMONSTRATION ===
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found.
  exit /b 1
)

echo.
echo [1/4] Deterministic list-merge/date-range gate...
.venv\Scripts\python.exe two_agent_feedback_selftest.py
if errorlevel 1 goto :fail

echo.
echo [2/4] Ollama connection check...
.venv\Scripts\python.exe check_ollama.py
if errorlevel 1 goto :fail

echo.
echo [3/4] Live two-agent MCP service demonstration...
.venv\Scripts\python.exe two_agent_feedback_demo.py --capital Paris --start-date 2009-01-01 --end-date 2010-12-31
if errorlevel 1 goto :fail

echo.
echo [4/4] Validate saved evidence against feedback requirements...
.venv\Scripts\python.exe two_agent_feedback_validate.py
if errorlevel 1 goto :fail

echo.
echo FULL-FEEDBACK TWO-AGENT DEMO PASSED
echo Trace: %CD%\two_agent_feedback_trace.json
echo Pipeline explanation: %CD%\two_agent_pipeline_evidence.txt
exit /b 0

:fail
echo.
echo TWO-AGENT DEMO STOPPED BECAUSE A GATE FAILED.
exit /b 1
