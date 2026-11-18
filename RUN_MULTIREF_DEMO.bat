@echo off
cd /d E:\MCP-Thesis
call .venv\Scripts\activate
python -m pytest -q
if errorlevel 1 goto :fail
python check_ollama.py
if errorlevel 1 goto :fail
python experiment.py --mode ollama --model qwen2.5:1.5b --benchmark multiref --tasks 5 --replicates 1 --suite baselines --faults none --retry-budgets 2 --out results_multiref_demo
if errorlevel 1 goto :fail
python analyse.py --results results_multiref_demo
if errorlevel 1 goto :fail
echo.
echo MULTI-REFERENCE DEMO COMPLETE
exit /b 0
:fail
echo.
echo STOP: a command failed. Copy the error shown above.
exit /b 1
