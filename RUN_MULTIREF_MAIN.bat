@echo off
cd /d E:\MCP-Thesis
call .venv\Scripts\activate
python run_multiref_final.py
if errorlevel 1 goto :fail
echo.
echo FINAL MULTI-REFERENCE RUN PASSED
exit /b 0
:fail
echo.
echo STOP: The guarded run stopped before accepting bad evidence. Copy the last 20 lines if needed.
exit /b 1
