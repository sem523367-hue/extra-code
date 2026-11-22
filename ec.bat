@echo off
rem Extra Code Agent - terminal coding agent over the local stack.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set SCRIPT_DIR=%~dp0
"C:\Users\Sem52\AppData\Local\Programs\Python\Python311\python.exe" "%SCRIPT_DIR%tools\ec_agent.py" %*
