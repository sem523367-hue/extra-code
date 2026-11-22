@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PY=C:\Users\Sem52\AppData\Local\Programs\Python\Python311\python.exe
if not exist "%PY%" set PY=python
"%PY%" -m tools.ask %*
endlocal
