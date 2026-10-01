@echo off
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv || (echo Python 3.9+ is required & pause & exit /b 1) )
call .venv\Scripts\activate
python -m pip install -q -r requirements.txt
start "" http://127.0.0.1:5000
python backend\app.py
pause
