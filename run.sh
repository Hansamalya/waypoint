#!/usr/bin/env bash
set -e; cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate; python -m pip install -q -r requirements.txt
( sleep 2; (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:5000) || (command -v open >/dev/null && open http://127.0.0.1:5000) ) >/dev/null 2>&1 &
python backend/app.py
