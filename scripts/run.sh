#!/usr/bin/env bash
# Start the app locally. Reads ANTHROPIC_API_KEY from .env if present.
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
exec ./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
