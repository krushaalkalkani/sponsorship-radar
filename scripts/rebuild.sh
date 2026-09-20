#!/usr/bin/env bash
# Full pipeline: download -> fact tables -> employer rollups -> embeddings.
set -e
cd "$(dirname "$0")/.."
./scripts/download.sh
./.venv/bin/python ingest/build.py
./.venv/bin/python ingest/rollup.py
./.venv/bin/python ingest/embed.py
./.venv/bin/python ingest/pack.py
echo "rebuild complete"
