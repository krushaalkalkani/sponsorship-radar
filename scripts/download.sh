#!/usr/bin/env bash
# Downloads the raw government disclosure files. Resumable: re-running skips complete files.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
while read -r kind tag url; do
  [ -z "${kind:-}" ] && continue
  out="data/raw/${kind}_${tag}.xlsx"
  if [ -s "$out" ]; then echo "have $out"; continue; fi
  echo "GET $out"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$out" "$url" || echo "FAILED $url"
done < ingest/sources.txt
ls -lh data/raw
