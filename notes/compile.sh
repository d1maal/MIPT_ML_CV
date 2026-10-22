#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v tectonic >/dev/null 2>&1; then
  tectonic -X compile robust-stability-mikhailov.tex
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode robust-stability-mikhailov.tex
  xelatex -interaction=nonstopmode robust-stability-mikhailov.tex
else
  echo "Need tectonic or xelatex" >&2
  exit 1
fi
echo "Wrote $(pwd)/robust-stability-mikhailov.pdf"
