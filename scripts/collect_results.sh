#!/usr/bin/env bash
# Package all run logs (CSV / JSON / stdout) into one tarball for download.
# Checkpoints (*.pkl) are excluded to keep it small.
#
# Usage: bash scripts/collect_results.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="results_$(date +%Y%m%d_%H%M%S).tar.gz"
find exp -type f \( -name '*.csv' -o -name '*.json' -o -name 'stdout_*.log' \) -print0 \
  | tar --null -czf "${OUT}" -T -
echo "[collect_results] wrote ${OUT} ($(du -h "${OUT}" | cut -f1))"
echo "[collect_results] download this file and send it back for analysis."
