#!/usr/bin/env bash
# Start the Voila app on the host (the host python env has all app deps).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -f "$REPO_ROOT/deploy/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/deploy/.env"
  set +a
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$REPO_ROOT/Frontend"
exec voila --Voila.ip=0.0.0.0 --port=8866 tweet_browser.ipynb
