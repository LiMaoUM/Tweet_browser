#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
exec vllm serve "${STANCE_MODEL:-google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant}" \
  --port 8001 \
  --gpu-memory-utilization 0.68 \
  --max-model-len 16384 \
  --api-key "${LLM_API_KEY:-token-census}"
