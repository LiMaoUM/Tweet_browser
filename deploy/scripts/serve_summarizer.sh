#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
exec vllm serve /home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm \
  --served-model-name Lllama3TS_unsloth_vllm \
  --port 8000 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --api-key "${LLM_API_KEY:-token-census}"
