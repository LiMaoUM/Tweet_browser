"""Central configuration for Tweet Browser backends and data paths.

Every value can be overridden with an environment variable of the same name.
Defaults match the historical hardcoded behavior so running with no env set
changes nothing. See deploy/.env.example for documentation of each variable.
"""
import os

SUMMARIZER_BASE_URL = os.environ.get("SUMMARIZER_BASE_URL", "http://localhost:8000/v1")
SUMMARIZER_MODEL = os.environ.get("SUMMARIZER_MODEL", "Lllama3TS_unsloth_vllm")
STANCE_BASE_URL = os.environ.get("STANCE_BASE_URL", "http://localhost:8001/v1")
STANCE_MODEL = os.environ.get(
    "STANCE_MODEL", "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized"
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "token-census")
SUMMARY_TIMEOUT_S = float(os.environ.get("SUMMARY_TIMEOUT_S", "120"))
STANCE_TIMEOUT_S = float(os.environ.get("STANCE_TIMEOUT_S", "180"))
DATASET_PATH = os.environ.get("DATASET_PATH", "allCensus_sample_with_demographics.csv")
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "allCensus_sample_embeddings.csv")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or None
