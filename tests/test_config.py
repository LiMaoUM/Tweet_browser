def test_defaults_match_current_behavior(fresh_config):
    config = fresh_config()
    assert config.SUMMARIZER_BASE_URL == "http://localhost:8000/v1"
    assert config.SUMMARIZER_MODEL == "Lllama3TS_unsloth_vllm"
    assert config.STANCE_BASE_URL == "http://localhost:8001/v1"
    assert config.STANCE_MODEL == "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
    assert config.LLM_API_KEY == "token-census"
    assert config.SUMMARY_TIMEOUT_S == 120.0
    assert config.STANCE_TIMEOUT_S == 180.0
    assert config.DATASET_PATH == "allCensus_sample_with_demographics.csv"
    assert config.EMBEDDINGS_PATH == "allCensus_sample_embeddings.csv"
    assert config.EMBEDDING_DEVICE is None


def test_env_overrides(fresh_config):
    config = fresh_config(
        SUMMARIZER_BASE_URL="http://gpu2:9000/v1",
        SUMMARY_TIMEOUT_S="5",
        EMBEDDING_DEVICE="cpu",
    )
    assert config.SUMMARIZER_BASE_URL == "http://gpu2:9000/v1"
    assert config.SUMMARY_TIMEOUT_S == 5.0
    assert config.EMBEDDING_DEVICE == "cpu"
