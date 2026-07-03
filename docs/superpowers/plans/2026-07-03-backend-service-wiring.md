# Backend Service Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Tweet Browser a launchable service: summarization and stance detection wired to self-hosted vLLM with config, health checks, and error handling; embeddings fast path; CSV-backed demographics provider; one-command deploy.

**Architecture:** Keep the in-process Voila -> `Session` -> OpenAI-compatible-client shape. Add `Frontend/config.py` as the single source of endpoints/paths, refactor `Frontend/prompts.py` around it with typed errors and hardened parsing, and add a `deploy/` directory (docker compose for the two vLLM servers, host script for Voila).

**Tech Stack:** Python 3.10 (system interpreter `/usr/bin/python3`, already has torch 2.6/transformers/openai/pandas), openai SDK, vLLM via `vllm/vllm-openai` Docker image, Voila, pytest.

**Spec:** `docs/superpowers/specs/2026-07-03-backend-service-wiring-design.md`

## Global Constraints

- All config defaults MUST match current behavior exactly: `http://localhost:8000/v1`, `Lllama3TS_unsloth_vllm`, `http://localhost:8001/v1`, api key `token-census`.
- Stance model: `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant`. Summarizer weights: `/home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm` (verified on disk).
- Both vLLM services on GPU 1: summarizer `--gpu-memory-utilization 0.25`, stance `0.68`.
- Ports: summarizer 8000, stance 8001, Voila 8866.
- No new runtime dependencies (`httpx` already ships with the `openai` package). pytest is dev-only.
- Tests run with the system interpreter: `python3 -m pytest`. (Deviation from the user's uv preference: this project is pip-managed with heavy preinstalled system deps; converting to uv is out of scope.)
- Deviation from spec, flagged at plan review: docker-compose contains only the two vLLM services; Voila runs on the host via `deploy/scripts/serve_app.sh`. Reason: the host Python env is the working app env; the host `vllm` CLI is broken (torch ABI mismatch), so vLLM goes in Docker (nvidia runtime is installed), while the app does not need a container for intranet use.
- Existing `tests/tweet_tester.py` must remain untouched and passing.
- Working directory for all commands: repo root `/home/maolee/projects/Tweet_browser`.

---

### Task 1: Test infrastructure + `Frontend/config.py`

**Files:**
- Create: `Frontend/config.py`
- Create: `deploy/.env.example`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces module `config` with attributes (all read from env at import time, with these defaults):
  `SUMMARIZER_BASE_URL: str = "http://localhost:8000/v1"`, `SUMMARIZER_MODEL: str = "Lllama3TS_unsloth_vllm"`, `STANCE_BASE_URL: str = "http://localhost:8001/v1"`, `STANCE_MODEL: str = "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"`, `LLM_API_KEY: str = "token-census"`, `SUMMARY_TIMEOUT_S: float = 120.0`, `STANCE_TIMEOUT_S: float = 180.0`, `DATASET_PATH: str = "allCensus_sample_with_demographics.csv"`, `EMBEDDINGS_PATH: str = "allCensus_sample_embeddings.csv"`, `EMBEDDING_DEVICE: str | None = None`.
- Produces `tests/conftest.py` putting `Frontend/` on `sys.path` and a `fresh_config` fixture used by later tasks.

- [ ] **Step 1: Install pytest (dev only)**

Run: `python3 -m pip install --user pytest`
Expected: `Successfully installed pytest-...` (or already satisfied)

- [ ] **Step 2: Write conftest and the failing test**

Create `tests/conftest.py`:

```python
import importlib
import os
import sys

import pytest

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "Frontend")
sys.path.insert(0, os.path.abspath(FRONTEND_DIR))


@pytest.fixture
def fresh_config(monkeypatch):
    """Reload the config module with a controlled environment, restoring defaults after."""
    import config

    def _load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(config)

    yield _load
    monkeypatch.undo()
    importlib.reload(config)
```

Create `tests/test_config.py`:

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Write `Frontend/config.py`**

```python
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
    "STANCE_MODEL", "google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"
)
LLM_API_KEY = os.environ.get("LLM_API_KEY", "token-census")
SUMMARY_TIMEOUT_S = float(os.environ.get("SUMMARY_TIMEOUT_S", "120"))
STANCE_TIMEOUT_S = float(os.environ.get("STANCE_TIMEOUT_S", "180"))
DATASET_PATH = os.environ.get("DATASET_PATH", "allCensus_sample_with_demographics.csv")
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", "allCensus_sample_embeddings.csv")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE") or None
```

Create `deploy/.env.example`:

```bash
# Tweet Browser backend configuration.
# Copy to deploy/.env and adjust; deploy/scripts/serve_app.sh sources deploy/.env.
# Every variable is optional; the values shown are the defaults baked into Frontend/config.py.

# vLLM summarizer (fine-tuned Llama-3-8B)
SUMMARIZER_BASE_URL=http://localhost:8000/v1
SUMMARIZER_MODEL=Lllama3TS_unsloth_vllm

# vLLM stance model (Gemma 4 26B MoE)
STANCE_BASE_URL=http://localhost:8001/v1
STANCE_MODEL=google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant

# Shared API key both vLLM servers are started with
LLM_API_KEY=token-census

# Request timeouts (seconds); stance timeout applies per 50-tweet batch
SUMMARY_TIMEOUT_S=120
STANCE_TIMEOUT_S=180

# Data files, relative to the Frontend/ working directory
DATASET_PATH=allCensus_sample_with_demographics.csv
EMBEDDINGS_PATH=allCensus_sample_embeddings.csv

# Device for the BGE embedding model; unset = cuda if available, else cpu
#EMBEDDING_DEVICE=cuda

# HuggingFace token for pulling the (license-gated) Gemma stance model
#HF_TOKEN=
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add Frontend/config.py deploy/.env.example tests/conftest.py tests/test_config.py
git commit -m "feat: central env-driven config for backends and data paths"
```

---

### Task 2: Refactor `Frontend/prompts.py` (lazy clients, BackendError, hardened parsing, health check)

**Files:**
- Rewrite: `Frontend/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `config` module from Task 1.
- Produces (used by Tasks 3, 4, 6, 7):
  - `class BackendError(Exception)` with attributes `backend: str`, `url: str`; message reads `"<backend> backend at <url>: <detail>"`.
  - `ai_summarize(tweets: str) -> str` — raises `BackendError` on any `openai.OpenAIError`.
  - `async stance_annotation(tweets: str, topic, stances, examples) -> str` — tries `response_format={"type": "json_object"}` first, retries once without it on `openai.BadRequestError`; raises `BackendError` on any other `openai.OpenAIError`.
  - `parse_stance_response(text: str, start: int, end: int) -> dict[int, int]` — always returns a stance for every id in `[start, end)`; anything missing/unparseable maps to `-1`.
  - `check_backends() -> dict` — `{"summarizer": bool, "stance": bool}` via GET `<base_url>/models`, 5s timeout, never raises.
  - `_get_summarizer_client()` / `_get_stance_client()` — lazy singletons (monkeypatch targets for tests).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prompts.py`:

```python
import asyncio

import httpx
import openai
import pytest


def _connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1"))


# ---------- parse_stance_response ----------

def test_parse_plain_json():
    import prompts
    text = '{"tweet-0": "1", "tweet-1": 0, "tweet-2": "-1"}'
    assert prompts.parse_stance_response(text, 0, 3) == {0: 1, 1: 0, 2: -1}


def test_parse_json_wrapped_in_prose():
    import prompts
    text = 'Sure! Here are the stances:\n{"tweet-5": 2, "tweet-6": 0}\nLet me know.'
    assert prompts.parse_stance_response(text, 5, 7) == {5: 2, 6: 0}


def test_parse_missing_and_invalid_keys_default_to_minus_one():
    import prompts
    text = '{"tweet-0": "yes", "tweet-2": 1}'
    assert prompts.parse_stance_response(text, 0, 3) == {0: -1, 1: -1, 2: 1}


def test_parse_garbage_returns_all_minus_one():
    import prompts
    assert prompts.parse_stance_response("no json here", 0, 2) == {0: -1, 1: -1}


# ---------- ai_summarize ----------

def test_ai_summarize_success(monkeypatch):
    import prompts

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "Lllama3TS_unsloth_vllm"
            msg = type("M", (), {"content": "a summary (0, 1)"})
            choice = type("C", (), {"message": msg})
            return type("R", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(prompts, "_get_summarizer_client", lambda: FakeClient())
    assert prompts.ai_summarize("0-[hello] 1-[world]") == "a summary (0, 1)"


def test_ai_summarize_raises_backend_error(monkeypatch):
    import prompts

    class FakeCompletions:
        def create(self, **kwargs):
            raise _connection_error()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(prompts, "_get_summarizer_client", lambda: FakeClient())
    with pytest.raises(prompts.BackendError) as exc:
        prompts.ai_summarize("0-[hello]")
    assert exc.value.backend == "summarizer"
    assert "8000" in exc.value.url


# ---------- stance_annotation ----------

class _FakeAsyncCompletions:
    def __init__(self, fail_json_object=False, fail_always=False):
        self.fail_json_object = fail_json_object
        self.fail_always = fail_always
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_always:
            raise _connection_error()
        if self.fail_json_object and "response_format" in kwargs:
            raise openai.BadRequestError(
                "response_format unsupported",
                response=httpx.Response(400, request=httpx.Request("POST", "http://x/v1")),
                body=None,
            )
        msg = type("M", (), {"content": '{"tweet-0": 1}'})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice]})


def _fake_stance_client(completions):
    return type("Client", (), {"chat": type("Chat", (), {"completions": completions})})


def test_stance_annotation_uses_guided_json(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions()
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    result = asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro", "anti"], {}))
    assert result == '{"tweet-0": 1}'
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_stance_annotation_falls_back_without_response_format(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions(fail_json_object=True)
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    result = asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro", "anti"], {}))
    assert result == '{"tweet-0": 1}'
    assert len(fake.calls) == 2
    assert "response_format" not in fake.calls[1]


def test_stance_annotation_raises_backend_error(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions(fail_always=True)
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    with pytest.raises(prompts.BackendError) as exc:
        asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro"], {}))
    assert exc.value.backend == "stance"


# ---------- check_backends ----------

def test_check_backends_all_up(monkeypatch):
    import prompts

    def fake_get(url, headers=None, timeout=None):
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(prompts.httpx, "get", fake_get)
    assert prompts.check_backends() == {"summarizer": True, "stance": True}


def test_check_backends_down(monkeypatch):
    import prompts

    def fake_get(url, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(prompts.httpx, "get", fake_get)
    assert prompts.check_backends() == {"summarizer": False, "stance": False}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_prompts.py -v`
Expected: FAIL / ERROR (`AttributeError: module 'prompts' has no attribute 'parse_stance_response'` and similar)

- [ ] **Step 3: Rewrite `Frontend/prompts.py`**

Replace the entire file with:

```python
import json
import re

import httpx
import openai
from openai import AsyncOpenAI

import config

AI_SUMMARY_PROMPT = """"I would like you to help me by summarizing a group of tweets, delimited by triple backticks, and each tweet is labeled by a number in a given format: number-[tweet]. Give me a comprehensive summary in a concise paragraph and as you generate each sentence, provide the identifying number of tweets on which that sentence is based:"""


class BackendError(Exception):
    """An LLM backend is unreachable or returned an unusable response."""

    def __init__(self, backend, url, message):
        self.backend = backend
        self.url = url
        super().__init__(f"{backend} backend at {url}: {message}")


_summarizer_client = None
_stance_client = None


def _get_summarizer_client():
    global _summarizer_client
    if _summarizer_client is None:
        _summarizer_client = openai.OpenAI(
            base_url=config.SUMMARIZER_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.SUMMARY_TIMEOUT_S,
            max_retries=1,
        )
    return _summarizer_client


def _get_stance_client():
    global _stance_client
    if _stance_client is None:
        _stance_client = AsyncOpenAI(
            base_url=config.STANCE_BASE_URL,
            api_key=config.LLM_API_KEY,
            timeout=config.STANCE_TIMEOUT_S,
            max_retries=1,
        )
    return _stance_client


def check_backends():
    """Ping both vLLM servers. Returns {"summarizer": bool, "stance": bool}; never raises."""
    status = {}
    for name, base_url in (
        ("summarizer", config.SUMMARIZER_BASE_URL),
        ("stance", config.STANCE_BASE_URL),
    ):
        try:
            resp = httpx.get(
                base_url.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {config.LLM_API_KEY}"},
                timeout=5.0,
            )
            status[name] = resp.status_code == 200
        except httpx.HTTPError:
            status[name] = False
    return status


def ai_summarize(tweets):
    llama3_gen_prompt = """system

{}user

{}assistant {}"""
    input_text = llama3_gen_prompt.format(
        AI_SUMMARY_PROMPT,
        tweets,
        ""
    )
    client = _get_summarizer_client()
    try:
        completion = client.chat.completions.create(
            model=config.SUMMARIZER_MODEL,
            messages=[{"role": "user", "content": input_text}],
            temperature=0,
        )
    except openai.OpenAIError as e:
        raise BackendError("summarizer", config.SUMMARIZER_BASE_URL, str(e)) from e
    return completion.choices[0].message.content


def parse_stance_response(text, start, end):
    """Map tweet ids in [start, end) to int stances from a model response.

    Extracts the first {...} block; any id that is missing, out of range of the
    JSON, or has a non-integer value gets -1 instead of raising.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    parsed = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    results = {}
    for j in range(start, end):
        raw = parsed.get(f"tweet-{j}", -1)
        try:
            results[j] = int(raw)
        except (TypeError, ValueError):
            results[j] = -1
    return results


async def stance_annotation(tweets, topic, stances, examples):
    formattedStances = []
    examplePrompt = "### Examples: \ninput:\n"
    for i in range(len(stances)):
        if stances[i] != "":
            currStanceNum = len(formattedStances)
            formattedStances.append(f"{currStanceNum}: {stances[i]}")

    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f"{exampleCount}-{key}\n"
        exampleCount += 1
    examplePrompt += "output:\n{\n"
    exampleCount = 0
    for key in examples.keys():
        examplePrompt += f'"tweet-{exampleCount}": {examples[key]},\n'
        exampleCount += 1
    examplePrompt += "}"
    if len(examples) == 0:
        examplePrompt = ""

    prompt = [
        {
            "role": "system",
            "content": f"You are a human annotator. You will be presented with a list of tweets (labeled with id numbers), delimited by triple backticks, concerning '{topic}'. Please make the following assessment without further commentary:",
        },
        {
            "role": "user",
            "content": f"""
    Determine whether each tweet discusses the topic of '{topic}'. If it does, indicate the stance of the Twitter user who posted the tweet as one of '{formattedStances}', otherwise label the stance as -1. Your response should be in JSON format as shown below, do not provide any other output:
    {{
        "tweet-<tweetID>" : "stance_number"
    }}

    The stance number must be between -1 and {len(formattedStances)-1}, no other text should be in the stance number field.

    {examplePrompt}

    ### Your Task:
    Tweets: ```{tweets}```
    """,
        },
    ]
    client = _get_stance_client()
    request = {"model": config.STANCE_MODEL, "messages": prompt}
    try:
        try:
            completion = await client.chat.completions.create(
                **request, response_format={"type": "json_object"}
            )
        except openai.BadRequestError:
            completion = await client.chat.completions.create(**request)
    except openai.OpenAIError as e:
        raise BackendError("stance", config.STANCE_BASE_URL, str(e)) from e

    return completion.choices[0].message.content
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_prompts.py tests/test_config.py -v`
Expected: 13 passed

- [ ] **Step 5: Commit**

```bash
git add Frontend/prompts.py tests/test_prompts.py
git commit -m "feat: config-driven LLM clients with BackendError, hardened stance parsing, health check"
```

---

### Task 3: Embeddings fast path and tensor fixes in `Frontend/tweet_browser.py`

**Files:**
- Modify: `Frontend/tweet_browser.py` (module top ~L26-28, `Session.__init__` ~L102, `getCentral` ~L505, `semanticSearch` ~L515-521)
- Test: `tests/test_embeddings.py`

**Interfaces:**
- Consumes: `config` (Task 1). `BackendError`/`parse_stance_response` arrive via the existing `from prompts import *` (Task 2).
- Produces (used by Task 6 notebook wiring):
  - `get_embedding_model() -> SentenceTransformer` — lazy singleton honoring `config.EMBEDDING_DEVICE`.
  - `load_or_compute_embeddings(data: pd.DataFrame, path: str) -> torch.Tensor` — loads `path` as a header-less float CSV if row count matches `len(data)`; otherwise encodes `data["Message"]` (normalized, batch 128) and saves to `path`. Always returns a CPU float tensor.

Note: `tweet_browser.py` imports torch/transformers at module import; the system interpreter has them, so these tests run without GPU but take a few seconds to import.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
import numpy as np
import pandas as pd
import pytest
import torch


def _df(n):
    return pd.DataFrame(
        {
            "Message": [f"tweet number {i}" for i in range(n)],
            "CreatedTime": pd.date_range("2020-01-01", periods=n).astype(str),
            "State": ["michigan"] * n,
        }
    )


class FakeModel:
    """Stands in for SentenceTransformer; returns deterministic embeddings."""

    def __init__(self, dim=4):
        self.dim = dim
        self.encode_calls = 0

    def encode(self, texts, **kwargs):
        self.encode_calls += 1
        arr = np.arange(len(texts) * self.dim, dtype=np.float32).reshape(len(texts), self.dim)
        return torch.from_numpy(arr)


def test_load_when_file_matches(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "emb.csv"
    np.savetxt(path, np.ones((3, 4), dtype=np.float32), delimiter=",")

    result = tb.load_or_compute_embeddings(_df(3), str(path))
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 4)
    assert fake.encode_calls == 0  # loaded, not recomputed


def test_recompute_and_save_on_row_mismatch(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "emb.csv"
    np.savetxt(path, np.ones((2, 4), dtype=np.float32), delimiter=",")  # wrong row count

    result = tb.load_or_compute_embeddings(_df(3), str(path))
    assert result.shape == (3, 4)
    assert fake.encode_calls == 1
    saved = np.loadtxt(path, delimiter=",")
    assert saved.shape == (3, 4)  # file was refreshed


def test_compute_when_file_missing(tmp_path, monkeypatch):
    import tweet_browser as tb

    fake = FakeModel()
    monkeypatch.setattr(tb, "get_embedding_model", lambda: fake)
    path = tmp_path / "does_not_exist.csv"

    result = tb.load_or_compute_embeddings(_df(2), str(path))
    assert result.shape == (2, 4)
    assert path.exists()


def test_get_central_works_with_tensor_embeddings(monkeypatch):
    import tweet_browser as tb

    data = _df(4)
    embeddings = torch.eye(4)  # 4 rows, trivially normalized
    s = tb.Session(data, False, embeddings=embeddings)
    result = s.getCentral()
    assert "centrality" in result.columns
    assert len(result) == 4


def test_semantic_search_with_cpu_tensor(monkeypatch):
    import tweet_browser as tb

    data = _df(4)
    embeddings = torch.eye(4)
    s = tb.Session(data, False, embeddings=embeddings)

    class QueryModel:
        def encode(self, query, **kwargs):
            return torch.tensor([1.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(tb, "get_embedding_model", lambda: QueryModel())
    s.semanticSearch("anything", 0.5)
    assert s.currentSet.size == 2  # top 50% of 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_embeddings.py -v`
Expected: FAIL with `AttributeError: module 'tweet_browser' has no attribute 'load_or_compute_embeddings'` (and `get_embedding_model`)

- [ ] **Step 3: Apply the edits to `Frontend/tweet_browser.py`**

Edit A — replace the module-level model with lazy accessors. Old:

```python
embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5", device="cuda" if torch.cuda.is_available() else "cpu"
)
```

New:

```python
import config

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        device = config.EMBEDDING_DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        _embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    return _embedding_model

def load_or_compute_embeddings(data, path):
    """Load precomputed embeddings if the file matches the dataset, else compute and save."""
    if path and os.path.isfile(path):
        arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == len(data):
            return torch.from_numpy(arr)
        print(f"Embeddings file {path} has {arr.shape[0]} rows but dataset has {len(data)}; recomputing.")
    embeddings = get_embedding_model().encode(
        data["Message"].astype(str).tolist(),
        convert_to_tensor=True,
        show_progress_bar=False,
        batch_size=128,
        normalize_embeddings=True,
    ).cpu()
    if path:
        np.savetxt(path, embeddings.numpy(), delimiter=",")
    return embeddings
```

Edit B — `Session.__init__` (~L103). Old: `embeddings = embedding_model.encode(` New: `embeddings = get_embedding_model().encode(`

Edit C — `semanticSearch` (~L515). Old: `query_embedding = embedding_model.encode(` New: `query_embedding = get_embedding_model().encode(`

Edit D — `semanticSearch` device fix (~L520-521). Old:

```python
        indices_tensor = torch.tensor(inputSet.indices)
        embeddingTensor = self.embeddings[indices_tensor].float()
```

New:

```python
        indices_tensor = torch.tensor(np.asarray(inputSet.indices))
        embeddingTensor = self.embeddings[indices_tensor].float().to(query_embedding.device)
```

Edit E — `getCentral` tensor fix (~L505; today it calls `.iloc` on a tensor and crashes). Old:

```python
        input = self.embeddings.iloc[inputSet.indices]
```

New:

```python
        input = self.embeddings[torch.tensor(np.asarray(inputSet.indices))].cpu().numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_embeddings.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add Frontend/tweet_browser.py tests/test_embeddings.py
git commit -m "feat: embeddings fast path (load-or-compute) and tensor fixes for getCentral/semanticSearch"
```

---

### Task 4: Per-batch degradation in `Session.stanceAnalysis`

**Files:**
- Modify: `Frontend/tweet_browser.py` (`Session.stanceAnalysis`, ~L530-554)
- Test: `tests/test_stance_analysis.py`

**Interfaces:**
- Consumes: `stance_annotation`, `parse_stance_response`, `BackendError` (all in the `tweet_browser` namespace via its existing `from prompts import *`).
- Produces: `Session.stanceAnalysis` with the same signature; new attribute `Session.lastStanceFailedBatches: int` set on every run (used by Task 6). Failure semantics: `BackendError` on the FIRST batch propagates (backend is down); on later batches the batch's tweets get stance `-1` and the run continues.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stance_analysis.py`:

```python
import asyncio

import pandas as pd
import pytest
import torch


def _session(n):
    import tweet_browser as tb

    data = pd.DataFrame(
        {
            "Message": [f"tweet {i}" for i in range(n)],
            "CreatedTime": pd.date_range("2020-01-01", periods=n).astype(str),
            "State": ["michigan"] * n,
        }
    )
    return tb.Session(data, False, embeddings=torch.zeros((n, 4)))


def _response_for(tweets_str, stance=1):
    """Build a valid JSON response covering exactly the ids present in the batch."""
    import re, json
    ids = [int(m) for m in re.findall(r"^(\d+)-\[", tweets_str, re.M)]
    return json.dumps({f"tweet-{i}": stance for i in ids})


def test_all_batches_succeed():
    import tweet_browser as tb

    s = _session(60)  # 2 batches of 50 + 10

    async def fake_annotation(tweets, topic, stances, examples):
        return _response_for(tweets)

    tb.stance_annotation = fake_annotation
    df = asyncio.run(s.stanceAnalysis("census", ["pro", "anti"], {}, updateAllData=True))
    assert list(df["stance"]) == [1] * 60
    assert s.lastStanceFailedBatches == 0
    assert list(s.allData["stance"])[:3] == [1, 1, 1]


def test_second_batch_fails_degrades_to_minus_one():
    import tweet_browser as tb
    from prompts import BackendError

    s = _session(60)
    calls = {"n": 0}

    async def fake_annotation(tweets, topic, stances, examples):
        calls["n"] += 1
        if calls["n"] == 2:
            raise BackendError("stance", "http://localhost:8001/v1", "boom")
        return _response_for(tweets)

    tb.stance_annotation = fake_annotation
    df = asyncio.run(s.stanceAnalysis("census", ["pro"], {}))
    assert list(df["stance"])[:50] == [1] * 50
    assert list(df["stance"])[50:] == [-1] * 10
    assert s.lastStanceFailedBatches == 1


def test_first_batch_failure_raises():
    import tweet_browser as tb
    from prompts import BackendError

    s = _session(10)

    async def fake_annotation(tweets, topic, stances, examples):
        raise BackendError("stance", "http://localhost:8001/v1", "down")

    tb.stance_annotation = fake_annotation
    with pytest.raises(BackendError):
        asyncio.run(s.stanceAnalysis("census", ["pro"], {}))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stance_analysis.py -v`
Expected: FAIL (current code does `json.loads` directly and has no `lastStanceFailedBatches`; the degradation test raises instead of degrading)

- [ ] **Step 3: Replace the `stanceAnalysis` method**

In `Frontend/tweet_browser.py`, replace the whole `async def stanceAnalysis(...)` method body with:

```python
    async def stanceAnalysis(self, topic, stances, examples, updateAllData = False, inputSet = None):
        if inputSet == None or type(inputSet) != Subset:
            inputSet = self.currentSet
        df = self.allData.iloc[inputSet.indices]
        results = []
        failedBatches = 0
        i = 0
        while i < len(inputSet.indices):
            start = i
            tweets = ""
            while i < len(inputSet.indices) and i - start < 50:
                tweet = self.allData.iloc[inputSet.indices[i]]['Message']
                tweets += f"{i}-[{tweet}]\n"
                i += 1
            try:
                batchResult = await stance_annotation(tweets, topic, stances, examples)
                batchStances = parse_stance_response(batchResult, start, i)
            except BackendError:
                if start == 0:
                    raise  # backend is down; nothing to salvage
                failedBatches += 1
                batchStances = {j: -1 for j in range(start, i)}
            for j in range(start, i):
                results.append(batchStances[j])
        self.lastStanceFailedBatches = failedBatches
        df["stance"] = results
        if updateAllData:
            self.allData["stance"] = None
            self.allData.update(df)
        return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stance_analysis.py tests/test_embeddings.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add Frontend/tweet_browser.py tests/test_stance_analysis.py
git commit -m "feat: stance analysis degrades per batch instead of dying on one bad response"
```

---

### Task 5: Demographics provider + results page

**Files:**
- Create: `Frontend/demographics.py`
- Modify: `Frontend/ui_modules.py` (replace `InferDemographicsModule`, ~L195-215)
- Test: `tests/test_demographics.py`

**Interfaces:**
- Produces (used by Task 6):
  - `demographics.DEMOGRAPHIC_COLUMNS = ["sex", "age", "education", "income", "party_identification", "ideology", "urbanicity", "metro"]` (exact column names in `allCensus_sample_with_demographics.csv`).
  - `class DemographicsProvider` with `infer(df) -> pd.DataFrame`; implementation `PrecomputedProvider` returns the demographics columns or raises `ValueError` naming missing columns.
  - `demographics.distributions(attributes: pd.DataFrame) -> dict[str, dict[str, int]]` — value counts per column, NaN mapped to "unknown".
  - `InferDemographicsModule` gains `backButton` (ipywidgets Button), `resultsPage` (VBox), and `showResults(dists: dict, sampleSize: int) -> None` which renders tables into the page.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_demographics.py`:

```python
import numpy as np
import pandas as pd
import pytest


def test_precomputed_provider_returns_columns():
    import demographics

    df = pd.DataFrame({c: ["a", "b"] for c in demographics.DEMOGRAPHIC_COLUMNS})
    df["Message"] = ["x", "y"]
    result = demographics.PrecomputedProvider().infer(df)
    assert list(result.columns) == demographics.DEMOGRAPHIC_COLUMNS
    assert len(result) == 2


def test_precomputed_provider_raises_on_missing_columns():
    import demographics

    df = pd.DataFrame({"Message": ["x"], "sex": ["f"]})
    with pytest.raises(ValueError) as exc:
        demographics.PrecomputedProvider().infer(df)
    assert "age" in str(exc.value)


def test_distributions_counts_and_unknowns():
    import demographics

    attrs = pd.DataFrame({"sex": ["f", "f", "m", np.nan]})
    dists = demographics.distributions(attrs)
    assert dists["sex"]["f"] == 2
    assert dists["sex"]["m"] == 1
    assert dists["sex"]["unknown"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_demographics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demographics'`

- [ ] **Step 3: Create `Frontend/demographics.py`**

```python
"""Demographics inference providers.

v1 reads the demographics columns precomputed in the dataset CSV. A future
model-backed provider implements the same interface and swaps in without UI
changes (see docs/superpowers/specs/2026-07-03-backend-service-wiring-design.md).
"""
import pandas as pd

DEMOGRAPHIC_COLUMNS = [
    "sex",
    "age",
    "education",
    "income",
    "party_identification",
    "ideology",
    "urbanicity",
    "metro",
]


class DemographicsProvider:
    """Interface: per-tweet demographic attributes for a dataframe of posts."""

    def infer(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class PrecomputedProvider(DemographicsProvider):
    """Reads demographics columns already present in the dataset CSV."""

    def infer(self, df):
        missing = [c for c in DEMOGRAPHIC_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                "Dataset lacks precomputed demographics columns: " + ", ".join(missing)
            )
        return df[DEMOGRAPHIC_COLUMNS].copy()


def distributions(attributes: pd.DataFrame) -> dict:
    """Value counts per attribute column, for the results display."""
    return {
        c: attributes[c].fillna("unknown").astype(str).value_counts().to_dict()
        for c in attributes.columns
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_demographics.py -v`
Expected: 3 passed

- [ ] **Step 5: Replace `InferDemographicsModule` in `Frontend/ui_modules.py`**

Replace the entire `InferDemographicsModule` class (from `class InferDemographicsModule():` to the end of the file) with:

```python
class InferDemographicsModule():
    def __init__(self):
        self.inferDemographics = InferDemographics()
        self.page = self.inferDemographics

        self.backButton = widgets.Button(description="< New Demographic Analysis", layout=widgets.Layout(margin="0 auto 0 0", flex="0 0")).add_class("clear-button")
        self.resultsTitle = widgets.HTML("").add_class("display-count")
        self.resultsHTML = widgets.HTML("")
        self.resultsPage = widgets.VBox([self.backButton, self.resultsTitle, self.resultsHTML])

    def showResults(self, dists, sampleSize):
        self.resultsTitle.value = f"Inferred demographics for {sampleSize:,d} posts"
        parts = []
        for attr, counts in dists.items():
            total = sum(counts.values()) or 1
            rows = "".join(
                f"<tr><td>{value}</td><td>{count}</td><td>{count / total:.0%}</td></tr>"
                for value, count in sorted(counts.items(), key=lambda kv: -kv[1])
            )
            parts.append(
                f"<h4>{attr.replace('_', ' ').title()}</h4>"
                f"<table><tr><th>Value</th><th>Posts</th><th>Share</th></tr>{rows}</table>"
            )
        self.resultsHTML.value = "<div class='demographics-results'>" + "".join(parts) + "</div>"
```

(The removed `setupEventHandlers`/`onInferenceTriggered` only printed to stdout; the notebook attaches its own observer.)

- [ ] **Step 6: Verify ui_modules still imports**

Run: `cd Frontend && python3 -c "import ui_modules; m = ui_modules.InferDemographicsModule(); m.showResults({'sex': {'f': 2, 'm': 1}}, 3); assert 'Inferred demographics for 3' in m.resultsTitle.value; print('OK')" && cd ..`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add Frontend/demographics.py Frontend/ui_modules.py tests/test_demographics.py
git commit -m "feat: demographics provider interface with precomputed-CSV implementation and results page"
```

---

### Task 6: Notebook wiring (config, embeddings, health banner, error handling, demographics)

**Files:**
- Modify: `Frontend/tweet_browser.ipynb` (cell 0 only)

**Interfaces:**
- Consumes: `config` (Task 1), `prompts.check_backends`/`BackendError` (Task 2), `tb.load_or_compute_embeddings` (Task 3), `s.lastStanceFailedBatches` (Task 4), `demographics.PrecomputedProvider`/`distributions` and `demographicsModule.showResults`/`backButton`/`resultsPage` (Task 5).
- Produces: the runnable app. New `Browser.updateBackendBanner()` method and `Browser.backendStatus` dict.

Every edit below uses this pattern (fill in OLD/NEW verbatim from the sub-steps; each OLD string must occur exactly once):

```bash
python3 - <<'PYEOF'
import json
p = 'Frontend/tweet_browser.ipynb'
nb = json.load(open(p))
src = ''.join(nb['cells'][0]['source'])
old = """<OLD>"""
new = """<NEW>"""
assert src.count(old) == 1, f"expected 1 occurrence of edit anchor, got {src.count(old)}"
src = src.replace(old, new)
nb['cells'][0]['source'] = src.splitlines(keepends=True)
json.dump(nb, open(p, 'w'), indent=1)
print('edited ok')
PYEOF
```

If an assert fires (0 occurrences), the OLD block's whitespace differs from the file (the notebook has trailing spaces on some lines). Print the exact bytes around the anchor and copy them verbatim into OLD:

```bash
python3 -c "
import json
src = ''.join(json.load(open('Frontend/tweet_browser.ipynb'))['cells'][0]['source'])
i = src.find('ANCHOR SUBSTRING HERE')
print(repr(src[i-100:i+400]))
"
```

- [ ] **Step 1: Add imports and provider instance**

OLD:
```python
from custom_widgets import *
from ui_modules import *
```
NEW:
```python
from custom_widgets import *
from ui_modules import *
import config as app_config
import prompts
import demographics
```

Then a second edit. OLD:
```python
demographicsModule = InferDemographicsModule()
```
NEW:
```python
demographicsModule = InferDemographicsModule()
demographicsProvider = demographics.PrecomputedProvider()
```

- [ ] **Step 2: Embeddings fast path + config dataset path**

OLD:
```python
def autoStartSession(fileName):
    data = tb.parse_data(fileName)
    # Try to load embeddings, but continue without them if they don't exist
    try:
        embeddings = None
        s = tb.Session(data, False, embeddings=embeddings)
    except FileNotFoundError:
        print("Embeddings file not found, continuing without embeddings...")
        s = tb.Session(data, False)
    dummyEl.fileName = fileName
    global browser
    browser = Browser(s, out)
```
NEW:
```python
def autoStartSession(fileName):
    data = tb.parse_data(fileName)
    embeddings = tb.load_or_compute_embeddings(data, app_config.EMBEDDINGS_PATH)
    s = tb.Session(data, False, embeddings=embeddings)
    dummyEl.fileName = fileName
    global browser
    browser = Browser(s, out)
```

Then. OLD:
```python
autoStartSession("allCensus_sample_with_demographics.csv")
```
NEW:
```python
autoStartSession(app_config.DATASET_PATH)
```

- [ ] **Step 3: Backend banner in the layout**

OLD:
```python
        self.topBar = widgets.HBox([widgets.HTML("Social Media Browser").add_class("title"), self.datasetDisplay]).add_class("top-bar")
        self.mainPage = widgets.VBox([self.topBar, self.advancedBar, widgets.HBox([self.filterBox, self.tabs])])
```
NEW:
```python
        self.topBar = widgets.HBox([widgets.HTML("Social Media Browser").add_class("title"), self.datasetDisplay]).add_class("top-bar")
        self.backendBanner = widgets.HTML("")
        self.updateBackendBanner()
        self.mainPage = widgets.VBox([self.topBar, self.backendBanner, self.advancedBar, widgets.HBox([self.filterBox, self.tabs])])
```

- [ ] **Step 4: Add `updateBackendBanner` method**

OLD:
```python
    def alertHandler(self, change):
```
NEW:
```python
    def updateBackendBanner(self):
        self.backendStatus = prompts.check_backends()
        down = [name for name, ok in self.backendStatus.items() if not ok]
        if down:
            self.backendBanner.value = (
                "<div style='background:#FDECEA;color:#611A15;padding:8px 16px;border-radius:4px'>"
                + "AI backend unavailable: " + ", ".join(down)
                + ". AI Summary / Stance Analysis are disabled; search, filters and samples still work.</div>"
            )
        else:
            self.backendBanner.value = ""

    def alertHandler(self, change):
```

- [ ] **Step 5: Guard + error handling in `generateSummary`**

OLD:
```python
    def generateSummary(self, change = None):
        if AiSummary.aiSummary.rerender == 0:
            return
```
NEW:
```python
    def generateSummary(self, change = None):
        if AiSummary.aiSummary.rerender == 0:
            return
        if not self.backendStatus.get("summarizer", False):
            self.updateBackendBanner()
        if not self.backendStatus.get("summarizer", False):
            with out:
                display(Javascript('alert("Summarization backend is unreachable. See the banner at the top of the page.");'))
            return
```

Then. OLD:
```python
        self.tabs.children = [*self.tabs.children[:2], AiSummary.loadingPage, *self.tabs.children[3:]]
        summary = self.s.summarize()
        self.tabs.children = [*self.tabs.children[:2], self.summaryTab, *self.tabs.children[3:]]
```
NEW:
```python
        self.tabs.children = [*self.tabs.children[:2], AiSummary.loadingPage, *self.tabs.children[3:]]
        try:
            summary = self.s.summarize()
        except tb.BackendError as e:
            self.tabs.children = [*self.tabs.children[:2], self.summaryTab, *self.tabs.children[3:]]
            self.updateBackendBanner()
            with out:
                display(Javascript('alert(' + json.dumps("AI summary failed: " + str(e)) + ');'))
            return
        self.tabs.children = [*self.tabs.children[:2], self.summaryTab, *self.tabs.children[3:]]
```

- [ ] **Step 6: Guard + error handling in `startStanceAnalysis`**

OLD:
```python
    async def startStanceAnalysis(self, keepCurrResults = False):
        sampleSize = stanceModule.sampleSelector.value
```
NEW:
```python
    async def startStanceAnalysis(self, keepCurrResults = False):
        if not self.backendStatus.get("stance", False):
            self.updateBackendBanner()
        if not self.backendStatus.get("stance", False):
            self.stanceAnalysis.pageNumber = 0
            with out:
                display(Javascript('alert("Stance backend is unreachable. See the banner at the top of the page.");'))
            return
        sampleSize = stanceModule.sampleSelector.value
```

Then surface partial failures. OLD:
```python
            await self.s.stanceAnalysis(self.stanceAnalysis.topic, self.stanceAnalysis.stances, stanceExamples, updateAllData=True)
            self.getStanceTweets()
```
NEW:
```python
            await self.s.stanceAnalysis(self.stanceAnalysis.topic, self.stanceAnalysis.stances, stanceExamples, updateAllData=True)
            if self.s.lastStanceFailedBatches:
                with out:
                    display(Javascript('alert(' + json.dumps(f"{self.s.lastStanceFailedBatches} batch(es) failed and their posts were marked Irrelevant (-1).") + ');'))
            self.getStanceTweets()
```

Then a typed except branch before the bare one. OLD:
```python
        except:
            self.stanceAnalysisPage = self.stanceAnalysis
            self.tabs.children = [*self.tabs.children[:3], self.stanceAnalysisPage, *self.tabs.children[4:]]
            self.stanceAnalysis.pageNumber = 0
            dummyEl.activeStanceAnalysis = 0
            with out:
                display(Javascript('alert("Error genearting stance analysis");'))
```
NEW:
```python
        except tb.BackendError as e:
            self.stanceAnalysisPage = self.stanceAnalysis
            self.tabs.children = [*self.tabs.children[:3], self.stanceAnalysisPage, *self.tabs.children[4:]]
            self.stanceAnalysis.pageNumber = 0
            dummyEl.activeStanceAnalysis = 0
            self.updateBackendBanner()
            with out:
                display(Javascript('alert(' + json.dumps("Stance analysis failed: " + str(e)) + ');'))
        except:
            self.stanceAnalysisPage = self.stanceAnalysis
            self.tabs.children = [*self.tabs.children[:3], self.stanceAnalysisPage, *self.tabs.children[4:]]
            self.stanceAnalysis.pageNumber = 0
            dummyEl.activeStanceAnalysis = 0
            with out:
                display(Javascript('alert("Error genearting stance analysis");'))
```

- [ ] **Step 7: Real demographics instead of simulation**

OLD:
```python
    def onDemographicsInference(self, change):
        """Handle demographics inference trigger"""
        if change['new'] == 1:  # When pageNumber becomes 1, start inference
            print("Demographics inference triggered")
            # Here you would implement the actual demographics inference logic
            # For now, simulate the process
            self.simulateDemographicsInference()
    
    def simulateDemographicsInference(self):
        """Simulate the demographics inference process"""
        import threading
        import time
        
        def inference_worker():
            try:
                # Switch to loading screen if needed
                time.sleep(2)  # Simulate processing
                
                # Switch to results page in the tab
                self.tabs.children = [*self.tabs.children[:4], demographicsModule.resultsPage, *self.tabs.children[5:]]
                
            except Exception as e:
                print(f"Error in demographics inference: {e}")
        
        # Start the worker thread
        thread = threading.Thread(target=inference_worker)
        thread.daemon = True
        thread.start()
    
    def regenerateDemographics(self, change):
        """Handle regenerate demographics button"""
        # Reset to initial page and trigger new inference
        self.tabs.children = [*self.tabs.children[:4], demographicsModule.page, *self.tabs.children[5:]]
        demographicsModule.inferDemographics.pageNumber = 0
        demographicsModule.inferDemographicsWidget.showResults = False
```
NEW:
```python
    def onDemographicsInference(self, change):
        """Handle demographics inference trigger"""
        if change['new'] == 1:
            try:
                sample = self.s.getCurrentSubset()
                attrs = demographicsProvider.infer(sample)
                demographicsModule.showResults(demographics.distributions(attrs), len(attrs))
                self.tabs.children = [*self.tabs.children[:4], demographicsModule.resultsPage, *self.tabs.children[5:]]
            except ValueError as e:
                demographicsModule.inferDemographics.pageNumber = 0
                with out:
                    display(Javascript('alert(' + json.dumps("Demographics unavailable: " + str(e)) + ');'))

    def regenerateDemographics(self, change):
        """Handle regenerate demographics button"""
        self.tabs.children = [*self.tabs.children[:4], demographicsModule.page, *self.tabs.children[5:]]
        demographicsModule.inferDemographics.pageNumber = 0
```

Then wire the back button. OLD:
```python
        # demographicsModule.regenerateButton.on_click(self.regenerateDemographics)
```
NEW:
```python
        demographicsModule.backButton.on_click(self.regenerateDemographics)
```

- [ ] **Step 8: Verify cell 0 still compiles**

Run:
```bash
python3 - <<'PYEOF'
import json
src = ''.join(json.load(open('Frontend/tweet_browser.ipynb'))['cells'][0]['source'])
compile(src, 'cell0', 'exec')
assert 'simulateDemographicsInference' not in src
assert 'updateBackendBanner' in src
print('cell 0 compiles, wiring present')
PYEOF
```
Expected: `cell 0 compiles, wiring present`

- [ ] **Step 9: Run the full test suite (regression)**

Run: `python3 -m pytest tests/ --ignore=tests/tweet_tester.py --ignore=tests/jupyter_notebook_config.py -v`
Expected: all passed

- [ ] **Step 10: Commit**

```bash
git add Frontend/tweet_browser.ipynb
git commit -m "feat: wire notebook to config, embeddings fast path, health banner, typed errors, real demographics provider"
```

---

### Task 7: Deployment (compose, scripts, smoke test, README)

**Files:**
- Create: `deploy/docker-compose.yml`
- Create: `deploy/scripts/serve_app.sh`
- Create: `deploy/scripts/serve_summarizer.sh`
- Create: `deploy/scripts/serve_stance.sh`
- Create: `deploy/scripts/smoke_test.py`
- Create: `deploy/README.md`

**Interfaces:**
- Consumes: `prompts.check_backends`, `prompts.ai_summarize`, `prompts.stance_annotation`, `prompts.parse_stance_response` (Task 2).
- Produces: `docker compose -f deploy/docker-compose.yml up -d` brings up both vLLM servers on GPU 1; `deploy/scripts/serve_app.sh` starts Voila on the host; `python3 deploy/scripts/smoke_test.py` exits 0 iff both backends answer real requests.

- [ ] **Step 1: Write `deploy/docker-compose.yml`**

```yaml
# Tweet Browser model backends. Voila runs on the host: deploy/scripts/serve_app.sh
# Both services are pinned to GPU 1 per the design spec; memory budgets 0.25 + 0.68.
services:
  vllm-summarizer:
    image: vllm/vllm-openai:latest
    container_name: tweet-browser-summarizer
    ports:
      - "8000:8000"
    volumes:
      - /home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm:/models/Lllama3TS_unsloth_vllm:ro
    command: >
      --model /models/Lllama3TS_unsloth_vllm
      --served-model-name Lllama3TS_unsloth_vllm
      --gpu-memory-utilization 0.25
      --max-model-len 8192
      --api-key ${LLM_API_KEY:-token-census}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 60
      start_period: 300s
    restart: unless-stopped

  vllm-stance:
    image: vllm/vllm-openai:latest
    container_name: tweet-browser-stance
    ports:
      - "8001:8000"
    environment:
      - HF_TOKEN=${HF_TOKEN:-}
    volumes:
      - ${HF_CACHE:-/home/maolee/.cache/huggingface}:/root/.cache/huggingface
    command: >
      --model ${STANCE_MODEL:-google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant}
      --gpu-memory-utilization 0.68
      --max-model-len 16384
      --api-key ${LLM_API_KEY:-token-census}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 60
      start_period: 600s
    restart: unless-stopped
```

- [ ] **Step 2: Write the serve scripts**

`deploy/scripts/serve_app.sh`:

```bash
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
```

`deploy/scripts/serve_summarizer.sh` (bare-metal alternative; see README note about the host vllm install):

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
exec vllm serve /home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm \
  --served-model-name Lllama3TS_unsloth_vllm \
  --port 8000 \
  --gpu-memory-utilization 0.25 \
  --max-model-len 8192 \
  --api-key "${LLM_API_KEY:-token-census}"
```

`deploy/scripts/serve_stance.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
exec vllm serve "${STANCE_MODEL:-google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant}" \
  --port 8001 \
  --gpu-memory-utilization 0.68 \
  --max-model-len 16384 \
  --api-key "${LLM_API_KEY:-token-census}"
```

Run: `chmod +x deploy/scripts/serve_app.sh deploy/scripts/serve_summarizer.sh deploy/scripts/serve_stance.sh`

- [ ] **Step 3: Write `deploy/scripts/smoke_test.py`**

```python
#!/usr/bin/env python3
"""Post-deploy smoke test: exercises both vLLM backends end to end.

Run from anywhere: python3 deploy/scripts/smoke_test.py
Exit code 0 = service is usable; 1 = something is down or broken.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Frontend")))

import prompts  # noqa: E402

SUMMARY_TWEETS = (
    "0-[The census helps allocate federal funding to communities.] "
    "1-[I filled out my census form online today, it took five minutes.] "
    "2-[Census data should be kept private and secure.]"
)
STANCE_TWEETS = (
    "0-[The census is great and everyone should participate.]\n"
    "1-[The census is a waste of taxpayer money.]\n"
    "2-[Nice weather in Michigan today.]\n"
    "3-[Just reminded my family to fill out the census.]\n"
    "4-[I do not trust the census with my data.]\n"
)


def main():
    status = prompts.check_backends()
    print("backend status:", status)
    down = [name for name, ok in status.items() if not ok]
    if down:
        print("FAIL: backend(s) unreachable:", ", ".join(down))
        return 1

    summary = prompts.ai_summarize(SUMMARY_TWEETS)
    print("summary ok:", summary[:120].replace("\n", " "))

    raw = asyncio.run(
        prompts.stance_annotation(STANCE_TWEETS, "the US census", ["support", "oppose"], {})
    )
    stances = prompts.parse_stance_response(raw, 0, 5)
    print("stances:", stances)
    if all(v == -1 for v in stances.values()):
        print("FAIL: stance response produced no usable labels; raw response:")
        print(raw[:500])
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Run: `chmod +x deploy/scripts/smoke_test.py`

- [ ] **Step 4: Write `deploy/README.md`**

```markdown
# Deploying Tweet Browser

Three processes on the GPU server (all model traffic stays on `localhost`):

| Service | Port | GPU | What |
| --- | --- | --- | --- |
| vllm-summarizer | 8000 | 1 (25% mem) | fine-tuned Llama-3-8B, name `Lllama3TS_unsloth_vllm` |
| vllm-stance | 8001 | 1 (68% mem) | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` |
| Voila app | 8866 | 1 (embeddings) | `Frontend/tweet_browser.ipynb` |

## Quickstart

```bash
# 0. One-time: config. The Gemma stance model is license-gated on HuggingFace,
#    so accept its license with your HF account and set HF_TOKEN.
cp deploy/.env.example deploy/.env   # then edit; at minimum set HF_TOKEN

# 1. Model servers (first start downloads Gemma, ~50 GB; watch progress with docker logs)
docker compose --env-file deploy/.env -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml ps   # wait for both to be "healthy"

# 2. App
./deploy/scripts/serve_app.sh    # http://<server>:8866

# 3. Verify
python3 deploy/scripts/smoke_test.py   # prints PASS
```

## Notes

- **First app start after a dataset change** recomputes embeddings and rewrites
  `Frontend/allCensus_sample_embeddings.csv`; later starts load it in seconds.
- **GPU budget (GPU 1, H100 80 GB):** 0.25 + 0.68 = 0.93 of the card. If vLLM
  OOMs at startup, move one service to a free GPU: change `device_ids: ["1"]`
  in `deploy/docker-compose.yml`.
- **Bare-metal alternative:** `deploy/scripts/serve_summarizer.sh` /
  `serve_stance.sh` run vLLM without Docker. As of 2026-07-03 the host `vllm`
  install is broken (torch ABI mismatch: `undefined symbol ... ConstantString`),
  so the Docker path is the supported one; reinstall vllm against the system
  torch before using the scripts.
- **App is down / banner shows a backend as unavailable:** check
  `docker logs tweet-browser-summarizer` / `tweet-browser-stance`, then
  `python3 deploy/scripts/smoke_test.py`.
- **Pinning:** `vllm/vllm-openai:latest` is used because Gemma 4 support is
  recent. After a successful deploy, pin the image digest in the compose file.
```

- [ ] **Step 5: Validate compose file syntax**

Run: `docker compose -f deploy/docker-compose.yml config --quiet && echo COMPOSE_OK`
Expected: `COMPOSE_OK` (warnings about unset `HF_TOKEN`/`HF_CACHE` are fine)

- [ ] **Step 6: Full local verification (no GPU services needed)**

Run: `python3 -m pytest tests/test_config.py tests/test_prompts.py tests/test_embeddings.py tests/test_stance_analysis.py tests/test_demographics.py -v`
Expected: all passed

Run: `python3 deploy/scripts/smoke_test.py; echo "exit=$?"`
Expected (backends not running yet): prints `backend status: {'summarizer': False, 'stance': False}`, `FAIL: backend(s) unreachable...`, `exit=1` — this confirms the failure path works.

- [ ] **Step 7: Commit**

```bash
git add deploy/
git commit -m "feat: one-command deploy (vLLM compose on GPU 1, host Voila script, smoke test)"
```

---

### Task 8: On-server verification (requires GPU, run on the lab server)

**Files:** none (verification only)

- [ ] **Step 1: Bring up the model servers**

Run: `cp deploy/.env.example deploy/.env` (edit: set `HF_TOKEN`), then
`docker compose --env-file deploy/.env -f deploy/docker-compose.yml up -d`
Then poll: `docker compose -f deploy/docker-compose.yml ps`
Expected: both services eventually `healthy` (stance first start downloads ~50 GB)

- [ ] **Step 2: Smoke test**

Run: `python3 deploy/scripts/smoke_test.py`
Expected: `backend status: {'summarizer': True, 'stance': True}`, a real summary line, non-trivial stances, `PASS`

- [ ] **Step 3: Start the app and walk through the UI**

Run: `./deploy/scripts/serve_app.sh` and open `http://<server>:8866`.
Checklist:
- No red banner at the top (backends up)
- Random Posts, search, filters work
- Typical Posts tab renders (getCentral fix)
- AI Summary tab produces a summary with clickable contributing posts
- Stance Analysis on a 50-post sample completes and colors posts
- Infer Demographics shows distribution tables for the current subset
- Stop the compose services, click Generate AI Summary: banner appears, alert explains, app stays alive

- [ ] **Step 4: Regression: legacy tester**

Run: `cd tests && python3 tweet_tester.py && cd ..`
Expected: same behavior as before this project (it runs `test3`); no import errors from the refactor

- [ ] **Step 5: Final commit if any fixups were needed**

```bash
git add -A -- Frontend deploy tests docs
git commit -m "fix: post-deploy verification fixups"
```
