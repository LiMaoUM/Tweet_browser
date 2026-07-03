# Backend Service Wiring Design

Date: 2026-07-03
Status: Approved approach (Approach A: config + orchestration layer around the existing architecture)

## Problem

The Tweet Browser UI and the `Session` engine (search, filters, sampling, semantic search, history tree) are complete, but the app is not a runnable service:

- Summarization (`Frontend/prompts.py`) calls a hardcoded OpenAI-compatible server at `localhost:8000` expecting a model named `Lllama3TS_unsloth_vllm`. No launch definition exists in the repo.
- Stance detection calls a second hardcoded server at `localhost:8001`.
- Demographics inference is a simulation stub reading nothing real.
- LLM failures (connection refused, malformed JSON) kill async tasks silently; the loading spinner never resolves.
- `Session.__init__` re-embeds the entire dataset on every kernel start even though a precomputed embeddings CSV exists in the repo.

## Decisions (agreed in brainstorming)

| Question | Decision |
| --- | --- |
| LLM serving | Self-hosted vLLM on the lab GPU server |
| Summarizer model | Merged fine-tuned Llama-3-8B at `/home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm` (verified on disk, ~15 GB fp16) |
| Stance model | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant` (26B MoE, ~4B active, bf16 ~50 GB; vLLM supports it natively) |
| GPU placement | Both vLLM services on GPU 1 (H100 80 GB). Summarizer `--gpu-memory-utilization 0.25`, stance `0.68`. Fallback: move one service to another GPU via compose `device_ids`. |
| Demographics | Keep precomputed CSV columns behind a provider interface; real model is a later swap |
| Audience | Lab members on the intranet; Voila kernel per visitor; no auth work in this phase |

## Architecture

One GPU server runs three processes, brought up together:

```
deploy/docker-compose.yml  (or deploy/scripts/*.sh without Docker)
├── vllm-summarizer  :8000  GPU 1, served-model-name Lllama3TS_unsloth_vllm
├── vllm-stance      :8001  GPU 1, google/gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant
└── voila app        :8866  Frontend/tweet_browser.ipynb
```

The app keeps its in-process shape: Voila kernel -> `Session` -> OpenAI-compatible clients. No REST extraction in this phase.

## Components

1. **`Frontend/config.py`** (new). Env-var driven with defaults matching current behavior:
   - `SUMMARIZER_BASE_URL` (default `http://localhost:8000/v1`), `SUMMARIZER_MODEL` (default `Lllama3TS_unsloth_vllm`)
   - `STANCE_BASE_URL` (default `http://localhost:8001/v1`), `STANCE_MODEL` (default the Gemma 4 ID)
   - `LLM_API_KEY` (default `token-census`), `SUMMARY_TIMEOUT_S` (120), `STANCE_TIMEOUT_S` (180)
   - `DATASET_PATH`, `EMBEDDINGS_PATH`, `EMBEDDING_DEVICE`
   - `.env.example` documents every variable.
2. **`Frontend/prompts.py`** (refactor):
   - Clients constructed lazily from config (no hardcoded URLs, no client construction at import time).
   - Timeouts and one retry on connection errors and JSON parse failures.
   - `check_backends()` pings both `/v1/models` endpoints, returns per-backend status.
   - Stance request adds vLLM guided JSON (`response_format`/structured output) since Gemma 4 supports it; the hardened parser remains the fallback if guided decoding is unavailable.
   - Hardened stance parsing: extract first `{...}` block, validate expected `tweet-<id>` keys, map missing keys to stance -1 instead of raising.
   - Typed `BackendError` raised after retries are exhausted.
3. **Embeddings fast path** (notebook `autoStartSession` / `createSession`): load `EMBEDDINGS_PATH` when the file exists and row count matches the dataset; otherwise compute once and save it. The embedding model (`BAAI/bge-base-en-v1.5`, ~0.4 GB) loads on `EMBEDDING_DEVICE`; deployment sets the Voila service to GPU 1 as well so everything shares one card.
4. **`Frontend/demographics.py`** (new): `DemographicsProvider` interface with `infer(df) -> df`; single implementation `PrecomputedProvider` reading the demographics columns already present in `allCensus_sample_with_demographics.csv`. Replaces `simulateDemographicsInference()` in the notebook. A model-backed provider can be added later without UI changes.
5. **`deploy/`** (new):
   - `docker-compose.yml`: two `vllm/vllm-openai` services pinned to GPU 1 (`device_ids: ['1']`), weights volume-mounted, `--served-model-name Lllama3TS_unsloth_vllm` for the summarizer, healthchecks on `/v1/models`; Voila app service depends on both being healthy.
   - `scripts/serve_summarizer.sh`, `scripts/serve_stance.sh`, `scripts/serve_app.sh`, `scripts/smoke_test.py` for running without Docker (`CUDA_VISIBLE_DEVICES=1`).
   - `deploy/README.md`: one-command bring-up, port map, GPU budget table, troubleshooting.

## Data flow

- **Startup**: compose up -> vLLM servers load weights -> Voila starts -> config read -> dataset + embeddings load (fast path) -> `check_backends()` -> UI ready. If a backend is down, a visible banner names it and the Generate buttons are disabled; search/filter/sample/semantic search remain fully usable.
- **Summarize**: button -> `Session.summarize()` (<=100 tweets) -> vLLM :8000 -> `parseSummary()` -> summary widgets.
- **Stance**: form -> `Session.stanceAnalysis()` batches of 50 -> vLLM :8001 (guided JSON) -> `stance` column -> results tab.
- **Demographics**: tab -> `PrecomputedProvider` -> existing results UI.
- **Unchanged and local**: keyword/advanced/regex search, filters, sampling, semantic search, FastLexRank centrality, word cloud, time series.

## Error handling

- Every LLM call: timeout (`SUMMARY_TIMEOUT_S` / `STANCE_TIMEOUT_S` per batch) + one retry, then `BackendError`.
- `Browser` catches `BackendError`, hides the loading screen, and shows the existing alert dialog with a plain message naming the backend and URL.
- Stance batches degrade per batch: a failed batch after retry marks its tweets stance -1 and surfaces a warning count, rather than losing the whole run.
- Startup health check result drives a UI banner + disabled Generate buttons (no silent spinner).

## GPU budget (GPU 1, H100 80 GB)

| Consumer | Weights | vLLM memory budget |
| --- | --- | --- |
| Summarizer (Llama-3-8B fp16) | ~15 GB | `--gpu-memory-utilization 0.25` (~20 GB) |
| Stance (Gemma 4 26B bf16) | ~50 GB | `--gpu-memory-utilization 0.68` (~54 GB) |
| Embedding model (bge-base) | ~0.4 GB | shares remainder |

Contexts are short (<=100 tweets per summary, 50 per stance batch), so small KV budgets suffice. If OOM occurs, move one service to another GPU by editing `device_ids` in compose.

## Testing

- Unit tests for `prompts.py` parse/retry/`check_backends` logic against a monkeypatched client (no GPU required).
- `deploy/scripts/smoke_test.py`: pings both endpoints, runs one real summary and one 5-tweet stance batch. The post-deploy "is it actually up" check.
- Existing `tests/tweet_tester.py` must pass unchanged.
- Manual: `docker compose up` on the server, walk through every tab.

## Out of scope

- The file-upload UI limitation noted in the README.
- Multi-user resource quotas, auth, public exposure.
- A real demographics inference model (interface is ready for it).
- REST/FastAPI extraction of the Session engine.
