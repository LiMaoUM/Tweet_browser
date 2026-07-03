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
