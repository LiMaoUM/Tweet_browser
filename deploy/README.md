# Deploying Tweet Browser

Three processes on the GPU server (all model traffic stays on `localhost`):

| Service | Port | GPU | What |
| --- | --- | --- | --- |
| vllm-summarizer | 8000 | 1 (25% mem) | fine-tuned Llama-3-8B, name `Lllama3TS_unsloth_vllm` |
| vllm-stance | 8001 | 1 (68% mem) | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` |
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

## Live deployment on gpusrv (2026-07-03)

The compose path is currently blocked on this box: `nvidia-container-runtime`
is not installed (needs sudo: `apt-get install nvidia-container-toolkit`,
then `systemctl restart docker`). The running deployment instead uses:

- Summarizer: bare-metal vLLM from `/home/maolee/venvs/vllm` on **port 8002**
  (port 8000 is occupied by an unrelated sglang TTS server), GPU 1, util 0.25.
- Stance: the lab's existing `google/gemma-4-31B-it` vLLM on **port 8800**
  (no new GPU cost). Any OpenAI-compatible server works; that is what
  `STANCE_BASE_URL`/`STANCE_MODEL` in `deploy/.env` are for.
- `deploy/.env` on the server holds these overrides; restart Voila after
  changing it (`serve_app.sh` sources it at startup).

Restart commands (bare-metal):

```bash
CUDA_VISIBLE_DEVICES=1 nohup /home/maolee/venvs/vllm/bin/vllm serve \
  /home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm \
  --served-model-name Lllama3TS_unsloth_vllm --port 8002 \
  --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --api-key token-census > /tmp/vllm-summarizer.log 2>&1 &
./deploy/scripts/serve_app.sh
```

## Notes

- **Model default:** the stance default is the standard
  `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized`. The `-assistant` variant
  is NOT servable (its `gemma4_assistant` architecture has no vLLM/AutoModel
  implementation as of vLLM 0.24).
- **Port conflicts:** if 8000/8001 are taken (shared box), pick free ports in
  `deploy/.env` and mirror them in the compose `ports:` mapping.
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

## systemd user services (installed 2026-07-04)

All three processes now run as systemd user units with `Restart=always`
(this box kills GPU jobs unpredictably). Copies live in `deploy/systemd/`;
installed at `~/.config/systemd/user/` with linger enabled.

```bash
systemctl --user status tweet-summarizer tweet-stance tweet-voila
systemctl --user restart tweet-voila            # after editing serve_app.sh / .env
journalctl --user -u tweet-summarizer -n 50     # logs
```

Gotcha: units get a bare PATH, so each unit sets PATH to its venv bin
(vLLM shells out to `ninja` during startup compile) and `~/.local/bin`
(voila). HF_HOME is set explicitly (units do not read .bashrc).
