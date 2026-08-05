# CLAUDE.md

Project-level instructions for Tweet Browser. Loaded by Claude Code in every
session working in this directory.

## Environment

- Runs on the shared lab GPU box `gpusrv.isr.umich.edu` (8x GPU, indices
  0-7). GPUs are shared with other users/jobs; nothing here owns the box.
- Three long-running processes, all managed as **systemd --user units**
  (`~/.config/systemd/user/`, linger enabled), each with `Restart=always`:
  `tweet-summarizer.service`, `tweet-stance.service`, `tweet-voila.service`.
  Unit files are checked into `deploy/systemd/`.
- The `deploy/docker-compose.yml` path is **not the live deployment**. It
  was the original design (see `docs/superpowers/specs/2026-07-03-backend-
  service-wiring-design.md`) but is blocked on this box (`nvidia-container-
  runtime` not installed) and was superseded by the bare-metal systemd
  units. Do not assume compose is running; check `systemctl --user status`.
- App code: `Frontend/` (Voila notebook `tweet_browser.ipynb` + `prompts.py`,
  `config.py`, `ai_summary.py`, `demographics.py`, `ui_modules.py`).
  Deploy/ops code: `deploy/` (scripts, systemd units, `.env`). New ops
  tooling (this hardening pass) lives in `scripts/`.

## App Python environment (matters for widget/version debugging)

- Voila and its kernels run **system `/usr/bin/python3` (3.10) with the
  `~/.local` user site**. Packages there shadow system dist-packages.
  Live widget stack (verified 2026-08-05): **anywidget 0.9.18**,
  ipywidgets 8.1.7, voila 0.5.12, all in
  `/home/maolee/.local/lib/python3.10/site-packages`.
- Installing a different anywidget/ipywidgets version into conda, the repo
  `.venv`, or any other env does NOT change what the app runs. To actually
  change the app's version: `python3 -m pip install --user anywidget==X`,
  then `systemctl --user restart tweet-voila` and hard-refresh the page.
- Quick check of what the app env resolves:
  `python3 deploy/scripts/smoke_test.py` prints interpreter + versions
  first, or one-liner
  `python3 -c "import anywidget; print(anywidget.__version__, anywidget.__file__)"`.

## Backend topology

**Ports/GPU assignment are genuinely ad hoc and have already drifted at
least twice.** Three different documents in this repo disagree with each
other and with what's actually running:

| Source | Summarizer | Stance |
| --- | --- | --- |
| `docker-compose.yml` / `deploy/.env.example` / design doc (original plan, never live) | port 8000, GPU 1, util 0.25 | port 8001, GPU 1, util 0.68, `gemma-4-26B-A4B-it-...` |
| `deploy/scripts/serve_summarizer.sh` / `serve_stance.sh` (bare-metal, unused since 2026-07-04) | port 8000, GPU 1 | port 8001, GPU 1 |
| **`deploy/systemd/*.service` + `deploy/.env` (LIVE, verified 2026-07-06 via `systemctl --user status`)** | **port 8002, GPU 3** | **port 8800, GPU 4+5 (TP2), model `google/gemma-4-31B-it`** |

**Treat the systemd units + `deploy/.env` as ground truth.** The
docker-compose file, `.env.example`, `serve_summarizer.sh`/`serve_stance.sh`,
and the design doc all describe the abandoned GPU-1-for-everything plan and
are stale; do not "fix" the live config to match them. `scripts/
health_supervisor.sh`'s config block is kept in sync with the live systemd
units, not the docs — if you change a port or GPU, update all three: the
systemd unit file, `deploy/.env`, and `scripts/health_supervisor.sh`.

Also note: the live stance model (`google/gemma-4-31B-it`) is a different
model from the one named throughout the design doc / README
(`gemma-4-26B-A4B-it-qat-...`) and is **not** served from
`/home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm` — only the
summarizer uses that path. Only the summarizer is the fine-tuned
Llama-3-8B; the stance backend runs a separate, larger vLLM instance
(`/home/maolee/projects/llm-deploy/.venv`) that has nothing to do with that
model directory.

| Service | Purpose | Launch (live) | Port | Health endpoint | GPU (live) |
| --- | --- | --- | --- | --- | --- |
| `tweet-summarizer` | Tweet summarization, fine-tuned Llama-3-8B `Lllama3TS_unsloth_vllm` | systemd unit `deploy/systemd/tweet-summarizer.service` → `/home/maolee/venvs/vllm/bin/vllm serve /home/maolee/projects/llm-deploy/Lllama3TS_unsloth_vllm --served-model-name Lllama3TS_unsloth_vllm --port 8002 --gpu-memory-utilization 0.25 --max-model-len 8192 --api-key token-census` | 8002 | `GET /v1/models` (Bearer `LLM_API_KEY`) — same probe `Frontend/prompts.py:check_backends()` uses; vLLM also exposes plain `GET /health` | GPU 3, util 0.25 |
| `tweet-stance` | Stance detection, `google/gemma-4-31B-it` | systemd unit `deploy/systemd/tweet-stance.service` → `/home/maolee/projects/llm-deploy/.venv/bin/vllm serve google/gemma-4-31B-it --tensor-parallel-size 2 --port 8800 --trust-remote-code --gpu-memory-utilization 0.90` | 8800 | `GET /v1/models` (Bearer `LLM_API_KEY`); `GET /health` also available | GPU 4+5, tensor-parallel-size 2, util 0.90 |
| `tweet-voila` | Frontend app (Voila-rendered `Frontend/tweet_browser.ipynb`) | systemd unit `deploy/systemd/tweet-voila.service` → `deploy/scripts/serve_app.sh` | 8866 | plain `GET /` (200 if the kernel/server answers); no JSON health route | GPU 1 for the BGE embedding model (best-effort; falls back to CPU), or unset/CPU |

Frontend URLs/model names/timeouts are all overridable via env vars read by
`Frontend/config.py` (`SUMMARIZER_BASE_URL`, `SUMMARIZER_MODEL`,
`STANCE_BASE_URL`, `STANCE_MODEL`, `LLM_API_KEY`, `SUMMARY_TIMEOUT_S`,
`STANCE_TIMEOUT_S`); the live values are set in `deploy/.env`, sourced by
`deploy/scripts/serve_app.sh` at Voila startup — **restart `tweet-voila`
after editing `deploy/.env`, it is only read at process start.**

Port 8000 on this box is occupied by an unrelated sglang TTS server —
do not "reclaim" it back to the documented-but-stale default.

## Operations

- **Use `scripts/health_supervisor.sh` instead of manually killing GPU
  processes.** It probes each backend's actual health endpoint (not just
  "is the process alive," which systemd's `Restart=always` already
  guarantees) and restarts via `systemctl --user restart <unit>` after
  `FAIL_THRESHOLD` consecutive failures, with a cooldown to avoid
  restart-storming a backend that's crash-looping (e.g. GPU OOM). Modes:
  `--once` (single pass, cron-friendly), `--dry-run` (prints resolved
  config, no network calls or restarts, always safe to run), no args
  (foreground poll loop). Config (ports, health URLs, systemd unit names,
  thresholds) lives in one block at the top of the file — edit that, not
  the logic below it.
- Manual status check (read-only, always safe):
  `systemctl --user status tweet-summarizer tweet-stance tweet-voila`
- Manual restart of one service (prefer the supervisor, but if doing it by
  hand): `systemctl --user restart tweet-<summarizer|stance|voila>`
- Logs: `journalctl --user -u tweet-<summarizer|stance|voila> -n 100`
- Post-restart verification: `python3 deploy/scripts/smoke_test.py` (real
  summarize + stance call, not just a health probe; exits 0 = usable).
- `deploy/scripts/serve_summarizer.sh` / `serve_stance.sh` are stale
  (bare-metal alternative from before systemd units existed, and per
  `deploy/README.md` the host `vllm` install they'd use was broken as of
  2026-07-03 — torch ABI mismatch). Don't use them to "fix" a downed
  backend; use the systemd unit / supervisor instead.

## Known failure modes

- **"AI backend unavailable: summarizer" / "...: stance" banner** in the
  Voila UI: `Frontend/prompts.py:check_backends()` failed its `GET
  /v1/models` probe against `SUMMARIZER_BASE_URL` / `STANCE_BASE_URL`. Means
  the app could not reach the backend at startup — check
  `systemctl --user status`, then `scripts/health_supervisor.sh --once`.
  Generate buttons stay disabled; search/filter/sample/semantic search
  remain usable (only summarize/stance are gated on backend health).
- **Health check exits 1 ("Wait for stance server"-style failures)**: could
  not be located as a named check anywhere in this repo (no CI workflow,
  no cron/systemd timer found on the box as of 2026-07-06). If this is a
  step in an external CI/runbook, it most likely wraps a probe equivalent
  to `curl http://localhost:8800/v1/models` or vLLM's own `GET /health` —
  point it at the **live** port (8800), not the stale documented default
  (8001).
- **Process alive but unresponsive**: systemd's `Restart=always` only
  restarts a process that has exited; a vLLM process that is running but
  wedged (hung request queue, GPU error that didn't crash the process)
  will pass `systemctl status` as `active (running)` while the app still
  sees it as down. This is what `scripts/health_supervisor.sh` is for.
- **GPU OOM / crash loop**: stance is TP2 across GPU 4+5 at
  `--gpu-memory-utilization 0.90` — very little headroom. If another job
  lands on GPU 4 or 5, stance will fail to (re)load. The supervisor's
  restart cooldown prevents it from restart-storming in this case, but it
  cannot fix a genuinely occupied GPU; check `nvidia-smi` by hand.
- **Editing `deploy/.env` and nothing changes**: it's only sourced by
  `deploy/scripts/serve_app.sh` at process start; restart `tweet-voila`.
- **Gemma variant confusion**: `deploy/README.md` notes the
  `-assistant` suffixed stance model variant is not servable under vLLM
  (no matching architecture implementation). The live deployment sidesteps
  this entirely by running a different model (`google/gemma-4-31B-it`)
  from a different vLLM install than the one described in the design docs.

## Facts noted but not independently verified

- Whether `deploy/.env`'s `LLM_API_KEY` (`token-census`) is actually
  enforced by the stance server: `tweet-stance.service`'s `ExecStart` does
  not pass `--api-key`, so the stance vLLM instance may accept
  unauthenticated requests (harmless — the app still sends a Bearer token,
  vLLM just won't require it).
