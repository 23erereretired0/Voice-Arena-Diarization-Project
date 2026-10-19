# Reproducibility record

This file records exactly what was run, where, with which versions, and what remains to be pinned. Values marked **RECORD AT RUN** are written automatically by the scripts into `mini_eval/results/environment.json`, `mini_eval/predictions/<system>/run_status.json` and `mini_eval/results/system_configs.json`.

## 1. Build environment for this submission (actually used)

| Item | Value (captured by `scripts/env_report.py`, 2026-10-14 UTC) |
|---|---|
| OS | Linux 6.18 x86_64, Ubuntu 24.04 userland (glibc 2.39) |
| Python | 3.12.3 |
| Packages | numpy 2.4.4, scipy 1.17.1, pandas 3.0.2, requests 2.33.1, tabulate 0.10.0; torch, pyannote.audio, NeMo, intervaltree **not installed** |
| Hardware | 1 CPU core, ~3 GB RAM, **no GPU** |
| Tools | ffmpeg 6.1.1, Perl 5.38.2, git, Node.js; Playwright Chromium (used only to render-test the dashboard) |
| Network | **No outbound access**: AMI, VoxConverse, GitHub, Hugging Face and vendor APIs were unreachable |
| Credentials | None (no HF_TOKEN, DEEPGRAM_API_KEY, ASSEMBLYAI_API_KEY, PYANNOTEAI_API_KEY) |
| dscore | Not cloned (no network) |

**Build environment (sandbox).** No network, GPU, model packages or credentials, so all code was first tested on synthetic data only.

## 1b. Evaluation run that produced the reported results (2026-10-18)

| Item | Value (from `results/environment.json` and each `run_status.json`) |
|---|---|
| Machine | Windows 11 (`Windows-10-10.0.26100`), AMD Ryzen laptop CPU (AMD64 Family 25 Model 80). An NVIDIA RTX 3050 Laptop GPU is present but was **not used**: the PyTorch build installed was CPU-only |
| Python | 3.11.9 in all three virtual environments |
| System A | `pyannote/speaker-diarization-community-1` at commit `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`; pyannote.audio 4.0.7; torch 2.14.1+cpu. Model load 2.2 s and warm-up 0.05 s, both excluded from runtime |
| System B | Deepgram `/v1/listen` with `model=nova-3`, `diarize_model=v2`, `language=en`. Response metadata: model `general-nova-3` version `2025-07-31.0`, diarizer arch `v2`, UUID `6946b038-8264-4505-acf7-9822c009fecb`; region "provider default"; requests 2.34.2; runtime end to end, including upload |
| Scorer | dscore commit `e02f949ac6592279300a2c33d03daf9e0c12fd27` with `patch_dscore_windows.py` applied (launch-only change); md-eval-22 under Strawberry Perl 5.42.3; scorer env numpy 1.23.5, scipy 1.10.1, intervaltree 3.1.0, tabulate 0.9.0 |
| Audio tools | ffmpeg 9.0.2 (gyan.dev full build) |
| Settings | Collar 0 ms; overlap included; UEM = full 180 s excerpt; no speaker count supplied; one attempt per recording, 0 retries needed |

**What was executed:**
- `prepare_data.py`. VoxConverse selection succeeded, choosing `bgvvt`, `aepyx`, `aggyz`, `wcxfk` and `ptses`; the 232 reference RTTMs were fetched with `git clone` to speed this up. All 5 AMI excerpts were prepared. The VoxConverse audio zip returned **HTTP 500**, so those 5 recordings failed (recorded in `audio/manifest_resolved.json`).
- `run_pyannote.py`: 5/5 ok, 135–148 s per 180 s excerpt (RTF 0.78, CPU).
- `run_api.py --provider deepgram`: 5/5 ok, 3.8–5.6 s per excerpt.
- `validate_predictions.py`: 0 hard errors. `score.py --diagnostics`: dscore DER and component-rebuilt DER agree to 0.0000 points on all 10 files. Then `bootstrap.py` (`insufficient_sample`, n = 5), `make_results.py`, `analyze.py` and `export_dashboard.py`.
- NeMo: **not attempted** (not installed; needs Linux/WSL2).
- A documented inspection of the ES2004a pair (ffmpeg `volumedetect` and Deepgram word counts), recorded in `configs/inspections.json`.

**Tests passing after the run:**
- 45 Python unit tests and 11 Node aggregation tests, all on synthetic data.
- 69 Playwright browser checks on the measured data (75 in the not-run state).
- A check that every dashboard metric equals `results/summary.csv`: 0 mismatches.

**Still not verified:** upload-server inference with a real model, and `/api/score` against real dscore.

## 2. Target environments for a real run

| Component | Environment | Pin / record |
|---|---|---|
| Orchestration, validation, statistics | Python ≥ 3.10 + `requirements.txt` | `env_report.py --tag orchestrator` |
| pyannote Community-1 | Separate venv, `requirements-pyannote.txt` (pyannote.audio 4.x per the model card), CUDA GPU recommended | pyannote.audio, torch, CUDA versions; HF model commit SHA (or the local clone's git SHA when using `--model-path`) |
| NeMo MSDD | NVIDIA NeMo NGC container (preferred) or `requirements-nemo.txt` | Container digest, nemo_toolkit version, base-config tag, resolved YAML (`results/configs/nemo_msdd_resolved.yaml`, written by the NeMo runner at run time) |
| Commercial API | Orchestration env + API key | Provider, model and diarizer strings sent; response metadata (e.g. Deepgram `diarize_info.model_uuid`); request IDs; region; execution timestamps |
| Scorer | Python 3.10 venv, `requirements-scorer.txt`, Perl 5 | dscore git commit SHA (`DSCORE_REF=<sha>`) |

## 3. Data provenance

Ten recordings were declared in `mini_eval/configs/recordings.json`; **the five AMI excerpts were used**:

| Recording ID | Source | Window | Speakers / overlap (measured) | Condition |
|---|---|---|---|---|
| `ami_ES2004a_ihm_0300_0480` | AMI ES2004a, Mix-Headset (CC BY 4.0) | 300–480 s | 4 / 20.3% | close-talk |
| `ami_ES2004a_sdm_0300_0480` | AMI ES2004a, Array1-01 | same window | 4 / 20.3% | far-field (paired with the row above) |
| `ami_IS1009a_sdm_0300_0480` | AMI IS1009a, Array1-01 | 300–480 s | 4 / 13.2% | far-field |
| `ami_TS3003a_ihm_0300_0480` | AMI TS3003a, Mix-Headset | 300–480 s | 3 / 1.0% | close-talk |
| `ami_EN2002a_sdm_0300_0480` | AMI EN2002a, Array1-01 | 300–480 s | 4 / 18.4% | far-field, non-scenario meeting |
| `vox_{bgvvt,aepyx,aggyz,wcxfk,ptses}_0000_0180` | VoxConverse v0.3 test | 0–180 s | — | **not used**: audio download failed (HTTP 500) |

AMI references come from AMI-diarization-setup `only_words/rttms/test/` at commit `2509d8933721023fab4def2618aabd5c28eb82e9`; VoxConverse references from `joonson/voxconverse/test/`.

`prepare_data.py` records the following for each recording:
- dataset version, source URL and licence;
- excerpt window;
- SHA-256 of raw and prepared audio;
- reference repository commit;
- the selected VoxConverse IDs with their selection statistics;
- speaker count, overlap ratio and speaker-change rate;
- condition tags with their evidence.

Audio quality is "unknown" for all ten until a documented listening check is added. Telephone audio is not represented.

Preprocessing: `ffmpeg -ss <start> -t 180 -ac 1 -ar 16000 -c:a pcm_s16le`. There is no denoising, normalization or VAD. References are cropped to the window and shifted to t = 0, so turns crossing the window edge are truncated (a documented edge effect).

## 4. Fixed settings

| Setting | Value |
|---|---|
| Collar | 0 ms (diagnostic 250 ms reported separately) |
| Overlap | Included |
| Scoring region | UEM `[0, clip duration]` |
| Speaker count | Not provided (primary). Assisted run → separate directory and track |
| Word → segment conversion (STT APIs) | Bridge same-speaker gaps ≤ 0.2 s |
| Prediction normalization | Merge same-speaker overlapping segments only |
| Failure policy | Missing, invalid, empty or failed output → empty hypothesis (100 % missed) |
| API retries / timeout | 3 attempts on network errors, 429 and 5xx; backoff 2 s, 4 s; 900 s end to end per file |
| Bootstrap | 10,000 replicates, seed 20261015, 95 % percentile CI, Holm correction, min 20 recordings, practical threshold 1.0 pp |
| Runtime | `time.perf_counter` around inference; model load and warm-up excluded and recorded separately; API timing is end to end including network |

## 5. Commands

See `mini_eval/README.md` (end-to-end run) and `mini_eval/score_commands.txt` (exact scorer calls).

## 6. Known limitations of the reproducibility setup

- Hosted APIs can change behaviour under the same parameters. Only pinned version parameters and archived response metadata mitigate this.
- GPU inference can be non-deterministic at the last digit; DER differences below 0.1 pp between reruns are not meaningful.
- The 180 s excerpts and edge truncation make these numbers incomparable with published full-meeting AMI or VoxConverse results.
- The md-eval output parser was written against md-eval's documented field names. A built-in check flags any recording where md-eval-derived DER differs from dscore DER by more than 0.05 pp, which would expose a parsing problem on the first real run.
