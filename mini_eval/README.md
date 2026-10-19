# Voice Arena mini benchmark

A small, reproducible diarization experiment designed for **ten public excerpts** (5 AMI, 5 VoxConverse) and **up to three systems**, run so far on the 5 AMI excerpts with 2 systems, scored with a **pinned dscore** at strict settings (0 ms collar, overlap included, no speaker count).

> **Status: run on 2026-10-18 (provisional).** 5 AMI excerpts × 2 systems were scored with dscore: pyannote Community-1 pooled DER **22.2%**, Deepgram nova-3 **47.0%**. VoxConverse audio could not be downloaded (HTTP 500) and NeMo was not attempted. See `results/` and `../docs/benchmark_report.md`.
>
> **Five recordings give only a provisional, exploratory ranking.** The confidence-interval and ranking columns deliberately say so.

## Layout

```
mini_eval/
├── audio/                 prepared 16 kHz mono excerpts + manifest_resolved.json
├── reference/             cropped reference RTTM + UEM per recording
├── predictions/
│   ├── pyannote-community-1/   <rid>.rttm + run_status.json
│   ├── nemo-msdd/
│   └── api-system/             + raw/ (archived API responses)
├── configs/               recordings.json (clip windows), systems.json (declarations)
├── scripts/               see table below
├── tests/                 unit tests (synthetic data only) + scorer-output fixtures
├── third_party/           dscore checkout goes here (pinned commit)
├── results/               summary.csv, per_recording.csv, error_breakdown.csv,
│                          system_configs.json, pairwise_bootstrap.csv, scores_raw.json,
│                          environment.json, README.md; validation_report.json is written
│                          by validate_predictions.py once data has been prepared
├── README.md  requirements*.txt  score_commands.txt
```

| Script | Purpose |
|---|---|
| `prepare_data.py` | Select the 5 VoxConverse files by annotation rules, download, hash, convert to 16 kHz mono, cut the 180 s windows, crop references, write UEMs, condition tags and the resolved manifest |
| `run_pyannote.py` | pyannote/speaker-diarization-community-1 (primary; `--assisted` → separate track) |
| `run_nemo_msdd.py` | NeMo MSDD + TitaNet-L + MarbleNet VAD; saves the resolved YAML |
| `run_api.py` | `--provider deepgram\|assemblyai\|pyannoteai` with fixed retries and timeouts |
| `validate_predictions.py` | RTTM checks: file IDs, timestamps, duration bounds, emptiness, unknown files |
| `score.py` | dscore DER/JER, plus components, overlap and non-overlap DER from the bundled md-eval; failure policy; diagnostics |
| `bootstrap.py` | Paired recording-level bootstrap with Holm correction (returns `insufficient_sample` for n < 20) |
| `make_results.py` | The required CSVs and `system_configs.json` |
| `analyze.py` | Per-recording summaries and cross-recording findings from the real results → `results/recording_summaries.json`, `docs/benchmark_report.md` |
| `export_dashboard.py` | Writes `../dashboard/data/benchmark_results.js` (recording-level seconds, tags and segments for client-side reranking) |
| `env_report.py` | Records OS, Python, packages, GPU, ffmpeg, Perl and the dscore commit |
| `patch_dscore_windows.py` | Windows only: makes dscore launch md-eval through `perl` (launch-only change) |
| `run_all.sh` | Runs every stage. Blocked stages are recorded, not faked |

## Install

```bash
cd mini_eval
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt          # orchestration
python3 -m venv .venv-pyannote && .venv-pyannote/bin/pip install -r requirements-pyannote.txt
python3.10 -m venv .venv-scorer && .venv-scorer/bin/pip install -r requirements-scorer.txt
# NeMo: use the NVIDIA NeMo NGC container (record its digest), or requirements-nemo.txt
git clone https://github.com/nryant/dscore.git third_party/dscore && git -C third_party/dscore checkout <SHA>
```

Credentials (only for the systems you run):

```bash
export HF_TOKEN=...            # after accepting the Community-1 conditions on Hugging Face
export DEEPGRAM_API_KEY=...    # or ASSEMBLYAI_API_KEY / PYANNOTEAI_API_KEY
```

## Run

```bash
cd mini_eval
PY=.venv/bin/python PY_PYANNOTE=.venv-pyannote/bin/python PY_NEMO=python3 \
DSCORE_PYTHON=.venv-scorer/bin/python DSCORE_REF=<sha> API_PROVIDER=deepgram \
bash scripts/run_all.sh
# Skip stages you cannot run: SKIP_NEMO=1 SKIP_API=1 bash scripts/run_all.sh
# Optional assisted (known speaker count) run, kept in its own directory and track:
.venv-pyannote/bin/python scripts/run_pyannote.py --assisted
```

Tests (no models or network needed): `python3 -m unittest discover -s tests -v`

## Handling of problem cases

| Case | Behaviour |
|---|---|
| Missing data / no network | `prepare_data.py` records the error per recording; runners write `blocked_data_unavailable` |
| Missing credentials | Runner writes `blocked_missing_credentials` for every recording; nothing is scored |
| Missing packages | `blocked_missing_dependency` |
| No GPU | Falls back to CPU with a warning; RTF labelled CPU |
| Incompatible audio | Per-file `failed` with the reason (sample rate, channels, sample width) |
| API 429 / 5xx / network error | Retried up to 3 attempts with backoff; other 4xx errors are not retried |
| Timeout | `timeout` status → failure → scored as empty hypothesis |
| Empty or invalid RTTM, mismatched IDs | Validation error → failure → scored as empty hypothesis, never dropped |
| Scorer missing | Predictions validated; scores remain `not_run` |
| dscore / md-eval disagreement | Recording flagged `scorer_mismatch` |

## Planned analysis (fill in only after a real run)

`scripts/analyze.py` writes this analysis automatically from the real results (`docs/benchmark_report.md`); review it before quoting.

1. Lowest pooled DER on these excerpts, stated as "on these excerpts", not as a general ranking.
2. Whether JER orders the systems the same way as DER.
3. Which component dominates per recording (see `error_breakdown.csv`).
4. Close-talk versus far-field DER change on the *same* AMI audio window.
5. Whether RTF or cost changes the practical choice.
6. An explicit statement that n = 10 is below the 20-recording minimum, so no interval and only a provisional ranking.
