# Voice Arena: Speaker Diarization Benchmark and Leaderboard

An evaluation methodology, a reproducible mini benchmark (run on 5 AMI meeting excerpts with 2 systems), a simplified leaderboard with condition-based reranking, and a local upload service for diarizing your own audio.

> **Status: measured, provisional (run on 2026-10-18).** Two systems were run on the same 5 public AMI meeting excerpts (15 minutes) and scored with the pinned dscore scorer. Strict settings: 0 ms collar, overlap included, no speaker count given.
>
> | System | Pooled DER | Macro DER | JER | Missed | False alarm | Confusion | RTF |
> |---|---|---|---|---|---|---|---|
> | pyannote Community-1 (local, CPU) | **22.2%** | 23.6% | 46.6% | 12.3% | 3.7% | 6.2% | 0.78 |
> | Deepgram nova-3, diarizer v2 (API) | **47.0%** | 48.5% | 66.4% | 38.1% | 1.9% | 7.0% | 0.02 (end-to-end) |
>
> Five recordings are below the 20 needed for confidence intervals, so this ranking is **exploratory, not a general result**. The 5 VoxConverse recordings were selected but could not be downloaded (the dataset server returned HTTP 500). NeMo was not attempted. Full findings: `docs/benchmark_report.md`.

## Contents

| Item | Location |
|---|---|
| Methodology (metrics, protocol, statistics, reflection) | `docs/methodology.md` |
| One-page report | `docs/one_page_report.md` |
| Benchmark report (generated from the real results) | `docs/benchmark_report.md` |
| Reproducibility record | `docs/reproducibility.md` |
| Two-minute video script | `docs/video_script.md` |
| Recording manifest (10 planned, 5 prepared) and documented inspections | `mini_eval/configs/recordings.json`, `mini_eval/configs/inspections.json` |
| Benchmark pipeline, results, tests | `mini_eval/` |
| Simplified dashboard | `dashboard/index.html` (+ `aggregate.js`, `data/benchmark_results.js`) |
| Upload / diarization backend | `server/app.py` |

```
voice-arena/
├── README.md
├── docs/            methodology.md  one_page_report.md  benchmark_report.md  reproducibility.md  video_script.md
├── dashboard/       index.html  aggregate.js  README.md  data/benchmark_results.js
│                    tests/aggregate.test.js  tests/browser_check.py
├── server/          app.py   (standard library only)
└── mini_eval/
    ├── configs/     recordings.json (10 planned)  systems.json (3 systems)  inspections.json
    ├── scripts/     prepare_data  run_pyannote  run_nemo_msdd  run_api  validate_predictions  score
    │                bootstrap  make_results  analyze  export_dashboard  env_report  run_all.sh  common  rttm_utils
    │                patch_dscore_windows (Windows only)
    ├── tests/       test_mini_eval.py  test_selection_analysis_server.py  fixtures/
    ├── audio/ reference/ predictions/{pyannote-community-1,nemo-msdd,api-system}/ third_party/
    └── results/     summary.csv  per_recording.csv  error_breakdown.csv  system_configs.json
                     pairwise_bootstrap.csv  scores_raw.json  recording_summaries.json  environment.json
```

## Requirements

- Python 3.10+, ffmpeg, git and Perl 5 (for dscore's md-eval).
- A CUDA GPU is recommended; CPU works, but slowly.
- Credentials:
  - `HF_TOKEN`, after accepting the model conditions at https://huggingface.co/pyannote/speaker-diarization-community-1.
  - Optionally one API key: `DEEPGRAM_API_KEY`, `ASSEMBLYAI_API_KEY` or `PYANNOTEAI_API_KEY`.
- Network access to github.com, groups.inf.ed.ac.uk (AMI), robots.ox.ac.uk (VoxConverse) and huggingface.co.

## Install

Linux / macOS:

```bash
cd voice-arena/mini_eval
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3 -m venv .venv-pyannote && .venv-pyannote/bin/pip install -r requirements-pyannote.txt
python3.10 -m venv .venv-scorer && .venv-scorer/bin/pip install -r requirements-scorer.txt
git clone https://github.com/nryant/dscore.git third_party/dscore
git -C third_party/dscore checkout <commit-sha>      # pin it; it is recorded in the results
# NeMo: use the NVIDIA NeMo NGC container (record its digest), or: pip install -r requirements-nemo.txt
```

Windows (PowerShell; install ffmpeg, Git, Perl, e.g. Strawberry Perl, and Python 3.10+ first):

```powershell
cd voice-arena\mini_eval
py -3 -m venv .venv;          .venv\Scripts\pip install -r requirements.txt
py -3 -m venv .venv-pyannote; .venv-pyannote\Scripts\pip install -r requirements-pyannote.txt
py -3.10 -m venv .venv-scorer; .venv-scorer\Scripts\pip install -r requirements-scorer.txt
git clone https://github.com/nryant/dscore.git third_party\dscore
git -C third_party\dscore checkout e02f949ac6592279300a2c33d03daf9e0c12fd27   # the commit used for the reported results
.venv\Scripts\python scripts\patch_dscore_windows.py   # Windows only: dscore must start md-eval via perl
```

NeMo is easiest under WSL2 or Linux. `run_all.sh` needs bash, so on Windows run the stages one by one (below) or use Git Bash.

## Run the benchmark

Linux / macOS (all stages):

```bash
cd voice-arena/mini_eval
export HF_TOKEN=...        DEEPGRAM_API_KEY=...      # API key optional
PY=.venv/bin/python PY_PYANNOTE=.venv-pyannote/bin/python DSCORE_PYTHON=.venv-scorer/bin/python \
DSCORE_REF=<commit-sha> API_PROVIDER=deepgram bash scripts/run_all.sh
# skip what you cannot run: SKIP_NEMO=1 SKIP_API=1 bash scripts/run_all.sh
```

Any OS, stage by stage (Windows: replace `.venv/bin/python` with `.venv\Scripts\python`):

```bash
.venv/bin/python scripts/env_report.py --tag orchestrator
.venv/bin/python scripts/prepare_data.py                 # downloads, selects, crops, hashes, tags
.venv-pyannote/bin/python scripts/run_pyannote.py        # System A
python scripts/run_nemo_msdd.py                          # System B (inside the NeMo env/container)
.venv/bin/python scripts/run_api.py --provider deepgram  # System C (optional)
.venv/bin/python scripts/validate_predictions.py
DSCORE_PYTHON=.venv-scorer/bin/python .venv/bin/python scripts/score.py --diagnostics
.venv/bin/python scripts/bootstrap.py
.venv/bin/python scripts/make_results.py
.venv/bin/python scripts/analyze.py                      # summaries + docs/benchmark_report.md
.venv/bin/python scripts/export_dashboard.py             # dashboard/data/benchmark_results.js
```

Exact scorer calls are listed in `mini_eval/score_commands.txt`.

## Open the dashboard and upload audio

```bash
# results only
open dashboard/index.html            # or double-click it
# with audio upload (pyannote env, HF_TOKEN set):
cd voice-arena
mini_eval/.venv-pyannote/bin/python server/app.py      # Windows: mini_eval\.venv-pyannote\Scripts\python server\app.py
# then browse to http://127.0.0.1:8765/
```

How uploads are handled:
- Audio is processed only on that machine and deleted after each request. Uploads are never added to the benchmark.
- Limits are 50 MB and 20 minutes; set `VA_MAX_UPLOAD_MB` and `VA_MAX_DURATION_S` to change them.
- Scoring an upload against your own reference RTTM needs the dscore checkout above.

## Tests

```bash
cd mini_eval && python3 -m unittest discover -s tests -v     # 45 tests, synthetic data
node dashboard/tests/aggregate.test.js                        # 11 tests, ranking maths
python dashboard/tests/browser_check.py                       # 69 checks with measured data, 75 in the not-run state (needs playwright + chromium)
```

## Status of each stage

| Stage | Status |
|---|---|
| Recordings | **5 of 10 prepared and used** (AMI ES2004a close-talk and far-field, IS1009a far-field, TS3003a close-talk, EN2002a far-field). VoxConverse files `bgvvt`, `aepyx`, `aggyz`, `wcxfk` and `ptses` were chosen by the annotation rules, but the audio download failed with HTTP 500 |
| System A: pyannote Community-1 | **Run**: 5/5 recordings, 0 failures. pyannote.audio 4.0.7, model commit `3533c8c…`, CPU (PyTorch 2.14.1+cpu) |
| System B: Deepgram | **Run**: 5/5, 0 failures. `general-nova-3` version 2025-07-31.0, diarizer v2 |
| System C: NeMo MSDD | **Not attempted.** It needs Linux/WSL2 (ideally the NVIDIA container); the assignment required two systems |
| Scoring | **Run**: dscore commit `e02f949…` with md-eval-22. dscore DER and component-rebuilt DER agree to 0.0000 points on all 10 scored files |
| Statistics | Paired bootstrap run: `insufficient_sample` (5 < 20), so no CI or significance is claimed |
| Recording-level analysis | Generated from the real results, plus one documented audio inspection (`configs/inspections.json`) |
| Dashboard | Shows the measured results. Every displayed metric was checked against `summary.csv`, with 0 mismatches |
| Upload backend | Implemented and tested for every error path. **Not yet used for real inference** on the evaluation machine |
| Cost per audio hour | Not reported: no documented price inputs |

**Bugs found only by the real run, all fixed:**
1. md-eval prints `FALARM SPEAKER TIME`, not `FALSE ALARM SPEAKER TIME`.
2. This dscore version has no `--table_format` option.
3. On Windows, dscore launches `md-eval-22.pl` directly (WinError 193); `patch_dscore_windows.py` makes it start through `perl`.
4. pyannote 4 decodes audio files through torchcodec, which needs FFmpeg DLLs, so audio is now passed in memory.
5. The bootstrap treated differently worded "unknown" speaker-count conditions as different.
6. The VoxConverse annotation fetch was slow and silent.
7. Failed VoxConverse entries kept a placeholder ID.
8. The export and the report hard-coded the recording count.

## Manual steps still required (optional)

1. To add VoxConverse: download `voxconverse_test_wav.zip` when the Oxford server works, run `prepare_data.py --voxconverse-zip <path>`, then re-run both systems and the scoring.
2. To run NeMo: use Linux/WSL2 with the NVIDIA NeMo container, then `scripts/run_nemo_msdd.py`.
3. To use the GPU for pyannote: install a CUDA build of PyTorch in `.venv-pyannote`.
4. To try uploads for real: `python server/app.py` with `HF_TOKEN` set.
5. Delete the Hugging Face and Deepgram keys used for this run, since they were shared during setup.

## Known limitations

- Five 3-minute excerpts from one corpus give a provisional, exploratory ranking only.
- All recordings are AMI meetings with 3–4 speakers. There is no telephone, media or 6+ speaker audio, and audio quality is unknown.
- Excerpt edges truncate turns.
- Hosted APIs can drift between runs.
- Deepgram is scored from word-level labels: it cannot represent overlap, and unrecognised speech counts as missed.
- DER weighs every second equally.
