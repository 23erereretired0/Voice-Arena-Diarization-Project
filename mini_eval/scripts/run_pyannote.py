#!/usr/bin/env python3
"""Run pyannote/speaker-diarization-community-1 on every prepared recording.

Primary run (default): the true number of speakers is NOT provided.
Assisted run (--assisted): num_speakers is taken from the reference and the
output goes to a separate directory labelled as the assisted track. The two
must never be ranked together.

Model access: the checkpoint is gated on Hugging Face. Either
  * export HF_TOKEN=<token> after accepting the model conditions, or
  * clone the model at a pinned commit and pass --model-path (offline use,
    as documented on the model card).

Outputs: predictions/<system>/<recording_id>.rttm and run_status.json.
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (PRED_DIR, ROOT, STATUS_BLOCKED_CREDENTIALS,  # noqa: E402
                    STATUS_BLOCKED_DEPENDENCY, STATUS_EMPTY_OUTPUT, STATUS_FAILED,
                    STATUS_OK, RecordingRun, SystemRunStatus, blocked_status,
                    check_audio_compatible, get_logger, load_manifest, manifest_or_blocked, utc_now)
from rttm_utils import Segment, sanitize_speaker, write_rttm  # noqa: E402

log = get_logger("run_pyannote")
MODEL_ID = "pyannote/speaker-diarization-community-1"


def _git_sha(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=False)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _hf_sha(repo: str, token: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo, token=token).sha
    except Exception as exc:  # noqa: BLE001
        log.warning("could not resolve HF revision for %s: %s", repo, exc)
        return None


def load_pcm16(path: Path, torch):
    """Read a 16-bit PCM WAV into a (channels, samples) float32 tensor in [-1, 1]."""
    import wave
    import numpy as np
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {8 * wf.getsampwidth()}-bit")
        ch, sr = wf.getnchannels(), wf.getframerate()
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).reshape(-1, ch).T.contiguous()
    return audio, sr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", help="Local clone of the pipeline (offline, pinned)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--assisted", action="store_true",
                    help="Give the reference speaker count (ASSISTED track, separate output)")
    ap.add_argument("--system-id", default=None)
    args = ap.parse_args()

    system_id = args.system_id or ("pyannote-community-1-assisted" if args.assisted
                                   else "pyannote-community-1")
    track = "assisted_known_speaker_count_batch" if args.assisted else "audio_only_batch"
    condition = "known (reference count supplied)" if args.assisted else "unknown"
    out_dir = PRED_DIR / system_id
    out_dir.mkdir(parents=True, exist_ok=True)
    recordings = manifest_or_blocked(system_id, "run_pyannote.py", track, condition,
                                     MODEL_ID, out_dir)
    if recordings is None:
        return 5

    # ---- dependencies & credentials -------------------------------------
    try:
        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline
    except ImportError as exc:
        reason = f"pyannote.audio/torch not importable: {exc}"
        log.error(reason)
        blocked_status(system_id, "run_pyannote.py", track, condition, MODEL_ID,
                       reason, STATUS_BLOCKED_DEPENDENCY, recordings).save(out_dir)
        return 2
    token = os.environ.get("HF_TOKEN")
    if not args.model_path and not token:
        reason = ("HF_TOKEN not set and --model-path not given; the model is gated "
                  "(accept conditions on Hugging Face, then export HF_TOKEN).")
        log.error(reason)
        blocked_status(system_id, "run_pyannote.py", track, condition, MODEL_ID,
                       reason, STATUS_BLOCKED_CREDENTIALS, recordings).save(out_dir)
        return 3

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            log.warning("CUDA requested but unavailable; falling back to CPU")
            device = "cpu"

    status = SystemRunStatus(
        system_id=system_id, runner="run_pyannote.py", track=track,
        speaker_count_condition=condition, model_identifier=args.model_path or MODEL_ID,
        package_versions={"pyannote.audio": pyannote.audio.__version__,
                          "torch": torch.__version__,
                          "python": platform.python_version()},
        hardware={"device": device,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                  "cpu": platform.processor() or platform.machine(),
                  "os": platform.platform()},
        settings={"num_speakers": "reference" if args.assisted else None,
                  "min_speakers": None, "max_speakers": None,
                  "scored_output": "speaker_diarization (overlap-aware)",
                  "exclusive_output_used": False},
        run_started_utc=utc_now(),
    )
    status.model_revision = (_git_sha(Path(args.model_path)) if args.model_path
                             else _hf_sha(MODEL_ID, token))

    # ---- load + warm-up (timed separately, excluded from RTF) -----------
    t0 = time.perf_counter()
    try:
        if args.model_path:
            pipeline = Pipeline.from_pretrained(args.model_path)
        else:
            pipeline = Pipeline.from_pretrained(MODEL_ID, token=token)
        if pipeline is None:
            raise RuntimeError("Pipeline.from_pretrained returned None (access not granted?)")
        pipeline.to(torch.device(device))
    except Exception as exc:  # noqa: BLE001
        reason = f"model load failed: {exc}"
        log.error(reason)
        status.blocked_reason = reason
        for r in recordings:
            status.recordings.append(RecordingRun(r["recording_id"], status=STATUS_FAILED,
                                                  error=reason, audio_seconds=r["duration_s"]))
        status.run_finished_utc = utc_now()
        status.save(out_dir)
        return 4
    status.model_load_seconds = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    try:
        silence = torch.zeros(1, 16000 * 5)
        pipeline({"waveform": silence, "sample_rate": 16000})
        status.warmup_seconds = round(time.perf_counter() - t0, 3)
    except Exception as exc:  # noqa: BLE001
        log.warning("warm-up failed (continuing): %s", exc)

    # ---- inference --------------------------------------------------------
    for rec in recordings:
        rid = rec["recording_id"]
        run = RecordingRun(rid, audio_seconds=rec["duration_s"], started_utc=utc_now())
        wav = ROOT / rec["audio_path"]
        problems = check_audio_compatible(wav)
        if problems:
            run.status, run.error = STATUS_FAILED, f"incompatible audio: {problems}"
            status.recordings.append(run)
            continue
        kwargs = {}
        if args.assisted:
            kwargs["num_speakers"] = int(rec["n_reference_speakers"])
        try:
            # Pass audio in memory instead of a file path: pyannote 4 decodes files with
            # torchcodec, which needs shared FFmpeg libraries (often missing on Windows).
            # Our prepared files are 16 kHz mono 16-bit PCM, so the stdlib reads them exactly.
            audio, sr = load_pcm16(wav, torch)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            output = pipeline({"waveform": audio, "sample_rate": sr}, **kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            run.processing_seconds = round(time.perf_counter() - t0, 3)
            ann = getattr(output, "speaker_diarization", output)  # 4.x vs 3.x API
            segs = [Segment(rid, float(turn.start), float(turn.end - turn.start),
                            sanitize_speaker(spk))
                    for turn, _, spk in ann.itertracks(yield_label=True)
                    if turn.end - turn.start > 0]
            write_rttm(segs, out_dir / f"{rid}.rttm")
            run.status = STATUS_OK if segs else STATUS_EMPTY_OUTPUT
            run.extra["n_predicted_speakers"] = len({s.speaker for s in segs})
        except Exception as exc:  # noqa: BLE001
            run.status = STATUS_FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s\n%s", rid, exc, traceback.format_exc())
        run.finished_utc = utc_now()
        run.attempts = 1
        status.recordings.append(run)
        log.info("%s: %s (%.1fs)", rid, run.status, run.processing_seconds or -1)

    status.run_finished_utc = utc_now()
    status.save(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
