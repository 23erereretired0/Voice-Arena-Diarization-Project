"""Shared helpers for the Voice Arena mini benchmark.

Everything here is dependency-light (standard library only) so that
validation, scoring orchestration and result aggregation can run in a
plain Python environment without torch / pyannote / NeMo installed.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import shutil
import subprocess
import time
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]          # mini_eval/
AUDIO_DIR = ROOT / "audio"
REF_DIR = ROOT / "reference"
PRED_DIR = ROOT / "predictions"
RESULTS_DIR = ROOT / "results"
CONFIG_DIR = ROOT / "configs"
THIRD_PARTY = ROOT / "third_party"

RECORDINGS_CONFIG = CONFIG_DIR / "recordings.json"
SYSTEMS_CONFIG = CONFIG_DIR / "systems.json"
RESOLVED_MANIFEST = AUDIO_DIR / "manifest_resolved.json"

# Status vocabulary used everywhere (CSV, JSON, dashboard).
STATUS_OK = "ok"
STATUS_NOT_RUN = "not_run"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"
STATUS_BLOCKED_CREDENTIALS = "blocked_missing_credentials"
STATUS_BLOCKED_DEPENDENCY = "blocked_missing_dependency"
STATUS_BLOCKED_DATA = "blocked_data_unavailable"
STATUS_INVALID_OUTPUT = "invalid_output"
STATUS_EMPTY_OUTPUT = "empty_output"
STATUS_MISSING_PREDICTION = "missing_prediction"

FAILURE_STATUSES = {
    STATUS_FAILED, STATUS_TIMEOUT, STATUS_INVALID_OUTPUT,
    STATUS_MISSING_PREDICTION,
}


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=os.environ.get("VA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return logging.getLogger(name)


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def load_manifest() -> list[dict]:
    """Return the resolved recording manifest written by prepare_data.py."""
    rel = RESOLVED_MANIFEST.relative_to(ROOT)
    if not RESOLVED_MANIFEST.exists():
        raise FileNotFoundError(
            f"{rel} not found. Run scripts/prepare_data.py first "
            "(requires network access to AMI / VoxConverse)."
        )
    manifest = read_json(RESOLVED_MANIFEST)
    if not manifest.get("recordings"):
        failed = manifest.get("failed_recordings") or []
        cause = ""
        if failed:
            err = str(failed[0].get("error", ""))
            cause = " First preparation error: " + (err[:160] + "..." if len(err) > 160 else err)
        raise ValueError(f"{rel} lists 0 prepared recordings "
                         f"({len(failed)} failed in prepare_data.py).{cause}")
    return manifest["recordings"]


def load_systems() -> list[dict]:
    return read_json(SYSTEMS_CONFIG)["systems"]


def wav_duration(path: Path) -> float:
    """Duration in seconds of a PCM WAV file (stdlib only)."""
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        if rate <= 0:
            raise ValueError(f"Invalid sample rate in {path}")
        return frames / float(rate)


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as wf:
        return {
            "sample_rate": wf.getframerate(),
            "channels": wf.getnchannels(),
            "sample_width_bytes": wf.getsampwidth(),
            "duration_s": wf.getnframes() / float(wf.getframerate()),
        }


def check_audio_compatible(path: Path, expected_rate: int = 16000,
                           expected_channels: int = 1) -> list[str]:
    """Return a list of problems (empty list means compatible)."""
    problems: list[str] = []
    if not path.exists():
        return [f"audio file missing: {path}"]
    try:
        info = wav_info(path)
    except (wave.Error, EOFError, ValueError) as exc:
        return [f"unreadable WAV ({exc}); expected 16-bit PCM WAV"]
    if info["sample_rate"] != expected_rate:
        problems.append(f"sample rate {info['sample_rate']} != {expected_rate}")
    if info["channels"] != expected_channels:
        problems.append(f"channels {info['channels']} != {expected_channels}")
    if info["sample_width_bytes"] != 2:
        problems.append(f"sample width {info['sample_width_bytes']} bytes != 2")
    if info["duration_s"] <= 0:
        problems.append("zero-length audio")
    return problems


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"Required executable '{name}' not found on PATH.")
    return path


def run_cmd(cmd: list[str], timeout: float | None = None,
            cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True,
        timeout=timeout, check=False,
    )


def retry(fn: Callable[[], Any], *, attempts: int = 3, base_delay: float = 2.0,
          retry_on: tuple[type[BaseException], ...] = (Exception,),
          logger: logging.Logger | None = None) -> Any:
    """Call fn with exponential backoff. Re-raises the last exception."""
    last: BaseException | None = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except retry_on as exc:  # noqa: PERF203
            last = exc
            if i == attempts:
                break
            delay = base_delay * (2 ** (i - 1))
            if logger:
                logger.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                               i, attempts, exc, delay)
            time.sleep(delay)
    assert last is not None
    raise last


@dataclass
class RecordingRun:
    """Outcome of running one system on one recording."""
    recording_id: str
    status: str = STATUS_NOT_RUN
    processing_seconds: float | None = None
    audio_seconds: float | None = None
    started_utc: str | None = None
    finished_utc: str | None = None
    attempts: int = 0
    error: str | None = None
    notes: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float | None:
        if self.processing_seconds is None or not self.audio_seconds:
            return None
        return self.processing_seconds / self.audio_seconds


@dataclass
class SystemRunStatus:
    """Written to predictions/<system>/run_status.json by every runner."""
    system_id: str
    runner: str
    track: str
    speaker_count_condition: str
    model_identifier: str
    model_revision: str | None = None
    package_versions: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)
    model_load_seconds: float | None = None
    warmup_seconds: float | None = None
    run_started_utc: str | None = None
    run_finished_utc: str | None = None
    blocked_reason: str | None = None
    recordings: list[RecordingRun] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for rec, obj in zip(d["recordings"], self.recordings):
            rec["rtf"] = obj.rtf
        return d

    def save(self, out_dir: Path) -> Path:
        path = out_dir / "run_status.json"
        write_json(path, self.to_dict())
        return path


def blocked_status(system_id: str, runner: str, track: str, condition: str,
                   model_identifier: str, reason: str, status: str,
                   recordings: Iterable[dict]) -> SystemRunStatus:
    """Build a run-status record where every recording is blocked."""
    st = SystemRunStatus(system_id=system_id, runner=runner, track=track,
                         speaker_count_condition=condition,
                         model_identifier=model_identifier,
                         blocked_reason=reason,
                         run_started_utc=utc_now(), run_finished_utc=utc_now())
    for r in recordings:
        st.recordings.append(RecordingRun(recording_id=r["recording_id"],
                                          status=status, error=reason,
                                          audio_seconds=r.get("duration_s")))
    return st


def planned_recordings() -> list[dict]:
    """Declared recordings (used only to label blocked runs before data prep)."""
    cfg = read_json(RECORDINGS_CONFIG)
    return [{"recording_id": r["recording_id"].replace("{vox_id}", f"slot{r.get('slot')}" if r.get("slot") else "AUTO"), "duration_s": None}
            for r in cfg["recordings"]]


def manifest_or_blocked(system_id: str, runner: str, track: str, condition: str,
                        model_identifier: str, out_dir: Path) -> list[dict] | None:
    """Return the resolved manifest, or write a blocked run_status and return None."""
    try:
        return load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        reason = f"data not prepared: {exc}"
        get_logger(runner).error(reason)
        out_dir.mkdir(parents=True, exist_ok=True)
        blocked_status(system_id, runner, track, condition, model_identifier, reason,
                       STATUS_BLOCKED_DATA, planned_recordings()).save(out_dir)
        return None
