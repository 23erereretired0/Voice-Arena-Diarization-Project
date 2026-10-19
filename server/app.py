#!/usr/bin/env python3
"""Voice Arena local backend: serves the dashboard and runs diarization on uploads.

Standard library only (no web framework). Start it from the repository root:

    python server/app.py            # http://127.0.0.1:8765
    python server/app.py --port 9000 --host 127.0.0.1

Endpoints
  GET  /                 dashboard (static files from ./dashboard)
  GET  /api/status       which systems / scorer are usable, limits, privacy note
  POST /api/diarize      body = raw audio bytes; headers X-Filename, X-System
                         -> speaker segments, RTTM text, model/version, runtime
  POST /api/score        JSON {"prediction_rttm", "reference_rttm", "duration_s"}
                         -> DER, JER, components (same settings as the benchmark)

Privacy and limits
  * Audio is processed only on the machine running this server; nothing is sent elsewhere.
  * Uploads are written to a temporary directory that is deleted after each request.
  * Personal uploads are never added to the benchmark or the public ranking.
  * Limits: VA_MAX_UPLOAD_MB (default 50) and VA_MAX_DURATION_S (default 1200).

Models
  * pyannote-community-1 needs pyannote.audio 4.x plus HF_TOKEN (gated model) or
    VA_PYANNOTE_MODEL_PATH pointing to a local clone. If unavailable, /api/diarize
    returns HTTP 503 with the exact reason. Nothing is simulated.
  * Scoring needs a dscore checkout (VA_DSCORE_DIR, default mini_eval/third_party/dscore) and Perl.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "dashboard"
sys.path.insert(0, str(REPO / "mini_eval" / "scripts"))

from rttm_utils import Segment, parse_rttm_lines, validate_rttm, write_rttm, write_uem  # noqa: E402

MODEL_ID = "pyannote/speaker-diarization-community-1"
UPLOAD_FILE_ID = "upload"
SAFE_SUFFIX = re.compile(r"^\.(wav|flac|mp3|m4a|ogg|opus|webm|aac)$", re.I)


def limits() -> dict:
    return {"max_upload_mb": float(os.environ.get("VA_MAX_UPLOAD_MB", "50")),
            "max_duration_s": float(os.environ.get("VA_MAX_DURATION_S", "1200"))}


class ApiError(Exception):
    def __init__(self, status: HTTPStatus, message: str, fix: str | None = None):
        super().__init__(message)
        self.status, self.message, self.fix = status, message, fix


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------
def decode_to_wav(src: Path, dst: Path) -> float:
    """Decode any ffmpeg-readable file to 16 kHz mono PCM WAV; return duration (s)."""
    try:
        proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                               "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)],
                              capture_output=True, text=True, timeout=300, check=False)
    except FileNotFoundError as exc:
        raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "ffmpeg is not installed on the server.",
                       "Install ffmpeg and make sure it is on PATH.") from exc
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size <= 44:
        raise ApiError(HTTPStatus.BAD_REQUEST, "The file could not be decoded as audio.",
                       "Upload a WAV, FLAC, MP3, M4A, OGG/Opus or WebM audio file.")
    import wave
    with wave.open(str(dst), "rb") as wf:
        dur = wf.getnframes() / float(wf.getframerate())
    if dur <= 0.5:
        raise ApiError(HTTPStatus.BAD_REQUEST, "The audio is shorter than 0.5 s.")
    return dur


# ---------------------------------------------------------------------------
# pyannote (loaded once, lazily)
# ---------------------------------------------------------------------------
class Pyannote:
    _lock = threading.Lock()
    _pipeline = None
    _info: dict = {}

    @classmethod
    def availability(cls) -> tuple[bool, str]:
        try:
            import pyannote.audio  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            return False, (f"pyannote.audio/torch not installed ({exc}). "
                           "Install with: pip install -r mini_eval/requirements-pyannote.txt")
        if not (os.environ.get("HF_TOKEN") or os.environ.get("VA_PYANNOTE_MODEL_PATH")):
            return False, ("HF_TOKEN is not set (the model is gated) and VA_PYANNOTE_MODEL_PATH is not set. "
                           "Accept the model conditions on Hugging Face, then set HF_TOKEN and restart the server.")
        return True, "ready"

    @classmethod
    def run(cls, wav: Path) -> tuple[list[Segment], dict]:
        ok, why = cls.availability()
        if not ok:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, why)
        import torch
        import pyannote.audio
        from pyannote.audio import Pipeline
        with cls._lock:
            if cls._pipeline is None:
                t0 = time.perf_counter()
                path = os.environ.get("VA_PYANNOTE_MODEL_PATH")
                try:
                    pipe = Pipeline.from_pretrained(path) if path else \
                        Pipeline.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
                except Exception as exc:  # noqa: BLE001
                    raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, f"Model load failed: {exc}") from exc
                if pipe is None:
                    raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, "Model access not granted on Hugging Face.")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                pipe.to(torch.device(device))
                revision = None
                try:
                    from huggingface_hub import HfApi
                    revision = None if path else HfApi().model_info(MODEL_ID, token=os.environ.get("HF_TOKEN")).sha
                except Exception:  # noqa: BLE001
                    pass
                cls._pipeline = pipe
                cls._info = {"system": "pyannote-community-1", "model": path or MODEL_ID,
                             "model_revision": revision or "unrecorded",
                             "package_versions": {"pyannote.audio": pyannote.audio.__version__,
                                                  "torch": torch.__version__, "python": platform.python_version()},
                             "device": device, "model_load_seconds": round(time.perf_counter() - t0, 3)}
            import wave
            import numpy as np
            with wave.open(str(wav), "rb") as wf:
                sr = wf.getframerate()
                pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            audio = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)  # in memory: no torchcodec needed
            t0 = time.perf_counter()
            out = cls._pipeline({"waveform": audio, "sample_rate": sr})
            elapsed = time.perf_counter() - t0
        ann = getattr(out, "speaker_diarization", out)
        segs = [Segment(UPLOAD_FILE_ID, float(t.start), float(t.end - t.start), str(spk).replace(" ", "_"))
                for t, _, spk in ann.itertracks(yield_label=True) if t.end > t.start]
        return segs, dict(cls._info, processing_seconds=round(elapsed, 3))


SYSTEMS = {"pyannote-community-1": Pyannote}


# ---------------------------------------------------------------------------
# Scoring (same scorer and settings as the benchmark)
# ---------------------------------------------------------------------------
def dscore_dir() -> Path:
    return Path(os.environ.get("VA_DSCORE_DIR", REPO / "mini_eval" / "third_party" / "dscore"))


def scorer_availability() -> tuple[bool, str]:
    d = dscore_dir()
    if not (d / "score.py").exists() or not (d / "scorelib" / "md-eval-22.pl").exists():
        return False, (f"dscore not found at {d}. Clone it: git clone https://github.com/nryant/dscore.git "
                       "mini_eval/third_party/dscore (or set VA_DSCORE_DIR).")
    return True, "ready"


def normalize_rttm(text: str, label: str) -> list[Segment]:
    try:
        segs, _ = parse_rttm_lines(text.splitlines(), source=label)
    except ValueError as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Invalid {label} RTTM: {exc}") from exc
    if not segs:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"The {label} RTTM contains no SPEAKER segments.")
    ids = {s.file_id for s in segs}
    if len(ids) > 1:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"The {label} RTTM contains several file IDs {sorted(ids)}; use one recording.")
    return [Segment(UPLOAD_FILE_ID, s.start, s.duration, s.speaker) for s in segs]


def score(body: dict) -> dict:
    from score import DER_MATCH_TOL_PP, PRIMARY_COLLAR, Scorer, overlap_uem
    try:
        duration = float(body.get("duration_s"))
    except (TypeError, ValueError) as exc:
        raise ApiError(HTTPStatus.BAD_REQUEST, "duration_s (audio duration in seconds) is required.") from exc
    pred = normalize_rttm(str(body.get("prediction_rttm", "")), "prediction")
    ref = normalize_rttm(str(body.get("reference_rttm", "")), "reference")
    with tempfile.TemporaryDirectory(prefix="va_score_") as td:
        td = Path(td)
        rp, sp, up = td / "ref.rttm", td / "sys.rttm", td / "all.uem"
        write_rttm(ref, rp)
        write_rttm(pred, sp)
        write_uem(up, UPLOAD_FILE_ID, 0.0, duration)
        for path, label in ((rp, "reference"), (sp, "prediction")):
            rep = validate_rttm(path, UPLOAD_FILE_ID, duration)
            if not rep.ok:
                raise ApiError(HTTPStatus.BAD_REQUEST, f"{label} RTTM failed validation: {'; '.join(rep.errors)}")
        ok, why = scorer_availability()
        if not ok:
            raise ApiError(HTTPStatus.SERVICE_UNAVAILABLE, why)
        sc = Scorer(dscore_dir(), os.environ.get("DSCORE_PYTHON", sys.executable))
        ds = sc.dscore(rp, sp, up, PRIMARY_COLLAR)[UPLOAD_FILE_ID]
        comp = sc.md_eval_run(rp, sp, up, PRIMARY_COLLAR)
        tot = comp["scored_speaker_s"]
        der_c = 100 * (comp["missed_speaker_s"] + comp["false_alarm_speaker_s"] + comp["speaker_error_s"]) / tot
        res = {"DER": ds["DER"], "JER": ds["JER"],
               "missed_pct": 100 * comp["missed_speaker_s"] / tot,
               "false_alarm_pct": 100 * comp["false_alarm_speaker_s"] / tot,
               "confusion_pct": 100 * comp["speaker_error_s"] / tot,
               "components_reconcile": abs(der_c - ds["DER"]) <= DER_MATCH_TOL_PP,
               "n_reference_speakers": len({s.speaker for s in ref}),
               "n_predicted_speakers": len({s.speaker for s in pred}),
               "settings": {"collar_s": PRIMARY_COLLAR, "overlap": "included", "uem": f"[0, {duration:.3f}]"},
               "scorer_commit": sc.commit,
               "note": "Personal recording: scored against your reference only. Not part of the benchmark or ranking."}
        ov = td / "ov.uem"
        if overlap_uem(ref, UPLOAD_FILE_ID, duration, ov):
            o = sc.md_eval_run(rp, sp, ov, PRIMARY_COLLAR)
            res["overlap_DER"] = 100 * (o["missed_speaker_s"] + o["false_alarm_speaker_s"] + o["speaker_error_s"]) / o["scored_speaker_s"]
        n = sc.md_eval_run(rp, sp, up, PRIMARY_COLLAR, single_speaker_only=True)
        res["non_overlap_DER"] = 100 * (n["missed_speaker_s"] + n["false_alarm_speaker_s"] + n["speaker_error_s"]) / n["scored_speaker_s"]
        return res


def diarize(raw: bytes, filename: str, system: str) -> dict:
    lim = limits()
    if not raw:
        raise ApiError(HTTPStatus.BAD_REQUEST, "No audio data received.")
    if len(raw) > lim["max_upload_mb"] * 1024 * 1024:
        raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"File exceeds the {lim['max_upload_mb']:.0f} MB limit.")
    if system not in SYSTEMS:
        raise ApiError(HTTPStatus.BAD_REQUEST, f"Unknown system '{system}'. Available: {sorted(SYSTEMS)}")
    suffix = Path(filename or "audio").suffix
    suffix = suffix if SAFE_SUFFIX.match(suffix or "") else ".bin"
    with tempfile.TemporaryDirectory(prefix="va_upload_") as td:   # deleted after the request
        src, wav = Path(td) / f"in{suffix}", Path(td) / "audio16k.wav"
        src.write_bytes(raw)
        duration = decode_to_wav(src, wav)
        if duration > lim["max_duration_s"]:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                           f"Audio is {duration:.0f} s; the limit is {lim['max_duration_s']:.0f} s.")
        segs, info = SYSTEMS[system].run(wav)
        rttm_path = Path(td) / "pred.rttm"
        write_rttm(segs, rttm_path)
        rttm = rttm_path.read_text(encoding="utf-8")
    return dict(info, status="ok", filename=Path(filename or "audio").name, duration_s=round(duration, 3),
                rtf=round(info["processing_seconds"] / duration, 4),
                n_speakers=len({s.speaker for s in segs}),
                segments=[{"start": round(s.start, 3), "end": round(s.end, 3), "speaker": s.speaker}
                          for s in sorted(segs, key=lambda x: x.start)],
                rttm=rttm,
                note="No DER or rank: this recording has no reference annotation unless you upload one.")


def status() -> dict:
    systems = {}
    for sid, cls in SYSTEMS.items():
        ok, why = cls.availability()
        systems[sid] = {"available": ok, "detail": why}
    ok, why = scorer_availability()
    return {"status": "ok", "systems": systems, "scorer": {"available": ok, "detail": why}, "limits": limits(),
            "privacy": "Audio is processed on this machine only and deleted after each request. "
                       "Uploads are never added to the benchmark."}


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------
class Handler(SimpleHTTPRequestHandler):
    server_version = "VoiceArena/1.0"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DASHBOARD), **kw)

    def log_message(self, fmt, *args):  # quieter, still logs to stderr
        sys.stderr.write("[voice-arena] " + (fmt % args) + "\n")

    def _json(self, code: int, obj: dict) -> None:
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _body(self, max_bytes: int) -> bytes:
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(HTTPStatus.LENGTH_REQUIRED, "Content-Length header is required.")
        if n > max_bytes:
            raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"Request exceeds {max_bytes // (1024 * 1024)} MB.")
        return self.rfile.read(n) if n else b""

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/api/status":
            return self._json(200, status())
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        route = self.path.split("?")[0]
        try:
            if route == "/api/diarize":
                max_b = int(limits()["max_upload_mb"] * 1024 * 1024)
                raw = self._body(max_b + 1)
                return self._json(200, diarize(raw, self.headers.get("X-Filename", "audio"),
                                               self.headers.get("X-System", "pyannote-community-1")))
            if route == "/api/score":
                try:
                    body = json.loads(self._body(10 * 1024 * 1024) or b"{}")
                except json.JSONDecodeError as exc:
                    raise ApiError(HTTPStatus.BAD_REQUEST, f"Body is not valid JSON: {exc}") from exc
                return self._json(200, score(body))
            return self._json(404, {"status": "error", "error": "Unknown endpoint."})
        except ApiError as exc:
            return self._json(int(exc.status), {"status": "error", "error": exc.message, "fix": exc.fix})
        except Exception as exc:  # noqa: BLE001
            self.log_message("unhandled error: %r", exc)
            return self._json(500, {"status": "error", "error": f"Internal error: {type(exc).__name__}: {exc}"})


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default: localhost only)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = make_server(args.host, args.port)
    print(f"Voice Arena: http://{args.host}:{srv.server_port}/  (Ctrl+C to stop)")
    print("status:", json.dumps(status()["systems"]), "| scorer:", status()["scorer"]["detail"])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
