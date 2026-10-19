#!/usr/bin/env python3
"""Run ONE commercial diarization API on every prepared recording.

Providers (choose with --provider):
  deepgram    POST /v1/listen?model=nova-3&diarize_model=v2   (env DEEPGRAM_API_KEY)
  assemblyai  /v2/upload + /v2/transcript speaker_labels=true (env ASSEMBLYAI_API_KEY)
  pyannoteai  /v1/media/input + /v1/diarize model=precision-3  (env PYANNOTEAI_API_KEY)

Primary-track rules enforced here:
  * no speaker-count parameters are sent (no speakers_expected, numSpeakers, ...)
  * same language hint ("en") for every STT-based provider
  * fixed timeouts and retry policy (below), decided before running
  * every raw response is archived under predictions/api-system/raw/ for audit
  * the resolved model/diarizer version reported by the provider is recorded

Retry policy: up to --max-attempts attempts per request on network errors,
HTTP 429 and HTTP 5xx, exponential backoff starting at 2 s. Other 4xx are
not retried. A recording that exceeds --timeout seconds end-to-end is marked
"timeout" and counted as a failure (scored as an empty prediction).

Endpoint shapes follow each provider's public documentation at the time of
writing; they must be re-verified on the execution date and any change
recorded in run_status.json -> settings.notes.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (PRED_DIR, ROOT, STATUS_BLOCKED_CREDENTIALS,  # noqa: E402
                    STATUS_BLOCKED_DEPENDENCY, STATUS_EMPTY_OUTPUT, STATUS_FAILED,
                    STATUS_OK, STATUS_TIMEOUT, RecordingRun, SystemRunStatus,
                    blocked_status, check_audio_compatible, get_logger, load_manifest, manifest_or_blocked,
                    utc_now, write_json)
from rttm_utils import Segment, sanitize_speaker, words_to_segments, write_rttm  # noqa: E402

log = get_logger("run_api")
SYSTEM_ID = "api-system"

PROVIDERS = {
    "deepgram": {"env": "DEEPGRAM_API_KEY", "company": "Deepgram",
                 "model": "nova-3 + diarize_model=v2", "granularity": "word-level",
                 "overlap": False},
    "assemblyai": {"env": "ASSEMBLYAI_API_KEY", "company": "AssemblyAI",
                   "model": "default speech model + speaker_labels=true",
                   "granularity": "word/utterance-level", "overlap": False},
    "pyannoteai": {"env": "PYANNOTEAI_API_KEY", "company": "pyannoteAI",
                   "model": "precision-3", "granularity": "segment-level",
                   "overlap": True},
}


class Deadline(Exception):
    pass


class Http:
    """Thin requests wrapper implementing the declared retry policy."""

    def __init__(self, max_attempts: int, deadline: float):
        import requests
        self.requests = requests
        self.max_attempts = max_attempts
        self.deadline = deadline
        self.attempts = 0

    def call(self, method: str, url: str, **kw):
        last = None
        for i in range(1, self.max_attempts + 1):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise Deadline("end-to-end timeout reached")
            kw.setdefault("timeout", (15, max(5.0, min(remaining, 600.0))))
            self.attempts += 1
            try:
                r = self.requests.request(method, url, **kw)
            except (self.requests.ConnectionError, self.requests.Timeout) as exc:
                last = exc
            else:
                if r.status_code == 429 or r.status_code >= 500:
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                elif r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                else:
                    return r
            if i < self.max_attempts:
                time.sleep(min(2.0 * 2 ** (i - 1), max(0.0, self.deadline - time.monotonic())))
        raise RuntimeError(f"request failed after {self.max_attempts} attempts: {last}")


# ---------------------------------------------------------------------------
# Provider adapters: return (segments, raw_json, version_info)
# ---------------------------------------------------------------------------
def run_deepgram(http: Http, key: str, wav: Path, rid: str, gap: float, args):
    params = {"model": args.deepgram_model, "diarize_model": args.deepgram_diarizer,
              "language": "en", "punctuate": "false", "utterances": "false"}
    with wav.open("rb") as fh:
        data = fh.read()
    r = http.call("POST", "https://api.deepgram.com/v1/listen", params=params, data=data,
                  headers={"Authorization": f"Token {key}", "Content-Type": "audio/wav"})
    js = r.json()
    words = js["results"]["channels"][0]["alternatives"][0].get("words", [])
    md = js.get("metadata", {})
    version = {"request_id": md.get("request_id"), "model_info": md.get("model_info"),
               "diarize_info": md.get("diarize_info"), "params": params}
    if "diarize_info" not in md:
        version["warning"] = "diarize_info absent: diarizer may not have run"
    return words_to_segments(words, rid, max_gap=gap), js, version


def run_assemblyai(http: Http, key: str, wav: Path, rid: str, gap: float, args):
    base, hdr = "https://api.assemblyai.com", {"authorization": key}
    with wav.open("rb") as fh:
        up = http.call("POST", f"{base}/v2/upload", headers=hdr, data=fh.read()).json()
    body = {"audio_url": up["upload_url"], "speaker_labels": True, "language_code": "en"}
    job = http.call("POST", f"{base}/v2/transcript", headers=hdr, json=body).json()
    while True:
        js = http.call("GET", f"{base}/v2/transcript/{job['id']}", headers=hdr).json()
        if js.get("status") == "completed":
            break
        if js.get("status") == "error":
            raise RuntimeError(f"AssemblyAI job error: {js.get('error')}")
        if time.monotonic() > http.deadline:
            raise Deadline("polling exceeded end-to-end timeout")
        time.sleep(args.poll_interval)
    words = [{"start": w["start"] / 1000.0, "end": w["end"] / 1000.0, "speaker": w.get("speaker")}
             for w in js.get("words") or []]
    version = {"transcript_id": js.get("id"), "speech_model": js.get("speech_model"),
               "request_body": {k: v for k, v in body.items() if k != "audio_url"}}
    return words_to_segments(words, rid, max_gap=gap), js, version


def run_pyannoteai(http: Http, key: str, wav: Path, rid: str, gap: float, args):
    base = "https://api.pyannote.ai/v1"
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    media_url = f"media://voicearena-{rid}"
    up = http.call("POST", f"{base}/media/input", headers=hdr, json={"url": media_url}).json()
    with wav.open("rb") as fh:
        http.call("PUT", up["url"], data=fh.read())
    body = {"url": media_url, "model": args.pyannoteai_model}
    job = http.call("POST", f"{base}/diarize", headers=hdr, json=body).json()
    job_id = job.get("jobId") or job.get("id")
    while True:
        js = http.call("GET", f"{base}/jobs/{job_id}", headers=hdr).json()
        st = js.get("status")
        if st == "succeeded":
            break
        if st in {"failed", "canceled"}:
            raise RuntimeError(f"pyannoteAI job {st}: {json.dumps(js)[:300]}")
        if time.monotonic() > http.deadline:
            raise Deadline("polling exceeded end-to-end timeout")
        time.sleep(args.poll_interval)
    turns = (js.get("output") or {}).get("diarization") or []
    segs = [Segment(rid, float(t["start"]), float(t["end"]) - float(t["start"]),
                    f"spk_{sanitize_speaker(t['speaker'])}")
            for t in turns if float(t["end"]) > float(t["start"])]
    version = {"job_id": job_id, "model": args.pyannoteai_model, "request_body": body}
    return segs, js, version


ADAPTERS = {"deepgram": run_deepgram, "assemblyai": run_assemblyai,
            "pyannoteai": run_pyannoteai}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=sorted(PROVIDERS))
    ap.add_argument("--region", default="provider default",
                    help="API region/endpoint used (recorded; endpoint override not implemented)")
    ap.add_argument("--timeout", type=float, default=900.0, help="end-to-end seconds per file")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--poll-interval", type=float, default=3.0)
    ap.add_argument("--word-gap", type=float, default=0.2,
                    help="bridge same-speaker word gaps <= this (s)")
    ap.add_argument("--deepgram-model", default="nova-3")
    ap.add_argument("--deepgram-diarizer", default="v2", help="pin, never 'latest'")
    ap.add_argument("--pyannoteai-model", default="precision-3")
    args = ap.parse_args()

    prov = PROVIDERS[args.provider]
    out_dir = PRED_DIR / SYSTEM_ID
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    track, condition = "audio_only_batch", "unknown (no speaker-count parameters sent)"
    model_identifier = f"{prov['company']}: {prov['model']}"
    recordings = manifest_or_blocked(SYSTEM_ID, "run_api.py", track, condition,
                                     model_identifier, out_dir)
    if recordings is None:
        return 5

    try:
        import requests
    except ImportError as exc:
        blocked_status(SYSTEM_ID, "run_api.py", track, condition, model_identifier,
                       f"requests missing: {exc}", STATUS_BLOCKED_DEPENDENCY,
                       recordings).save(out_dir)
        return 2
    key = os.environ.get(prov["env"])
    if not key:
        reason = f"{prov['env']} is not set; API run not executed."
        log.error(reason)
        blocked_status(SYSTEM_ID, "run_api.py", track, condition, model_identifier,
                       reason, STATUS_BLOCKED_CREDENTIALS, recordings).save(out_dir)
        return 3

    status = SystemRunStatus(
        system_id=SYSTEM_ID, runner=f"run_api.py --provider {args.provider}", track=track,
        speaker_count_condition=condition, model_identifier=model_identifier,
        package_versions={"requests": requests.__version__, "python": platform.python_version()},
        hardware={"type": "remote API", "region": args.region, "client_os": platform.platform()},
        settings={"provider": args.provider, "timeout_s": args.timeout,
                  "max_attempts": args.max_attempts, "word_merge_gap_s": args.word_gap,
                  "output_granularity": prov["granularity"], "overlap_capable": prov["overlap"],
                  "timing": "end-to-end wall clock incl. upload, queueing, polling and network",
                  "deepgram_model": args.deepgram_model,
                  "deepgram_diarizer": args.deepgram_diarizer,
                  "pyannoteai_model": args.pyannoteai_model,
                  "notes": "verify endpoint shapes against provider docs on the run date"},
        run_started_utc=utc_now(),
    )
    versions = {}
    for rec in recordings:
        rid = rec["recording_id"]
        run = RecordingRun(rid, audio_seconds=rec["duration_s"], started_utc=utc_now())
        wav = ROOT / rec["audio_path"]
        problems = check_audio_compatible(wav)
        if problems:
            run.status, run.error = STATUS_FAILED, f"incompatible audio: {problems}"
            status.recordings.append(run)
            continue
        http = Http(args.max_attempts, time.monotonic() + args.timeout)
        t0 = time.perf_counter()
        try:
            segs, raw, version = ADAPTERS[args.provider](http, key, wav, rid, args.word_gap, args)
            run.processing_seconds = round(time.perf_counter() - t0, 3)
            write_json(raw_dir / f"{rid}.json", raw)
            write_rttm(segs, out_dir / f"{rid}.rttm")
            versions[rid] = version
            run.status = STATUS_OK if segs else STATUS_EMPTY_OUTPUT
            run.extra["n_predicted_speakers"] = len({s.speaker for s in segs})
        except Deadline as exc:
            run.status, run.error = STATUS_TIMEOUT, str(exc)
        except Exception as exc:  # noqa: BLE001
            run.status, run.error = STATUS_FAILED, f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s\n%s", rid, exc, traceback.format_exc())
        run.attempts = http.attempts
        run.finished_utc = utc_now()
        status.recordings.append(run)
        log.info("%s: %s", rid, run.status)

    status.model_revision = json.dumps(versions, default=str)[:4000] if versions else None
    status.run_finished_utc = utc_now()
    status.save(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
