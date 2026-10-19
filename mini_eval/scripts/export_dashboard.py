#!/usr/bin/env python3
"""Export mini-benchmark CSVs to dashboard/data/benchmark_results.js.

The dashboard loads this file with a plain <script> tag (works from file://).
Values are copied verbatim from the CSVs; "not_run" stays "not_run".
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRED_DIR, RESOLVED_MANIFEST, RESULTS_DIR, ROOT, load_systems, read_json, utc_now  # noqa: E402
from rttm_utils import read_rttm  # noqa: E402

DASH = ROOT.parent / "dashboard" / "data" / "benchmark_results.js"


def rows(name: str) -> list[dict]:
    p = RESULTS_DIR / name
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def segments(path: Path) -> list | None:
    """[[start, end, speaker], ...] from an RTTM, or None if absent/invalid."""
    if not path.exists():
        return None
    try:
        segs, _ = read_rttm(path)
    except Exception:  # noqa: BLE001
        return None
    return [[round(x.start, 3), round(x.end, 3), x.speaker] for x in sorted(segs, key=lambda x: x.start)]


def recordings_block(scores: dict) -> list[dict]:
    """Recording-level facts the dashboard aggregates client-side (real values only)."""
    manifest = read_json(RESOLVED_MANIFEST, default={}) if RESOLVED_MANIFEST.exists() else {}
    recs = manifest.get("recordings") or []
    prepared = bool(recs)
    if not prepared:
        recs = [dict(r, recording_id=r["recording_id"].replace("{vox_id}", f"slot{r.get('slot')}" if r.get("slot") else "AUTO"),
                     tags={k: v for k, v in (r.get("tags_declared") or {}).items()})
                for r in read_json(ROOT / "configs" / "recordings.json")["recordings"]]
    systems = load_systems()
    runs = {s["system_id"]: {r["recording_id"]: r for r in
                             read_json(PRED_DIR / s["system_id"] / "run_status.json", default={}).get("recordings", [])}
            for s in systems}
    out = []
    for rec in recs:
        rid = rec["recording_id"]
        tags = dict({"recording_type": "unknown", "n_speakers": "unknown", "overlap_level": "unknown",
                     "audio_quality": "unknown", "mic_distance": "unknown"}, **(rec.get("tags") or {}))
        entry = {"recording_id": rid, "prepared": prepared, "source": rec.get("source"),
                 "dataset_version": rec.get("dataset_version"), "source_url": rec.get("source_url"),
                 "license": rec.get("license"), "excerpt": [rec.get("excerpt_start_s"), rec.get("excerpt_end_s")],
                 "duration_s": rec.get("duration_s"), "n_reference_speakers": rec.get("n_reference_speakers"),
                 "overlap_ratio": rec.get("overlap_ratio"), "tags": tags, "tag_evidence": rec.get("tag_evidence", {}),
                 "selection_reason": rec.get("selection_reason"), "audio_sha256": rec.get("audio_sha256"),
                 "reference_segments": segments(ROOT / rec["reference_rttm"]) if prepared else None, "results": {}}
        for s in systems:
            sid = s["system_id"]
            sc = next((r for r in scores.get("systems", {}).get(sid, {}).get("recordings", [])
                       if r.get("recording_id") == rid), {})
            run = runs.get(sid, {}).get(rid, {})
            entry["results"][sid] = {
                "run_status": run.get("status", "not_run"), "score_status": sc.get("score_status", "not_run"),
                "failed": bool(sc.get("failure_policy_applied")), "error": run.get("error"),
                "DER": sc.get("DER"), "JER": sc.get("JER"), "components": sc.get("components"),
                "overlap_region": sc.get("overlap_region"), "non_overlap": sc.get("non_overlap"),
                "n_predicted_speakers": sc.get("n_predicted_speakers"),
                "boundary_distances_s": (sc.get("boundary_error") or {}).get("distances_s"),
                "processing_seconds": run.get("processing_seconds"), "audio_seconds": run.get("audio_seconds"),
                "segments": segments(PRED_DIR / sid / f"{rid}.rttm") if sc.get("components") else None,
            }
        out.append(entry)
    return out


def dataset_version(recs: list[dict]) -> str:
    """Describe the recordings actually present (counted, never hard-coded)."""
    from collections import Counter
    c = Counter(r.get("source") or "unknown" for r in recs)
    return " + ".join(f"{src} ({n} excerpt{'s' if n != 1 else ''})" for src, n in c.items()) or "no recordings"


def system_block() -> list[dict]:
    out = []
    for s in load_systems():
        st = read_json(PRED_DIR / s["system_id"] / "run_status.json", default={})
        name = s["display_name"]
        if s["system_id"] == "api-system" and st.get("model_identifier") and not st.get("blocked_reason"):
            name = st["model_identifier"]          # e.g. "Deepgram: nova-3 + diarize_model=v2"
        out.append({"system_id": s["system_id"], "name": name, "company": s["company"],
                    "deployment": s["deployment"], "mode": s["mode"], "track": s["track"],
                    "declared_model": s["model_identifier"], "model_revision": st.get("model_revision"),
                    "package_versions": st.get("package_versions"), "hardware": st.get("hardware"),
                    "settings": st.get("settings") or s.get("settings"), "speaker_count_condition": s["speaker_count_condition"],
                    "run_started_utc": st.get("run_started_utc"), "run_finished_utc": st.get("run_finished_utc"),
                    "model_load_seconds": st.get("model_load_seconds"), "blocked_reason": st.get("blocked_reason")})
    return out


def main() -> int:
    summary = rows("summary.csv")
    measured = any(r.get("Pooled DER") not in ("not_run", "", None) for r in summary)
    scores = read_json(RESULTS_DIR / "scores_raw.json", default={"systems": {}})
    recs = recordings_block(scores)
    n_sys_scored = sum(1 for s in load_systems()
                       if any(r["results"][s["system_id"]]["components"] for r in recs))
    status = "not_run" if not measured else ("measured" if n_sys_scored >= 2 else "partial")
    summaries = read_json(RESULTS_DIR / "recording_summaries.json", default={})
    payload = {
        "exported_utc": utc_now(),
        "status": status,
        "provisional": True,
        "evaluated_utc": scores.get("created_utc") if measured else None,
        "n_recordings": len(recs),
        "dataset_version": dataset_version(recs),
        "scoring": scores.get("settings") or {"collar_s": 0.0, "overlap": "included"},
        "scorer": scores.get("scorer"),
        "min_recordings_for_ci": 20,
        "status_note": ({"measured": f"Measured on {len(recs)} short public excerpts. Provisional and exploratory: not a general ranking.",
                         "partial": "Partially measured: fewer than two systems produced scored predictions.",
                         "not_run": "Benchmark code is implemented and unit-tested on synthetic data, but no model has been run and nothing has been scored."}[status]),
        "systems": system_block(),
        "recordings": recs,
        "recording_summaries": summaries.get("recordings", []),
        "cross_recording": summaries.get("cross_recording", []),
        "blockers": summaries.get("blockers", []),
        "summary": summary,
        "per_recording": rows("per_recording.csv"),
        "error_breakdown": rows("error_breakdown.csv"),
        "pairwise": rows("pairwise_bootstrap.csv"),
        "configs": read_json(RESULTS_DIR / "system_configs.json", default={}),
    }
    DASH.parent.mkdir(parents=True, exist_ok=True)
    DASH.write_text("/* Generated by mini_eval/scripts/export_dashboard.py. Do not edit by hand. */\n"
                    "window.VA_BENCHMARK = " + json.dumps(payload, indent=1) + ";\n", encoding="utf-8")
    print(f"wrote {DASH} (status={payload['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
