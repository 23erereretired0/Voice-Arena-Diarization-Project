#!/usr/bin/env python3
"""Aggregate scores into the result tables required by the assignment.

Inputs : results/scores_raw.json (from score.py; may be absent),
         predictions/<system>/run_status.json, configs/*.json,
         results/environment.json (from env_report.py; optional)
Outputs: results/summary.csv, results/per_recording.csv,
         results/error_breakdown.csv, results/system_configs.json

Any value that was not measured is written as the literal string
"not_run" (never 0, never blank-filled with a guess).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (CONFIG_DIR, PRED_DIR, RECORDINGS_CONFIG, RESULTS_DIR,  # noqa: E402
                    RESOLVED_MANIFEST, STATUS_NOT_RUN, get_logger, load_systems,
                    read_json, utc_now, write_json)
from rttm_utils import percentile  # noqa: E402

log = get_logger("make_results")
NR = "not_run"
COLLAR = "0 ms"
OVERLAP = "included"

PER_REC_COLS = [
    "System", "Exact model/version", "Recording", "DER", "JER", "Missed speech",
    "False alarm", "Speaker confusion", "Runtime", "Real-time factor",
    "Predicted speaker count", "Collar", "Overlap policy", "Speaker-count condition",
    # extra, documented columns
    "Track", "Run status", "Score status", "Reference speaker count", "Audio seconds",
    "Overlap ratio", "Overlap-region DER", "Non-overlap DER", "Boundary error median (s)",
    "Boundary error p95 (s)", "Units", "Notes",
]
SUMMARY_COLS = [
    "System", "Exact model/version", "Track", "Speaker-count condition", "Collar",
    "Overlap policy", "Recordings", "Audio hours", "Pooled DER", "Macro DER", "Median DER",
    "Mean JER", "Pooled missed", "Pooled false alarm", "Pooled confusion",
    "Speaker-count MAE", "Total runtime (s)", "Pooled RTF", "Failed recordings",
    "Failure rate", "95% CI", "Ranking statement",
]
BREAKDOWN_COLS = [
    "System", "Recording", "Scored speaker time (s)", "Missed (s)", "False alarm (s)",
    "Confusion (s)", "Missed (%)", "False alarm (%)", "Confusion (%)", "DER (%)",
    "Dominant component", "Overlap-region DER (%)", "Non-overlap DER (%)",
    "DER 250 ms collar (%) [diagnostic]",
]


def fmt(x, nd=2):
    if x is None or x == "" or x == NR:
        return NR
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def comp_pct(c: dict | None, key: str):
    if not c or not c.get("scored_speaker_s"):
        return None
    return 100.0 * c[key] / c["scored_speaker_s"]


def der_of(c: dict | None):
    if not c or not c.get("scored_speaker_s"):
        return None
    return 100.0 * (c["missed_speaker_s"] + c["false_alarm_speaker_s"]
                    + c["speaker_error_s"]) / c["scored_speaker_s"]


def planned_recordings() -> list[dict]:
    if RESOLVED_MANIFEST.exists():
        recs = read_json(RESOLVED_MANIFEST).get("recordings") or []
        if recs:
            return recs
    # Not prepared yet: list the declared recordings so tables show the plan.
    return [{"recording_id": r["recording_id"].replace("{vox_id}", f"slot{r.get('slot')}" if r.get("slot") else "AUTO"), "duration_s": None}
            for r in read_json(RECORDINGS_CONFIG)["recordings"]]


def model_version(decl: dict, st: dict) -> str:
    rev = st.get("model_revision")
    ident = st.get("model_identifier") or decl.get("model_identifier")
    pk = st.get("package_versions") or {}
    pk_s = ", ".join(f"{k} {v}" for k, v in pk.items() if k != "python")
    if not st or st.get("blocked_reason"):
        return f"{decl.get('model_identifier')} (version {NR})"
    return f"{ident}; revision {rev or 'unrecorded'}; {pk_s}".strip("; ")


def main() -> int:
    scores = read_json(RESULTS_DIR / "scores_raw.json", default={"systems": {}})
    systems = load_systems()
    recs = planned_recordings()
    per_rec, summary, breakdown, configs = [], [], [], []

    for decl in systems:
        sid = decl["system_id"]
        st = read_json(PRED_DIR / sid / "run_status.json", default={})
        runs = {r["recording_id"]: r for r in st.get("recordings", [])}
        scored = {r["recording_id"]: r for r in
                  scores.get("systems", {}).get(sid, {}).get("recordings", [])}
        mv = model_version(decl, st)
        cond = st.get("speaker_count_condition") or decl["speaker_count_condition"]
        track = st.get("track") or decl["track"]
        ders, jers, mae_terms, err_rows = [], [], [], []
        tot_proc = tot_audio = 0.0
        n_fail = 0
        any_scored = False
        for rec in recs:
            rid = rec["recording_id"]
            run = runs.get(rid, {})
            sc = scored.get(rid, {})
            c = sc.get("components")
            der, jer = sc.get("DER"), sc.get("JER")
            proc, aud = run.get("processing_seconds"), run.get("audio_seconds") or rec.get("duration_s")
            rtf = (proc / aud) if (proc and aud) else None
            run_status = run.get("status", STATUS_NOT_RUN)
            score_status = sc.get("score_status", NR)
            if der is not None:
                any_scored = True
                ders.append(der)
                jers.append(jer)
                if c:
                    err_rows.append(c)
                if sc.get("failure_policy_applied"):
                    n_fail += 1
                ref_n, hyp_n = sc.get("n_reference_speakers"), sc.get("n_predicted_speakers")
                if ref_n is not None and hyp_n is not None:
                    mae_terms.append(abs(ref_n - hyp_n))
                if proc and aud:
                    tot_proc += proc
                    tot_audio += aud
            be = sc.get("boundary_error") or {}
            per_rec.append({
                "System": sid, "Exact model/version": mv, "Recording": rid,
                "DER": fmt(der), "JER": fmt(jer),
                "Missed speech": fmt(comp_pct(c, "missed_speaker_s")),
                "False alarm": fmt(comp_pct(c, "false_alarm_speaker_s")),
                "Speaker confusion": fmt(comp_pct(c, "speaker_error_s")),
                "Runtime": fmt(proc, 1), "Real-time factor": fmt(rtf, 3),
                "Predicted speaker count": sc.get("n_predicted_speakers", NR),
                "Collar": COLLAR, "Overlap policy": OVERLAP, "Speaker-count condition": cond,
                "Track": track, "Run status": run_status, "Score status": score_status,
                "Reference speaker count": sc.get("n_reference_speakers", rec.get("n_reference_speakers", NR)),
                "Audio seconds": fmt(aud, 1),
                "Overlap ratio": fmt(sc.get("overlap_ratio", rec.get("overlap_ratio")), 3),
                "Overlap-region DER": fmt(der_of(sc.get("overlap_region"))),
                "Non-overlap DER": fmt(der_of(sc.get("non_overlap"))),
                "Boundary error median (s)": fmt(be.get("median_s"), 3),
                "Boundary error p95 (s)": fmt(be.get("p95_s"), 3),
                "Units": "DER/JER/components in % of scored reference speaker time; runtime in s",
                "Notes": run.get("error") or st.get("blocked_reason") or "",
            })
            dom = NR
            if c and der_of(c):
                dom = max((("missed", c["missed_speaker_s"]), ("false alarm", c["false_alarm_speaker_s"]),
                           ("confusion", c["speaker_error_s"])), key=lambda kv: kv[1])[0]
            breakdown.append({
                "System": sid, "Recording": rid,
                "Scored speaker time (s)": fmt((c or {}).get("scored_speaker_s"), 1),
                "Missed (s)": fmt((c or {}).get("missed_speaker_s"), 1),
                "False alarm (s)": fmt((c or {}).get("false_alarm_speaker_s"), 1),
                "Confusion (s)": fmt((c or {}).get("speaker_error_s"), 1),
                "Missed (%)": fmt(comp_pct(c, "missed_speaker_s")),
                "False alarm (%)": fmt(comp_pct(c, "false_alarm_speaker_s")),
                "Confusion (%)": fmt(comp_pct(c, "speaker_error_s")),
                "DER (%)": fmt(der_of(c)), "Dominant component": dom,
                "Overlap-region DER (%)": fmt(der_of(sc.get("overlap_region"))),
                "Non-overlap DER (%)": fmt(der_of(sc.get("non_overlap"))),
                "DER 250 ms collar (%) [diagnostic]": fmt(der_of(sc.get("diag_collar_250ms"))),
            })

        if any_scored and err_rows:
            tot = sum(c["scored_speaker_s"] for c in err_rows)
            pooled = {k: 100.0 * sum(c[k] for c in err_rows) / tot for k in
                      ("missed_speaker_s", "false_alarm_speaker_s", "speaker_error_s")}
            pooled_der = sum(pooled.values())
            n = len(ders)
            summary.append({
                "System": sid, "Exact model/version": mv, "Track": track,
                "Speaker-count condition": cond, "Collar": COLLAR, "Overlap policy": OVERLAP,
                "Recordings": n, "Audio hours": fmt(sum((r.get("duration_s") or 0) for r in recs) / 3600, 3),
                "Pooled DER": fmt(pooled_der), "Macro DER": fmt(sum(ders) / n),
                "Median DER": fmt(percentile(ders, 50)), "Mean JER": fmt(sum(jers) / n),
                "Pooled missed": fmt(pooled["missed_speaker_s"]),
                "Pooled false alarm": fmt(pooled["false_alarm_speaker_s"]),
                "Pooled confusion": fmt(pooled["speaker_error_s"]),
                "Speaker-count MAE": fmt(sum(mae_terms) / len(mae_terms)) if mae_terms else NR,
                "Total runtime (s)": fmt(tot_proc, 1),
                "Pooled RTF": fmt(tot_proc / tot_audio, 3) if tot_audio else NR,
                "Failed recordings": n_fail, "Failure rate": fmt(100.0 * n_fail / n, 1) + "%",
                "95% CI": f"not reported (n={n} recordings < 20 minimum)",
                "Ranking statement": (f"Provisional, exploratory ranking only: {n} recordings is below the "
                                      "20-recording minimum for a reliable ranking"),
            })
        else:
            summary.append({k: NR for k in SUMMARY_COLS} | {
                "System": sid, "Exact model/version": mv, "Track": track,
                "Speaker-count condition": cond, "Collar": COLLAR, "Overlap policy": OVERLAP,
                "Recordings": f"{len(recs)} planned",
                "95% CI": "not_run",
                "Ranking statement": f"not_run: {st.get('blocked_reason') or 'system has not been executed'}",
            })
        configs.append({"declaration": decl, "run_status": st or {"status": NR}})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, cols, rows in (("per_recording.csv", PER_REC_COLS, per_rec),
                             ("summary.csv", SUMMARY_COLS, summary),
                             ("error_breakdown.csv", BREAKDOWN_COLS, breakdown)):
        with (RESULTS_DIR / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        log.info("wrote results/%s (%d rows)", name, len(rows))
    write_json(RESULTS_DIR / "system_configs.json", {
        "generated_utc": utc_now(),
        "measured_results_present": any(r.get("Pooled DER") not in (NR, None) for r in summary),
        "scoring": scores.get("settings") or {"collar_s": 0.0, "overlap": "included",
                                              "scorer": "dscore (pinned commit recorded at run time)",
                                              "status": NR},
        "scorer": scores.get("scorer", {"status": NR}),
        "recordings_config": str(RECORDINGS_CONFIG.relative_to(CONFIG_DIR.parent)),
        "environment": read_json(RESULTS_DIR / "environment.json", default={"status": NR}),
        "systems": configs,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
