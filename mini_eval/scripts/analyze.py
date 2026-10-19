#!/usr/bin/env python3
"""Generate recording-level and cross-recording summaries from REAL result files.

Inputs : results/scores_raw.json (score.py), audio/manifest_resolved.json,
         predictions/<system>/run_status.json
Outputs: results/recording_summaries.json (consumed by export_dashboard.py)
         ../docs/benchmark_report.md

Honesty rules implemented here:
  * Every sentence is built from a computed number or a documented tag.
  * Interpretations are emitted with label "hypothesis"; measurements with "measured".
  * No causal claim about noise, microphones or voices unless the recording
    carries a documented tag, and even then it is phrased as a hypothesis.
  * Failed recordings are reported, never skipped.
  * With no scored results, the report says so and lists the blockers.
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRED_DIR, RESULTS_DIR, ROOT, RESOLVED_MANIFEST, read_json, utc_now, write_json  # noqa: E402

REPORT = ROOT.parent / "docs" / "benchmark_report.md"
COMP_NAMES = {"missed_speaker_s": "missed speech", "false_alarm_speaker_s": "false alarm",
              "speaker_error_s": "speaker confusion"}
COMP_MEANING = {"missed_speaker_s": "reference speech the system did not output",
                "false_alarm_speaker_s": "speech the system output where the reference has none",
                "speaker_error_s": "speech attributed to the wrong speaker"}
MIN_OVERLAP_S = 5.0     # below this much reference overlap, overlap DER is flagged as thin
MIN_TAG_RECORDINGS = 2  # condition comparisons need at least this many recordings per tag value


def der_of(c: dict | None) -> float | None:
    if not c or not c.get("scored_speaker_s"):
        return None
    return 100.0 * (c["missed_speaker_s"] + c["false_alarm_speaker_s"] + c["speaker_error_s"]) / c["scored_speaker_s"]


def pct(c: dict, key: str) -> float:
    return 100.0 * c[key] / c["scored_speaker_s"] if c.get("scored_speaker_s") else 0.0


def summarize_system_on_recording(name: str, row: dict, rec: dict) -> list[dict]:
    """Statements about one system on one recording. Each: {text, basis}."""
    out = []
    if row.get("failure_policy_applied"):
        out.append({"basis": "measured", "text":
                    f"{name} did not produce a valid prediction (status: {row.get('score_status')}). "
                    "Under the failure policy it is scored as an empty output, so all reference speech counts as missed (DER 100%)."})
        return out
    c = row.get("components")
    if not c or row.get("DER") is None:
        return [{"basis": "status", "text": f"{name}: no score available ({row.get('score_status', 'not_run')})."}]
    dom = max(COMP_NAMES, key=lambda k: c[k])
    out.append({"basis": "measured", "text":
                f"{name} had a DER of {row['DER']:.1f}%. Its largest error component was {COMP_NAMES[dom]} "
                f"({pct(c, dom):.1f}% of reference speaker time), i.e. {COMP_MEANING[dom]}."})
    ov, nov = der_of(row.get("overlap_region")), der_of(row.get("non_overlap"))
    ov_s = (row.get("overlap_region") or {}).get("scored_speaker_s", 0.0)
    if ov is not None and nov is not None:
        rel = "higher" if ov > nov else "lower" if ov < nov else "equal to"
        thin = f" The reference contains only {ov_s:.1f} s of overlapped speaker time, so the overlap figure is based on little data." \
            if ov_s < MIN_OVERLAP_S else ""
        out.append({"basis": "measured", "text":
                    f"Its overlap-region DER ({ov:.1f}%) was {rel} than its non-overlap DER ({nov:.1f}%).{thin}"})
    elif nov is not None:
        out.append({"basis": "measured", "text": "The reference has no overlapping speech in this excerpt, so only non-overlap DER applies."})
    ref_n, hyp_n = row.get("n_reference_speakers"), row.get("n_predicted_speakers")
    if ref_n is not None and hyp_n is not None:
        if hyp_n == ref_n:
            out.append({"basis": "measured", "text": f"It found the correct number of speakers ({ref_n})."})
        else:
            out.append({"basis": "measured", "text":
                        f"It output {hyp_n} speaker{'s' if hyp_n != 1 else ''} versus {ref_n} in the reference "
                        f"({'over' if hyp_n > ref_n else 'under'}-counting by {abs(hyp_n - ref_n)})."})
            if dom == "speaker_error_s":
                out.append({"basis": "hypothesis", "text":
                            "The speaker-count error and the dominant confusion component are consistent with "
                            f"{'split' if hyp_n > ref_n else 'merged'} speaker clusters; this is an interpretation, not verified by listening."})
    tags = rec.get("tags", {})
    if tags.get("mic_distance") == "far-field" and dom == "missed_speaker_s" and not rec.get("_inspected_" + name):
        out.append({"basis": "hypothesis", "text":
                    "The recording uses a documented far-field microphone; quieter distant speech may contribute to missed speech. Not verified."})
    return out


def recording_summary(rec: dict, by_system: dict[str, dict], names: dict[str, str]) -> dict:
    scored = {sid: r for sid, r in by_system.items() if r.get("DER") is not None}
    statements = []
    for sid, row in by_system.items():
        statements.extend(summarize_system_on_recording(names.get(sid, sid), row, rec))
    comparison = None
    ok = {sid: r for sid, r in scored.items() if not r.get("failure_policy_applied")}
    if len(scored) >= 2:
        best = min(scored, key=lambda s: scored[s]["DER"])
        worst = max(scored, key=lambda s: scored[s]["DER"])
        comparison = (f"Lowest DER on this recording: {names.get(best, best)} ({scored[best]['DER']:.1f}%); "
                      f"highest: {names.get(worst, worst)} ({scored[worst]['DER']:.1f}%). "
                      "A single recording does not establish which system is better in general.")
    main_error = None
    if ok:
        tot = {k: sum(r["components"][k] for r in ok.values()) for k in COMP_NAMES}
        main_error = COMP_NAMES[max(tot, key=tot.get)]
    return {"recording_id": rec["recording_id"], "main_error_category": main_error,
            "comparison": comparison, "statements": statements,
            "n_systems_scored": len(scored)}


def cross_summary(recs: list[dict], data: dict[str, dict], names: dict[str, str], runs: dict) -> dict:
    rows = {}
    for sid, block in data.items():
        rs = [r for r in block if r.get("components")]
        if not rs:
            continue
        tot = sum(r["components"]["scored_speaker_s"] for r in rs)
        err = {k: sum(r["components"][k] for r in rs) for k in COMP_NAMES}
        ders = [r["DER"] for r in rs]
        ovs = [r["overlap_region"] for r in rs if r.get("overlap_region")]
        ov_t = sum(o["scored_speaker_s"] for o in ovs)
        proc = [x.get("processing_seconds") for x in runs.get(sid, {}).values() if x.get("processing_seconds")]
        aud = [x.get("audio_seconds") for x in runs.get(sid, {}).values() if x.get("processing_seconds")]
        rows[sid] = {"pooled_der": 100 * sum(err.values()) / tot if tot else None,
                     "per_recording": ders, "iqr": (statistics.quantiles(ders, n=4)[2] - statistics.quantiles(ders, n=4)[0]) if len(ders) >= 4 else None,
                     "stdev": statistics.pstdev(ders) if len(ders) > 1 else None,
                     "overlap_der": 100 * sum(o["missed_speaker_s"] + o["false_alarm_speaker_s"] + o["speaker_error_s"] for o in ovs) / ov_t if ov_t else None,
                     "shares": {COMP_NAMES[k]: 100 * v / sum(err.values()) if sum(err.values()) else 0 for k, v in err.items()},
                     "rtf": sum(proc) / sum(aud) if proc and sum(aud) else None,
                     "n": len(rs), "failed": sum(1 for r in rs if r.get("failure_policy_applied"))}
    if not rows:
        return {}
    n = lambda s: names.get(s, s)  # noqa: E731
    lines = []
    best = min(rows, key=lambda s: rows[s]["pooled_der"])
    lines.append(("Lowest pooled DER", f"{n(best)} ({rows[best]['pooled_der']:.1f}%). "
                  + "; ".join(f"{n(s)} {r['pooled_der']:.1f}%" for s, r in sorted(rows.items(), key=lambda kv: kv[1]['pooled_der']))))
    cons = [s for s in rows if rows[s]["stdev"] is not None]
    if cons:
        c = min(cons, key=lambda s: rows[s]["stdev"])
        lines.append(("Most consistent across recordings", f"{n(c)} (standard deviation of per-recording DER {rows[c]['stdev']:.1f} points)."))
    ovc = [s for s in rows if rows[s]["overlap_der"] is not None]
    if ovc:
        o = min(ovc, key=lambda s: rows[s]["overlap_der"])
        lines.append(("Best on overlapping speech", f"{n(o)} (pooled overlap-region DER {rows[o]['overlap_der']:.1f}%)."))
    # hardest conditions: mean per-recording DER across systems, per tag value
    rec_mean = {}
    for rec in recs:
        vals = [r["DER"] for block in data.values() for r in block if r.get("recording_id") == rec["recording_id"] and r.get("DER") is not None]
        if vals:
            rec_mean[rec["recording_id"]] = statistics.mean(vals)
    tagv = {}
    for rec in recs:
        for k, v in rec.get("tags", {}).items():
            if v != "unknown" and rec["recording_id"] in rec_mean:
                tagv.setdefault(f"{k}={v}", []).append(rec_mean[rec["recording_id"]])
    cmp_ = {k: statistics.mean(v) for k, v in tagv.items()
            if MIN_TAG_RECORDINGS <= len(v) < len(rec_mean)}   # a value shared by every recording is no contrast
    if cmp_:
        hard = sorted(cmp_.items(), key=lambda kv: -kv[1])[:3]
        lines.append(("Hardest conditions (mean DER across systems)", "; ".join(f"{k} {v:.1f}% (n={len(tagv[k])})" for k, v in hard)
                      + ". Condition groups overlap and are small, so this is descriptive only."))
    lines.append(("Dominant errors", "; ".join(f"{n(s)}: " + ", ".join(f"{k} {v:.0f}%" for k, v in sorted(r["shares"].items(), key=lambda kv: -kv[1])) for s, r in rows.items())))
    tr = [f"{n(s)} DER {r['pooled_der']:.1f}% at RTF {r['rtf']:.3f}" if r["rtf"] is not None else f"{n(s)} DER {r['pooled_der']:.1f}% (runtime unavailable)" for s, r in rows.items()]
    lines.append(("Accuracy / runtime / cost", "; ".join(tr) + ". Cost per audio hour is not reported: no documented price inputs were recorded for this run."))
    lines.append(("What cannot be concluded", f"With {len(recs)} recordings (below the 20-recording minimum in methodology §10.4) no confidence interval or significance "
                  "is reported, rankings are provisional, and results do not generalise to telephone audio, other languages, or conditions absent from the sample."))
    return {"findings": [{"question": q, "answer": a} for q, a in lines], "per_system": rows}


def write_report(status: str, recs, summaries, cross, names, blockers: list[str]) -> None:
    L = ["# Voice Arena mini benchmark report", "",
         f"_Generated by `mini_eval/scripts/analyze.py` on {utc_now()} from `mini_eval/results/`. Do not edit by hand._", "",
         f"**Status: {status}.**", ""]
    if status == "not_run":
        L += ["No system has produced scored predictions, so this report contains **no results**. "
              "Nothing below is a measurement.", "", "## Blockers recorded by the pipeline", ""]
        L += [f"- {b}" for b in blockers] or ["- (none recorded)"]
        L += ["", "## Planned recordings", "", "| Recording slot | Source | Window | Declared tags | Selection |", "|---|---|---|---|---|"]
        for r in recs:
            L.append(f"| {r['recording_id']} | {r.get('source')} | {r.get('excerpt_start_s')}–{r.get('excerpt_end_s')} s | "
                     f"{', '.join(f'{k}={v}' for k, v in (r.get('tags_declared') or r.get('tags') or {}).items())} | {r.get('selection_reason', '')} |")
        L += ["", "Coverage gaps: telephone audio is not represented; audio quality is unknown for every file until a documented listening check is added.", ""]
    else:
        n_scored = sum(1 for s in summaries if s["n_systems_scored"])
        L += [f"Provisional, exploratory results on {n_scored} public excerpts. Not a general ranking.", "", "## Cross-recording findings", ""]
        for f in cross.get("findings", []):
            L.append(f"- **{f['question']}.** {f['answer']}")
        L += ["", "## Recording-level observations", ""]
        for s in summaries:
            L.append(f"### {s['recording_id']}")
            for st in s["statements"]:
                L.append(f"- [{st['basis']}] {st['text']}")
            if s["comparison"]:
                L.append(f"- [measured] {s['comparison']}")
            L.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    scores = read_json(RESULTS_DIR / "scores_raw.json", default={"systems": {}})
    manifest = read_json(RESOLVED_MANIFEST, default={}) if RESOLVED_MANIFEST.exists() else {}
    recs = manifest.get("recordings") or []
    if not recs:
        recs = read_json(ROOT / "configs" / "recordings.json")["recordings"]
    sysdecl = read_json(ROOT / "configs" / "systems.json")["systems"]
    names = {s["system_id"]: s["display_name"] for s in sysdecl}
    for s in sysdecl:  # show the provider/model that actually ran for the API slot
        st = read_json(PRED_DIR / s["system_id"] / "run_status.json", default={})
        if s["system_id"] == "api-system" and st.get("model_identifier") and not st.get("blocked_reason"):
            names[s["system_id"]] = st["model_identifier"]
    inspections = read_json(ROOT / "configs" / "inspections.json", default={"inspections": []})["inspections"]
    data = {sid: blk.get("recordings", []) for sid, blk in scores.get("systems", {}).items()}
    runs = {}
    blockers = []
    for s in sysdecl:
        st = read_json(PRED_DIR / s["system_id"] / "run_status.json", default={})
        runs[s["system_id"]] = {r["recording_id"]: r for r in st.get("recordings", [])}
        if st.get("blocked_reason"):
            blockers.append(f"{s['system_id']}: {st['blocked_reason']}")
    if scores.get("status") == "not_run" and scores.get("reason"):
        blockers.insert(0, f"scorer: {scores['reason']}")
    any_scored = any(r.get("DER") is not None for blk in data.values() for r in blk)
    summaries = []
    if any_scored:
        for rec in recs:
            by_sys = {sid: next((r for r in blk if r.get("recording_id") == rec["recording_id"]), {}) for sid, blk in data.items()}
            for ins in inspections:
                if rec["recording_id"] in ins.get("recording_ids", []):
                    rec["_inspected_" + names.get(ins.get("system_id"), "")] = True
            summ = recording_summary(rec, by_sys, names)
            for ins in inspections:
                if rec["recording_id"] in ins.get("recording_ids", []):
                    summ["statements"].append({"basis": "inspection", "text":
                        f"Documented inspection ({ins['date']}, {names.get(ins.get('system_id'), ins.get('system_id'))}): {ins['finding']}"})
            summaries.append(summ)
    cross = cross_summary(recs, data, names, runs) if any_scored else {}
    status = "measured (provisional)" if any_scored else "not_run"
    write_json(RESULTS_DIR / "recording_summaries.json", {"generated_utc": utc_now(), "status": status,
                                                           "recordings": summaries,
                                                           "cross_recording": cross.get("findings", []),
                                                           "blockers": blockers})
    write_report(status, recs, summaries, cross, names, blockers)
    print(f"wrote results/recording_summaries.json and docs/benchmark_report.md (status={status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
