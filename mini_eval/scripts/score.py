#!/usr/bin/env python3
"""Score every system's predictions with ONE pinned scorer family (dscore).

* DER and JER per recording: dscore `score.py` (collar 0, overlap included,
  explicit UEM). dscore prints DER/JER but not the DER components.
* DER components (missed / false alarm / speaker confusion, in seconds):
  the `md-eval-22.pl` script bundled inside the SAME dscore checkout, called
  with the same collar, reference, system and UEM. DER recomputed from
  md-eval components must match dscore DER (|diff| <= 0.05 pp) or the
  recording is flagged `scorer_mismatch`.
* Overlap-region DER: md-eval with a UEM restricted to reference regions
  with >= 2 active speakers. Non-overlap DER: md-eval with `-1` (the flag
  dscore uses for --ignore_overlaps).
* Diagnostics only (never the headline): 250 ms collar DER.

Failure policy (declared before running): a missing, invalid or empty
prediction for a recording whose run did not succeed is scored as an EMPTY
hypothesis (DER = 100 %, all reference speech missed, JER = 100 %). It is
never dropped from the denominator. The failure is also counted in the
failure rate.

Usage:
  python scripts/score.py --dscore-dir third_party/dscore [--dscore-python python3.10]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (FAILURE_STATUSES, PRED_DIR, RESULTS_DIR, ROOT,  # noqa: E402
                    STATUS_EMPTY_OUTPUT, STATUS_INVALID_OUTPUT, STATUS_MISSING_PREDICTION,
                    STATUS_NOT_RUN, STATUS_OK, get_logger, load_manifest, load_systems,
                    read_json, run_cmd, utc_now, write_json)
from rttm_utils import (activity_profile, boundary_errors, merge_same_speaker,  # noqa: E402
                        read_rttm, validate_rttm, write_rttm)

log = get_logger("score")
PRIMARY_COLLAR = 0.0
DIAG_COLLAR = 0.25
DER_MATCH_TOL_PP = 0.05


# ---------------------------------------------------------------------------
# Output parsers (unit-tested against fixtures in tests/fixtures/)
# ---------------------------------------------------------------------------
_MDEVAL_PATTERNS = {
    "scored_speaker_s": r"SCORED SPEAKER TIME\s*=\s*([0-9.]+)",
    "missed_speaker_s": r"MISSED SPEAKER TIME\s*=\s*([0-9.]+)",
    "false_alarm_speaker_s": r"(?:FALSE ALARM|FALARM) SPEAKER TIME\s*=\s*([0-9.]+)",  # real md-eval-22 prints "FALARM"
    "speaker_error_s": r"SPEAKER ERROR TIME\s*=\s*([0-9.]+)",
    "der_percent": r"OVERALL SPEAKER DIARIZATION ERROR\s*=\s*([0-9.]+)\s*percent",
}


def parse_md_eval(stdout: str) -> dict:
    """Parse md-eval-22.pl output. Uses the LAST occurrence of each field
    (the overall block) so that per-file blocks, if printed, are ignored."""
    out = {}
    for key, pat in _MDEVAL_PATTERNS.items():
        matches = re.findall(pat, stdout)
        if not matches:
            raise ValueError(f"md-eval output missing '{key}'")
        out[key] = float(matches[-1])
    return out


def parse_dscore_table(stdout: str) -> dict[str, dict]:
    """Parse dscore's per-file table (any tabulate format with a 'File' header).

    Returns {file_id: {"DER": float, "JER": float}} plus key "*** OVERALL ***".
    """
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    header_idx = next((i for i, ln in enumerate(lines) if ln.strip().startswith("File")), None)
    if header_idx is None:
        raise ValueError("dscore output has no 'File' header")

    def split(line: str) -> list[str]:
        if "\t" in line:
            return [c.strip() for c in line.split("\t")]
        return [c for c in re.split(r"\s{2,}", line.strip()) if c]

    header = split(lines[header_idx])
    try:
        i_der, i_jer = header.index("DER"), header.index("JER")
    except ValueError as exc:
        raise ValueError(f"dscore header lacks DER/JER: {header}") from exc
    rows = {}
    for ln in lines[header_idx + 1:]:
        if set(ln.strip()) <= set("-+ |"):
            continue
        cells = split(ln)
        if len(cells) <= max(i_der, i_jer):
            continue
        try:
            rows[cells[0]] = {"DER": float(cells[i_der]), "JER": float(cells[i_jer])}
        except ValueError:
            continue
    if not rows:
        raise ValueError("no numeric rows parsed from dscore output")
    return rows


# ---------------------------------------------------------------------------
# Scorer invocation
# ---------------------------------------------------------------------------
class Scorer:
    def __init__(self, dscore_dir: Path, python: str):
        dscore_dir = Path(dscore_dir).resolve()
        self.dir = dscore_dir
        self.python = python
        self.score_py = dscore_dir / "score.py"
        self.md_eval = dscore_dir / "scorelib" / "md-eval-22.pl"
        for p in (self.score_py, self.md_eval):
            if not p.exists():
                raise FileNotFoundError(f"dscore component missing: {p}")
        proc = run_cmd(["git", "-C", str(dscore_dir), "rev-parse", "HEAD"], timeout=10)
        self.commit = proc.stdout.strip() or "unknown (not a git checkout)"

    def dscore(self, ref: Path, sys_: Path, uem: Path, collar: float) -> dict[str, dict]:
        cmd = [self.python, str(self.score_py), "--collar", f"{collar}", "-u", str(uem),
               "-r", str(ref), "-s", str(sys_), "--n_digits", "4"]  # default table; this dscore version has no --table_format option
        proc = run_cmd(cmd, timeout=600, cwd=self.dir)
        if proc.returncode != 0:
            raise RuntimeError(f"dscore failed: {proc.stderr.strip()[:800]}")
        return parse_dscore_table(proc.stdout)

    def md_eval_run(self, ref: Path, sys_: Path, uem: Path, collar: float,
                    single_speaker_only: bool = False) -> dict:
        cmd = ["perl", str(self.md_eval), "-c", f"{collar}", "-r", str(ref),
               "-s", str(sys_), "-u", str(uem)]
        if single_speaker_only:
            cmd.append("-1")
        proc = run_cmd(cmd, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"md-eval failed: {proc.stderr.strip()[:800]}")
        return parse_md_eval(proc.stdout)


def empty_hypothesis_scores(ref_speaker_seconds: float) -> dict:
    """Exact scores for an empty prediction: everything is missed."""
    return {"DER": 100.0, "JER": 100.0,
            "components": {"scored_speaker_s": ref_speaker_seconds,
                           "missed_speaker_s": ref_speaker_seconds,
                           "false_alarm_speaker_s": 0.0, "speaker_error_s": 0.0,
                           "der_percent": 100.0 if ref_speaker_seconds > 0 else 0.0}}


def overlap_uem(ref_segs, file_id: str, duration: float, path: Path) -> bool:
    """Write a UEM covering reference regions with >= 2 active speakers."""
    events = []
    for s in ref_segs:
        a, b = max(0.0, s.start), min(duration, s.end)
        if b > a:
            events += [(a, 1), (b, -1)]
    events.sort(key=lambda e: (e[0], e[1]))
    regions, active, start = [], 0, None
    for t, d in events:
        prev = active
        active += d
        if prev < 2 <= active:
            start = t
        elif prev >= 2 > active and start is not None:
            if t - start > 1e-3:
                regions.append((start, t))
            start = None
    if not regions:
        return False
    with path.open("w", encoding="utf-8") as fh:
        for a, b in regions:
            fh.write(f"{file_id} 1 {a:.3f} {b:.3f}\n")
    return True


def score_recording(scorer: Scorer | None, rec: dict, sys_dir: Path, run: dict | None,
                    diagnostics: bool, tmp: Path) -> dict:
    rid = rec["recording_id"]
    ref = ROOT / rec["reference_rttm"]
    uem = ROOT / rec["reference_uem"]
    pred = sys_dir / f"{rid}.rttm"
    run_status = (run or {}).get("status", STATUS_NOT_RUN)
    ref_segs, _ = read_rttm(ref)
    ref_prof = activity_profile(ref_segs, region=(0.0, rec["duration_s"]))
    row = {"recording_id": rid, "run_status": run_status, "score_status": None,
           "n_reference_speakers": ref_prof["n_speakers"],
           "reference_speaker_s": round(ref_prof["speaker_seconds"], 3),
           "overlap_ratio": round(ref_prof["overlap_ratio"], 4)}

    if run_status == STATUS_NOT_RUN or run_status.startswith("blocked"):
        row["score_status"] = STATUS_NOT_RUN
        return row

    report = validate_rttm(pred, expected_file_id=rid, audio_duration=rec["duration_s"])
    row["validation"] = report.as_dict()
    use_empty = False
    if not pred.exists():
        row["score_status"], use_empty = STATUS_MISSING_PREDICTION, True
    elif not report.ok:
        row["score_status"], use_empty = STATUS_INVALID_OUTPUT, True
    elif report.empty:
        row["score_status"], use_empty = STATUS_EMPTY_OUTPUT, True
    elif run_status in FAILURE_STATUSES:
        row["score_status"], use_empty = run_status, True
    else:
        row["score_status"] = STATUS_OK

    if use_empty:
        row.update(empty_hypothesis_scores(row["reference_speaker_s"]))
        row["failure_policy_applied"] = "scored as empty hypothesis"
        row["n_predicted_speakers"] = 0
        return row

    hyp_segs, _ = read_rttm(pred)
    norm = tmp / f"{rid}.norm.rttm"
    write_rttm(merge_same_speaker(hyp_segs), norm)   # documented normalization only
    if scorer is None:
        row["score_status"] = "scorer_unavailable"
        return row
    ds = scorer.dscore(ref, norm, uem, PRIMARY_COLLAR)
    if rid not in ds:
        raise RuntimeError(f"dscore output lacks row for {rid}: {list(ds)}")
    comp = scorer.md_eval_run(ref, norm, uem, PRIMARY_COLLAR)
    row["DER"], row["JER"] = ds[rid]["DER"], ds[rid]["JER"]
    row["components"] = comp
    tot = comp["scored_speaker_s"]
    der_from_comp = 100.0 * (comp["missed_speaker_s"] + comp["false_alarm_speaker_s"]
                             + comp["speaker_error_s"]) / tot if tot else float("nan")
    row["der_from_components"] = round(der_from_comp, 4)
    if abs(der_from_comp - row["DER"]) > DER_MATCH_TOL_PP:
        row["score_status"] = "scorer_mismatch"
        log.warning("%s: dscore DER %.3f vs md-eval %.3f", rid, row["DER"], der_from_comp)

    # Slice DERs via md-eval (same engine).
    ov_uem = tmp / f"{rid}.overlap.uem"
    if overlap_uem(ref_segs, rid, rec["duration_s"], ov_uem):
        row["overlap_region"] = scorer.md_eval_run(ref, norm, ov_uem, PRIMARY_COLLAR)
    row["non_overlap"] = scorer.md_eval_run(ref, norm, uem, PRIMARY_COLLAR,
                                            single_speaker_only=True)
    if diagnostics:
        row["diag_collar_250ms"] = scorer.md_eval_run(ref, norm, uem, DIAG_COLLAR)

    # Voice Arena diagnostics (not from dscore).
    row["n_predicted_speakers"] = len({s.speaker for s in hyp_segs})
    row["boundary_error"] = boundary_errors(ref_segs, hyp_segs)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dscore-dir", default=str(ROOT / "third_party" / "dscore"))
    ap.add_argument("--dscore-python", default=os.environ.get("DSCORE_PYTHON", sys.executable))
    ap.add_argument("--diagnostics", action="store_true", help="also compute 250 ms collar DER")
    args = ap.parse_args()

    systems = load_systems()
    try:
        recordings = load_manifest()
    except (FileNotFoundError, ValueError) as exc:
        log.error("nothing to score: %s", exc)
        write_json(RESULTS_DIR / "scores_raw.json", {
            "created_utc": utc_now(), "status": "not_run", "reason": str(exc),
            "settings": {"collar_s": PRIMARY_COLLAR, "overlap": "included"},
            "scorer": {"name": "dscore", "status": "not_run"}, "systems": {}})
        return 4
    extra = sorted(p.name for p in PRED_DIR.iterdir()
                   if p.is_dir() and p.name not in {s["system_id"] for s in systems}
                   and (p / "run_status.json").exists())
    try:
        scorer = Scorer(Path(args.dscore_dir), args.dscore_python)
        scorer_info = {"name": "dscore", "commit": scorer.commit,
                       "components_from": "scorelib/md-eval-22.pl (bundled)",
                       "python": args.dscore_python}
    except FileNotFoundError as exc:
        log.error("%s -- predictions will be validated but not scored", exc)
        scorer, scorer_info = None, {"name": "dscore", "error": str(exc)}

    out = {"created_utc": utc_now(), "scorer": scorer_info,
           "settings": {"collar_s": PRIMARY_COLLAR, "overlap": "included",
                        "uem": "full excerpt [0, duration]",
                        "failure_policy": "failed/invalid/empty -> empty hypothesis",
                        "normalization": "same-speaker overlapping segments merged"},
           "systems": {}}
    with tempfile.TemporaryDirectory() as td:
        for sid in [s["system_id"] for s in systems] + extra:
            sys_dir = PRED_DIR / sid
            st = read_json(sys_dir / "run_status.json", default={})
            runs = {r["recording_id"]: r for r in st.get("recordings", [])}
            rows = []
            for rec in recordings:
                try:
                    rows.append(score_recording(scorer, rec, sys_dir, runs.get(rec["recording_id"]),
                                                args.diagnostics, Path(td)))
                except Exception as exc:  # noqa: BLE001
                    log.error("%s/%s scoring error: %s", sid, rec["recording_id"], exc)
                    rows.append({"recording_id": rec["recording_id"],
                                 "score_status": "scoring_error", "error": str(exc)})
            out["systems"][sid] = {"run_status": st, "recordings": rows}
    write_json(RESULTS_DIR / "scores_raw.json", out)
    log.info("wrote %s", RESULTS_DIR / "scores_raw.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
