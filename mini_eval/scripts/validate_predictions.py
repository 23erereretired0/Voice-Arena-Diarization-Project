#!/usr/bin/env python3
"""Validate every prediction RTTM before scoring.

Checks per system and recording: file exists, parses, file IDs match the
recording ID, no segment ends after the audio duration (+0.5 s tolerance),
non-empty, plus warnings for same-speaker overlaps and odd field counts.
Also checks that no prediction exists for an unknown recording ID.
Writes results/validation_report.json. Exit code 1 if any hard error.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PRED_DIR, RESULTS_DIR, get_logger, load_manifest, read_json, utc_now, write_json  # noqa: E402
from rttm_utils import validate_rttm  # noqa: E402

log = get_logger("validate")


def main() -> int:
    try:
        recs = {r["recording_id"]: r for r in load_manifest()}
    except (FileNotFoundError, ValueError) as exc:
        log.error("nothing to validate: %s", exc)
        return 2
    report, hard_errors = {"created_utc": utc_now(), "systems": {}}, 0
    for sys_dir in sorted(p for p in PRED_DIR.iterdir() if p.is_dir()):
        st = read_json(sys_dir / "run_status.json", default={})
        entry = {"run_status_present": bool(st), "blocked_reason": st.get("blocked_reason"),
                 "recordings": {}, "unexpected_files": []}
        for rttm in sys_dir.glob("*.rttm"):
            if rttm.stem not in recs:
                entry["unexpected_files"].append(rttm.name)
                hard_errors += 1
        for rid, rec in recs.items():
            rep = validate_rttm(sys_dir / f"{rid}.rttm", expected_file_id=rid,
                                audio_duration=rec["duration_s"])
            entry["recordings"][rid] = rep.as_dict()
            if (sys_dir / f"{rid}.rttm").exists() and not rep.ok:
                hard_errors += 1
                log.error("%s/%s: %s", sys_dir.name, rid, rep.errors)
        report["systems"][sys_dir.name] = entry
    write_json(RESULTS_DIR / "validation_report.json", report)
    log.info("validation complete: %d hard error(s)", hard_errors)
    return 1 if hard_errors else 0


if __name__ == "__main__":
    sys.exit(main())
