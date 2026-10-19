#!/usr/bin/env python3
"""Browser checks for the simplified dashboard (requires: pip install playwright; playwright install chromium).

  python dashboard/tests/browser_check.py

Scenarios
  1. Shipped data (whatever data/benchmark_results.js currently contains), at 1366 px and 390 px.
  2. A SYNTHETIC measured dataset in a temporary copy (never written to the project) to check
     ranking, filtered reranking and recording summaries. Its numbers are invented test inputs.
  3. Upload flow against the real local backend (server/app.py) for error paths, plus a mocked
     /api/diarize success response to check the result UI. No real inference happens here.
"""
from __future__ import annotations

import json
import math
import shutil
import struct
import sys
import tempfile
import threading
import wave
from pathlib import Path

from playwright.sync_api import sync_playwright

DASH = Path(__file__).resolve().parents[1]
REPO = DASH.parent
sys.path.insert(0, str(REPO / "server"))
sys.path.insert(0, str(REPO / "mini_eval" / "scripts"))
RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), str(detail)[:300]))


def attach(pg, errs):
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" and "/api/" not in m.text else None)


def unnamed(pg):
    return pg.evaluate("""() => [...document.querySelectorAll('input,select,button')].filter(e => e.offsetParent !== null)
      .filter(e => !(e.getAttribute('aria-label') || e.getAttribute('aria-labelledby') || (e.id && document.querySelector('label[for="'+e.id+'"]'))
                    || e.closest('label') || (e.textContent||'').trim())).map(e => e.outerHTML.slice(0, 80))""")


def shipped(p, width):
    tag = f"shipped@{width}"
    errs = []
    pg = p.chromium.launch().new_page(viewport={"width": width, "height": 900})
    attach(pg, errs)
    pg.goto((DASH / "index.html").as_uri())
    pg.wait_for_timeout(400)
    data = pg.evaluate("window.VA_BENCHMARK")
    status = data["status"]
    check(f"[{tag}] header shows name, count, status and date", "Voice Arena" in pg.inner_text("header")
          and f"{data['n_recordings']} recordings" in pg.inner_text("#meta") and "Last evaluated" in pg.inner_text("#meta"))
    check(f"[{tag}] Customize Ranking + Upload Audio buttons present",
          pg.is_visible("#customizeBtn") and pg.is_visible("#uploadJump"))
    rows = pg.locator("#overall tbody tr:not(.details)").count()
    check(f"[{tag}] leaderboard lists every declared system", rows == len(data["systems"]), rows)
    heads = pg.locator("#overall thead th").all_inner_texts()
    want = ["DER", "95% CI", "Macro", "Median", "JER", "Missed", "False alarm", "Confusion", "Overlap DER", "Non-overlap",
            "Spk MAE", "Bnd med", "Bnd p95", "Runtime", "RTF", "Cost/h", "Failures", "Recs"]
    check(f"[{tag}] all required metric columns present", all(any(w in h for h in heads) for w in want), heads)
    if status == "not_run":
        body = pg.inner_text("#overall tbody")
        import re
        nums = re.findall(r"\b\d+\.\d+\b", body)
        check(f"[{tag}] not_run: no numeric scores shown", not nums, nums[:5])
        check(f"[{tag}] not_run: status says no measured scores", "Not run" in pg.inner_text("#statusTag"))
    # tooltips: hover, keyboard focus, tap
    b = pg.locator("#overall thead [data-m='der']")
    b.hover(); pg.wait_for_timeout(80)
    t = pg.inner_text("#tip")
    check(f"[{tag}] DER tooltip on hover has name, formula, units, direction, protocol",
          all(k in t for k in ["diarization error rate", "missed + false-alarm + speaker-confusion time", "Units: %", "Lower is better", "collar"]), t)
    for key, frag in [("jer", "intersection / union"), ("mae", "mean |predicted − reference speakers|"),
                      ("rtf", "inference runtime (s) / audio duration (s)"), ("fail", "failed recording attempts / total recording attempts")]:
        pg.locator(f"#overall thead [data-m='{key}']").focus(); pg.wait_for_timeout(60)
        check(f"[{tag}] {key} tooltip via keyboard shows formula", frag in pg.inner_text("#tip") and pg.is_visible("#tip"))
    pg.keyboard.press("Escape")
    pg.locator("#overall thead [data-m='rtf']").click()
    check(f"[{tag}] tooltip opens on click/tap", pg.is_visible("#tip"))
    pg.mouse.click(5, 5); pg.wait_for_timeout(50)
    # details
    pg.locator("#overall [data-details]").first.click()
    check(f"[{tag}] system details expand with version/scoring/config", pg.locator("#overall tr.details:not([hidden])").count() == 1
          and "Scoring" in pg.inner_text("#overall tr.details:not([hidden])"))
    # customize panel
    pg.click("#customizeBtn")
    check(f"[{tag}] customize panel has exactly five filters", pg.locator("#filterPanel select").count() == 5)
    pg.select_option("[data-f='n_speakers']", "6+"); pg.click("#applyBtn"); pg.wait_for_timeout(100)
    ctext = pg.inner_text("#customSection")
    check(f"[{tag}] filter shows active chips + matching count/duration", "Speakers: 6+ speakers" in ctext and "matching recording" in ctext)
    check(f"[{tag}] small/unscored subset -> 'Insufficient evidence'", "Insufficient evidence for a reliable comparison" in ctext, ctext[:200])
    check(f"[{tag}] overall leaderboard still present while filtered", pg.is_visible("#overall"))
    pg.select_option("[data-f='audio_quality']", "clean"); pg.click("#applyBtn")
    check(f"[{tag}] unknown tags never match specific filter", "0 matching recording" in pg.inner_text("#customSection"))
    pg.click("#backBtn"); pg.wait_for_timeout(50)
    check(f"[{tag}] one-click return to overall", pg.is_hidden("#customSection") and pg.input_value("[data-f='n_speakers']") == "all")
    # recordings
    n = pg.locator("#recTable tbody tr").count()
    check(f"[{tag}] recordings table lists {data['n_recordings']} recordings", n == data["n_recordings"], n)
    pg.locator("[data-rec]").nth(1).click(); pg.wait_for_timeout(80)
    rd = pg.inner_text("#recDetail")
    check(f"[{tag}] recording panel opens", "Conditions:" in rd)
    if status == "not_run":
        check(f"[{tag}] not_run: no summary text invented", "No measured result for this recording yet" in rd)
    # upload (file:// mode)
    check(f"[{tag}] file:// upload explains that the local server is required", "python server/app.py" in pg.inner_text("#runStatus"))
    pg.click("#uploadJump")
    check(f"[{tag}] Upload Audio jumps to the picker", pg.evaluate("document.activeElement.id") == "audioFile")
    check(f"[{tag}] no page-level horizontal overflow", pg.evaluate("document.documentElement.scrollWidth") <= width,
          pg.evaluate("document.documentElement.scrollWidth"))
    check(f"[{tag}] every visible control has an accessible name", not unnamed(pg), unnamed(pg))
    check(f"[{tag}] no JS errors", not errs, errs)
    pg.context.browser.close()


def synthetic_payload():
    """Invented numbers for a UI test only."""
    import analyze
    tags = [("meeting", "3-5", "high", "unknown", "far-field"), ("meeting", "3-5", "low", "unknown", "close-talk"),
            ("meeting", "3-5", "medium", "unknown", "far-field"), ("interview/media", "2", "low", "unknown", "unknown"),
            ("interview/media", "6+", "high", "unknown", "unknown"), ("meeting", "6+", "high", "unknown", "far-field")]
    recs, sums = [], []
    for i, tg in enumerate(tags):
        res = {}
        for sid, k in (("pyannote-community-1", 1.0), ("nemo-msdd", 1.4 if i % 2 else 0.8), ("api-system", None)):
            if k is None:
                res[sid] = {"run_status": "blocked_missing_credentials", "components": None}
                continue
            c = {"missed_speaker_s": 10 * k + i, "false_alarm_speaker_s": 4 * k, "speaker_error_s": 6 * k + i, "scored_speaker_s": 200.0}
            der = 100 * (c["missed_speaker_s"] + c["false_alarm_speaker_s"] + c["speaker_error_s"]) / 200
            res[sid] = {"run_status": "ok", "score_status": "ok", "failed": False, "DER": der, "JER": der + 5, "components": c,
                        "overlap_region": {"missed_speaker_s": 6, "false_alarm_speaker_s": 1, "speaker_error_s": 3, "scored_speaker_s": 30},
                        "non_overlap": {"missed_speaker_s": 4, "false_alarm_speaker_s": 3, "speaker_error_s": 3, "scored_speaker_s": 170},
                        "n_predicted_speakers": 4, "boundary_distances_s": [0.1, 0.3, 1.2], "processing_seconds": 9 * k, "audio_seconds": 180,
                        "segments": [[0, 40, "s1"], [35, 90, "s2"], [90, 180, "s1"]]}
        r = {"recording_id": f"SYNTH_{i}", "prepared": True, "source": "SYNTHETIC", "duration_s": 180, "n_reference_speakers": 4,
             "tags": dict(zip(["recording_type", "n_speakers", "overlap_level", "audio_quality", "mic_distance"], tg)),
             "tag_evidence": {}, "reference_segments": [[0, 45, "A"], [40, 100, "B"], [100, 180, "A"]], "results": res, "excerpt": [0, 180]}
        recs.append(r)
        sums.append(analyze.recording_summary(r, {s: dict(v, n_reference_speakers=4) for s, v in res.items()},
                                              {"pyannote-community-1": "pyannote", "nemo-msdd": "NeMo", "api-system": "API"}))
    systems = [{"system_id": s, "name": n, "declared_model": "SYNTHETIC", "deployment": "x", "mode": "batch", "track": "audio_only_batch",
                "speaker_count_condition": "unknown", "blocked_reason": b}
               for s, n, b in (("pyannote-community-1", "pyannote", None), ("nemo-msdd", "NeMo", None), ("api-system", "API", "no key (synthetic)"))]
    return {"status": "measured", "provisional": True, "evaluated_utc": "SYNTHETIC-TEST", "n_recordings": len(recs), "systems": systems,
            "recordings": recs, "recording_summaries": sums, "cross_recording": [{"question": "Lowest pooled DER", "answer": "SYNTHETIC"}],
            "scoring": {"collar_s": 0.0, "overlap": "included"}, "status_note": "SYNTHETIC TEST DATA"}


def synthetic_run(p):
    tag = "synthetic"
    errs = []
    td = tempfile.mkdtemp()
    try:
        d = Path(td) / "dashboard"
        shutil.copytree(DASH, d, ignore=shutil.ignore_patterns("tests"))
        payload = synthetic_payload()
        (d / "data" / "benchmark_results.js").write_text("window.VA_BENCHMARK = " + json.dumps(payload) + ";", encoding="utf-8")
        pg = p.chromium.launch().new_page(viewport={"width": 1366, "height": 900})
        attach(pg, errs)
        pg.goto((d / "index.html").as_uri()); pg.wait_for_timeout(400)
        first = pg.locator("#overall tbody tr:not(.details)").all_inner_texts()
        check(f"[{tag}] ranked by pooled DER; blocked system unranked last", first[0].startswith("1.") and first[1].startswith("2.") and "API" in first[2] and "Not run" in first[2], first)
        # independent check of pooled DER for rank 1
        def pooled(sid, recs):
            e = sum(sum(r["results"][sid]["components"][k] for k in ("missed_speaker_s", "false_alarm_speaker_s", "speaker_error_s")) for r in recs)
            return 100 * e / sum(r["results"][sid]["components"]["scored_speaker_s"] for r in recs)
        exp = sorted(((pooled(s, payload["recordings"]), s) for s in ("pyannote-community-1", "nemo-msdd")))
        shown = pg.locator("#overall tbody tr:not(.details)").first.locator("td").nth(1).inner_text()
        check(f"[{tag}] displayed rank-1 DER equals independently computed pooled DER", shown == f"{exp[0][0]:.1f}", (shown, exp))
        check(f"[{tag}] CI shown as not reported for n<20", "not reported (n=6 < 20)" in pg.inner_text("#overall"))
        pg.click("#customizeBtn"); pg.select_option("[data-f='mic_distance']", "far-field"); pg.click("#applyBtn"); pg.wait_for_timeout(80)
        sub = [r for r in payload["recordings"] if r["tags"]["mic_distance"] == "far-field"]
        ct = pg.inner_text("#customSection")
        check(f"[{tag}] far-field filter: 3 matching, ranked table shown", "3 matching recording(s) · 9.0 min" in ct and pg.locator("#custom").count() == 1, ct[:160])
        exp_sub = sorted(((pooled(s, sub), s) for s in ("pyannote-community-1", "nemo-msdd")))
        got = pg.locator("#custom tbody tr:not(.details)").first.locator("td").nth(1).inner_text()
        check(f"[{tag}] filtered DER recomputed from matching recordings only", got == f"{exp_sub[0][0]:.1f}", (got, exp_sub))
        overall_after = pg.locator("#overall tbody tr:not(.details)").first.locator("td").nth(1).inner_text()
        check(f"[{tag}] overall scores unchanged by filtering", overall_after == shown)
        pg.select_option("[data-f='n_speakers']", "2"); pg.click("#applyBtn")
        check(f"[{tag}] 1 matching recording -> insufficient evidence, no ranking table", "Insufficient evidence" in pg.inner_text("#customSection") and pg.locator("#custom").count() == 0)
        pg.click("#resetBtn")
        pg.locator("[data-rec='SYNTH_0']").click(); pg.wait_for_timeout(80)
        rd = pg.inner_text("#recDetail")
        check(f"[{tag}] recording summary shows measured statements, comparison and timelines",
              "measured" in rd and "largest error component" in rd and "does not establish" in rd and pg.locator("#recDetail .tl").count() >= 3, rd[:200])
        check(f"[{tag}] cross-recording findings panel rendered", "Across all recordings" in pg.inner_text("body"))
        check(f"[{tag}] no JS errors", not errs, errs)
        pg.context.browser.close()
    finally:
        shutil.rmtree(td, ignore_errors=True)


def make_wav(path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))))


def upload(p):
    import app
    tag = "upload"
    srv = app.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}/"
    errs = []
    td = Path(tempfile.mkdtemp())
    try:
        wav, bad = td / "tone.wav", td / "notaudio.wav"
        make_wav(wav, 2.0); bad.write_bytes(b"definitely not audio" * 100)
        pg = p.chromium.launch().new_page(viewport={"width": 390, "height": 844})
        attach(pg, errs)
        pg.goto(base); pg.wait_for_timeout(500)
        st = pg.inner_text("#runStatus")
        check(f"[{tag}] served by backend; status reports model availability honestly",
              ("model is unavailable" in st) or ("Server ready" in st), st)
        pg.set_input_files("#audioFile", str(bad)); pg.wait_for_timeout(300)
        pg.click("#runBtn"); pg.wait_for_timeout(1500)
        check(f"[{tag}] invalid file -> readable backend error", "could not be decoded" in pg.inner_text("#runStatus"), pg.inner_text("#runStatus"))
        pg.set_input_files("#audioFile", str(wav)); pg.wait_for_timeout(500)
        check(f"[{tag}] filename and duration shown", "tone.wav" in pg.inner_text("#fileInfo") and "2.0 s" in pg.inner_text("#fileInfo"), pg.inner_text("#fileInfo"))
        if not app.Pyannote.availability()[0]:
            pg.click("#runBtn"); pg.wait_for_timeout(1500)
            check(f"[{tag}] model unavailable -> 503 message, no fake segments",
                  "Failed:" in pg.inner_text("#runStatus") and pg.locator("#runResult table").count() == 0, pg.inner_text("#runStatus"))
        # mocked success (UI rendering only; NOT inference)
        mock = {"status": "ok", "system": "pyannote-community-1", "model": "MOCK (UI test)", "model_revision": "MOCK", "package_versions": {"mock": "1"},
                "device": "none", "filename": "tone.wav", "duration_s": 2.0, "processing_seconds": 0.5, "rtf": 0.25, "n_speakers": 2,
                "segments": [{"start": 0.0, "end": 0.9, "speaker": "SPEAKER_00"}, {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}],
                "rttm": "SPEAKER upload 1 0.000 0.900 <NA> <NA> SPEAKER_00 <NA> <NA>\n", "note": "No DER or rank: no reference."}
        pg.route("**/api/diarize", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps(mock)))
        pg.click("#runBtn"); pg.wait_for_timeout(500)
        rr = pg.inner_text("#runResult")
        check(f"[{tag}] (mocked) result shows model, segments, timeline, no DER/rank claim",
              "MOCK (UI test)" in rr and "SPEAKER_01" in rr and pg.locator("#runResult .tl i").count() == 2 and "No DER or rank" in rr)
        with pg.expect_download() as d1:
            pg.click("#dlRttm")
        check(f"[{tag}] RTTM download works", d1.value.suggested_filename.endswith(".rttm"))
        with pg.expect_download() as d2:
            pg.click("#dlJson")
        check(f"[{tag}] JSON download works", d2.value.suggested_filename.endswith(".json"))
        pg.locator("[data-t='1']").click(); pg.wait_for_timeout(200)
        check(f"[{tag}] timestamp click seeks the audio player", abs(pg.evaluate("document.getElementById('player').currentTime") - 1.0) < 0.3)
        ref = td / "ref.rttm"; ref.write_text("SPEAKER x 1 0 9 <NA> <NA> A <NA> <NA>\n")
        pg.set_input_files("#refFile", str(ref)); pg.click("#scoreBtn"); pg.wait_for_timeout(800)
        so = pg.inner_text("#scoreOut")
        check(f"[{tag}] reference RTTM is validated server-side (segment beyond audio rejected)", "after audio duration" in so, so)
        check(f"[{tag}] no horizontal overflow at 390 px with results", pg.evaluate("document.documentElement.scrollWidth") <= 390)
        check(f"[{tag}] no JS errors", not [e for e in errs if "Failed to load resource" not in e], errs)
        pg.context.browser.close()
    finally:
        srv.shutdown(); srv.server_close(); shutil.rmtree(td, ignore_errors=True)


def main():
    with sync_playwright() as p:
        shipped(p, 1366)
        shipped(p, 390)
        synthetic_run(p)
        upload(p)
    bad = [r for r in RESULTS if not r[1]]
    for n, ok, d in RESULTS:
        print(("PASS " if ok else "FAIL ") + n + ("" if ok else "  | " + d))
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} browser checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
