"""Tests for rule-based recording selection, condition tags, summary generation
and the upload backend. All data is SYNTHETIC; nothing here is a benchmark result.
"""
from __future__ import annotations

import io
import json
import math
import os
import struct
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(REPO / "server"))

import analyze  # noqa: E402
import prepare_data as pdm  # noqa: E402
from rttm_utils import Segment, boundary_errors  # noqa: E402

FIX = HERE / "fixtures"


def stats(n, ov, ch, speech=60.0):
    return {"n_speakers": n, "overlap_ratio": ov, "speaker_changes_per_min": ch, "speech_s": speech}


class TestSelection(unittest.TestCase):
    SLOTS = [{"slot": 1, "vox_select": {"n_speakers": [2, 2], "order": "file_id"}},
             {"slot": 2, "vox_select": {"n_speakers": [3, 5], "order": "file_id"}},
             {"slot": 3, "vox_select": {"n_speakers": [6, 999], "order": "file_id"}},
             {"slot": 4, "vox_select": {"n_speakers": [2, 999], "order": "overlap_ratio_desc"}},
             {"slot": 5, "vox_select": {"n_speakers": [2, 999], "order": "speaker_changes_per_min_desc"}}]

    def test_rules_distinct_and_ordered(self):
        st = {"aaa": stats(2, 0.01, 3), "bbb": stats(2, 0.30, 4), "ccc": stats(4, 0.02, 9),
              "ddd": stats(7, 0.05, 5), "eee": stats(3, 0.20, 20), "fff": stats(1, 0.0, 0)}
        got = {k: v[0] for k, v in pdm.select_vox(self.SLOTS, st).items()}
        self.assertEqual(got, {1: "aaa", 2: "ccc", 3: "ddd", 4: "bbb", 5: "eee"})
        self.assertEqual(len(set(got.values())), 5)

    def test_missing_candidate_is_omitted_not_guessed(self):
        got = pdm.select_vox(self.SLOTS, {"aaa": stats(2, 0.1, 3)})
        self.assertEqual(set(got), {1})

    def test_window_stats_and_bins(self):
        segs = [Segment("x", 0, 10, "A"), Segment("x", 8, 4, "B"), Segment("x", 12, 6, "A"), Segment("x", 200, 5, "C")]
        w = pdm.window_stats(segs, 0, 60)
        self.assertEqual(w["n_speakers"], 2)
        self.assertEqual(w["speaker_changes"], 2)
        self.assertAlmostEqual(w["overlap_ratio"], 2 / 18)
        self.assertEqual([pdm.n_speakers_bin(n) for n in (1, 2, 4, 6)], ["1", "2", "3-5", "6+"])
        self.assertEqual([pdm.overlap_level(r) for r in (0.01, 0.10, 0.15)], ["low", "medium", "high"])

    def test_boundary_distances_kept_for_pooling(self):
        be = boundary_errors([Segment("r", 1.0, 2.0, "A")], [Segment("r", 1.1, 2.2, "X")])
        self.assertEqual(be["distances_s"], [0.1, 0.3])


def row(miss, fa, conf, total=100.0, ov=None, nov=None, ref_n=3, hyp_n=3, failed=False):
    c = {"missed_speaker_s": miss, "false_alarm_speaker_s": fa, "speaker_error_s": conf, "scored_speaker_s": total}
    r = {"DER": 100 * (miss + fa + conf) / total, "components": c, "n_reference_speakers": ref_n,
         "n_predicted_speakers": hyp_n, "score_status": "ok"}
    if ov:
        r["overlap_region"] = ov
    if nov:
        r["non_overlap"] = nov
    if failed:
        r["failure_policy_applied"] = "scored as empty hypothesis"
    return r


class TestSummaries(unittest.TestCase):
    REC = {"recording_id": "r1", "tags": {"mic_distance": "close-talk"}}

    def test_dominant_component_and_overlap_statement(self):
        ov = {"missed_speaker_s": 6, "false_alarm_speaker_s": 0, "speaker_error_s": 4, "scored_speaker_s": 20}
        nov = {"missed_speaker_s": 2, "false_alarm_speaker_s": 2, "speaker_error_s": 6, "scored_speaker_s": 80}
        st = analyze.summarize_system_on_recording("Sys", row(8, 2, 10, ov=ov, nov=nov), self.REC)
        text = " ".join(s["text"] for s in st)
        self.assertIn("largest error component was speaker confusion (10.0%", text)
        self.assertIn("overlap-region DER (50.0%) was higher than its non-overlap DER (12.5%)", text)
        self.assertTrue(all(s["basis"] == "measured" for s in st))

    def test_thin_overlap_flagged(self):
        ov = {"missed_speaker_s": 1, "false_alarm_speaker_s": 0, "speaker_error_s": 0, "scored_speaker_s": 2}
        nov = {"missed_speaker_s": 1, "false_alarm_speaker_s": 1, "speaker_error_s": 1, "scored_speaker_s": 98}
        text = " ".join(s["text"] for s in analyze.summarize_system_on_recording("Sys", row(2, 1, 1, ov=ov, nov=nov), self.REC))
        self.assertIn("based on little data", text)

    def test_interpretation_is_labelled_hypothesis(self):
        st = analyze.summarize_system_on_recording("Sys", row(1, 1, 20, ref_n=3, hyp_n=5), self.REC)
        self.assertIn("hypothesis", {s["basis"] for s in st})
        self.assertTrue(any("over-counting by 2" in s["text"] for s in st))

    def test_no_unsupported_causal_claims(self):
        st = analyze.summarize_system_on_recording("Sys", row(30, 1, 1), {"recording_id": "r", "tags": {}})
        text = " ".join(s["text"] for s in st).lower()
        for word in ("noise", "noisy", "murmur", "microphone", "similar voices"):
            self.assertNotIn(word, text)

    def test_singular_speaker_and_status_basis(self):
        st = analyze.summarize_system_on_recording("Sys", row(1, 1, 1, ref_n=3, hyp_n=1), self.REC)
        self.assertTrue(any("output 1 speaker versus 3" in s["text"] for s in st))
        nr = analyze.summarize_system_on_recording("Sys", {"score_status": "not_run"}, self.REC)
        self.assertEqual(nr[0]["basis"], "status")

    def test_inspection_suppresses_generic_hypothesis(self):
        rec = {"recording_id": "r", "tags": {"mic_distance": "far-field"}}
        st = analyze.summarize_system_on_recording("Sys", row(30, 1, 1), rec)
        self.assertIn("hypothesis", {x["basis"] for x in st})
        rec["_inspected_Sys"] = True
        st = analyze.summarize_system_on_recording("Sys", row(30, 1, 1), rec)
        self.assertNotIn("hypothesis", {x["basis"] for x in st})

    def test_failure_is_reported(self):
        st = analyze.summarize_system_on_recording("Sys", row(100, 0, 0, failed=True), self.REC)
        self.assertIn("DER 100%", st[0]["text"])

    def test_recording_summary_comparison_hedged(self):
        s = analyze.recording_summary(self.REC, {"a": row(5, 0, 0), "b": row(20, 0, 0)}, {"a": "A", "b": "B"})
        self.assertIn("Lowest DER on this recording: A (5.0%)", s["comparison"])
        self.assertIn("does not establish", s["comparison"])


def make_wav(path: Path, seconds=1.0, rate=16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                               for i in range(int(seconds * rate))))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.app = app
        cls.srv = app.make_server("127.0.0.1", 0)
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def call(self, path, data=None, headers=None):
        req = urllib.request.Request(self.base + path, data=data, headers=headers or {},
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_status_and_static(self):
        code, body = self.call("/api/status")
        js = json.loads(body)
        self.assertEqual(code, 200)
        self.assertIn("pyannote-community-1", js["systems"])
        self.assertIn("deleted after each request", js["privacy"])
        code, body = self.call("/")
        self.assertEqual(code, 200)
        self.assertIn(b"Voice Arena", body)

    def test_invalid_audio_rejected(self):
        code, body = self.call("/api/diarize", b"this is not audio" * 50, {"X-Filename": "x.wav"})
        self.assertEqual(code, 400)
        self.assertIn("could not be decoded", json.loads(body)["error"])

    def test_empty_and_oversize(self):
        self.assertEqual(self.call("/api/diarize", b"", {"X-Filename": "x.wav"})[0], 400)
        os.environ["VA_MAX_UPLOAD_MB"] = "0.001"
        try:
            self.assertEqual(self.call("/api/diarize", b"0" * 5000, {"X-Filename": "x.wav"})[0], 413)
        finally:
            del os.environ["VA_MAX_UPLOAD_MB"]

    def test_valid_audio_without_model_is_503_not_simulated(self):
        if self.app.Pyannote.availability()[0]:
            self.skipTest("pyannote available: real inference would run")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "tone.wav"
            make_wav(p)
            code, body = self.call("/api/diarize", p.read_bytes(), {"X-Filename": "tone.wav"})
        js = json.loads(body)
        self.assertEqual(code, 503)
        self.assertNotIn("segments", js)

    def test_score_validation_errors(self):
        bad = json.dumps({"prediction_rttm": "garbage", "reference_rttm": "", "duration_s": 10}).encode()
        self.assertEqual(self.call("/api/score", bad)[0], 400)
        self.assertEqual(self.call("/api/score", b"{not json")[0], 400)
        late = json.dumps({"prediction_rttm": "SPEAKER upload 1 0 50 <NA> <NA> A <NA> <NA>",
                           "reference_rttm": "SPEAKER upload 1 0 5 <NA> <NA> A <NA> <NA>", "duration_s": 10}).encode()
        code, body = self.call("/api/score", late)
        self.assertEqual(code, 400)
        self.assertIn("after audio duration", json.loads(body)["error"])

    def test_score_without_scorer_is_503(self):
        os.environ["VA_DSCORE_DIR"] = "/nonexistent/dscore"
        try:
            ok = json.dumps({"prediction_rttm": "SPEAKER any 1 0 5 <NA> <NA> A <NA> <NA>",
                             "reference_rttm": "SPEAKER ref 1 0 5 <NA> <NA> X <NA> <NA>", "duration_s": 10}).encode()
            code, body = self.call("/api/score", ok)
            self.assertEqual(code, 503)
            self.assertIn("dscore not found", json.loads(body)["error"])
        finally:
            del os.environ["VA_DSCORE_DIR"]

    def test_score_endpoint_with_STUB_scorer_synthetic(self):
        """Exercises the endpoint plumbing with a stub that prints fixture output (not dscore)."""
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "dscore"
            (d / "scorelib").mkdir(parents=True)
            (d / "score.py").write_text("import sys\nprint(open(%r).read().replace('rec1','upload'))\n"
                                        % str(FIX / "dscore_tsv_sample.txt"))
            (d / "scorelib" / "md-eval-22.pl").write_text(
                'open(F,"<","%s"); print while <F>;\n' % (FIX / "md_eval_sample.txt").as_posix())  # forward slashes: Perl treats "\" as an escape on Windows
            os.environ["VA_DSCORE_DIR"] = str(d)
            try:
                body = json.dumps({"prediction_rttm": "SPEAKER a 1 0 5 <NA> <NA> A <NA> <NA>",
                                   "reference_rttm": "SPEAKER b 1 0 4 <NA> <NA> X <NA> <NA>\nSPEAKER b 1 3 4 <NA> <NA> Y <NA> <NA>",
                                   "duration_s": 10}).encode()
                code, raw = self.call("/api/score", body)
            finally:
                del os.environ["VA_DSCORE_DIR"]
        js = json.loads(raw)
        self.assertEqual(code, 200, js)
        self.assertAlmostEqual(js["DER"], 19.6429, places=3)
        self.assertTrue(js["components_reconcile"])
        self.assertIn("Not part of the benchmark", js["note"])


if __name__ == "__main__":
    unittest.main()
