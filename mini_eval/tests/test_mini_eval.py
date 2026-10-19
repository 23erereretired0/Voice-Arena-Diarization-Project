"""Unit tests for the mini benchmark plumbing.

All data here is SYNTHETIC and exists only to test code paths. Nothing in
this file is a benchmark measurement. Run with:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import bootstrap as bs  # noqa: E402
import score as sc  # noqa: E402
from rttm_utils import (RTTMError, Segment, activity_profile, boundary_errors,  # noqa: E402
                        crop_segments, merge_same_speaker, parse_rttm_lines,
                        validate_rttm, words_to_segments, write_rttm)

FIX = HERE / "fixtures"


def rttm(lines: str):
    return parse_rttm_lines(io.StringIO(lines))


class TestParsing(unittest.TestCase):
    def test_valid(self):
        segs, warns = rttm("SPEAKER r1 1 0.50 1.00 <NA> <NA> A <NA> <NA>\n"
                           "SPKR-INFO r1 1 <NA> <NA> <NA> unknown A <NA> <NA>\n")
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0].end, 1.5)
        self.assertTrue(any("non-SPEAKER" in w for w in warns))

    def test_negative_onset_raises(self):
        with self.assertRaises(RTTMError):
            rttm("SPEAKER r1 1 -1.0 1.0 <NA> <NA> A <NA> <NA>\n")

    def test_non_numeric_raises(self):
        with self.assertRaises(RTTMError):
            rttm("SPEAKER r1 1 abc 1.0 <NA> <NA> A <NA> <NA>\n")

    def test_too_few_fields_raises(self):
        with self.assertRaises(RTTMError):
            rttm("SPEAKER r1 1 0.0\n")

    def test_zero_duration_dropped(self):
        segs, warns = rttm("SPEAKER r1 1 0.0 0.0 <NA> <NA> A <NA> <NA>\n")
        self.assertEqual(segs, [])
        self.assertTrue(warns)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file(self):
        rep = validate_rttm(self.d / "nope.rttm", "r1", 10.0)
        self.assertFalse(rep.ok)

    def test_file_id_mismatch_and_late_segment(self):
        p = self.d / "x.rttm"
        write_rttm([Segment("other", 0, 1, "A"), Segment("other", 9, 5, "B")], p)
        rep = validate_rttm(p, "r1", 10.0)
        self.assertFalse(rep.ok)
        self.assertEqual(len(rep.errors), 2)

    def test_empty_prediction_is_flagged_not_error(self):
        p = self.d / "e.rttm"
        p.write_text("")
        rep = validate_rttm(p, "r1", 10.0)
        self.assertTrue(rep.ok)
        self.assertTrue(rep.empty)


class TestNormalization(unittest.TestCase):
    def test_crop_and_shift(self):
        segs = [Segment("m", 290, 20, "A"), Segment("m", 400, 10, "B"), Segment("m", 595, 30, "C")]
        out = crop_segments(segs, 300, 600, "clip")
        self.assertEqual([(s.start, round(s.duration, 6), s.speaker) for s in out],
                         [(0.0, 10.0, "A"), (100.0, 10.0, "B"), (295.0, 5.0, "C")])
        self.assertTrue(all(s.file_id == "clip" for s in out))

    def test_merge_same_speaker(self):
        segs = [Segment("r", 0, 1, "A"), Segment("r", 0.5, 1, "A"), Segment("r", 3, 1, "A")]
        self.assertEqual(len(merge_same_speaker(segs)), 2)
        self.assertEqual(len(merge_same_speaker(segs, max_gap=2.0)), 1)

    def test_words_to_segments_bridges_small_gaps_only(self):
        words = [{"start": 0.0, "end": 0.4, "speaker": 0}, {"start": 0.5, "end": 0.9, "speaker": 0},
                 {"start": 1.5, "end": 2.0, "speaker": 0}, {"start": 2.0, "end": 2.5, "speaker": 1},
                 {"start": 3.0, "end": 3.0, "speaker": 1}, {"start": 4.0, "end": 4.5}]
        segs = words_to_segments(words, "r", max_gap=0.2)
        self.assertEqual([(s.speaker, s.start, round(s.end, 3)) for s in segs],
                         [("spk_0", 0.0, 0.9), ("spk_0", 1.5, 2.0), ("spk_1", 2.0, 2.5)])


class TestDiagnostics(unittest.TestCase):
    def test_overlap_ratio_exact(self):
        segs = [Segment("r", 0, 10, "A"), Segment("r", 8, 4, "B")]
        prof = activity_profile(segs)
        self.assertAlmostEqual(prof["speech_seconds"], 12.0)
        self.assertAlmostEqual(prof["overlap_seconds"], 2.0)
        self.assertAlmostEqual(prof["speaker_seconds"], 14.0)
        self.assertAlmostEqual(prof["overlap_ratio"], 2 / 12)

    def test_boundary_errors(self):
        ref = [Segment("r", 1.0, 2.0, "A")]       # boundaries 1.0, 3.0
        hyp = [Segment("r", 1.1, 2.2, "X")]       # boundaries 1.1, 3.3
        be = boundary_errors(ref, hyp)
        self.assertAlmostEqual(be["median_s"], 0.2)
        self.assertEqual(be["n_ref_boundaries"], 2)
        empty = boundary_errors(ref, [])
        self.assertEqual(empty["n_capped"], 2)

    def test_overlap_uem(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "o.uem"
            ok = sc.overlap_uem([Segment("r", 0, 10, "A"), Segment("r", 8, 4, "B")], "r", 20.0, p)
            self.assertTrue(ok)
            self.assertEqual(p.read_text().split(), ["r", "1", "8.000", "10.000"])
            self.assertFalse(sc.overlap_uem([Segment("r", 0, 5, "A")], "r", 20.0, Path(td) / "n.uem"))


class TestScorerParsers(unittest.TestCase):
    def test_md_eval_uses_overall_block(self):
        out = sc.parse_md_eval((FIX / "md_eval_sample.txt").read_text())
        self.assertAlmostEqual(out["scored_speaker_s"], 280.0)
        self.assertAlmostEqual(out["missed_speaker_s"], 20.0)
        self.assertAlmostEqual(out["false_alarm_speaker_s"], 14.0)
        self.assertAlmostEqual(out["speaker_error_s"], 21.0)
        recomputed = 100 * (20 + 14 + 21) / 280
        self.assertAlmostEqual(recomputed, out["der_percent"], places=2)

    def test_md_eval_real_output_format(self):
        # Exact line format printed by md-eval-22.pl (observed on a real run, Windows/Strawberry Perl)
        real = ("SCORED SPEAKER TIME =10.000000 secs\nMISSED SPEAKER TIME =0.000000 secs\n"
                "FALARM SPEAKER TIME =0.000000 secs\nSPEAKER ERROR TIME =0.000000 secs\n"
                " OVERALL SPEAKER DIARIZATION ERROR = 0.00 percent of scored speaker time  `(ALL)\n")
        out = sc.parse_md_eval(real)
        self.assertEqual(out["scored_speaker_s"], 10.0)
        self.assertEqual(out["false_alarm_speaker_s"], 0.0)

    def test_md_eval_missing_field(self):
        with self.assertRaises(ValueError):
            sc.parse_md_eval("nothing useful")

    def test_dscore_tsv_and_plain(self):
        for name in ("dscore_tsv_sample.txt", "dscore_plain_sample.txt"):
            rows = sc.parse_dscore_table((FIX / name).read_text())
            self.assertAlmostEqual(rows["rec1"]["DER"], 19.64, places=2)
            self.assertAlmostEqual(rows["rec1"]["JER"], 27.10, places=2)
            self.assertIn("*** OVERALL ***", rows)

    def test_empty_hypothesis_policy(self):
        s = sc.empty_hypothesis_scores(123.0)
        self.assertEqual(s["DER"], 100.0)
        self.assertEqual(s["components"]["missed_speaker_s"], 123.0)
        self.assertEqual(s["components"]["false_alarm_speaker_s"], 0.0)


class TestBootstrap(unittest.TestCase):
    def test_identical_systems(self):
        err, tot = [10, 20, 30, 40] * 10, [100] * 40
        r = bs.paired_bootstrap(err, err, tot, b=2000)
        self.assertEqual(r["delta_pp"], 0.0)
        self.assertEqual(r["ci95_low"], 0.0)
        self.assertEqual(r["ci95_high"], 0.0)
        self.assertEqual(r["p_two_sided"], 1.0)

    def test_clear_difference_is_significant_and_deterministic(self):
        tot = [100.0] * 60
        a = [10.0 + (i % 5) for i in range(60)]
        b = [x + 3.0 for x in a]
        r1 = bs.paired_bootstrap(a, b, tot, b=2000, seed=7)
        r2 = bs.paired_bootstrap(a, b, tot, b=2000, seed=7)
        self.assertEqual(r1, r2)
        self.assertAlmostEqual(r1["delta_pp"], -3.0)
        self.assertLess(r1["ci95_high"], 0)
        self.assertLess(r1["p_two_sided"], 0.01)

    def test_pooled_is_ratio_of_sums(self):
        import numpy as np
        self.assertAlmostEqual(bs.pooled_der(np.array([1.0, 50.0]), np.array([10.0, 100.0])),
                               100 * 51 / 110)

    def test_pooled_vs_macro_worked_example(self):
        # Methodology section 9.2 worked example: one scored file + one failed short file.
        import numpy as np
        err, tot = np.array([55.0, 20.0]), np.array([280.0, 20.0])
        self.assertAlmostEqual(bs.pooled_der(err, tot), 25.0)
        macro = float(np.mean(100 * err / tot))
        self.assertAlmostEqual(round(macro, 1), 59.8)

    def test_holm(self):
        adj = bs.holm([0.01, 0.04, 0.03])
        self.assertAlmostEqual(adj[0], 0.03)
        self.assertAlmostEqual(adj[2], 0.06)
        self.assertAlmostEqual(adj[1], 0.06)

    def test_condition_class_groups_equivalent_wording(self):
        self.assertEqual(bs.condition_class("unknown"), bs.condition_class("unknown (no speaker-count parameters sent)"))
        self.assertEqual(bs.condition_class("unknown (oracle_num_speakers=false)"), "unknown")
        self.assertEqual(bs.condition_class("known (reference count supplied)"), "known")
        self.assertNotEqual(bs.condition_class("unknown"), bs.condition_class("known"))

    def test_verdicts(self):
        self.assertEqual(bs.verdict(3, 0.001, -5.0), "insufficient_sample")
        self.assertEqual(bs.verdict(50, 0.001, -5.0), "practically_different")
        self.assertEqual(bs.verdict(50, 0.001, -0.4), "statistically_only")
        self.assertEqual(bs.verdict(50, 0.20, -5.0), "not_distinguishable")


if __name__ == "__main__":
    unittest.main()
