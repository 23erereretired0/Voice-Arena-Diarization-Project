// node dashboard/tests/aggregate.test.js  — SYNTHETIC data only; tests the math, not models.
"use strict";
const assert = require("node:assert/strict");
const A = require("../aggregate.js");
let n = 0; const t = (name, fn) => { fn(); n++; console.log("ok -", name); };
const C = (m, f, c, tot) => ({ missed_speaker_s: m, false_alarm_speaker_s: f, speaker_error_s: c, scored_speaker_s: tot });
const res = (comp, extra = {}) => Object.assign({ run_status: "ok", components: comp, DER: 100 * (comp.missed_speaker_s + comp.false_alarm_speaker_s + comp.speaker_error_s) / comp.scored_speaker_s, JER: 30, n_predicted_speakers: 3, processing_seconds: 10, audio_seconds: 100, boundary_distances_s: [0.1, 0.2] }, extra);
const recs = [
  { recording_id: "r1", duration_s: 100, n_reference_speakers: 3, tags: { recording_type: "meeting", n_speakers: "3-5", overlap_level: "high", audio_quality: "unknown", mic_distance: "far-field" },
    results: { a: res(C(20, 15, 20, 280)), b: res(C(10, 5, 5, 280)) } },
  { recording_id: "r2", duration_s: 20, n_reference_speakers: 2, tags: { recording_type: "interview/media", n_speakers: "2", overlap_level: "low", audio_quality: "unknown", mic_distance: "unknown" },
    results: { a: res(C(20, 0, 0, 20), { run_status: "failed", failed: true, n_predicted_speakers: 0, processing_seconds: null }), b: res(C(1, 1, 0, 20)) } },
  { recording_id: "r3", duration_s: 60, n_reference_speakers: 6, tags: { recording_type: "meeting", n_speakers: "6+", overlap_level: "medium", audio_quality: "unknown", mic_distance: "close-talk" },
    results: { a: res(C(5, 5, 5, 100)), b: res(C(30, 5, 5, 100)) } },
];
t("pooled vs macro worked example (methodology §9.2)", () => {
  const two = [recs[0], recs[1]];
  const g = A.aggregate(two, "a");
  assert.equal(g.pooled_der.toFixed(1), "25.0");
  assert.equal(g.macro_der.toFixed(1), "59.8");
});
t("components reconcile with pooled DER", () => {
  const g = A.aggregate(recs, "a");
  assert.ok(Math.abs(g.missed + g.false_alarm + g.confusion - g.pooled_der) < 1e-9);
});
t("failed recording stays in denominator and failure rate", () => {
  const g = A.aggregate(recs, "a");
  assert.equal(g.n_scored, 3); assert.equal(g.n_failed, 1);
  assert.equal(g.failure_rate.toFixed(1), "33.3");
  assert.equal(g.speaker_mae.toFixed(3), ((0 + 2 + 3) / 3).toFixed(3));   // failed counts as 0 predicted
});
t("RTF uses only recordings with measured runtime", () => {
  const g = A.aggregate(recs, "a"); assert.equal(g.rtf, 20 / 200); assert.equal(g.runtime_s, 20);
});
t("boundary median/p95 pooled across recordings", () => {
  const g = A.aggregate(recs, "a"); assert.equal(g.boundary_median.toFixed(2), "0.15");
});
t("rank follows pooled DER (lower first)", () => {
  const r = A.rank(recs, ["a", "b"]);
  assert.deepEqual(r.map(x => [x.system_id, x.rank]), [["b", 1], ["a", 2]]);   // b 52/400=13.0, a 90/400=22.5
});
t("filters use only matching recordings; unknown never matches a specific value", () => {
  assert.deepEqual(A.filterRecordings(recs, { recording_type: "meeting" }).map(r => r.recording_id), ["r1", "r3"]);
  assert.equal(A.filterRecordings(recs, { audio_quality: "clean" }).length, 0);
  assert.equal(A.filterRecordings(recs, { mic_distance: "close-talk" }).length, 1);
  assert.equal(A.filterRecordings(recs, { recording_type: "all", n_speakers: "all" }).length, 3);
});
t("filtered ranking recomputed from the subset (can differ from overall)", () => {
  const sub = A.filterRecordings(recs, { n_speakers: "6+" });
  assert.equal(A.rank(sub, ["a", "b"])[0].system_id, "a");
});
t("insufficient evidence below 3 scored recordings", () => {
  assert.equal(A.evidence(A.filterRecordings(recs, { recording_type: "meeting" }), ["a", "b"]).sufficient, false);
  assert.equal(A.evidence(recs, ["a", "b"]).sufficient, true);
  assert.match(A.evidence([], ["a"]).reason, /No recording/);
});
t("CI only at >= 20 recordings", () => { assert.equal(A.aggregate(recs, "a").ci, null); });
t("not-run data yields no numbers and no ranks", () => {
  const nr = [{ recording_id: "x", tags: {}, results: { a: { run_status: "blocked_data_unavailable", components: null } } }];
  const g = A.rank(nr, ["a"])[0];
  assert.equal(g.pooled_der, null); assert.equal(g.rank, null); assert.equal(g.failure_rate, null);
});
console.log(`\n${n} aggregate tests passed`);
