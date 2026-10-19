/*
 * Voice Arena aggregation: recomputes leaderboard metrics from recording-level
 * results (window.VA_BENCHMARK.recordings). Pure functions, no DOM.
 * Definitions follow docs/methodology.md §8.2:
 *   pooled DER  = Σ(miss + FA + conf) / Σ reference speaker time × 100   (official rank)
 *   macro DER   = mean of per-recording DER;  median DER = median of per-recording DER
 *   JER         = mean of per-recording JER (dscore)
 *   overlap / non-overlap DER pooled from md-eval seconds on those regions
 *   speaker-count MAE = mean |predicted − reference| speakers (failed file counts as 0 predicted)
 *   boundary median / p95 pooled over all reference boundaries
 *   RTF = Σ inference seconds / Σ audio seconds;  failure rate = failed / attempted × 100
 * Failed recordings were already scored by score.py as empty hypotheses (DER 100 %),
 * so they stay in every denominator.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.VAAggregate = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";
  var MIN_FILTER_RECORDINGS = 3;   // below this, no customized ranking is shown
  var MIN_CI_RECORDINGS = 20;      // methodology §10.4
  var FILTER_DIMS = ["recording_type", "n_speakers", "overlap_level", "audio_quality", "mic_distance"];
  var FAILED = { failed: 1, timeout: 1, invalid_output: 1, missing_prediction: 1, empty_output: 1 };

  function has(x) { return x !== null && x !== undefined && !(typeof x === "number" && isNaN(x)); }
  function err(c) { return c.missed_speaker_s + c.false_alarm_speaker_s + c.speaker_error_s; }
  function median(xs) { return percentile(xs, 50); }
  function percentile(xs, q) {
    if (!xs.length) return null;
    var s = xs.slice().sort(function (a, b) { return a - b; });
    if (s.length === 1) return s[0];
    var k = (s.length - 1) * q / 100, lo = Math.floor(k), hi = Math.ceil(k);
    return s[lo] + (s[hi] - s[lo]) * (k - lo);
  }
  function pooled(list) {
    var t = 0, e = 0;
    list.forEach(function (c) { if (c && c.scored_speaker_s > 0) { t += c.scored_speaker_s; e += err(c); } });
    return t > 0 ? 100 * e / t : null;
  }

  function filterRecordings(recs, filters) {
    filters = filters || {};
    return recs.filter(function (r) {
      return FILTER_DIMS.every(function (d) {
        var want = filters[d];
        if (!want || want === "all") return true;
        var have = r.tags ? r.tags[d] : undefined;
        return have === want;           // "unknown" or missing never matches a specific value
      });
    });
  }

  function aggregate(recs, sid) {
    var rows = recs.map(function (r) { return r.results ? r.results[sid] : null; }).filter(Boolean);
    var attempted = rows.filter(function (x) { return x.run_status && x.run_status !== "not_run" && x.run_status.indexOf("blocked") !== 0; });
    var scored = rows.filter(function (x) { return x.components && x.components.scored_speaker_s > 0; });
    var failed = rows.filter(function (x) { return x.failed || FAILED[x.run_status]; }).length;
    var comps = scored.map(function (x) { return x.components; });
    var tot = comps.reduce(function (a, c) { return a + c.scored_speaker_s; }, 0);
    var ders = scored.map(function (x) { return x.DER; }).filter(has);
    var jers = scored.map(function (x) { return x.JER; }).filter(has);
    var dists = [];
    scored.forEach(function (x) { (x.boundary_distances_s || []).forEach(function (d) { dists.push(d); }); });
    var mae = [];
    recs.forEach(function (r) {
      var x = r.results && r.results[sid];
      if (x && x.components && has(r.n_reference_speakers)) mae.push(Math.abs((x.n_predicted_speakers || 0) - r.n_reference_speakers));
    });
    var proc = 0, aud = 0;
    rows.forEach(function (x) { if (has(x.processing_seconds) && x.audio_seconds > 0) { proc += x.processing_seconds; aud += x.audio_seconds; } });
    var comp = function (k) { return tot > 0 ? 100 * comps.reduce(function (a, c) { return a + c[k]; }, 0) / tot : null; };
    return {
      system_id: sid, n_recordings: rows.length, n_attempted: attempted.length, n_scored: scored.length, n_failed: failed,
      pooled_der: pooled(comps), macro_der: ders.length ? ders.reduce(function (a, b) { return a + b; }, 0) / ders.length : null,
      median_der: median(ders), jer: jers.length ? jers.reduce(function (a, b) { return a + b; }, 0) / jers.length : null,
      missed: comp("missed_speaker_s"), false_alarm: comp("false_alarm_speaker_s"), confusion: comp("speaker_error_s"),
      overlap_der: pooled(scored.map(function (x) { return x.overlap_region; })),
      non_overlap_der: pooled(scored.map(function (x) { return x.non_overlap; })),
      speaker_mae: mae.length ? mae.reduce(function (a, b) { return a + b; }, 0) / mae.length : null,
      boundary_median: percentile(dists, 50), boundary_p95: percentile(dists, 95),
      runtime_s: aud > 0 ? proc : null, rtf: aud > 0 ? proc / aud : null,
      failure_rate: attempted.length ? 100 * failed / attempted.length : null,
      cost_per_hour: null, // no documented price inputs recorded for this benchmark
      ci: scored.length >= MIN_CI_RECORDINGS ? "see pairwise_bootstrap.csv" : null
    };
  }

  function rank(recs, systemIds) {
    var aggs = systemIds.map(function (s) { return aggregate(recs, s); });
    aggs.sort(function (a, b) {
      if (!has(a.pooled_der) && !has(b.pooled_der)) return 0;
      if (!has(a.pooled_der)) return 1;
      if (!has(b.pooled_der)) return -1;
      return a.pooled_der - b.pooled_der;
    });
    var r = 0;
    aggs.forEach(function (a) { a.rank = has(a.pooled_der) ? ++r : null; });
    return aggs;
  }

  function evidence(recs, systemIds) {
    var scoredRecs = recs.filter(function (r) {
      return systemIds.some(function (s) { return r.results && r.results[s] && r.results[s].components; });
    });
    var known = recs.filter(function (r) { return has(r.duration_s); });
    var dur = known.length ? known.reduce(function (a, r) { return a + r.duration_s; }, 0) : null;
    var ok = scoredRecs.length >= MIN_FILTER_RECORDINGS;
    return { n_matching: recs.length, n_scored: scoredRecs.length, duration_s: dur, sufficient: ok,
      reason: ok ? null : (recs.length === 0 ? "No recording matches these conditions."
        : "Only " + scoredRecs.length + " scored recording(s) match; at least " + MIN_FILTER_RECORDINGS +
          " are required before systems are ranked, and even then the result is exploratory.") };
  }

  return { filterRecordings: filterRecordings, aggregate: aggregate, rank: rank, evidence: evidence,
    percentile: percentile, FILTER_DIMS: FILTER_DIMS, MIN_FILTER_RECORDINGS: MIN_FILTER_RECORDINGS,
    MIN_CI_RECORDINGS: MIN_CI_RECORDINGS };
});
