#!/usr/bin/env python3
"""Paired, recording-level bootstrap for comparing systems on pooled DER.

Statistic
    pooled DER = 100 * sum_r(miss_r + fa_r + conf_r) / sum_r(ref_speaker_time_r)
    delta(A, B) = pooledDER_A - pooledDER_B   (negative => A is better)

Procedure (declared constants below; change only via a methodology version bump)
    1. Keep per-recording error seconds for A and B and the shared reference
       speaker time, aligned by recording ID (paired).
    2. Draw B_REPLICATES resamples of recordings with replacement using a
       numpy Generator seeded with SEED. The same index draw is used for A
       and B (pairing preserved).
    3. Recompute pooled DER for A and B on each resample (ratio of sums, not
       mean of ratios) and store delta.
    4. 95 % percentile interval of delta; two-sided bootstrap p-value
       p = min(1, 2 * min(P*(delta <= 0), P*(delta >= 0))) with +1 smoothing.
    5. Holm-Bonferroni across all pairwise comparisons within one track
       and speaker-count condition (the comparison family).

Verdict per pair (proposed Voice Arena policy, not an established standard)
    insufficient_sample   n_recordings < MIN_RECORDINGS: no CI, no verdict
    practically_different Holm-adjusted p < ALPHA AND |delta| >= PRACTICAL_PP
    statistically_only    Holm-adjusted p < ALPHA AND |delta| <  PRACTICAL_PP
    not_distinguishable   otherwise
"""
from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, get_logger, read_json  # noqa: E402

SEED = 20261015
B_REPLICATES = 10_000
ALPHA = 0.05
PRACTICAL_PP = 1.0
MIN_RECORDINGS = 20

log = get_logger("bootstrap")


def pooled_der(err: np.ndarray, tot: np.ndarray) -> float:
    s = float(np.sum(tot))
    return float("nan") if s <= 0 else 100.0 * float(np.sum(err)) / s


def paired_bootstrap(err_a, err_b, tot, *, b: int = B_REPLICATES, seed: int = SEED) -> dict:
    err_a, err_b, tot = (np.asarray(x, dtype=float) for x in (err_a, err_b, tot))
    n = len(tot)
    if not (len(err_a) == len(err_b) == n) or n == 0:
        raise ValueError("paired arrays must be non-empty and the same length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))
    t = tot[idx].sum(axis=1)
    valid = t > 0
    da = 100.0 * err_a[idx].sum(axis=1)[valid] / t[valid]
    db = 100.0 * err_b[idx].sum(axis=1)[valid] / t[valid]
    delta = da - db
    point = pooled_der(err_a, tot) - pooled_der(err_b, tot)
    m = len(delta)
    p_le = (np.sum(delta <= 0) + 1) / (m + 1)
    p_ge = (np.sum(delta >= 0) + 1) / (m + 1)
    return {"n": n, "replicates": int(m), "seed": seed, "delta_pp": point,
            "ci95_low": float(np.percentile(delta, 2.5)),
            "ci95_high": float(np.percentile(delta, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min(p_le, p_ge)))}


def bootstrap_ci_single(err, tot, *, b: int = B_REPLICATES, seed: int = SEED) -> tuple[float, float]:
    """95 % percentile CI of one system's pooled DER (recording resampling)."""
    err, tot = np.asarray(err, float), np.asarray(tot, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(tot), size=(b, len(tot)))
    t = tot[idx].sum(axis=1)
    v = t > 0
    d = 100.0 * err[idx].sum(axis=1)[v] / t[v]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def verdict(n: int, p_adj: float | None, delta: float | None) -> str:
    if n < MIN_RECORDINGS or p_adj is None or delta is None:
        return "insufficient_sample"
    if p_adj < ALPHA:
        return "practically_different" if abs(delta) >= PRACTICAL_PP else "statistically_only"
    return "not_distinguishable"


def condition_class(text: str | None) -> str:
    """Runners describe the speaker-count condition in free text, e.g. 'unknown' or
    'unknown (no speaker-count parameters sent)'. Compare by class, not wording."""
    t = (text or "").strip().lower()
    if t.startswith("unknown"):
        return "unknown"
    if t.startswith("known"):
        return "known"
    return t or "unspecified"


def collect(scores: dict) -> dict[str, dict]:
    """Per system: aligned error seconds & reference time for scored recordings."""
    out = {}
    for sid, block in scores.get("systems", {}).items():
        st = block.get("run_status", {})
        rows = {}
        for r in block.get("recordings", []):
            c = r.get("components")
            if not c:
                continue
            rows[r["recording_id"]] = (c["missed_speaker_s"] + c["false_alarm_speaker_s"]
                                       + c["speaker_error_s"], c["scored_speaker_s"])
        if rows:
            out[sid] = {"track": st.get("track"),
                        "condition": condition_class(st.get("speaker_count_condition")), "rows": rows}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default=str(RESULTS_DIR / "scores_raw.json"))
    ap.add_argument("--out", default=str(RESULTS_DIR / "pairwise_bootstrap.csv"))
    args = ap.parse_args()
    data = collect(read_json(Path(args.scores), default={"systems": {}}))
    pairs = []
    for a, b in itertools.combinations(sorted(data), 2):
        if (data[a]["track"], data[a]["condition"]) != (data[b]["track"], data[b]["condition"]):
            continue  # never compare across tracks / speaker-count conditions
        common_ids = sorted(set(data[a]["rows"]) & set(data[b]["rows"]))
        pairs.append((a, b, common_ids))
    rows = []
    for a, b, ids in pairs:
        n = len(ids)
        base = {"system_a": a, "system_b": b, "track": data[a]["track"], "n_recordings": n,
                "replicates": B_REPLICATES, "seed": SEED, "alpha": ALPHA,
                "practical_threshold_pp": PRACTICAL_PP, "min_recordings": MIN_RECORDINGS}
        if n < MIN_RECORDINGS:
            base.update({"delta_pp": "", "ci95_low": "", "ci95_high": "", "p_two_sided": "",
                         "p_holm": "", "verdict": "insufficient_sample",
                         "note": f"n={n} < {MIN_RECORDINGS}: CI and significance not reported"})
            rows.append(base)
            continue
        ea = [data[a]["rows"][i][0] for i in ids]
        eb = [data[b]["rows"][i][0] for i in ids]
        tt = [data[a]["rows"][i][1] for i in ids]
        base.update(paired_bootstrap(ea, eb, tt))
        rows.append(base)
    tested = [r for r in rows if r.get("p_two_sided") not in ("", None)]
    for r, p in zip(tested, holm([r["p_two_sided"] for r in tested])):
        r["p_holm"] = p
        r["verdict"] = verdict(r["n_recordings"], p, r["delta_pp"])
        r.setdefault("note", "")
    cols = ["system_a", "system_b", "track", "n_recordings", "delta_pp", "ci95_low",
            "ci95_high", "p_two_sided", "p_holm", "verdict", "replicates", "seed", "alpha",
            "practical_threshold_pp", "min_recordings", "note"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d comparisons)", args.out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
