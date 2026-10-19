"""RTTM utilities: parse, validate, write, crop, and diagnostic statistics.

RTTM line format (10 space-delimited fields, see dscore README / DIHARD III
evaluation plan appendix C):

    SPEAKER <file-id> <chan> <onset> <dur> <NA> <NA> <speaker> <NA> <NA>

Scoring itself is NOT implemented here: DER/JER/components come from the
pinned dscore checkout (score.py + bundled md-eval-22.pl). The functions
below are for validation, normalization and the Voice Arena *diagnostic*
statistics (speaker count, overlap ratio, boundary error) that dscore does
not report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# Floating-point tolerance used when comparing segment ends to durations.
EPS = 1e-6


class RTTMError(ValueError):
    """Raised when an RTTM file cannot be parsed."""


@dataclass(frozen=True)
class Segment:
    file_id: str
    start: float
    duration: float
    speaker: str

    @property
    def end(self) -> float:
        return self.start + self.duration


@dataclass
class ValidationReport:
    path: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_segments: int = 0
    n_speakers: int = 0
    file_ids: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def empty(self) -> bool:
        return self.n_segments == 0

    def as_dict(self) -> dict:
        return {"path": self.path, "ok": self.ok, "empty": self.empty,
                "errors": self.errors, "warnings": self.warnings,
                "n_segments": self.n_segments, "n_speakers": self.n_speakers,
                "file_ids": self.file_ids}


# --------------------------------------------------------------------------
# Parsing / writing
# --------------------------------------------------------------------------
def parse_rttm_lines(lines: Iterable[str], source: str = "<memory>",
                     strict: bool = True) -> tuple[list[Segment], list[str]]:
    """Parse RTTM lines. Returns (segments, warnings).

    In strict mode, malformed SPEAKER lines raise RTTMError. Non-SPEAKER
    record types (e.g. SPKR-INFO) are skipped with a warning.
    """
    segments: list[Segment] = []
    warnings: list[str] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        fields = line.split()
        if fields[0] != "SPEAKER":
            warnings.append(f"{source}:{lineno}: skipped non-SPEAKER record '{fields[0]}'")
            continue
        if len(fields) < 9:
            msg = f"{source}:{lineno}: expected 10 fields, found {len(fields)}"
            if strict:
                raise RTTMError(msg)
            warnings.append(msg)
            continue
        if len(fields) != 10:
            warnings.append(f"{source}:{lineno}: expected 10 fields, found {len(fields)}")
        try:
            onset = float(fields[3])
            dur = float(fields[4])
        except ValueError as exc:
            raise RTTMError(f"{source}:{lineno}: non-numeric onset/duration ({exc})") from exc
        if any(map(math.isnan, (onset, dur))) or any(map(math.isinf, (onset, dur))):
            raise RTTMError(f"{source}:{lineno}: NaN/inf onset or duration")
        if onset < -EPS:
            raise RTTMError(f"{source}:{lineno}: negative onset {onset}")
        if dur < 0:
            raise RTTMError(f"{source}:{lineno}: negative duration {dur}")
        if dur <= EPS:
            warnings.append(f"{source}:{lineno}: zero-duration segment dropped")
            continue
        segments.append(Segment(fields[1], max(0.0, onset), dur, fields[7]))
    return segments, warnings


def read_rttm(path: Path, strict: bool = True) -> tuple[list[Segment], list[str]]:
    if not Path(path).exists():
        raise FileNotFoundError(f"RTTM not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return parse_rttm_lines(fh, source=str(path), strict=strict)


def format_rttm_line(seg: Segment) -> str:
    return (f"SPEAKER {seg.file_id} 1 {seg.start:.3f} {seg.duration:.3f} "
            f"<NA> <NA> {seg.speaker} <NA> <NA>")


def write_rttm(segments: Sequence[Segment], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(segments, key=lambda s: (s.file_id, s.start, s.speaker))
    with path.open("w", encoding="utf-8") as fh:
        for seg in ordered:
            fh.write(format_rttm_line(seg) + "\n")


def write_uem(path: Path, file_id: str, start: float, end: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{file_id} 1 {start:.3f} {end:.3f}\n")


def sanitize_speaker(label: object) -> str:
    """RTTM speaker names may not contain whitespace."""
    text = str(label).strip()
    text = "_".join(text.split())
    return text or "unknown"


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def merge_same_speaker(segments: Sequence[Segment], max_gap: float = 0.0) -> list[Segment]:
    """Merge segments of the same speaker that overlap or are <= max_gap apart."""
    out: list[Segment] = []
    by_key: dict[tuple[str, str], list[Segment]] = {}
    for s in segments:
        by_key.setdefault((s.file_id, s.speaker), []).append(s)
    for (fid, spk), segs in by_key.items():
        segs = sorted(segs, key=lambda s: s.start)
        cur_start, cur_end = segs[0].start, segs[0].end
        for s in segs[1:]:
            if s.start <= cur_end + max_gap + EPS:
                cur_end = max(cur_end, s.end)
            else:
                out.append(Segment(fid, cur_start, cur_end - cur_start, spk))
                cur_start, cur_end = s.start, s.end
        out.append(Segment(fid, cur_start, cur_end - cur_start, spk))
    return sorted(out, key=lambda s: (s.file_id, s.start, s.speaker))


def crop_segments(segments: Sequence[Segment], start: float, end: float,
                  new_file_id: str) -> list[Segment]:
    """Keep the part of each segment inside [start, end) and shift to t=0."""
    if end <= start:
        raise ValueError(f"crop window end ({end}) must exceed start ({start})")
    out: list[Segment] = []
    for s in segments:
        a, b = max(s.start, start), min(s.end, end)
        if b - a > EPS:
            out.append(Segment(new_file_id, a - start, b - a, s.speaker))
    return out


def words_to_segments(words: Sequence[dict], file_id: str,
                      max_gap: float = 0.2) -> list[Segment]:
    """Convert word-level speaker labels (STT APIs) into RTTM segments.

    `words` items need keys: start (s), end (s), speaker. Consecutive words
    of the same speaker separated by <= max_gap seconds are bridged. The
    0.2 s default mirrors the DIHARD III convention of bridging pauses of
    200 ms or less; it is a declared Voice Arena normalization setting.
    """
    segs: list[Segment] = []
    for w in words:
        try:
            st, en = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError):
            continue
        spk = w.get("speaker")
        if spk is None or en - st <= EPS:
            continue
        segs.append(Segment(file_id, st, en - st, f"spk_{sanitize_speaker(spk)}"))
    return merge_same_speaker(segs, max_gap=max_gap)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_rttm(path: Path, expected_file_id: str | None = None,
                  audio_duration: float | None = None,
                  end_tolerance: float = 0.5) -> ValidationReport:
    """Validate a prediction/reference RTTM against Voice Arena rules."""
    rep = ValidationReport(path=str(path))
    p = Path(path)
    if not p.exists():
        rep.errors.append("file does not exist")
        return rep
    try:
        segs, warns = read_rttm(p, strict=True)
    except RTTMError as exc:
        rep.errors.append(f"parse error: {exc}")
        return rep
    except UnicodeDecodeError:
        rep.errors.append("file is not valid UTF-8 text")
        return rep
    rep.warnings.extend(warns)
    rep.n_segments = len(segs)
    rep.file_ids = sorted({s.file_id for s in segs})
    rep.n_speakers = len({s.speaker for s in segs})
    if not segs:
        rep.warnings.append("no SPEAKER segments (empty prediction)")
        return rep
    if expected_file_id is not None:
        wrong = [f for f in rep.file_ids if f != expected_file_id]
        if wrong:
            rep.errors.append(
                f"file-id mismatch: expected '{expected_file_id}', found {wrong}")
    if audio_duration is not None:
        late = [s for s in segs if s.end > audio_duration + end_tolerance]
        if late:
            rep.errors.append(
                f"{len(late)} segment(s) end after audio duration "
                f"{audio_duration:.3f}s (+{end_tolerance}s tolerance); "
                f"max end {max(s.end for s in late):.3f}s")
    # Same-speaker self-overlap is legal for md-eval but usually a bug.
    merged = merge_same_speaker(segs)
    if len(merged) < len(segs):
        rep.warnings.append(
            f"{len(segs) - len(merged)} same-speaker overlapping/adjacent "
            "segment(s); normalization will merge them")
    return rep


# --------------------------------------------------------------------------
# Diagnostics (Voice Arena-defined; NOT produced by dscore)
# --------------------------------------------------------------------------
def _boundaries(segments: Sequence[Segment]) -> list[float]:
    pts = []
    for s in segments:
        pts.extend((s.start, s.end))
    return sorted(pts)


def activity_profile(segments: Sequence[Segment],
                     region: tuple[float, float] | None = None) -> dict:
    """Return speech time, overlap time and overlap ratio via a sweep line.

    overlap_ratio = time with >=2 active speakers / time with >=1 active speaker
    """
    events: list[tuple[float, int]] = []
    for s in segments:
        a, b = s.start, s.end
        if region is not None:
            a, b = max(a, region[0]), min(b, region[1])
        if b - a > EPS:
            events.append((a, +1))
            events.append((b, -1))
    events.sort(key=lambda e: (e[0], e[1]))
    speech = overlap = speaker_time = 0.0
    active = 0
    prev_t = None
    for t, delta in events:
        if prev_t is not None and t > prev_t:
            span = t - prev_t
            if active >= 1:
                speech += span
                speaker_time += span * active
            if active >= 2:
                overlap += span
        active += delta
        prev_t = t
    return {
        "speech_seconds": speech,
        "overlap_seconds": overlap,
        "speaker_seconds": speaker_time,
        "overlap_ratio": (overlap / speech) if speech > 0 else 0.0,
        "n_speakers": len({s.speaker for s in segments}),
    }


def boundary_errors(reference: Sequence[Segment], hypothesis: Sequence[Segment],
                    max_distance: float = 5.0) -> dict:
    """Distance from each reference turn boundary to the nearest hypothesis boundary.

    Speaker identity is ignored: this measures timing only. Distances are
    capped at `max_distance` seconds so that a missing region does not
    produce unbounded values; capped boundaries are counted separately.
    Proposed Voice Arena diagnostic, not an established standard metric.
    """
    ref_b = _boundaries(merge_same_speaker(reference))
    hyp_b = _boundaries(merge_same_speaker(hypothesis))
    if not ref_b:
        return {"n_ref_boundaries": 0, "median_s": None, "p95_s": None,
                "n_capped": 0, "distances_s": []}
    if not hyp_b:
        return {"n_ref_boundaries": len(ref_b), "median_s": max_distance,
                "p95_s": max_distance, "n_capped": len(ref_b),
                "distances_s": [max_distance] * len(ref_b)}
    import bisect
    dists = []
    for b in ref_b:
        i = bisect.bisect_left(hyp_b, b)
        cands = []
        if i < len(hyp_b):
            cands.append(abs(hyp_b[i] - b))
        if i > 0:
            cands.append(abs(hyp_b[i - 1] - b))
        dists.append(min(min(cands), max_distance))
    return {
        "n_ref_boundaries": len(ref_b),
        "median_s": percentile(dists, 50),
        "p95_s": percentile(dists, 95),
        "n_capped": sum(1 for d in dists if d >= max_distance - EPS),
        # kept so boundary statistics can be pooled over any subset of recordings
        "distances_s": [round(d, 4) for d in dists],
    }


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (same convention as numpy default)."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * (q / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(xs[int(k)])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (k - lo))
