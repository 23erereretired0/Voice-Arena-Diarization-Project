#!/usr/bin/env python3
"""Prepare audio excerpts and reference RTTM/UEM files for the mini benchmark.

Steps per recording (configs/recordings.json):
  1. Download the original audio (or use --local-* overrides).
  2. Record SHA-256 of every downloaded file.
  3. Convert with ffmpeg to 16 kHz, mono, 16-bit PCM WAV and cut the
     declared excerpt window [excerpt_start_s, excerpt_end_s).
  4. Download the official reference RTTM, crop it to the same window,
     shift to t=0 and rename the file id to the recording id.
  5. Write a UEM covering [0, excerpt duration].
  6. Compute reference statistics (speakers, overlap ratio).
  7. Write audio/manifest_resolved.json (the only manifest systems read).

Requires network access and ffmpeg. Fails loudly per recording and keeps
going; a recording that cannot be prepared is listed with status "failed".
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (AUDIO_DIR, RECORDINGS_CONFIG, REF_DIR, RESOLVED_MANIFEST,  # noqa: E402
                    THIRD_PARTY, check_audio_compatible, get_logger, read_json,
                    require_executable, retry, run_cmd, utc_now, wav_duration,
                    write_json)
from rttm_utils import (activity_profile, crop_segments, parse_rttm_lines,  # noqa: E402
                        write_rttm, write_uem)

log = get_logger("prepare_data")
RAW_DIR = AUDIO_DIR / "raw"
HTTP_TIMEOUT = (15, 600)  # (connect, read) seconds


def _requests():
    try:
        import requests  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The 'requests' package is required: pip install requests") from exc
    return requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    requests = _requests()
    if dest.exists() and dest.stat().st_size > 0:
        log.info("cached: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _get():
        with requests.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        tmp.replace(dest)
        return dest

    log.info("downloading %s", url)
    return retry(_get, attempts=3, base_delay=3.0, logger=log)


def fetch_text(url: str) -> str:
    requests = _requests()

    def _get():
        r = requests.get(url, timeout=(15, 120))
        r.raise_for_status()
        return r.text

    return retry(_get, attempts=3, base_delay=2.0, logger=log)


def github_head_sha(repo: str, branch: str) -> str | None:
    try:
        requests = _requests()
        r = requests.get(f"https://api.github.com/repos/{repo}/commits/{branch}",
                         timeout=(10, 30))
        if r.ok:
            return r.json().get("sha")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not resolve commit for %s@%s: %s", repo, branch, exc)
    return None


def fetch_reference(ref_cfg: dict, substitutions: dict) -> tuple[str, dict]:
    """Download a reference RTTM from GitHub raw, trying declared branches."""
    path = ref_cfg["path"].format(**substitutions)
    errors = []
    for branch in ref_cfg.get("branches", ["main", "master"]):
        url = f"https://raw.githubusercontent.com/{ref_cfg['repo']}/{branch}/{path}"
        try:
            text = fetch_text(url)
            sha = github_head_sha(ref_cfg["repo"], branch)
            return text, {"url": url, "branch": branch, "commit": sha}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    raise RuntimeError("reference download failed: " + " | ".join(errors))


def resolve_vox_id(rec: dict, override: str | None) -> str:
    if override:
        return override
    if rec.get("vox_id") and rec["vox_id"] != "auto":
        return rec["vox_id"]
    requests = _requests()
    repo = rec["reference"]["repo"]
    r = requests.get(f"https://api.github.com/repos/{repo}/contents/test", timeout=(10, 60))
    r.raise_for_status()
    ids = sorted(item["name"][:-5] for item in r.json() if item["name"].endswith(".rttm"))
    if not ids:
        raise RuntimeError("no test RTTMs listed in VoxConverse repo")
    log.info("VoxConverse auto-selected id: %s (alphabetically first of %d)", ids[0], len(ids))
    return ids[0]


def extract_from_zip(zip_path: Path, member_basename: str, dest: Path) -> Path:
    with zipfile.ZipFile(zip_path) as zf:
        matches = [n for n in zf.namelist() if Path(n).name == member_basename]
        if not matches:
            raise RuntimeError(f"{member_basename} not found in {zip_path}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(matches[0]) as src, dest.open("wb") as out:
            out.write(src.read())
    return dest


def ffmpeg_excerpt(src: Path, dest: Path, start: float, end: float) -> None:
    ffmpeg = require_executable("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src),
           "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest)]
    proc = run_cmd(cmd, timeout=600)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()}")


def ffprobe_info(src: Path) -> dict:
    try:
        ffprobe = require_executable("ffprobe")
    except RuntimeError:
        return {}
    proc = run_cmd([ffprobe, "-v", "error", "-show_entries",
                    "stream=sample_rate,channels,codec_name:format=duration",
                    "-of", "json", str(src)], timeout=60)
    if proc.returncode != 0:
        return {}
    import json
    try:
        return json.loads(proc.stdout)
    except ValueError:
        return {}


# ---------------------------------------------------------------------------
# Condition tags and rule-based VoxConverse selection (pure, unit-tested)
# ---------------------------------------------------------------------------
def window_stats(segs, start: float, end: float) -> dict:
    """Reference statistics inside [start, end) after cropping."""
    cropped = crop_segments(segs, start, end, "w")
    prof = activity_profile(cropped, region=(0.0, end - start))
    ordered = sorted(cropped, key=lambda x: (x.start, x.speaker))
    changes = sum(1 for a, b in zip(ordered, ordered[1:]) if a.speaker != b.speaker)
    minutes = max((end - start) / 60.0, 1e-9)
    return {"n_speakers": prof["n_speakers"], "overlap_ratio": prof["overlap_ratio"],
            "speech_s": prof["speech_seconds"], "speaker_changes": changes,
            "speaker_changes_per_min": changes / minutes}


def n_speakers_bin(n: int) -> str:
    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    return "3-5" if n <= 5 else "6+"


def overlap_level(ratio: float) -> str:
    if ratio < 0.05:
        return "low"
    return "high" if ratio >= 0.15 else "medium"


def select_vox(slots: list[dict], stats_by_id: dict[str, dict]) -> dict[int, tuple[str, dict]]:
    """Apply each slot's rule in order; each file is used at most once.

    Rule: {"n_speakers": [lo, hi], "order": "file_id" | "overlap_ratio_desc" |
    "speaker_changes_per_min_desc"}. Returns {slot: (vox_id, stats)}; a slot
    with no candidate is omitted (the caller records it as failed).
    """
    used, out = set(), {}
    for slot in slots:
        rule = slot["vox_select"]
        lo, hi = rule["n_speakers"]
        cands = [i for i, st in stats_by_id.items()
                 if i not in used and lo <= st["n_speakers"] <= hi and st["speech_s"] > 0]
        order = rule.get("order", "file_id")
        if order == "overlap_ratio_desc":
            cands.sort(key=lambda i: (-stats_by_id[i]["overlap_ratio"], i))
        elif order == "speaker_changes_per_min_desc":
            cands.sort(key=lambda i: (-stats_by_id[i]["speaker_changes_per_min"], i))
        else:
            cands.sort()
        if cands:
            used.add(cands[0])
            out[slot["slot"]] = (cands[0], stats_by_id[cands[0]])
    return out


def resolve_vox_selections(recs: list[dict]) -> None:
    """Fetch all VoxConverse test RTTMs and fill vox_id for rule-based slots."""
    slots = [r for r in recs if r.get("vox_select")]
    if not slots:
        return
    requests = _requests()
    repo = slots[0]["reference"]["repo"]
    cache = REF_DIR / "raw" / "voxconverse_test"
    cache.mkdir(parents=True, exist_ok=True)
    r = requests.get(f"https://api.github.com/repos/{repo}/contents/test", timeout=(10, 60))
    r.raise_for_status()
    ids = sorted(item["name"][:-5] for item in r.json() if item["name"].endswith(".rttm"))
    stats = {}
    missing = [v for v in ids if not (cache / f"{v}.rttm").exists()]
    if missing:
        log.info("fetching %d VoxConverse reference RTTMs (one shared connection). Faster alternative: "
                 "git clone --depth 1 https://github.com/%s.git and copy test/*.rttm into %s", len(missing), repo, cache)
    session = requests.Session()
    for i, vid in enumerate(missing, 1):
        url = f"https://raw.githubusercontent.com/{repo}/master/test/{vid}.rttm"
        def _get(url=url):
            r = session.get(url, timeout=(15, 120)); r.raise_for_status(); return r.text
        (cache / f"{vid}.rttm").write_text(retry(_get, attempts=3, base_delay=2.0, logger=log), encoding="utf-8")
        if i % 25 == 0 or i == len(missing):
            log.info("  fetched %d/%d reference RTTMs", i, len(missing))
    for vid in ids:
        path = cache / f"{vid}.rttm"
        segs, _ = parse_rttm_lines(io.StringIO(path.read_text(encoding="utf-8")), source=str(path))
        w = slots[0]
        stats[vid] = window_stats(segs, w["excerpt_start_s"], w["excerpt_end_s"])
    chosen = select_vox(slots, stats)
    for rec in slots:
        if rec["slot"] in chosen:
            vid, st = chosen[rec["slot"]]
            rec["vox_id"] = vid
            rec["selection_stats"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in st.items()}
        else:
            rec["vox_id"] = None
            rec["selection_error"] = f"no VoxConverse test file satisfies rule {rec['vox_select']}"
    log.info("VoxConverse selections: %s", {r["slot"]: r["vox_id"] for r in slots})


def prepare_one(rec: dict, args) -> dict:
    out = dict(rec)
    out["prepared_utc"] = utc_now()
    subs = {}
    if rec.get("selection_error"):
        raise RuntimeError(rec["selection_error"])
    if rec["reference"]["kind"] == "voxconverse":
        vox_id = resolve_vox_id(rec, args.vox_id)
        subs["vox_id"] = vox_id
        out["vox_id"] = vox_id
        out["recording_id"] = rec["recording_id"].format(vox_id=vox_id)
    rid = out["recording_id"]

    # --- audio -------------------------------------------------------------
    if rec["reference"]["kind"] == "voxconverse":
        wav_name = f"{subs['vox_id']}.wav"
        raw = RAW_DIR / "voxconverse" / wav_name
        if not raw.exists():
            zip_path = Path(args.voxconverse_zip) if args.voxconverse_zip else \
                RAW_DIR / "voxconverse_test_wav.zip"
            if not zip_path.exists():
                download(rec["audio_zip_url"], zip_path)
            extract_from_zip(zip_path, wav_name, raw)
        out["audio_source"] = rec["audio_zip_url"]
    else:
        raw = RAW_DIR / "ami" / Path(rec["audio_url"]).name
        local = Path(args.ami_dir) / rec["meeting_id"] / "audio" / Path(rec["audio_url"]).name \
            if args.ami_dir else None
        if local and local.exists():
            raw = local
        else:
            download(rec["audio_url"], raw)
        out["audio_source"] = rec["audio_url"]
    out["raw_audio_sha256"] = sha256_file(raw)
    out["raw_audio_probe"] = ffprobe_info(raw)

    wav = AUDIO_DIR / f"{rid}.wav"
    ffmpeg_excerpt(raw, wav, rec["excerpt_start_s"], rec["excerpt_end_s"])
    problems = check_audio_compatible(wav)
    if problems:
        raise RuntimeError(f"prepared audio incompatible: {problems}")
    dur = wav_duration(wav)
    out["audio_path"] = str(wav.relative_to(AUDIO_DIR.parent))
    out["duration_s"] = round(dur, 3)
    out["audio_sha256"] = sha256_file(wav)
    out["preprocessing"] = ("ffmpeg: -ss start -t len; downmix -ac 1; resample -ar 16000; "
                            "pcm_s16le. No denoising, normalization or VAD.")

    # --- reference ---------------------------------------------------------
    text, ref_meta = fetch_reference(rec["reference"], subs)
    raw_ref = REF_DIR / "raw" / f"{rid}.orig.rttm"
    raw_ref.parent.mkdir(parents=True, exist_ok=True)
    raw_ref.write_text(text, encoding="utf-8")
    segs, warns = parse_rttm_lines(io.StringIO(text), source=ref_meta["url"])
    for w in warns:
        log.warning(w)
    end = rec["excerpt_start_s"] + dur  # actual excerpt length (file may be shorter)
    cropped = crop_segments(segs, rec["excerpt_start_s"], end, rid)
    if not cropped:
        raise RuntimeError("reference contains no speech inside the excerpt window")
    write_rttm(cropped, REF_DIR / f"{rid}.rttm")
    write_uem(REF_DIR / f"{rid}.uem", rid, 0.0, dur)
    stats = activity_profile(cropped, region=(0.0, dur))
    out["reference_meta"] = ref_meta
    out["reference_rttm"] = f"reference/{rid}.rttm"
    out["reference_uem"] = f"reference/{rid}.uem"
    out["n_reference_speakers"] = stats["n_speakers"]
    out["reference_speech_s"] = round(stats["speech_seconds"], 3)
    out["reference_speaker_time_s"] = round(stats["speaker_seconds"], 3)
    out["overlap_ratio"] = round(stats["overlap_ratio"], 4)
    wst = window_stats(cropped, 0.0, dur)
    out["speaker_changes_per_min"] = round(wst["speaker_changes_per_min"], 3)
    declared = rec.get("tags_declared", {})
    out["tags"] = {
        "recording_type": declared.get("recording_type", "unknown"),
        "n_speakers": n_speakers_bin(stats["n_speakers"]),
        "overlap_level": overlap_level(stats["overlap_ratio"]),
        "audio_quality": declared.get("audio_quality", "unknown"),
        "mic_distance": declared.get("mic_distance", "unknown"),
    }
    out["tag_evidence"] = dict(rec.get("tag_evidence", {}),
                               n_speakers=f"{stats['n_speakers']} speakers in cropped reference",
                               overlap_level=f"reference overlap ratio {stats['overlap_ratio']:.3f}")
    out["annotation_quality_status"] = "official public reference (not re-adjudicated by Voice Arena)"
    out["status"] = "ok"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vox-id", help="Deprecated: rejected (VoxConverse files are chosen by vox_select rules)")
    ap.add_argument("--voxconverse-zip", help="Path to an already-downloaded voxconverse_test_wav.zip")
    ap.add_argument("--ami-dir", help="Local AMI 'amicorpus' directory (skips audio download)")
    args = ap.parse_args()

    cfg = read_json(RECORDINGS_CONFIG)
    resolved, n_fail = [], 0
    if args.vox_id:
        log.error("--vox-id is not supported with rule-based selection (it would assign one file "
                  "to every VoxConverse slot). Edit vox_select rules in configs/recordings.json instead.")
        return 2
    if True:
        try:
            resolve_vox_selections(cfg["recordings"])
        except Exception as exc:  # noqa: BLE001
            log.error("VoxConverse selection failed: %s", exc)
            for r in cfg["recordings"]:
                if r.get("vox_select"):
                    r["selection_error"] = f"VoxConverse selection failed: {exc}"
    for rec in cfg["recordings"]:
        try:
            resolved.append(prepare_one(rec, args))
            log.info("prepared %s", resolved[-1]["recording_id"])
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            log.error("FAILED %s: %s", rec["recording_id"], exc)
            failed = dict(rec, status="failed", error=str(exc), prepared_utc=utc_now())
            if rec.get("vox_id") and "{vox_id}" in failed["recording_id"]:
                failed["recording_id"] = failed["recording_id"].format(vox_id=rec["vox_id"])
            resolved.append(failed)
    ok = [r for r in resolved if r.get("status") == "ok"]
    write_json(RESOLVED_MANIFEST, {
        "created_utc": utc_now(),
        "excerpt_policy": cfg.get("excerpt_policy"),
        "recordings": ok,
        "failed_recordings": [r for r in resolved if r.get("status") != "ok"],
    })
    log.info("wrote %s (%d ok, %d failed)", RESOLVED_MANIFEST, len(ok), n_fail)
    THIRD_PARTY.mkdir(exist_ok=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
