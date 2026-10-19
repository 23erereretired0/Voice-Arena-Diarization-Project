#!/usr/bin/env python3
"""Run NVIDIA NeMo MSDD diarization (TitaNet-L embeddings) on prepared recordings.

Configuration
-------------
Base config: examples/speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml
from the NeMo git tag matching the installed nemo_toolkit version (or a
local copy via --base-config). Voice Arena overrides (primary track):

    diarizer.oracle_vad                         = False   (system VAD)
    diarizer.clustering.parameters.oracle_num_speakers = False
    diarizer.clustering.parameters.max_num_speakers    = 8
    diarizer.vad.model_path                     = vad_multilingual_marblenet
    diarizer.speaker_embeddings.model_path      = titanet_large
    diarizer.msdd_model.model_path              = diar_msdd_telephonic

The fully resolved YAML is saved to results/configs/nemo_msdd_resolved.yaml.
Model names above are NGC/HF identifiers documented by NeMo; verify that
they resolve for the pinned NeMo version and record any substitution.

Timing note: NeuralDiarizer loads its sub-models when instantiated. To get
per-recording timings the diarizer is instantiated once and run on a
one-file manifest per recording; model download/loading is measured
separately as model_load_seconds. A GPU is strongly recommended; CPU runs
are allowed but their RTF must be labelled as CPU.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (PRED_DIR, RESULTS_DIR, ROOT, STATUS_BLOCKED_DEPENDENCY,  # noqa: E402
                    STATUS_EMPTY_OUTPUT, STATUS_FAILED, STATUS_OK, RecordingRun,
                    SystemRunStatus, blocked_status, check_audio_compatible,
                    get_logger, load_manifest, manifest_or_blocked, utc_now)
from rttm_utils import read_rttm, sanitize_speaker, write_rttm, Segment  # noqa: E402

log = get_logger("run_nemo_msdd")
SYSTEM_ID = "nemo-msdd"
CONFIG_URL = ("https://raw.githubusercontent.com/NVIDIA/NeMo/{tag}/examples/"
              "speaker_tasks/diarization/conf/inference/diar_infer_meeting.yaml")
OVERRIDES = {
    "diarizer.oracle_vad": False,
    "diarizer.clustering.parameters.oracle_num_speakers": False,
    "diarizer.clustering.parameters.max_num_speakers": 8,
    "diarizer.vad.model_path": "vad_multilingual_marblenet",
    "diarizer.speaker_embeddings.model_path": "titanet_large",
    "diarizer.msdd_model.model_path": "diar_msdd_telephonic",
}


def fetch_base_config(tag: str, dest: Path) -> Path:
    import requests
    url = CONFIG_URL.format(tag=tag)
    r = requests.get(url, timeout=(10, 60))
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(r.text, encoding="utf-8")
    log.info("fetched base config %s", url)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-config", help="Local diar_infer_meeting.yaml (else fetched by tag)")
    ap.add_argument("--nemo-tag", help="NeMo git tag for the base config, e.g. v2.x.y "
                                       "(default: v<installed nemo version>)")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    out_dir = PRED_DIR / SYSTEM_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    track, condition = "audio_only_batch", "unknown (oracle_num_speakers=false)"
    model_identifier = "NeMo NeuralDiarizer (MSDD) + titanet_large + vad_multilingual_marblenet"
    recordings = manifest_or_blocked(SYSTEM_ID, "run_nemo_msdd.py", track, condition,
                                     model_identifier, out_dir)
    if recordings is None:
        return 5

    try:
        import torch
        import nemo
        from omegaconf import OmegaConf
        from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    except ImportError as exc:
        reason = f"nemo_toolkit[asr]/torch/omegaconf not importable: {exc}"
        log.error(reason)
        blocked_status(SYSTEM_ID, "run_nemo_msdd.py", track, condition, model_identifier,
                       reason, STATUS_BLOCKED_DEPENDENCY, recordings).save(out_dir)
        return 2

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("No GPU in use: NeMo MSDD will be slow; RTF will be reported as CPU RTF.")

    tag = args.nemo_tag or f"v{nemo.__version__}"
    try:
        base = Path(args.base_config) if args.base_config else \
            fetch_base_config(tag, RESULTS_DIR / "configs" / f"diar_infer_meeting_{tag}.yaml")
        cfg = OmegaConf.load(base)
    except Exception as exc:  # noqa: BLE001
        reason = f"could not obtain base config (tag {tag}): {exc}"
        log.error(reason)
        blocked_status(SYSTEM_ID, "run_nemo_msdd.py", track, condition, model_identifier,
                       reason, STATUS_BLOCKED_DEPENDENCY, recordings).save(out_dir)
        return 3

    for key, val in OVERRIDES.items():
        OmegaConf.update(cfg, key, val, merge=True)
    cfg.device = device
    cfg.num_workers = 0
    cfg.batch_size = 32 if device == "cuda" else 8
    # Manifest + out_dir are set per recording below.

    status = SystemRunStatus(
        system_id=SYSTEM_ID, runner="run_nemo_msdd.py", track=track,
        speaker_count_condition=condition, model_identifier=model_identifier,
        model_revision=f"nemo_toolkit {nemo.__version__}; base config tag {tag}",
        package_versions={"nemo_toolkit": nemo.__version__, "torch": torch.__version__,
                          "python": platform.python_version()},
        hardware={"device": device,
                  "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
                  "os": platform.platform()},
        settings={"base_config": str(base), "overrides": OVERRIDES},
        run_started_utc=utc_now(),
    )

    for rec in recordings:
        rid = rec["recording_id"]
        run = RecordingRun(rid, audio_seconds=rec["duration_s"], started_utc=utc_now())
        wav = ROOT / rec["audio_path"]
        problems = check_audio_compatible(wav)
        if problems:
            run.status, run.error = STATUS_FAILED, f"incompatible audio: {problems}"
            status.recordings.append(run)
            continue
        rec_dir = work / rid
        shutil.rmtree(rec_dir, ignore_errors=True)
        rec_dir.mkdir(parents=True)
        manifest = rec_dir / "manifest.json"
        manifest.write_text(json.dumps({
            "audio_filepath": str(wav.resolve()), "offset": 0, "duration": None,
            "label": "infer", "text": "-", "num_speakers": None,
            "rttm_filepath": None, "uem_filepath": None}) + "\n", encoding="utf-8")
        cfg.diarizer.manifest_filepath = str(manifest)
        cfg.diarizer.out_dir = str(rec_dir)
        try:
            t0 = time.perf_counter()
            diarizer = NeuralDiarizer(cfg=cfg)
            if status.model_load_seconds is None:
                status.model_load_seconds = round(time.perf_counter() - t0, 3)
            if device == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            diarizer.diarize()
            if device == "cuda":
                torch.cuda.synchronize()
            run.processing_seconds = round(time.perf_counter() - t1, 3)
            run.notes = "processing_seconds excludes NeuralDiarizer construction"
            pred = rec_dir / "pred_rttms" / f"{wav.stem}.rttm"
            segs, _ = read_rttm(pred)
            segs = [Segment(rid, s.start, s.duration, sanitize_speaker(s.speaker)) for s in segs]
            write_rttm(segs, out_dir / f"{rid}.rttm")
            run.status = STATUS_OK if segs else STATUS_EMPTY_OUTPUT
            run.extra["n_predicted_speakers"] = len({s.speaker for s in segs})
        except Exception as exc:  # noqa: BLE001
            run.status = STATUS_FAILED
            run.error = f"{type(exc).__name__}: {exc}"
            log.error("%s failed: %s\n%s", rid, exc, traceback.format_exc())
        run.finished_utc = utc_now()
        run.attempts = 1
        status.recordings.append(run)

    resolved = RESULTS_DIR / "configs" / "nemo_msdd_resolved.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, resolved)
    status.settings["resolved_config"] = str(resolved.relative_to(ROOT))
    status.run_finished_utc = utc_now()
    status.save(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
