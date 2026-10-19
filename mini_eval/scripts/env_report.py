#!/usr/bin/env python3
"""Record the execution environment to results/environment.json.

Captures OS, Python, key package versions, GPU, ffmpeg/perl versions, the
pinned dscore commit and the UTC timestamp. Run it in EACH environment used
(pyannote env, NeMo env, scorer env) with --tag to keep them apart.
"""
from __future__ import annotations

import argparse
import importlib
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, ROOT, read_json, run_cmd, utc_now, write_json  # noqa: E402

PACKAGES = ["numpy", "scipy", "pandas", "requests", "torch", "torchaudio",
            "pyannote.audio", "pyannote.core", "pyannote.metrics", "nemo",
            "omegaconf", "huggingface_hub", "intervaltree", "tabulate"]


def version_of(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "installed (no __version__)")
    except Exception:  # noqa: BLE001
        return "not installed"


def first_line(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return "not found"
    p = run_cmd(cmd, timeout=20)
    text = (p.stdout or p.stderr).strip().splitlines()
    return next((t for t in text if t.strip()), "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="default", help="environment label, e.g. pyannote, nemo, scorer")
    args = ap.parse_args()
    gpu = "none detected"
    try:
        import torch
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
    except Exception:  # noqa: BLE001
        pass
    if gpu == "none detected" and shutil.which("nvidia-smi"):
        gpu = first_line(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    dscore = ROOT / "third_party" / "dscore"
    rec = {
        "captured_utc": utc_now(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu": platform.processor() or platform.machine(),
        "gpu": gpu,
        "packages": {p: version_of(p) for p in PACKAGES},
        "ffmpeg": first_line(["ffmpeg", "-version"]),
        "perl": first_line(["perl", "-e", "print $^V"]),
        "dscore_commit": run_cmd(["git", "-C", str(dscore), "rev-parse", "HEAD"], timeout=10).stdout.strip()
                          if (dscore / ".git").exists() else "dscore not cloned",
    }
    path = RESULTS_DIR / "environment.json"
    allenv = read_json(path, default={})
    allenv[args.tag] = rec
    write_json(path, allenv)
    print(f"recorded environment '{args.tag}' -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
