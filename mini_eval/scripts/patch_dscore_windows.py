#!/usr/bin/env python3
"""Windows only: make dscore launch md-eval-22.pl through perl.

dscore's scorelib/metrics.py runs md-eval-22.pl directly as an executable, which
works on Linux/macOS (shebang line) but fails on Windows with
"OSError: [WinError 193] %1 is not a valid Win32 application".

This rewrites the command list from  [MDEVAL_BIN, ...]  to  ['perl', MDEVAL_BIN, ...].
It changes only how the script is started, not how anything is scored.
Safe to run more than once. Record that the patch was applied alongside the dscore commit.

    python scripts/patch_dscore_windows.py [--dscore-dir third_party/dscore]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"\[\s*MDEVAL_BIN\s*,")
PATCHED = "['perl', MDEVAL_BIN,"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dscore-dir", default=str(Path(__file__).resolve().parents[1] / "third_party" / "dscore"))
    args = ap.parse_args()
    path = Path(args.dscore_dir) / "scorelib" / "metrics.py"
    if not path.exists():
        print(f"not found: {path}  (clone dscore first)")
        return 2
    text = path.read_text(encoding="utf-8")
    if PATCHED in text:
        print(f"already patched: {path}")
        return 0
    n = len(PATTERN.findall(text))
    if n == 0:
        print(f"no '[MDEVAL_BIN,' command found in {path}; lines mentioning md-eval:")
        for i, line in enumerate(text.splitlines(), 1):
            if "MDEVAL" in line or "md-eval" in line:
                print(f"  {i}: {line}")
        print("Send these lines to the maintainer; nothing was changed.")
        return 3
    path.write_text(PATTERN.sub(PATCHED, text), encoding="utf-8")
    print(f"patched {n} command(s) in {path}: md-eval now runs as 'perl md-eval-22.pl ...'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
