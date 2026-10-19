#!/usr/bin/env bash
# End-to-end mini benchmark. Each stage records its own status; a blocked
# stage (no GPU, no token, no network) does not stop later stages, and its
# results remain "not_run" or "blocked_*" rather than being invented.
#
# Usage (from mini_eval/):
#   bash scripts/run_all.sh                       # pyannote + nemo + api(deepgram)
#   API_PROVIDER=assemblyai bash scripts/run_all.sh
#   SKIP_NEMO=1 SKIP_API=1 bash scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
PY_PYANNOTE=${PY_PYANNOTE:-$PY}     # interpreter of the pyannote venv
PY_NEMO=${PY_NEMO:-$PY}             # interpreter of the NeMo venv / container
DSCORE_PYTHON=${DSCORE_PYTHON:-$PY} # interpreter of the scorer venv
API_PROVIDER=${API_PROVIDER:-deepgram}
DSCORE_REF=${DSCORE_REF:-master}    # pin to a commit sha for reproducible runs

stage() { echo; echo "=== $* ==="; }

stage "0. environment"
$PY scripts/env_report.py --tag orchestrator

stage "1. scorer checkout (dscore @ $DSCORE_REF)"
if [ ! -d third_party/dscore/.git ]; then
  git clone https://github.com/nryant/dscore.git third_party/dscore || echo "WARN: dscore clone failed"
fi
if [ -d third_party/dscore/.git ]; then
  git -C third_party/dscore fetch --all --tags -q || true
  git -C third_party/dscore checkout -q "$DSCORE_REF" || echo "WARN: cannot checkout $DSCORE_REF"
  echo "dscore commit: $(git -C third_party/dscore rev-parse HEAD)"
fi

stage "2. data preparation"
$PY scripts/prepare_data.py || { echo "ERROR: data preparation failed; nothing to run."; exit 1; }

stage "3a. pyannote community-1 (primary, speaker count unknown)"
if [ -z "${SKIP_PYANNOTE:-}" ]; then $PY_PYANNOTE scripts/run_pyannote.py || echo "pyannote stage ended with non-zero status (see run_status.json)"; else echo "skipped"; fi

stage "3b. NeMo MSDD (primary, speaker count unknown)"
if [ -z "${SKIP_NEMO:-}" ]; then $PY_NEMO scripts/run_nemo_msdd.py || echo "NeMo stage ended with non-zero status (see run_status.json)"; else echo "skipped"; fi

stage "3c. commercial API ($API_PROVIDER)"
if [ -z "${SKIP_API:-}" ]; then $PY scripts/run_api.py --provider "$API_PROVIDER" || echo "API stage ended with non-zero status (see run_status.json)"; else echo "skipped"; fi

stage "4. validate predictions"
$PY scripts/validate_predictions.py || echo "WARN: validation found hard errors (they will be scored as failures)"

stage "5. score (dscore, collar 0, overlap included)"
DSCORE_PYTHON=$DSCORE_PYTHON $PY scripts/score.py --diagnostics

stage "6. statistics + tables + dashboard export"
$PY scripts/bootstrap.py
$PY scripts/make_results.py
$PY scripts/analyze.py
$PY scripts/export_dashboard.py
echo; echo "Done. See results/summary.csv and results/README.md"
