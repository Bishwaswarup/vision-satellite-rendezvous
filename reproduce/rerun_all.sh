#!/usr/bin/env bash
# =============================================================================
# Rerun every test and experiment behind the paper, then check each number the
# paper quotes against the fresh output.
#
#   bash setup.sh                         # once: creates .venv/
#   bash reproduce/rerun_all.sh           # default: 4 parallel jobs, ~30 min
#   JOBS=8 bash reproduce/rerun_all.sh    # more cores = faster
#
# outputs/ is copied to outputs_before_rerun_<timestamp>/ first, so the archived
# figures and per-trial records can be compared trial by trial afterwards.
# Results: reproduce/claims_report.txt  (+ reproduce/logs/ for the raw output)
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No project venv found. Create it first:  bash setup.sh"
  exit 1
fi
JOBS="${JOBS:-4}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$HERE/logs"
mkdir -p "$LOG"
export MPLBACKEND=Agg PYTHONHASHSEED=0

echo "Python : $("$PY" --version 2>&1)   numpy $("$PY" -c 'import numpy;print(numpy.__version__)')   scipy $("$PY" -c 'import scipy;print(scipy.__version__)')"
echo "Root   : $ROOT"
echo "Jobs   : $JOBS"
echo

BK="$ROOT/outputs_before_rerun_$STAMP"
cp -R outputs "$BK"
echo "$BK" > "$LOG/backup_path.txt"
echo "Backed up outputs/ -> $(basename "$BK")"
echo

T0=$SECONDS
step () {   # step <name> <logfile> <command...>
  local name="$1" log="$2"; shift 2
  echo "──────── $name ────────"
  local t=$SECONDS
  "$@" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  echo "   [$name finished in $((SECONDS - t)) s, exit code $rc]"
  echo "$name $rc $((SECONDS - t))" >> "$LOG/exit_codes.txt"
  echo
}
: > "$LOG/exit_codes.txt"

step "Unit tests"   "$LOG/pytest.log" "$PY" -m pytest -q -p no:cacheprovider tests
step "Experiment A" "$LOG/A.log" "$PY" experiments/experiment_A.py
step "Experiment B" "$LOG/B.log" "$PY" experiments/experiment_B.py
step "Experiment C" "$LOG/C.log" "$PY" experiments/experiment_C.py
step "Experiment D" "$LOG/D.log" "$PY" experiments/experiment_D.py
step "Experiment E" "$LOG/E.log" "$PY" experiments/experiment_E.py
step "Experiment F (Monte Carlo, the long one)" "$LOG/F.log" \
     "$PY" experiments/experiment_F.py --jobs "$JOBS"
step "Experiment G (sensitivity + causality)" "$LOG/G.log" \
     "$PY" experiments/experiment_G.py --jobs "$JOBS"
step "Experiment G3 (RANSAC threshold)" "$LOG/G3.log" \
     "$PY" experiments/experiment_G.py --g3 --jobs "$JOBS"
step "MPC solver check" "$LOG/mpc.log" "$PY" "$HERE/check_mpc_solver.py"
step "Extra claims (needs opencv-python-headless)" "$LOG/extra.log" \
     "$PY" "$HERE/verify_extra_claims.py" --jobs "$JOBS"

echo "All runs finished in $(( (SECONDS - T0) / 60 )) min $(( (SECONDS - T0) % 60 )) s"
echo
"$PY" "$HERE/check_claims.py" --logs "$LOG" --outputs "$ROOT/outputs" --backup "$BK" \
  | tee "$HERE/claims_report.txt"
echo
echo "Report saved to: reproduce/claims_report.txt"
