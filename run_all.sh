#!/usr/bin/env bash
# Run the whole analysis.
#
#   ./run_all.sh               fit all 26 models, then forecast and score (the fits take ~27 hours on 16 cores)
#   ./run_all.sh --skip-fits   start from the fits already in fits/
#   ./run_all.sh --score-only  start from the forecasts already in forecasts/ (a couple of minutes, no Stan needed)
#
# JOBS = how many fits run at once. Each fit runs 4 chains on 4 cores, so the default (4) uses 16 cores.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
JOBS=${JOBS:-4}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   # keep numpy from grabbing extra cores

mode=${1:-all}

if [[ "$mode" == "all" ]]; then
  # Compile the Stan model once up front, so parallel fits don't all try to compile it at the same time.
  $PY -c "from cmdstanpy import CmdStanModel; CmdStanModel(stan_file='stan/jsdm_spec.stan')"
  # 1. Fit the full-A and diagonal-A JSDM at each forecast week (2019 weeks 2-14). Slowest ones first.
  { for o in 13 12 4 11 8 2 10 5 9 7 3 6 14; do echo "$o full"; done
    for o in 13 14 12 11 10 8 9 7 5 6 4 3 2; do echo "$o diag"; done; } |
    xargs -P "$JOBS" -L 1 $PY fit_jsdm.py
fi

if [[ "$mode" == "all" || "$mode" == "--skip-fits" ]]; then
  # 2. Roll each fit forward 1-4 weeks.
  $PY forecast_jsdm.py --all
fi

# 3. Carry-forward and T4 regression forecasts (seconds).
$PY regressions.py
# 4. Full vs diagonal A (pre-registered tests P1 and P2), and the comparison with the published model.
mkdir -p results
$PY score_jsdm.py > results/score_jsdm.txt
$PY compare_villanueva.py > results/villanueva_comparison.txt
cat fits/full_T??.txt fits/diag_T??.txt > results/convergence.txt 2>/dev/null || true
echo "done - see results/"
