"""Fit the JSDM at one forecast origin and save the posterior draws.

    python fit_jsdm.py 8 full                      # origin = 2019 week 8, full A matrix (4 chains, 4 cores)
    python fit_jsdm.py 8 diag                      # same, but A restricted to its diagonal (no cross-lags)
    python fit_jsdm.py 8 full --warmup 200 --samples 200    # quick test run

Writes fits/<model>_T<origin>.npz (the draws we need for forecasting, plus the beach labels) and a one-line
convergence report in fits/<model>_T<origin>.txt.

Settings are the ones used for the reported results: 4 chains, 2000 warm-up + 3000 sampling iterations each,
adapt_delta 0.95, max tree depth 12, seed 2026 + origin. A long-chain fit takes roughly 1.5-10 hours depending on
the origin and model (full A is the slow one).
"""
import argparse
import time

import numpy as np
from cmdstanpy import CmdStanModel

from common import FITS, HERE
from prep_data import make_stan_data

# Parameters the forecast needs (plus a few worth looking at). The per-week noise terms are left out - there are
# thousands of them and we never use them after fitting.
KEEP = ["A", "Sigma", "rhoA", "mu", "B", "log_kappa", "aV", "cV", "sV", "zT", "mu0", "tau_mu", "sigma"]
# Parameters we check convergence on (R-hat and effective sample size).
CHECK = ["A", "Sigma", "mu0", "tau_mu", "B", "log_kappa", "aV", "cV", "sV"]


def ebfmi(energy):
    """Energy Bayesian fraction of missing information for one chain. Values under ~0.3 mean the sampler is
    struggling to move between energy levels."""
    return np.sum(np.diff(energy) ** 2) / len(energy) / np.var(energy, ddof=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("origin", type=int, help="2019 week the forecast is issued from (2-14)")
    ap.add_argument("model", choices=["full", "diag"])
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--tag", help="name for the output files (default <model>_T<origin>)")
    args = ap.parse_args()
    tag = args.tag or f"{args.model}_T{args.origin:02d}"

    data, info = make_stan_data(args.origin, full_A=args.model == "full")
    model = CmdStanModel(stan_file=HERE / "stan" / "jsdm_spec.stan")  # compiles the first time, then reuses it

    start = time.time()
    fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=args.warmup, iter_sampling=args.samples,
                       adapt_delta=0.95, max_treedepth=12, seed=2026 + args.origin, show_progress=False)
    minutes = (time.time() - start) / 60

    # Flatten the kept parameters into one draws x parameters matrix, with Stan's own column names ("A[1,2]" etc.).
    draws = fit.draws_pd(vars=KEEP)
    draws = draws[[c for c in draws.columns if not c.endswith("__")]]   # drop chain/iteration bookkeeping columns
    np.savez(FITS / f"{tag}.npz", draws=draws.to_numpy(), columns=np.array(draws.columns, dtype=str),
             sites=np.array(info["sites"]), series_site=info["series"].site.to_numpy(dtype=str),
             series_year=info["series"].year.to_numpy(), origin=args.origin, model=args.model)

    # Convergence report. We want: no (or very few) divergences, R-hat < 1.01, a few hundred effective draws or more.
    summ = fit.summary()
    summ = summ[[name.split("[")[0] in CHECK for name in summ.index]]
    energy = fit.method_variables()["energy__"]          # iterations x chains
    report = (f"{tag}: {minutes:.1f} min | divergences {'/'.join(map(str, fit.divergences))} | "
              f"max-treedepth {'/'.join(map(str, fit.max_treedepths))} | "
              f"E-BFMI {'/'.join(f'{ebfmi(energy[:, c]):.2f}' for c in range(energy.shape[1]))} | "
              f"worst Rhat {summ['R_hat'].max():.3f} | min bulk ESS {summ['ESS_bulk'].min():.0f} | "
              f"min tail ESS {summ['ESS_tail'].min():.0f}")
    (FITS / f"{tag}.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
