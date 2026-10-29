"""Turn one fitted origin into 1-4 week ahead forecasts for every 2019 beach.

    python forecast_jsdm.py full_T08          # reads fits/full_T08.npz
    python forecast_jsdm.py --all             # every fit in fits/

For each of 1000 posterior draws we start from that draw's latent community at the origin week and step it forward
with the fitted dynamics, adding process noise each week:

    z[t+1] = mu_site + A (z[t] - mu_site) + B x + noise,   noise ~ N(0, Sigma)

The weather x is held at its origin-week value (we don't forecast weather). Each simulated state then gives

  * read forecasts for every library actually sequenced at that beach and week: the log predictive density of the
    observed counts (Dirichlet-multinomial, given the library's total reads), the beta-binomial version for
    Microcystis alone, and a simulated replicate library for the Microcystis read share (median, CRPS, coverage);
  * toxin forecasts for every toxin measurement at that beach and week: P(microcystin >= 0.3, 1.6, 8 µg/L) from the
    censored-normal toxin link  log V ~ N(aV + cV'z, sV).

Output: forecasts/<tag>_reads.csv and forecasts/<tag>_toxin.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import betaln, gammaln
from scipy.stats import norm

from common import FITS, FORECASTS, N_WEEKS, TEST_YEAR
from prep_data import load_raw, make_stan_data

N_DRAWS = 1000   # posterior draws used per forecast
MAX_LEAD = 4     # weeks ahead
TAU = 0.05       # "Microcystis signal" threshold on the read share (fixed in advance)
# µg/L. 0.3 and 1.6 are the EPA drinking-water health advisory levels; 8 is the EPA recreational criterion that
# beach advisories use.
LEVELS = {"p03": 0.3, "p16": 1.6, "p8": 8.0}


def log_mean_exp(x):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m)))


def dirmult_logpmf(n, alpha):
    """Dirichlet-multinomial log probability of counts n (length G) for each row of alpha (draws x G)."""
    M, a0 = n.sum(), alpha.sum(axis=1)
    return (gammaln(M + 1) - gammaln(n + 1).sum() + gammaln(a0) - gammaln(M + a0)
            + (gammaln(n + alpha) - gammaln(alpha)).sum(axis=1))


def betabinom_logpmf(y, M, a, b):
    return gammaln(M + 1) - gammaln(y + 1) - gammaln(M - y + 1) + betaln(y + a, M - y + b) - betaln(a, b)


def forecast(tag, seed=7):
    fit = np.load(FITS / f"{tag}.npz")
    origin, model = int(fit["origin"]), str(fit["model"])
    columns = {c: i for i, c in enumerate(fit["columns"])}
    sites = list(fit["sites"])
    series = list(zip(fit["series_site"], fit["series_year"]))   # the fit's own ordering of beach-seasons

    rng = np.random.default_rng(seed)
    draws = fit["draws"]
    use = rng.choice(len(draws), size=min(N_DRAWS, len(draws)), replace=False)
    D = draws[use]

    def par(name):
        return D[:, columns[name]]

    K = sum(1 for c in columns if c.startswith("cV["))           # number of modelled taxa (3)
    P = sum(1 for c in columns if c.startswith("B[1,"))          # number of weather covariates (2)
    A = np.stack([np.stack([par(f"A[{r},{c}]") for c in range(1, K + 1)], -1) for r in range(1, K + 1)], 1)
    Sigma = np.stack([np.stack([par(f"Sigma[{r},{c}]") for c in range(1, K + 1)], -1) for r in range(1, K + 1)], 1)
    B = np.stack([np.stack([par(f"B[{r},{q}]") for q in range(1, P + 1)], -1) for r in range(1, K + 1)], 1)
    chol = np.linalg.cholesky(Sigma)                             # to draw correlated process noise
    kappa = np.exp(par("log_kappa"))
    aV, sV = par("aV"), par("sV")
    cV = np.stack([par(f"cV[{k}]") for k in range(1, K + 1)], -1)

    # Weather at the origin, standardised the same way the fit saw it.
    data, info = make_stan_data(origin, full_A=model == "full")
    X = {(s, y): data["X"][i] for i, (s, y) in enumerate(zip(info["series"].site, info["series"].year))}

    # What actually happened after the origin (only used for scoring).
    lib, tox, _ = load_raw()
    lib = lib[(lib.year == TEST_YEAR) & (lib.week > origin)]
    tox = tox[(tox.year == TEST_YEAR) & (tox.week > origin)]
    groups = ["Microcystis", "Dolichospermum", "Planktothrix"][:K]

    read_rows, tox_rows = [], []
    test_series = [i for i, (_, y) in enumerate(series) if y == TEST_YEAR]
    for o, i in enumerate(test_series):        # o indexes zT, the saved state at the origin
        site = series[i][0]
        s = sites.index(site) + 1
        z = np.stack([par(f"zT[{o + 1},{k}]") for k in range(1, K + 1)], -1)
        mu = np.stack([par(f"mu[{s},{k}]") for k in range(1, K + 1)], -1)
        x = X[(site, TEST_YEAR)][origin - 1]   # weather in the origin week, carried forward

        for h in range(1, MAX_LEAD + 1):
            week = origin + h
            if week > N_WEEKS:
                break
            noise = np.einsum("dij,dj->di", chol, rng.standard_normal((len(use), K)))
            z = mu + np.einsum("dij,dj->di", A, z - mu) + np.einsum("dkp,p->dk", B, x) + noise

            # Group shares: softmax of (z, 0), with "Other" as the reference group.
            full = np.column_stack([z, np.zeros(len(z))])
            pi = np.exp(full - full.max(axis=1, keepdims=True))
            pi /= pi.sum(axis=1, keepdims=True)
            p = pi[:, 0]                                  # Microcystis share

            for _, lb in lib[(lib.site == site) & (lib.week == week)].iterrows():
                n = np.append(lb[groups].to_numpy(float), lb.prok - lb[groups].sum())
                M, y = n.sum(), n[0]
                lpd_joint = log_mean_exp(dirmult_logpmf(n, kappa[:, None] * pi))
                lpd_target = log_mean_exp(betabinom_logpmf(y, M, kappa * p, kappa * (1 - p)))
                # A simulated library of the same size: Microcystis share drawn from the predictive distribution.
                f_rep = rng.binomial(int(M), rng.beta(kappa * p, kappa * (1 - p))) / M
                f_obs = y / M
                q = np.quantile(f_rep, [.025, .1, .25, .5, .75, .9, .975])
                crps = np.mean(np.abs(f_rep - f_obs)) - 0.5 * np.mean(np.abs(f_rep[:200, None] - f_rep[None, :200]))
                read_rows.append(dict(tag=tag, model=model, origin=origin, h=h, site=site, week=week, run=lb.run, M=M,
                                      f_obs=f_obs, lpd_joint=lpd_joint, lpd_target=lpd_target, f_med=q[3], crps=crps,
                                      in50=q[2] <= f_obs <= q[4], in80=q[1] <= f_obs <= q[5], in95=q[0] <= f_obs <= q[6],
                                      p_read=np.mean(f_rep > TAU), p_signal=np.mean(p > TAU)))

            for _, tx in tox[(tox.site == site) & (tox.week == week)].iterrows():
                eta = aV + (cV * z).sum(axis=1)               # expected log toxin for each draw
                row = dict(tag=tag, model=model, origin=origin, h=h, site=site, week=week, mc_f=tx.mc)
                for name, level in LEVELS.items():
                    row[name] = np.mean(norm.sf((np.log(level) - eta) / sV))
                row["v_med"] = np.median(np.exp(eta + rng.normal(0, sV)))
                tox_rows.append(row)

    FORECASTS.mkdir(exist_ok=True)
    pd.DataFrame(read_rows).to_csv(FORECASTS / f"{tag}_reads.csv", index=False)
    pd.DataFrame(tox_rows).to_csv(FORECASTS / f"{tag}_toxin.csv", index=False)
    print(f"{tag}: {len(read_rows)} library forecasts, {len(tox_rows)} toxin forecasts")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="*", help="fit names, e.g. full_T08")
    ap.add_argument("--all", action="store_true", help="forecast from every fit in fits/")
    args = ap.parse_args()
    tags = args.tags
    if args.all:
        tags = sorted(p.stem for pattern in ("full_T??.npz", "diag_T??.npz") for p in FITS.glob(pattern))
    for t in tags:
        forecast(t)
