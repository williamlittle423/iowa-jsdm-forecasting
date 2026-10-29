"""Non-Bayesian comparison models, forecast the same way as the JSDM: issued week by week through 2019, each one using
only what was known at the time.

    python regressions.py          -> forecasts/regression_forecasts.csv

  CF  carry-forward: "next week looks like today". Its score is simply today's toxin level, so it ranks beaches by
      how close they already are to a threshold. It's the baseline everything has to beat.
  T4  censored (Tobit) regression on log10 microcystin, fitted fresh at every forecast week:
        log10 toxin at week t+h ~ toxin today + week + week^2 + last week's toxin + season-to-date mean and max
                                  + log10 read shares of all cyanobacteria, Microcystis, Planktothrix, Aphanizomenon,
                                    Dolichospermum (each + 1e-4 so zeros are allowed)
      Results under 0.1 µg/L are treated as "somewhere below 0.1" (left-censored), not as 0.1.

Training data at forecast week t0 for lead h: every 2018 pair, plus the 2019 pairs whose outcome week (issue week + h)
is no later than t0, i.e. pairs with issue week <= t0 - h.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr
from scipy.stats import norm

from common import DATA, DL, FORECASTS, TEST_YEAR, lake_of

LDL = np.log10(DL)
READS = ["cy", "g_Microcystis", "g_Planktothrix", "g_Aphanizomenon", "g_Dolichospermum"]
T4 = ["ly", "wk", "wk2", "ly_prev", "ly_mean", "ly_max"] + [f"r_{k}" for k in READS]


def weekly_features():
    """One row per beach-week with the toxin history and read-share features."""
    w = pd.read_csv(DATA / "weekly_table.csv")
    key = ["site", "year", "week"]
    # Two libraries from the same beach-week get averaged (same water sample, so the toxin is identical anyway).
    w = w.groupby(key, as_index=False)[["mc"] + READS].mean().sort_values(key).reset_index(drop=True)
    w["ly"] = np.log10(np.maximum(w.mc, DL))
    season = w.groupby(["site", "year"])["ly"]
    w["ly_prev"] = season.shift(1).fillna(w.ly)          # previous sample this season (first sample: itself)
    w["ly_mean"] = season.cumsum() / (season.cumcount() + 1)
    w["ly_max"] = season.cummax()
    w["wk"] = (w.week - 8) / 4                            # centred so the quadratic term behaves
    w["wk2"] = w.wk ** 2
    for k in READS:
        w[f"r_{k}"] = np.log10(w[k] + 1e-4)
    return w


def pairs(w, h):
    """Each beach-week paired with the toxin measured h weeks later at the same beach."""
    fut = w[["site", "year", "week", "mc"]].rename(columns={"mc": "mc_f"})
    fut["week"] -= h
    m = w.merge(fut, on=["site", "year", "week"])
    m = m[m.cy.notna()].copy()                            # need reads on the issue date
    m["h"] = h
    m["yf"] = np.log10(np.maximum(m.mc_f, DL))
    m["cens"] = m.mc_f < DL
    return m


def tobit_fit(X, y, cens):
    """Maximum-likelihood censored normal regression (left-censored at y for the censored rows).
    Returns (coefficients, sigma). Same model as survreg(Surv(..., type = 'interval2'), dist = 'gaussian') in R."""
    X = np.column_stack([np.ones(len(X)), X])
    beta0 = np.linalg.lstsq(X, y, rcond=None)[0]
    start = np.append(beta0, np.log(np.std(y - X @ beta0)))

    def negloglik(theta):
        beta, log_s = theta[:-1], theta[-1]
        s = np.exp(log_s)
        r = (y - X @ beta) / s
        ll_obs = -log_s - 0.5 * r ** 2 - 0.5 * np.log(2 * np.pi)
        ll_cen = log_ndtr(r)
        ll = np.where(cens, ll_cen, ll_obs)
        # gradient: observed rows contribute r/s * x and r^2 - 1; censored rows -lambda/s * x and -lambda * r,
        # where lambda = phi(r) / Phi(r) (the inverse Mills ratio)
        lam = np.exp(norm.logpdf(r) - ll_cen)
        g_mu = np.where(cens, -lam / s, r / s)
        g_ls = np.where(cens, -lam * r, r ** 2 - 1)
        return -ll.sum(), -np.append(X.T @ g_mu, g_ls.sum())

    res = minimize(negloglik, start, jac=True, method="BFGS", options={"gtol": 1e-9, "maxiter": 10000})
    return res.x[:-1], np.exp(res.x[-1])


def rolling_forecasts():
    w = weekly_features()
    rows = []
    for h in range(1, 5):
        m = pairs(w, h)
        for t0 in sorted(m.week[m.year == TEST_YEAR].unique()):
            test = m[(m.year == TEST_YEAR) & (m.week == t0)]
            train = m[(m.year != TEST_YEAR) | (m.week <= t0 - h)]
            if len(train) < 60:      # too little to fit a 12-parameter regression
                continue
            base = pd.DataFrame({"site": test.site, "year": test.year, "t": t0, "h": h, "mc_now": test.mc,
                                 "mc_f": test.mc_f, "yf": test.yf})
            rows.append(base.assign(model="CF", mu=test.ly.to_numpy(), sigma=np.nan))
            beta, sigma = tobit_fit(train[T4].to_numpy(), train.yf.to_numpy(), train.cens.to_numpy())
            mu = beta[0] + test[T4].to_numpy() @ beta[1:]
            rows.append(base.assign(model="T4", mu=mu, sigma=sigma))
    out = pd.concat(rows, ignore_index=True)
    out.insert(1, "lake", lake_of(out.site))
    # Probability of reaching each level. For CF there is no probability, so we keep today's log10 level as its
    # score (it ranks the same way as today's toxin).
    for name, level in {"p16": 1.6, "p8": 8.0}.items():
        out[name] = np.where(out.model == "CF", out.mu, norm.sf((np.log10(level) - out.mu) / out.sigma))
    return out


if __name__ == "__main__":
    FORECASTS.mkdir(exist_ok=True)
    f = rolling_forecasts()
    f.to_csv(FORECASTS / "regression_forecasts.csv", index=False)
    print(f"{len(f)} forecasts written ({(f.model == 'T4').sum()} T4, {(f.model == 'CF').sum()} carry-forward)")
