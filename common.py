"""Small helpers shared by all the scripts: file locations, the lake lookup, AUC and the lake bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FITS = HERE / "fits"
FORECASTS = HERE / "forecasts"
RESULTS = HERE / "results"

DL = 0.1          # microcystin reporting limit (µg/L). Anything under it is treated as censored, never filled in.
TEST_YEAR = 2019  # the season we forecast. 2018 is always fully available for training.
N_WEEKS = 15      # beaches were sampled weekly, weeks 1-15 of each season


def lake_of(sites):
    """Lake for each beach. Several beaches share a lake (four are on West Okoboji, for example), so the bootstrap
    resamples whole lakes instead of beaches; otherwise sister beaches would count as independent evidence."""
    lakes = pd.read_csv(DATA / "water_bodies.csv").set_index("site")["lake"]
    out = pd.Series(sites).map(lakes)
    assert out.notna().all(), "a beach is missing from water_bodies.csv"
    return out.to_numpy()


def auc(y, score):
    """Area under the ROC curve: the chance that a random event scores higher than a random non-event (ties count
    half). Returns NaN if there are no events or no non-events."""
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(score)  # average ranks, so ties get half credit
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def lake_bootstrap(df, stat, reps=2000):
    """Point estimate and 95% percentile interval for stat(df), resampling lakes with replacement.
    df needs a 'lake' column. Each replicate uses its own seed (1, 2, ...), so the intervals are reproducible."""
    groups = df.groupby("lake").indices  # lake -> row positions
    names = sorted(groups)
    est = stat(df)
    boots = []
    for i in range(1, reps + 1):
        pick = np.random.default_rng(i).choice(names, size=len(names), replace=True)
        boots.append(stat(df.iloc[np.concatenate([groups[k] for k in pick])]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return est, lo, hi


def fmt_ci(est, lo, hi, digits=3):
    return f"{est:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"
