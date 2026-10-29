"""Score the rolling JSDM forecasts: does the full A matrix (cross-lags between taxa) beat the diagonal one?

    python score_jsdm.py > results/score_jsdm.txt      (+ results/score_reads.csv, results/score_toxin.csv)

Cases: 2019 forecasts from origins T = 2..14, h = 1..4 weeks ahead (T + h <= 15), for every library and toxin value
measured at T + h. Both models are scored on exactly the same cases.

Primary tests, written down before the forecasts existed (h = 1, full minus diagonal):
  P1  mean log predictive density of the held-out read counts.  Positive = full A forecasts the community better.
  P2  Brier score for microcystin >= 1.6 µg/L.                   Negative = full A forecasts toxin better.
95% intervals come from resampling lakes (2000 times). Everything else in the output is secondary.

Also shown for context: a persistence forecast for the Microcystis read share (last observed share), and for toxin
the T4 regression and carry-forward from regressions.py.
"""
import numpy as np
import pandas as pd

from common import DATA, FORECASTS, RESULTS, TEST_YEAR, auc, fmt_ci, lake_bootstrap, lake_of

KEY = ["origin", "h", "site", "week"]


def load(kind):
    """All forecast files of one kind ('reads' or 'toxin') for the full and diagonal fits."""
    files = sorted(FORECASTS.glob(f"full_T??_{kind}.csv")) + sorted(FORECASTS.glob(f"diag_T??_{kind}.csv"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def wide(df, cols, keys):
    """Put the full and diagonal forecasts side by side (only cases both models have)."""
    full = df[df.model == "full"][keys + cols].rename(columns={c: f"{c}_full" for c in cols})
    diag = df[df.model == "diag"][keys + cols].rename(columns={c: f"{c}_diag" for c in cols})
    return full.merge(diag, on=keys)


def verdict(ci, higher_is_better):
    est, lo, hi = ci
    if (lo > 0) if higher_is_better else (hi < 0):
        return "full A (cross-lags) better"
    if (hi < 0) if higher_is_better else (lo > 0):
        return "full A worse"
    return "no detectable difference"


def main():
    # ---------------- reads ----------------
    reads = load("reads")
    score_cols = ["lpd_joint", "lpd_target", "f_med", "crps", "in50", "in80", "in95", "p_read"]
    W = wide(reads, score_cols, KEY + ["run"])
    W = W.merge(reads[reads.model == "full"][KEY + ["run", "f_obs"]], on=KEY + ["run"])
    W["lake"] = lake_of(W.site)

    # Persistence: the Microcystis read share at the most recent sample on or before the origin.
    w = pd.read_csv(DATA / "weekly_table.csv")
    lib = w[w.cy.notna() & (w.year == TEST_YEAR)][["run", "site", "week"]].merge(pd.read_csv(DATA / "read_counts.csv"), on="run")
    lib["f"] = lib.Microcystis / lib.prok

    def persistence(site, origin):
        z = lib[(lib.site == site) & (lib.week <= origin)]
        return z.f[z.week == z.week.max()].mean() if len(z) else np.nan

    W["f_pers"] = [persistence(s, o) for s, o in zip(W.site, W.origin)]

    print(f"Rolling-origin JSDM forecasts, {TEST_YEAR}: {len(W)} library forecasts, "
          f"{(load('toxin').model == 'full').sum()} toxin forecasts per model")
    print("\n==== Read-count forecasts (held-out libraries) ====")
    print(f"  {'h':<4} {'model':<7} {'lpd_joint':>10} {'lpd_target':>10} {'MAE':>8} {'CRPS':>8} {'in50':>6} {'in80':>6} {'in95':>6}")
    for h in range(1, 5):
        z = W[W.h == h]
        for m in ["full", "diag"]:
            print(f"  {h:<4} {m:<7} {z[f'lpd_joint_{m}'].mean():10.2f} {z[f'lpd_target_{m}'].mean():10.3f} "
                  f"{(z[f'f_med_{m}'] - z.f_obs).abs().mean():8.4f} {z[f'crps_{m}'].mean():8.4f} "
                  f"{z[f'in50_{m}'].mean():6.2f} {z[f'in80_{m}'].mean():6.2f} {z[f'in95_{m}'].mean():6.2f}")
        print(f"  {h:<4} {'persist':<7} {'':>10} {'':>10} {(z.f_pers - z.f_obs).abs().mean():8.4f}")

    def lpd_gain(d):
        return (d.lpd_joint_full - d.lpd_joint_diag).mean()

    p1 = lake_bootstrap(W[W.h == 1], lpd_gain)
    print(f"  P1 PRIMARY  h=1 joint lpd, full - diag: {fmt_ci(*p1, 4)} -> {verdict(p1, True)}")
    for h in range(2, 5):
        print(f"  h={h} joint lpd full - diag: {fmt_ci(*lake_bootstrap(W[W.h == h], lpd_gain), 4)}")

    # ---------------- toxin ----------------
    print("\n==== Toxin (censored-normal link) ====")
    tox = load("toxin")
    VW = wide(tox, ["p16", "p8"], KEY)
    VW = VW.merge(tox[tox.model == "full"][KEY + ["mc_f"]], on=KEY)
    VW["lake"] = lake_of(VW.site)
    reg = pd.read_csv(FORECASTS / "regression_forecasts.csv").rename(columns={"t": "origin"})
    reg["week"] = reg.origin + reg.h
    t4 = reg[reg.model == "T4"][KEY + ["p16", "p8"]].rename(columns={"p16": "p16_T4", "p8": "p8_T4"})
    cf = reg[reg.model == "CF"][KEY + ["mu", "mc_now"]].rename(columns={"mu": "score_CF"})
    VW = VW.merge(t4, on=KEY).merge(cf, on=KEY)        # identical cases for every model

    models = ["full", "diag", "T4"]
    for level, col in [(1.6, "p16"), (8, "p8")]:
        print(f"  >= {level} µg/L   (Brier: lower is better; new-event AUC: beach-weeks below {level} today)")
        for h in range(1, 5):
            z = VW[VW.h == h]
            y = z.mc_f >= level
            new = z.mc_now < level
            brier = "  ".join(f"{m} {np.mean((z[f'{col}_{m}'] - y) ** 2):.4f}" for m in models)
            aucs = "  ".join(f"{m} {auc(y[new], z[f'{col}_{m}'][new]):.3f}" for m in models)
            print(f"    h={h} Brier {brier} | new-event AUC {aucs}  CF {auc(y[new], z.score_CF[new]):.3f}")

    def brier_diff(a, b):
        return lambda d: (np.mean((d[f"p16_{a}"] - (d.mc_f >= 1.6)) ** 2) - np.mean((d[f"p16_{b}"] - (d.mc_f >= 1.6)) ** 2))

    p2 = lake_bootstrap(VW[VW.h == 1], brier_diff("full", "diag"))
    print(f"  P2 PRIMARY  h=1 Brier >= 1.6, full - diag: {fmt_ci(*p2, 4)} -> {verdict(p2, False)}")
    for a, b in [("full", "T4"), ("diag", "T4")]:
        for hs in [[1], [2, 3, 4]]:
            ci = lake_bootstrap(VW[VW.h.isin(hs)], brier_diff(a, b))
            print(f"  h={hs[0] if len(hs) == 1 else '2-4'} Brier >= 1.6 {a} - {b}: {fmt_ci(*ci, 4)}")

    RESULTS.mkdir(exist_ok=True)
    W.to_csv(RESULTS / "score_reads.csv", index=False)
    VW.to_csv(RESULTS / "score_toxin.csv", index=False)


if __name__ == "__main__":
    main()
