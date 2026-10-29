"""Head-to-head with the published Iowa beach model (Villanueva et al. 2023, ES&T 57:20636) on their task: will
microcystin be >= 8 µg/L one week from now? Everything is forecast forward through 2019, one week at a time.

    python compare_villanueva.py > results/villanueva_comparison.txt

Every model only uses what was known on the forecast date (all of 2018, plus 2019 outcomes measured by then).

  CF      carry-forward: score = today's toxin
  T4      Tobit regression on toxin history + read shares (regressions.py)
  JSDM    full-A JSDM, P(toxin >= level) (forecast_jsdm.py);  JSDM_D = diagonal A (secondary)
  V_XGB   XGBoost on Villanueva et al.'s 9 inputs, training set downsampled to balance the classes (their best
          setting): max depth 3, learning rate 0.05, 300 rounds, row subsample 0.8. Refit for every forecast week,
          lead and level.
  V_LR    plain logistic regression on the same 9 inputs and training rows (no downsampling)

Their 9 inputs: mcyA copies (Microcystis), mcyA/16S ratio, DOC, TKN, orthophosphate (these five as log10 with half
the smallest positive value added), pH, average dew point, % hay/pasture and % developed land in the watershed.

Endpoints (h = 1 week unless noted; AUC = chance a random event outscores a random non-event):
  E1  AUC for >= 8 µg/L over all beach-weeks (their task)     PRIMARY: JSDM - V_XGB, 95% lake bootstrap
  E2  AUC for >= 8, only beach-weeks below 8 today (new onsets)
  E3  AUC for >= 1.6, only beach-weeks below 1.6 today (new events)
  Brier scores at 8 and 1.6 for the probability models. V_XGB's are inflated by design: training on a balanced
  sample pushes its probabilities up.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import xgboost as xgb

from common import DATA, FORECASTS, auc, fmt_ci, lake_bootstrap, lake_of

KEY = ["site", "year", "week"]
VF = ["mcy_a_m", "mcya_16s", "doc_ppm", "tkn_mg_n_l", "ortho_p_mg_p_l", "p_h", "avg_dew", "hay_pasture", "developed_sum"]
MODELS = ["CF", "T4", "JSDM", "JSDM_D", "V_XGB", "V_LR"]


def published_inputs():
    """The 9 published inputs on each beach-week, paired with the toxin h weeks later (h = 1..4)."""
    w = pd.read_csv(DATA / "weekly_table.csv")
    tox = w.groupby(KEY, as_index=False)["mc"].mean()
    V = pd.read_csv(DATA / "villanueva2023_combined.csv")
    V["site"] = V.environmental_location.replace({"Pleasant Creek": "Pleasant Creek Lake Beach"})  # name differs
    for f in VF[:5]:
        x = V[f].clip(lower=0)
        V[f] = np.log10(x + x[x > 0].min() / 2)
    # Match on beach and sampling date, then average the odd duplicate.
    vm = w[["site", "date", "year", "week"]].drop_duplicates().merge(
        V[["site", "collected_date"] + VF], left_on=["site", "date"], right_on=["site", "collected_date"])
    vm = vm.groupby(KEY, as_index=False)[VF].mean().dropna().merge(tox, on=KEY)
    out = []
    for h in range(1, 5):
        fut = tox.rename(columns={"mc": "mc_f"}).assign(week=lambda d: d.week - h)
        out.append(vm.merge(fut, on=KEY).assign(h=h))
    return pd.concat(out, ignore_index=True)


def villanueva_forecasts(VP):
    """Refit both published-input models at each 2019 forecast week, lead and level."""
    rows = []
    for t0 in range(2, 15):
        for h in range(1, 5):
            test = VP[(VP.year == 2019) & (VP.week == t0) & (VP.h == h)].copy()
            if test.empty:
                continue
            train = VP[(VP.h == h) & ((VP.year == 2018) | (VP.week + h <= t0))]
            for level, tag in [(8, "p8"), (1.6, "p16")]:
                y = (train.mc_f >= level).astype(int).to_numpy()
                lr = sm.GLM(y, sm.add_constant(train[VF]), family=sm.families.Binomial()).fit()
                test[f"V_LR.{tag}"] = lr.predict(sm.add_constant(test[VF], has_constant="add")).to_numpy()
                # Balance the classes: keep every event and the same number of randomly chosen non-events.
                rng = np.random.default_rng(2026 + t0)
                pos = np.flatnonzero(y == 1)
                keep = np.concatenate([pos, rng.choice(np.flatnonzero(y == 0), size=len(pos), replace=False)])
                booster = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.05, "subsample": 0.8,
                                     "nthread": 1, "seed": 2026 + t0},
                                    xgb.DMatrix(train[VF].to_numpy()[keep], label=y[keep]), num_boost_round=300)
                test[f"V_XGB.{tag}"] = booster.predict(xgb.DMatrix(test[VF].to_numpy()))
            test["t"] = t0
            rows.append(test)
    return pd.concat(rows, ignore_index=True)


def endpoint(d, m, e):
    if e == "E1":
        return auc(d.mc_f >= 8, d[f"{m}.p8"])
    if e == "E2":
        z = d[d.mc < 8]
        return auc(z.mc_f >= 8, z[f"{m}.p8"])
    if e == "E3":
        z = d[d.mc < 1.6]
        return auc(z.mc_f >= 1.6, z[f"{m}.p16"])
    if e == "B8":
        return np.mean((d[f"{m}.p8"] - (d.mc_f >= 8)) ** 2)
    if e == "B16":
        return np.mean((d[f"{m}.p16"] - (d.mc_f >= 1.6)) ** 2)


def main():
    X = villanueva_forecasts(published_inputs())

    # JSDM forecasts (both versions) and the regressions, lined up on beach, forecast week and lead.
    jsdm = pd.concat([pd.read_csv(f) for f in sorted(FORECASTS.glob("*_T??_toxin.csv"))], ignore_index=True)
    for model, name in [("full", "JSDM"), ("diag", "JSDM_D")]:
        j = jsdm[jsdm.model == model][["site", "origin", "h", "p16", "p8"]]
        X = X.merge(j.rename(columns={"origin": "t", "p16": f"{name}.p16", "p8": f"{name}.p8"}), on=["site", "t", "h"])
    reg = pd.read_csv(FORECASTS / "regression_forecasts.csv")
    t4 = reg[reg.model == "T4"][["site", "t", "h", "p16", "p8"]].rename(columns={"p16": "T4.p16", "p8": "T4.p8"})
    X = X.merge(t4, on=["site", "t", "h"]).dropna()        # identical cases for every model
    X["lake"] = lake_of(X.site)
    X["CF.p8"] = X["CF.p16"] = X.mc                        # carry-forward ranks by today's toxin

    print("JSDM forecasts from forecasts/*_toxin.csv")
    for h in range(1, 5):
        d = X[X.h == h]
        print(f"\n==== h = {h}: {len(d)} cases, {d.lake.nunique()} lakes | >= 8: {(d.mc_f >= 8).sum()} events "
              f"({((d.mc_f >= 8) & (d.mc < 8)).sum()} new onsets) | >= 1.6 new events: "
              f"{((d.mc_f >= 1.6) & (d.mc < 1.6)).sum()} of {(d.mc < 1.6).sum()} below 1.6 ====")
        print(f"  {'model':<7} {'E1':>8} {'E2':>8} {'E3':>8} {'Brier8':>8} {'Brier1.6':>8}")
        for m in MODELS:
            b8 = "" if m == "CF" else f"{endpoint(d, m, 'B8'):.4f}"
            b16 = "" if m == "CF" else f"{endpoint(d, m, 'B16'):.4f}"
            print(f"  {m:<7} {endpoint(d, m, 'E1'):8.3f} {endpoint(d, m, 'E2'):8.3f} {endpoint(d, m, 'E3'):8.3f} {b8:>8} {b16:>8}")
        pairs = [("JSDM", "V_XGB"), ("JSDM", "V_LR"), ("JSDM", "CF"), ("JSDM", "T4"),
                 ("V_XGB", "CF"), ("T4", "V_XGB"), ("JSDM", "JSDM_D")]
        if h > 1:
            pairs = pairs[:4]
        for a, b in pairs:
            parts = []
            for e in (["E1", "E2", "E3"] if h == 1 else ["E1", "E3"]):
                ci = lake_bootstrap(d, lambda z: endpoint(z, a, e) - endpoint(z, b, e))
                parts.append(f"{e} {fmt_ci(*ci)}")
            flag = "  PRIMARY" if h == 1 and (a, b) == ("JSDM", "V_XGB") else ""
            print(f"  {a:<7} - {b:<7} {' | '.join(parts)}{flag}")


if __name__ == "__main__":
    main()
