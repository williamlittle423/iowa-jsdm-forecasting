"""Build the data list that stan/jsdm_spec.stan expects, for one forecast origin.

The idea of a "forecast origin" T: we pretend it's week T of the 2019 season. The model gets all of 2018 plus
whatever 2019 samples were collected in weeks 1..T, and nothing later. Calling this for T = 2..14 gives the
rolling forecasts we score.

    from prep_data import make_stan_data
    data, info = make_stan_data(origin=8, full_A=True)

Inputs (data/):
  weekly_table.csv  one row per sequencing library: beach, date, season week, microcystin (µg/L), weather, ...
  read_counts.csv   per library: total bacterial + archaeal reads (prok) and reads for the three cyanobacteria
"""
import numpy as np
import pandas as pd

from common import DATA, DL, N_WEEKS, TEST_YEAR

GROUPS = ["Microcystis", "Dolichospermum", "Planktothrix"]  # modelled taxa; everything else goes into "Other"
ENV = ["w_t7", "w_p30"]  # 7-day mean air temperature and 30-day rainfall before the sample date


def load_raw():
    """Read the tables and return (libraries, toxin, environment)."""
    w = pd.read_csv(DATA / "weekly_table.csv")
    counts = pd.read_csv(DATA / "read_counts.csv")

    # Libraries whose reads failed QC have no cyanobacteria share (cy is empty) - drop those, and any with zero reads.
    lib = w.loc[w.cy.notna(), ["run", "site", "year", "week"]].merge(counts, on="run")
    lib = lib[lib.prok > 0].sort_values("run").reset_index(drop=True)

    # Toxin and weather are per beach-week. Some beach-weeks have two libraries but they share one water sample,
    # so averaging just collapses the duplicates.
    key = ["site", "year", "week"]
    tox = w.groupby(key, as_index=False)["mc"].mean()
    env = w.groupby(key, as_index=False)[ENV].mean()
    return lib, tox, env


def make_stan_data(origin, full_A=True, sd_off=0.15):
    """Stan data for forecast origin `origin` (a 2019 week). Returns (data, info): `data` goes straight to Stan,
    `info` holds the labels we need later (which row is which beach-season, the scaling of the weather)."""
    lib, tox, env = load_raw()

    # What we're allowed to know at the origin: all of 2018, and 2019 up to and including week `origin`.
    def known(d):
        return (d.year != TEST_YEAR) | (d.week <= origin)

    lib_k, tox_k, env_k = lib[known(lib)], tox[known(tox)], env[known(env)]

    # Read counts: the three taxa plus "Other" = everything else. This way every library is split into the same
    # four groups that add up to its total, so the denominator never shifts when we change which taxa we model.
    n = lib_k[GROUPS].to_numpy()
    n = np.column_stack([n, lib_k.prok.to_numpy() - n.sum(axis=1)]).astype(int)
    assert (n >= 0).all() and (n.sum(axis=1) == lib_k.prok).all()

    # One latent time series per beach-season. We list every beach-season in the whole data set (even 2019 ones
    # with nothing observed yet) so that the model carries a state for every beach we want to forecast.
    series = (pd.concat([lib[["site", "year"]], tox[["site", "year"]]])
              .drop_duplicates().sort_values(["site", "year"]).reset_index(drop=True))
    sites = sorted(series.site.unique())
    row_of = {(s, y): i for i, (s, y) in enumerate(zip(series.site, series.year))}

    # Weather, standardised with the training-period mean and SD. Weeks without a sample just carry the last value
    # forward (and use 0, i.e. the training mean, before the first one). We never look past the origin.
    mean, sd = env_k[ENV].mean(), env_k[ENV].std()
    X = np.zeros((len(series), N_WEEKS, len(ENV)))
    for i, (site, year) in enumerate(zip(series.site, series.year)):
        e = env[(env.site == site) & (env.year == year)]
        if year == TEST_YEAR:
            e = e[e.week <= origin]
        for q, v in enumerate(ENV):
            weekly = np.full(N_WEEKS, np.nan)
            weekly[e.week.to_numpy() - 1] = ((e[v] - mean[v]) / sd[v]).to_numpy()
            last = 0.0
            for t in range(N_WEEKS):
                if np.isnan(weekly[t]):
                    weekly[t] = last
                else:
                    last = weekly[t]
            X[i, :, q] = weekly

    # Toxin on the natural-log scale. Values under the reporting limit (including reported zeros) are flagged as
    # censored and sit at log(0.1); the model treats them as "somewhere below 0.1", not as 0.1.
    lv = np.log(np.maximum(tox_k.mc.to_numpy(), DL))
    cens = (tox_k.mc.to_numpy() < DL).astype(int)

    # Prior centre for the average log-ratios: the overall read shares in the training data.
    share = n.sum(axis=0) / n.sum()
    K = len(GROUPS)

    # We save the latent state at the origin for every 2019 series; the forecast starts from there.
    out_rows = np.flatnonzero(series.year.to_numpy() == TEST_YEAR)

    data = {
        "G": K + 1, "S": len(sites), "L": len(series), "TT": N_WEEKS,
        "site": np.array([sites.index(s) + 1 for s in series.site]),          # Stan counts from 1
        "P": len(ENV), "X": X,
        "J": len(n), "n": n,
        "jl": np.array([row_of[k] + 1 for k in zip(lib_k.site, lib_k.year)]), "jt": lib_k.week.to_numpy(),
        "NV": len(lv), "vl": np.array([row_of[k] + 1 for k in zip(tox_k.site, tox_k.year)]), "vt": tox_k.week.to_numpy(),
        "lv": lv, "cens": cens,
        "mu00": np.log(share[:K] / share[K]), "aV0": lv.mean(),
        "full_A": int(full_A), "sd_off": sd_off, "use_lik": 1,
        "NO": len(out_rows), "ol": out_rows + 1, "ot": np.full(len(out_rows), max(1, origin)),
    }
    info = {"series": series, "sites": sites, "env_mean": mean, "env_sd": sd, "origin": origin}
    return data, info
