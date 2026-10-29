# Forecasting microcystin at Iowa beaches with a Bayesian dynamic JSDM

This folder is the Python version of the dynamic joint species distribution model (JSDM) analysis. It fits the model
to weekly 16S read counts and microcystin measurements from 37 Iowa beaches (31 lakes) in 2018 and 2019. It then
asks two questions:

1. **Do cross-lags between taxa matter?** Does a full community matrix *A* (taxa affect each other week to week)
   forecast better than a diagonal one (each taxon only follows itself)?
2. **How useful is the model for toxin?** How do its microcystin forecasts compare with carry-forward, with a
   simpler regression, and with the published Iowa beach model (Villanueva et al. 2023)?

The model itself is written in Stan ([stan/jsdm_spec.stan](stan/jsdm_spec.stan)). CmdStan compiles it to C++. Everything
else is Python.

## What's in the folder

| File | What it does |
|---|---|
| `stan/jsdm_spec.stan` | The model (read likelihood, latent community dynamics, toxin link, priors) |
| `prep_data.py` | Builds the Stan input for one forecast week: what was known at that point in 2019 |
| `fit_jsdm.py` | Fits the model at one forecast week through CmdStan and saves the posterior draws |
| `forecast_jsdm.py` | Rolls each beach's community forward 1-4 weeks from the posterior and scores the read and toxin forecasts |
| `regressions.py` | Comparison forecasts: carry-forward and the T4 censored regression |
| `score_jsdm.py` | Full vs diagonal *A*: the two pre-registered tests (P1 reads, P2 toxin) plus the secondary scores |
| `compare_villanueva.py` | Head-to-head with the published model on its own task (>= 8 µg/L one week ahead) |
| `common.py` | Small shared helpers (paths, lake lookup, AUC, lake bootstrap) |
| `run_all.sh` | Runs everything in order |
| `data/` | Input tables (see below) |
| `forecasts/` | Forecasts from our run, one CSV per fit (enough to redo all scoring without Stan) |
| `results/` | Output from our run |
| `fits/` | Posterior draws. **Not included in the download** (26 fits, ~650 MB). They're on the server in `iowa-beachs/jsdm_python/fits/`. |

**Data** (`data/`):
- `weekly_table.csv`: one row per sequencing library, with beach, date, season week, microcystin (µg/L), DNR water
  chemistry, weather (ERA5 via Open-Meteo) and genus read shares.
- `read_counts.csv`: per library, the total bacterial + archaeal reads and the *Microcystis*, *Dolichospermum* and
  *Planktothrix* counts (tronko-assign classifications).
- `water_bodies.csv`: which lake each beach is on.
- `villanueva2023_combined.csv`: the published model's input data.

## Setup

```bash
pip install -r requirements.txt
```

Fitting also needs CmdStan. Either run `python -m cmdstanpy.install_cmdstan`, or point `CMDSTAN` at an existing
install (we used 2.40.0). Nothing else needs CmdStan.

## Running it

```bash
./run_all.sh --score-only    # ~3 minutes: regressions + all scoring, from the forecasts in forecasts/
./run_all.sh --skip-fits     # also re-make the forecasts (needs fits/)
./run_all.sh                 # everything from scratch, including the 26 Stan fits (~27 hours on 16 cores)
```

Single steps, for example:

```bash
python fit_jsdm.py 8 full                           # fit at 2019 week 8, full A (4 chains, 4 cores)
python fit_jsdm.py 8 full --warmup 200 --samples 200   # quick test
python forecast_jsdm.py full_T08
python score_jsdm.py
```

## The model in brief

For each beach-season, a latent state z (log-ratios of *Microcystis*, *Dolichospermum* and *Planktothrix* against
everything else) moves week to week:

    z[t+1] = mu_beach + A (z[t] - mu_beach) + B x[t] + noise,     noise ~ N(0, Sigma)

- x is weather (7-day mean air temperature, 30-day rainfall).
- *A* is kept stable (spectral radius < 1) through the Ansley-Kohn parameterisation. The diagonal version just
  restricts *A* to its diagonal.
- **Reads:** each library's counts are Dirichlet-multinomial around softmax(z, 0). Replicate libraries share the
  beach-week's state.
- **Toxin:** log microcystin ~ Normal(aV + cV'z, sV). Results under 0.1 µg/L are treated as censored (somewhere
  below 0.1), not set to a value.
- **Priors:** listed at the top of the `.stan` file.

## How the forecasts were evaluated

We pretend to be at week T of 2019, for T = 2, ..., 14.

- **What each model sees:** all of 2018, plus 2019 data collected up to week T.
- **What it forecasts:** weeks T+1 to T+4 (up to week 15).
- **Refitting:** every model is refit at every T: the JSDM, T4, and both Villanueva models.
- **Scoring:** all models are scored on exactly the same cases.
- **Uncertainty:** 95% intervals come from resampling whole lakes (2000 times). Several beaches share a lake, so
  resampling beaches would overstate the evidence.

The main tests were written down before any forecasts existed:
- P1 and P2 for full vs diagonal *A*.
- JSDM minus Villanueva XGBoost for the published-model comparison.

## Results

Full output: `results/villanueva_comparison.txt`, `results/score_jsdm.txt` and `results/convergence.txt`.

> **Status (29 Oct 2026):** all 13 full-A fits are long-chain. Of the diagonal-A fits, weeks 11-14 are long-chain;
> weeks 2-10 are still from the earlier, shorter run (1000 + 1000 draws per chain) until their long-chain versions
> finish tonight. Only the diagonal-A numbers (and so P1/P2) can change when they do.

**Toxin, one week ahead, against the published model.** 327 beach-weeks in 31 lakes. 33 were >= 8 µg/L, 10 of
them new onsets; 19 of the 252 beach-weeks below 1.6 µg/L crossed it.

| Model | AUC >= 8, all | AUC >= 8, new onsets | AUC >= 1.6, new events | Brier >= 8 |
|---|---|---|---|---|
| Carry-forward (today's level) | 0.944 | 0.884 | 0.697 | – |
| T4 regression (toxin history + read shares) | **0.963** | **0.925** | 0.766 | **0.049** |
| JSDM, full A | 0.950 | 0.907 | **0.821** | 0.071 |
| JSDM, diagonal A | 0.948 | 0.907 | 0.825 | 0.070 |
| Villanueva et al., XGBoost | 0.873 | 0.725 | 0.737 | 0.199 |
| Villanueva et al., logistic | 0.898 | 0.746 | 0.762 | 0.064 |

AUC is the chance that a beach-week that went on to exceed the level outscores one that didn't. 0.5 is a coin flip.

- **Pre-registered test, JSDM minus Villanueva XGBoost (AUC >= 8, all beach-weeks):** +0.076 [+0.036, +0.171]. The
  JSDM is better.
- **JSDM vs carry-forward:** about the same at >= 8 (+0.006 [-0.019, +0.058]), because most >= 8 weeks were already
  high the week before. The JSDM is better at spotting new >= 1.6 events. At one week ahead the interval just touches
  zero (+0.124 [-0.007, +0.260]); at two weeks it doesn't (+0.167 [+0.072, +0.270]).
- **JSDM vs T4:** ranking is similar. T4's probabilities are better calibrated: Brier at >= 1.6, one week ahead,
  JSDM minus T4 = +0.012 [+0.002, +0.025], i.e. T4 is better.

**Full vs diagonal *A* (pre-registered, one week ahead).**
- P1 (reads): -0.79 [-2.07, +0.10]. No detectable difference; see "Things to know" about how unstable this one is.
- P2 (toxin, Brier >= 1.6): +0.0008 [-0.0016, +0.0031]. No difference.
- In short, there's no evidence that letting taxa affect each other improves the 1-4 week forecasts.

**Convergence of the long-chain fits** (2000 warm-up + 3000 draws, 4 chains):
- Worst R-hat is 1.003-1.009 across these fits, all under the usual 1.01 target.
- The lowest bulk effective sample size is 641.
- Divergent transitions: 7 in total out of 204,000 draws (the 17 long-chain fits so far).
- In three full-A fits (weeks 4, 10 and 12), two or three chains hit the maximum tree depth on most iterations.
  That makes them slow but, judging by R-hat and ESS, not wrong.

## Checks against the original R implementation

The analysis was first written in R. Every step of this port was checked against it.

| Step | Check | Result |
|---|---|---|
| Stan input (`prep_data.py`) | Origins 2, 8 and 14, both models; matched by beach name | All counts, indices and censoring flags identical. Real numbers agree to 6e-15, which is the precision R writes to Stan's input file. Beach order differs (R sorts names by locale), which doesn't change the model. |
| T4 and carry-forward (`regressions.py`) | All 1,807 forecasts per model | T4 within 2e-9 of R's `survreg`; carry-forward identical |
| JSDM forecasts (`forecast_jsdm.py`) | Same posterior draws as R | Toxin probabilities correlate 0.999 (Monte Carlo noise only, since the random numbers differ). Read scores: median per-library difference 0.01. |
| Published-model comparison | Same cases (327 at 1 week) | Carry-forward, T4 and logistic identical; JSDM within 0.002. XGBoost is slightly different because its random downsampling can't reproduce R's random number stream. |
| Fitting (`fit_jsdm.py`) | Test fit at week 3, diagonal A, 30 + 30 iterations | Runs end to end and saves the same 262 parameters as the R fits. Its toxin forecasts correlate 0.998 with those from the full-length fit, even though Python orders the beaches differently, so the labels line up. The reported fits themselves were sampled through R's cmdstanr with the same Stan file, data, seeds and settings, then converted to `.npz`. |

## Things to know

- **P1 depends on the forecast simulation's random seed.**
  - The read log predictive density averages each library's probability over 1000 simulated futures.
  - For a handful of libraries that surprised the model (e.g. 20% *Microcystis* when it expected ~1%), only a few
    simulations land anywhere near the observation. Their log-density changes by tens of units between seeds.
  - P1 is a small mean difference (under 1 unit), so those few libraries decide its sign. In a check on one set of
    posterior draws, seeds 7-11 gave P1 anywhere from -0.67 to +0.19. The original R run's +0.57 [+0.03, +1.67] (which
    looked like a win for full A) is within that kind of spread.
  - Read P1 as "no clear difference". A more stable version would use many more simulated futures per library.
- **The toxin results are stable.** Exceedance probabilities average a smooth function over draws. Changing the seed
  moves P(>= 1.6 µg/L) by 0.003 typically and 0.03 at most.
- **The published XGBoost model is sensitive to its random class-balancing downsample**, because there are few events
  to balance against.
  - At one week ahead, Python's random numbers give AUC 0.873 and R's give 0.886. The primary test favours the JSDM
    either way: +0.076 [+0.036, +0.171] in Python vs +0.063 [+0.014, +0.161] in R.
  - At four weeks ahead the two runs give 0.57 vs 0.76, so don't lean on the XGBoost numbers beyond one week.
  - The logistic version of their model has no randomness and matches R exactly.
- **Weather is held at its forecast-week value** for all four weeks ahead. We don't forecast weather.
- **This is one test season:** 2019, 31 lakes. At one week ahead there are only 33 beach-weeks at >= 8 µg/L, 10 of
  them new onsets, so the intervals are wide.

## Reference

Villanueva et al. (2023). *Environmental Science & Technology* 57, 20636.
