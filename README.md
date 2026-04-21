# NBA Draft Prospect Predictor
### CS 451 — End-to-End Data Science Final Project

**Problem.** NBA teams invest millions scouting draft prospects. This project
asks: how much of a player's four-year NBA value (VORP) can be recovered
from college box-score statistics and high-school recruiting rank alone?

**Dataset.** 1,042 college players drafted 2000–2022, scraped from
Basketball-Reference and the 247Sports composite recruiting index.

**Best model.** XGBoost regression on 41 engineered features, trained on
2000–2018 and evaluated on 2019–2022:

| Model            |  R²    |  MAE  |  RMSE |
| ---------------- | -----: | ----: | ----: |
| Ridge (baseline) | 0.162  | 1.78  | 2.35  |
| XGBoost          | **0.202** | **1.67**  | **2.29**  |
| Neural Network   | 0.152  | 1.65  | 2.36  |

On classification (positive VORP / negative VORP), XGBoost hits 56.5 %
accuracy and 0.591 AUC. All metrics are on a strictly held-out 2019–2022 test
set (temporal split — no leakage).

**Deployment.** An editorial-style single-page dashboard (`docs/index.html`)
that ships the Ridge coefficients to the browser so readers can drive a live
predictor. Also hosts a 2026 big board, a PCA scatter of all historical
players, and a drawer with SHAP drivers + same-position comps for every
prospect.

---

## Repository structure

```
CS 451 Final Project/
├── fixeddata.csv            source: 1,042 historical draftees (2000–2022)
├── 2026class.csv            source: 40 incoming 2026 prospects
├── requirements.txt
├── src/
│   ├── config.py            paths + constants (K_BY_POSITION, SPLIT_YEAR, …)
│   ├── preprocess.py        shared: load + clean + feature-engineer
│   ├── eda.py               generates figures 1–7
│   ├── train.py             MAIN pipeline — trains all models + exports
│   ├── generate_bigboard.py projects 2026 class using saved artifacts
│   ├── recluster.py         refits K-means only (skips retraining)
│   └── audit_data.py        data-quality auditor
├── models/                  .pkl artifacts (auto-created by train.py)
├── figures/                 EDA + modeling figures (auto-created)
├── docs/
│   ├── index.html           the dashboard
│   └── data/                JSON exported by train.py
└── notebooks/
    └── pipeline.ipynb       reference notebook (the scripts are primary)
```

## Reproducing results

```bash
pip install -r requirements.txt
python src/train.py               # trains all models + writes every JSON + figure
python src/generate_bigboard.py   # scores the 2026 class → docs/data/bigboard.json
python src/eda.py                 # (optional) regenerates EDA figures
cd docs && python -m http.server 8080
# open http://localhost:8080
```

`train.py` takes about two minutes (the XGBoost `RandomizedSearchCV` with
`n_iter=50` is the bottleneck). Everything downstream finishes in seconds.

To tweak clustering without retraining the predictive models:

```bash
python src/recluster.py           # refits per-position K-means, patches JSON
python src/generate_bigboard.py   # rerun so the board picks up new cluster IDs
```

## Pipeline at a glance

1. **Ingest** — `fixeddata.csv` (1,043 rows) → drop rows missing
   `nba_4yr_vorp` (n=1 dropped) → fix two scraping-error `fg_pct` values
   (Hamady N'Diaye, Tristan Thompson).
2. **Feature engineering** — 13 new columns beyond the 28 raw ones,
   including position-relative z-scores for 7 stats, a defensive-versatility
   interaction, a shrinkage-regularised volume-weighted 3-point percentage,
   and a position-relative build score.
3. **Temporal split** — train on classes 2000–2018 (n=851), test on 2019–2022
   (n=191). No draft year leaks across the split.
4. **Modeling** — three regressors trained in parallel:
   Ridge (interpretable baseline), XGBoost (randomised hyperparameter search,
   5-fold CV), and an MLP. Parallel classification heads on the binarised
   `PosVORP` target.
5. **Unsupervised archetype discovery** — K-means fit **within each position
   group** (k=3 guards, k=3 forwards, k=2 centers) for 8 interpretable
   archetypes. Global PCA to 2D for the dashboard scatter.
6. **Explain + export** — SHAP values per prospect, same-position nearest
   neighbours as "comps", Ridge coefficients and scaler parameters, plus a
   projection for every 2026-class player, all serialised to
   `docs/data/*.json` for the browser to consume.

## Models saved to `models/`

| File              | What it is                                       |
| ----------------- | ------------------------------------------------ |
| `xgboost_reg.pkl` | Regression head on 4-year VORP (best test R²)    |
| `xgboost_clf.pkl` | Classification head on positive 4-year VORP      |
| `nn_reg.pkl`      | MLP regressor for comparison                     |
| `nn_clf.pkl`      | MLP classifier for comparison                    |
| `ridge.pkl`       | Ridge regression — exported to the frontend      |
| `scaler.pkl`      | `StandardScaler` fit on the 41-feature matrix    |
| `kmeans.pkl`      | Dict `{position: {model, offset, k}}`            |

## Front-end artefacts (`docs/data/*.json`)

| File                  | What the dashboard uses it for                        |
| --------------------- | ----------------------------------------------------- |
| `prospects.json`      | 1,042 rows: stats, predictions, SHAP, comps           |
| `clusters.json`       | 8 archetypes with per-position mean profiles          |
| `position_stats.json` | Position means/stds for client-side z-score display   |
| `model_meta.json`     | Test metrics + Ridge coefs for the live predictor     |
| `bigboard.json`       | 40 projected 2026 prospects, ranked by predicted VORP |

## Limitations

* **R² ≈ 0.20.** Four-fifths of the variance in four-year VORP is not
  recoverable from the features we have. College box scores don't capture
  strength-of-schedule context, shot selection, defensive responsibility,
  injury history, or the human factors NBA scouts get paid to assess.
* **Small center sample.** Only 120 centers in the dataset; per-position
  K-means uses k=2 for that group.
* **Position labels are noisy.** "Forward" conflates wings and bigs. The
  `*_vs_pos` z-score features partially absorb this but don't fix it.
* **Era drift.** A player drafted in 2003 plays a very different game than
  one drafted in 2022. The model is agnostic to this.

## References

Training data: [basketball-reference.com](https://www.basketball-reference.com/)
(draft history, college and NBA advanced stats).
Recruiting: [247sports.com](https://247sports.com/) composite index.

---

*Colin Hanley · CS 451 · Spring 2026*
