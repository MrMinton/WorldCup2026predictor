# ⚽ FIFA World Cup 2026 Predictive Analytics & Operational Simulation Pipeline

**Framework:** Supervised Multi-Class Classification & Rolling Time-Series Feature Engineering

An end-to-end machine learning and data engineering pipeline designed to predict international football match outcomes and simulate the expanded 48-team **FIFA World Cup 2026** tournament.

This architecture blends long-term team skill vectors (historical international Elo ratings) with high-frequency, short-term momentum data (rolling goal-scoring and goal-conceding form) to map engineered match features to outcomes (Home Win, Draw, Away Win).

---

## 🛠️ Tech Stack & Frameworks
* **Data Engineering & Vectorization:** `Pandas`, `NumPy`, `Itertools`
* **Machine Learning Pipelines:** `Scikit-Learn` (Ensembles, Preprocessing, Metrics)
* **Gradient Boosting Frameworks:** `XGBoost` (eXtreme Gradient Boosting Classifier)
* **Statistical Diagnostics & Data Viz:** `Seaborn`, `Matplotlib`

---

## 📊 Datasets & Ingestion Sources

1. **International Football Results (1872 - Present):** [Kaggle Dataset by Mart Jürisoo](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
   * `results.csv` and `shootouts.csv` — chronological match metrics, scores, tournaments, and neutral-site markers.
2. **World Football Elo Ratings:** [Kaggle Dataset by Ernest Wonyaya](https://www.kaggle.com/datasets/saifalnimri/international-football-elo-ratings)
   * `eloratings.csv` — time-series global team skill ratings.
3. `former_names.csv` — geopolitical name-mapping table used to reconcile teams that have changed names over time.

---

## ⚙️ Core Engineering & Architecture Highlights

### 1. Data Cleaning & Standardization
* `eloratings.csv` is re-decoded from `latin-1` to proper `UTF-8` to fix corrupted characters in team names before ingestion.
* Historical name changes are reconciled with a `former → current` mapping (plus manual overrides, e.g. stripping stray non-breaking spaces and correcting the Eswatini/Swaziland split) so Elo history and match records refer to the same country consistently.
* Rows missing scores are dropped, score columns are cast to `int`, and duplicate fixtures (same date/home/away) are removed.
* Matches lacking a resolvable Elo rating on either side are excluded from the final training set; **Moldova** is dropped entirely due to incomplete Elo coverage.

### 2. Temporal Integrity & Look-Ahead Bias Prevention
Team-level momentum features are engineered using a strict chronological sort and a `.shift()` window constraint, ensuring a fixture at time $t$ only uses performance data from before $t$. The feature space is sealed against look-ahead leakage.

### 3. Feature Set (7 engineered dimensions)
The model is trained on a deliberately lean feature set:
`elo_diff`, `neutral`, `home_form`, `away_form`, `tournament_stage`, `defensive_fragility_diff`, `offensive_dominance_diff`

* **`tournament_stage`** — a weighting (1.0–3.0) reflecting match importance (World Cup > continental championship > qualifier > friendly), so a team's form is contextualized by the stakes it was tested under.
* **`defensive_fragility_diff`** / **`offensive_dominance_diff`** — rolling 5-match goals-conceded and goals-scored differentials between the two sides, replacing a simpler form metric with separate offensive and defensive signals.

### 4. Sequential Round-by-Round Tournament Simulation
The simulation is deliberately run round-by-round rather than as a single automated bracket pass: because `home_form`, `away_form`, `defensive_fragility_diff`, and `offensive_dominance_diff` are all rolling features, each round's matches are predicted only *after* the previous round's results are known, so those rolling windows genuinely update between rounds instead of using stale pre-tournament form for the whole bracket. Each round (Group Matchdays 1–3 → Round of 32 → Round of 16 → Quarter-Finals → Semi-Finals → Final) is run as its own matchup list against the trained Random Forest's `predict_proba` output, with winners fed forward into the next round's matchup list by hand. There is no automated group-standings or FIFA tie-break resolver in the current notebook.

---

## 📈 Model Evaluation & Trade-offs

### Holdout Split vs. 5-Fold Cross-Validation

| Predictive Model Configuration | Single Holdout Score | 5-Fold Cross-Validation Mean | Status |
| :--- | :---: | :---: | :--- |
| **Random Forest Ensemble** | 64.46% | **63.36% (+/- 0.84%)** | **Operational Champion** |
| **XGBoost Classifier** | 61.12% | **59.68% (+/- 1.25%)** | Competitive, more variance |

### Algorithmic Insight
With the richer 7-feature set (particularly the added offensive/defensive rolling differentials), both models improved substantially over earlier iterations — XGBoost in particular is no longer collapsing to a near coin-flip on unseen folds.

The **Random Forest**'s higher headline accuracy comes with a notable trade-off, visible in the classification report: it achieves 0% precision and recall on the **Draw** class — it essentially never predicts a draw, funneling every match into Home or Away Win. **XGBoost**, by contrast, still attempts to classify draws (27% precision / 15% recall on that class), which costs it some overall accuracy but gives more balanced coverage across all three outcomes. This is the same underlying "Draw Penalty" pattern in international football: draws are driven by high-entropy events (late equalizers, red cards, missed penalties) that structural features alone struggle to resolve.

---

## 🏆 Simulation Results

The trained Random Forest model was run through the sequential round predictor described above. The most recent saved run in the notebook reaches:

* **Predicted Final:** Spain vs. Argentina
* **Model Probabilities:** Spain 42.6% · Draw 16.9% · Argentina 40.6%
* **Predicted Winner:** Spain — but by an extremely narrow margin, with the model itself flagging the match as close to a toss-up.

Predictions for each round are written out to `predictions_log/final_predictions.csv`. Intermediate-round matchups (group stage through the semi-finals) are present in the notebook as hand-entered matchup lists, each run only after the prior round so its rolling-form features reflect real tournament results — but their prediction outputs weren't retained in this saved run, so only the Final result above is currently reproducible from the notebook as-is.

---

## 🚀 Pipeline Directory Structure
```text
├── data/
│   ├── results.csv          # Historical international fixtures
│   ├── shootouts.csv        # Historic penalty shootout outcomes
│   ├── eloratings.csv       # Chronological international Elo ratings
│   └── former_names.csv     # Geopolitical country name mapping
├── predictions_log/
│   └── final_predictions.csv # Output of the round-by-round simulation
├── wc2026.ipynb              # Integrated prototyping, engineering & validation notebook
└── README.md                 # Project documentation
```
