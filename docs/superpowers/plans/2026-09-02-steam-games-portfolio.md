# Steam Games Portfolio Project — Learning Plan

**Goal:** Build the EDA notebook, owners-bucket classifier, and Streamlit dashboard described in `docs/superpowers/specs/2026-09-02-steam-games-portfolio-design.md`, with the user writing the code and Claude guiding/reviewing (not writing it for them).

**Mode:** Interactive pairing, not autonomous execution. Steps below are checkpoints, not subagent tasks — worked through one at a time in conversation.

## Steps

1. **Environment setup** — venv, install pandas/matplotlib/scikit-learn/jupyter/streamlit, verify notebook runs
2. **Load & inspect raw data** — read `steam_games.csv`, check shape/dtypes/nulls, understand what we're working with
3. **Clean & engineer features** — drop unneeded columns, parse list-string columns, build engineered features (tag_count, platform_count, price_bucket, etc.)
4. **Export processed data** — chunk into `data/processed/games_part_*.csv` under GitHub's size limit
5. **EDA** — build the 4-6 storytelling charts (price/genre/owners relationships)
6. **Feature matrix for modeling** — encode categorical/multi-label features, explicit leakage exclusions
7. **Train baseline model** — `HistGradientBoostingClassifier` on owners bucket
8. **Evaluate & interpret** — classification report, confusion matrix, permutation importance
9. **Save model artifact** — `models/owners_classifier.pkl`
10. **Streamlit app — Explore tab** — reload processed data, interactive EDA charts
11. **Streamlit app — Predict tab** — form inputs → load model → prediction + probabilities
12. **Deploy to Streamlit Community Cloud**

Each step: user writes the code, Claude explains reasoning/reviews, commit when it runs.
