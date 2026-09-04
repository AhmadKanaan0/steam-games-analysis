# Steam Games Portfolio Project — Design

## Goal
A portfolio project built from `steam_games.csv` (~927MB, one row per Steam game) that demonstrates:
1. EDA and data storytelling — what drives a game's reach/reception on Steam
2. A supervised ML model — predicting a game's estimated-owners bucket from pre-launch-available features

Learning mode: the user is building this themselves with guidance (some pandas/plotting experience, less ML modeling experience). Explanations should focus more on modeling/feature-engineering reasoning than basic pandas syntax.

`steam_games_reviews.csv` (press quote text) is explicitly out of scope for this project.

## Deliverables
1. `notebook/steam_analysis.ipynb` — the main artifact: data cleaning, EDA, feature engineering, model training/evaluation, narrative
2. `data/processed/games.parquet` — cleaned, feature-engineered dataset, committed to the repo. Parquet compression brings 927MB of raw CSV down to ~32MB, so no chunking is needed (the originally planned `games_part_*` split was dropped as unnecessary).
3. `models/owners_classifier.pkl` — trained model artifact
4. `app/streamlit_app.py` — Streamlit dashboard (EDA tab + prediction tab), deployed to Streamlit Community Cloud from the repo

## Data prep
Source columns to drop (not needed for EDA or modeling, and bulk of the file size): `about_the_game`, `detailed_description`, `notes`, `header_image`, `screenshots`, `movies`, `website`, `support_url`, `support_email`, `metacritic_url`.

Columns to parse from stringified-list form into real lists/counts: `genres`, `categories`, `tags`, `developers`, `publishers`, `supported_languages`, `full_audio_languages`.

Engineered features:
- `tag_count`, `genre_count`, `category_count`, `language_count`
- `platform_count` (sum of windows/mac/linux booleans)
- `has_achievements` (achievements > 0)
- `description_length` (from `short_description`, the one text field we keep)
- `price_bucket` (free / budget / mid / premium)
- `release_year`, `release_month`
- `positive_ratio` = positive / (positive + negative) — EDA/target-adjacent, **not** used as a model input feature (see leakage note below)

Output: cleaned + engineered table saved as a single Parquet file at `data/processed/games.parquet` (~32MB). Parquet preserves the parsed list columns natively, so no re-parsing is needed on load.

Data quality issues found in the raw file (all handled during parsing):
- Some rows store a single value as a bare string instead of a JSON array (e.g. `supported_languages` = `English`) — parser falls back to wrapping it in a list.
- Some rows store `genres`/`categories` as lists of `{id, description}` objects instead of plain name strings — normalized by extracting `description`.
- ~23.5K rows store `tags` as a dict of tag name → vote count instead of a list — normalized by taking the dict keys (vote counts discarded, since the majority of rows don't carry them).
- `windows`/`mac`/`linux` contain NaN alongside True/False — filled with False and cast to bool.
- `estimated_owners` encodes the same bucket range in two different string formats (e.g. `"20000 - 50000"` vs `"20,000 .. 50,000"`), fragmenting 14 real categories into 26 — normalized via regex extraction of the two numeric bounds into one canonical `"{low} - {high}"` label.

## EDA (in notebook)
- Distributions: price, genres, tags, platform support, release timing
- Relationships: price vs estimated_owners, tag/genre count vs owners, platform_count vs owners, free vs paid, release year trends
- Relationships: what correlates with `positive_ratio` (price, platform_count, tag_count, genre)
- 4-6 sharpest charts selected for the narrative (e.g. "which genres have the best owners-per-game odds", "does going free-to-play change the owners distribution")

## Model
- **Target:** `estimated_owners`, normalized to canonical bucketed string ranges (e.g. `"0 - 20000"`), used as multiclass classification labels. Two preparation decisions, both made deliberately and documented in the notebook:
  - **Zero-owner rows are dropped.** The `0 - 0` bucket holds 21,185 games (15.18%) with literally zero owners — almost certainly unreleased, delisted, or broken catalog entries rather than games that launched and sold nothing. Including them would train the model on data artifacts rather than commercial outcomes. The model therefore answers: *among games that actually launched, how large an audience will this reach?*
  - **The five smallest buckets are merged into one `5000000+` class.** Buckets of 125, 51, 31, 9 and 4 games cannot be stratified or evaluated meaningfully — a 4-game class yields ~1 test sample, making its per-class F1 noise rather than a metric. Merging leaves 9 defensible classes (smallest: 220 games).
- **Class imbalance:** even after preparation the target is heavily imbalanced (~78% of remaining games sit in the lowest bucket). Evaluation uses class-aware metrics (per-class precision/recall/F1, confusion matrix) rather than raw accuracy, since a majority-class-only classifier would otherwise look deceptively good.
- **`release_year` is trained both ways.** It is technically pre-launch information, but section 12's maturity-bias finding means it largely proxies "how long this game has had to accumulate owners" rather than anything about the game itself. Two models are trained — with and without it — and their scores and feature importances compared; the contrast is itself a finding.
- **Features:** price, price_bucket, genres (multi-hot or top-N encoded), tags (multi-hot top-N), categories, platform_count, dlc_count, has_achievements, description_length, language_count, release_year, release_month
- **Explicitly excluded from features (leakage):** `positive`, `negative`, `recommendations`, `peak_ccu`, `user_score`, `metacritic_score`, all `*_playtime_*` columns — these are post-launch outcomes correlated with owners, not pre-launch predictors. This is the key modeling lesson of the project: a model trained on these would look great and mean nothing.
- **Algorithm:** `HistGradientBoostingClassifier` (scikit-learn) — handles mixed numeric/categorical features well, no heavy tuning needed, no extra dependency beyond sklearn
- **Evaluation:** train/test split, classification report (precision/recall/F1 per bucket), confusion matrix, feature importances (via permutation importance) as the narrative payoff — which pre-launch decisions actually move the needle

## Dashboard (Streamlit app)
- **Tab 1 — Explore:** key EDA charts from the notebook, with interactive filters (genre, price range, release year range)
- **Tab 2 — Predict:** form inputs mirroring the model's features (price, genre(s), tags, platforms, dlc count, achievements, release timing) → predicted owners bucket + class probabilities, using the saved model artifact
- Data loading: concatenate the `data/processed/games_part_*.csv` chunks at startup (cached with `st.cache_data`)
- Deployment: Streamlit Community Cloud, connected to the GitHub repo

## Process (learning mode)
Work happens step by step with the user writing code and the guide (Claude) explaining reasoning and reviewing:
1. Set up repo structure + git
2. Load raw data, explore shape/columns/dtypes
3. Clean + engineer features → save chunked processed data
4. EDA charts + narrative
5. Build feature matrix, explain leakage, train baseline model
6. Evaluate, interpret feature importances
7. Build Streamlit app (explore tab, then predict tab)
8. Deploy to Streamlit Community Cloud

Each step ends with working, runnable code before moving to the next.

## Out of scope
- `steam_games_reviews.csv` / press-quote NLP (future project)
- Hyperparameter tuning beyond sane defaults
- Any external data enrichment (e.g. live Steam API calls)
