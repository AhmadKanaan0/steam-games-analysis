# Can you predict a Steam game's audience before it launches?

An analysis of **139,580 Steam games**, ending in a model that predicts how many owners a game will reach using only information available *before* release — and an honest account of how well that actually works.

Short answer: partially. The model ranks games reasonably well (79.6% of predictions land within one owners bucket, and mega-hits are essentially never confused with flops) but cannot pin an exact figure. Most of what determines a game's reach — marketing, quality, timing, streamer coverage — is not in this dataset, and a model claiming otherwise would be worth distrusting.

📓 **[Read the notebook →](notebook/steam_analysis.ipynb)**

---

## What this project actually demonstrates

Not a high score. The final leak-free model reaches a macro F1 of 0.193, which is modest. What it demonstrates instead:

### The dataset lies in seven different ways, and none are visible in `head()`

Every one surfaced as a downstream error and was traced back to the offending value:

| # | Issue | Scale |
| --- | --- | --- |
| 1 | Single values stored as bare strings instead of JSON arrays (`supported_languages` = `English`) | a few rows |
| 2 | Genres/categories stored as `{id, description}` objects instead of names | a few rows |
| 3 | `tags` stored as a name→vote-count dict instead of a list | ~23,500 rows |
| 4 | `NaN` mixed into boolean platform columns, silently forcing them to `object` dtype | 15 rows |
| 5 | `estimated_owners` encoded in two formats (`"20000 - 50000"` vs `"20,000 .. 50,000"`), fragmenting 14 real categories into 26 | dataset-wide |
| 6 | `tags_count` hard-capped at 20 — 33,494 games pile up at exactly 20, against ~2,600 at 19 and **2** at 21 | 24% of rows |
| 7 | `price_status = "unavailable"` means neither "no price recorded" nor "delisted" — both hypotheses tested and ruled out | 4,230 rows |

Issue 4 is a good example of why this matters: adding three boolean columns to count platform support returns a *boolean* in pandas, not a 0-3 count. The bug throws no error — it silently produced a feature meaning "supports at least one platform," which is true of 139,563 of 139,580 games and therefore carries no information at all.

### An accuracy score that was worse than doing nothing

The first model scored **78.3% accuracy**. A `DummyClassifier` that ignores every feature and always answers "0 - 20000" scored **78.4%**.

The trained model was fractionally *worse than a constant*. Its recall on two entire classes — 3,356 test games — was exactly zero. Reporting "78% accuracy predicting game success" would have been technically true and completely misleading. Class weighting fixed it, at the cost of the headline number dropping to 52%.

### Finding my own data leakage, after the model was already working

Permutation importance ranked `tags_count` as the top feature by a factor of 4.6 over the next. But Steam tags are applied by *players*, accumulating over a game's life — so `tags_count` is a proxy for how many people engaged with a game, not a description of it. The model had located the best available stand-in for popularity and leaned on it hard.

Applying that test properly disqualified two more features that had passed a looser audit: `dlc_count` (DLC ships after the base game, mostly for successful ones) and the localisation counts (translations get added post-launch, mostly for games that sold well). All three accumulate over a game's lifetime, and accumulate faster for successful games — the same leakage as review scores, just better disguised.

**About 22% of the best model's macro F1 came from information a developer would not have on launch day.**

Dropping `tags_count` alone barely helped, because the summed tag indicator columns correlate with it at **0.936** — the leak survived through the indicators.

---

## Model results

Nine owners buckets, after dropping zero-owner entries (unreleased/delisted artifacts) and merging the five smallest buckets, which held as few as 4 games each and could not be evaluated meaningfully.

| Model | Features | Accuracy | Macro F1 | Within 1 bucket |
| --- | --- | --- | --- | --- |
| Unweighted baseline | 11 | 0.783 | 0.140 | — |
| Class-weighted | 11 | 0.523 | 0.193 | 0.787 |
| + genres, tags, categories, publisher scale | 117 | 0.613 | 0.248 | 0.855 |
| − leaky counts | 113 | 0.598 | 0.230 | 0.843 |
| **− all tag features (fully clean)** | **53** | **0.565** | **0.193** | **0.796** |

Macro F1 alone understates this, because it treats every error as equally bad while these classes are ordinal. The confusion matrix shows the model ranks games sensibly: of games that truly reached 5M+ owners, **0%** were predicted as `0 - 20000`, and vice versa.

`release_year` was trained both ways. Despite ranking 2nd by permutation importance, it contributed **+0.003 macro F1** — the maturity bias it encodes (older games have had longer to accumulate owners) is real in the data, but the model was not relying on it.

---

## Other findings

- **Indie is 66.5% of the entire catalog.** Steam is overwhelmingly small and solo developers.
- **Prices cluster at charm points** ($0.99 / $4.99 / $9.99 / $19.99) rather than spreading smoothly — a comb, not a bell curve.
- **Higher-priced games reach more owners**, not fewer — median price rises steadily with owners bucket, contrary to the intuition that cheap sells more. (Correlation, not causation: games with breakout potential can also charge more.)
- **Free-to-play dominates at the very top.** Free games are 13-17% of most buckets, but 44% of the 50-100M bucket and 100% of the 100-200M bucket.
- **110,512 games support exactly one platform** (almost certainly Windows) against 12,492 supporting all three.

---

## Repo layout

```
notebook/steam_analysis.ipynb   the whole analysis, cleaning through modelling
data/processed/games.parquet    cleaned dataset (~32MB, from 927MB of raw CSV)
models/owners_classifier.joblib both trained models + vocabularies + test metrics
docs/superpowers/specs/         design decisions and why they were made
docs/superpowers/plans/         build plan and running findings log
```

The raw `steam_games.csv` is not committed (927MB). Download it from the source below and place it in the repo root to rerun the notebook from scratch.

## Running it

```bash
py -m venv .venv
.venv/Scripts/python.exe -m pip install pandas numpy matplotlib seaborn scikit-learn pyarrow jupyter
```

Then place `steam_games.csv` in the repo root, open `notebook/steam_analysis.ipynb`, and run all cells.

The notebook builds everything from the raw CSV, so it needs that file to rerun end to end. `data/processed/games.parquet` is its *output*, committed so the dashboard (and anyone who just wants the cleaned data) can skip the 927MB download and the cleaning step entirely:

```python
df = pd.read_parquet("data/processed/games.parquet")  # list columns survive intact
```

## Data source

Steam Games Dataset, from Kaggle. <!-- TODO: add the exact dataset URL --> The quirks documented above are properties of that scrape, not of Steam itself — notably the 20-tag ceiling and the ambiguous `price_status` values, both of which appear to be artifacts of how the data was collected rather than facts about the games.

## Still to come

A Streamlit dashboard with an interactive EDA view and a prediction form that shows **both** models side by side — because the higher-scoring model carries a train/serve mismatch (it learned "many tags → popular" from mature catalog entries, so a developer entering the 4-5 tags they plan to launch with would be systematically under-predicted). Showing both makes that trade-off visible rather than hiding it behind one number.
