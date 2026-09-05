# Steam Games Portfolio Project — Learning Plan

**Goal:** Build the EDA notebook, owners-bucket classifier, and Streamlit dashboard described in `docs/superpowers/specs/2026-09-02-steam-games-portfolio-design.md`, with the user writing the code and Claude guiding/reviewing (not writing it for them).

**Mode:** Interactive pairing, not autonomous execution. Steps below are checkpoints, not subagent tasks — worked through one at a time in conversation.

## Steps

- [x] 1. **Environment setup** — venv at `.venv`, pandas/matplotlib/seaborn/scikit-learn/jupyter/streamlit installed. Note: run Python via `py` (the `python` alias hits a Windows Store stub), and VS Code must use the `.venv` interpreter.
- [x] 2. **Load & inspect raw data** — 139,580 games × 32 columns after dropping unused text/URL columns via `usecols`.
- [x] 3. **Clean & engineer features** — parsed list columns, engineered counts, `platform_count`, `price_bucket`, `release_year`/`month`, `description_length`, `positive_ratio`.
- [x] 4. **Export processed data** — single `data/processed/games.parquet` (~32MB); chunking proved unnecessary.
- [x] 5. **EDA** — notebook sections 7-12: genre/tag/platform distributions, price vs owners, counts vs owners, free vs paid, release-year trends.
- [x] 6. **Feature matrix + leakage audit** — sections 13-14, 21.
- [x] 7. **Train models** — sections 16-20: baseline, class-weighted, feature-rich, `release_year` variant, leak-free variants.
- [x] 8. **Evaluate & interpret** — classification reports, confusion matrix, off-by-N ordinal metrics, permutation importance.
- [x] 9. **Save model artifact** — `models/owners_classifier.joblib` (both models + vocabularies + metrics).
- [x] 10. **Streamlit app — Explore tab** — `app/streamlit_app.py`. Filters for genre/price/year, four Altair charts, top-games table. Loads only 18 of 46 parquet columns via `read_parquet(columns=...)`, cutting memory from 188MB to 50MB for Streamlit Cloud's ~1GB tier.
- [x] 11. **Streamlit app — Predict tab** — form → **both** models side by side with probability charts, metrics read from the saved bundle, and the train/serve mismatch explained inline.
- [ ] 12. **Deploy to Streamlit Community Cloud** — `requirements.txt` is ready (sklearn pinned to 1.9.0 to match the pickled models). Repo is public: github.com/AhmadKanaan0/steam-games-analysis.

### App gotchas found while building

- Streamlit renders markdown, so a **pair of unescaped `$` is parsed as LaTeX math** and silently swallows the text between them. All dollar signs in captions/help text are escaped as `\\$`.
- matplotlib/seaborn are deliberately **not** in `requirements.txt` — the app uses Altair (which Streamlit ships) instead, for lower memory on the free tier. They remain notebook-only dependencies.

Each step: user writes the code, Claude explains reasoning/reviews, commit when it runs.

## Key findings so far (for resuming context)

**Seven data-quality issues found**, five of them schema inconsistencies fixed during parsing, two documented as limitations:
1. bare strings instead of JSON arrays (`supported_languages`)
2. `{id, description}` objects instead of genre/category names
3. `tags` as a name→vote-count dict (~23.5K rows)
4. NaN mixed into boolean platform columns
5. `estimated_owners` encoded in two formats, fragmenting 14 buckets into 26
6. `tags_count` hard-capped at 20 (33,494 games pile up there) — collection artifact
7. `price_status = "unavailable"` (4,230 games) — two hypotheses tested and ruled out; unresolved, likely a scrape artifact

**Also unresolved:** 2025's share of lowest-bucket games (55.3%) sits below 2024's (66.6%), backwards for maturity bias. `steam_spy_available` was checked and is constant, so uninformative. Documented as an open question.

**Modelling results** (9 classes after dropping `0 - 0` and merging the five smallest buckets):

| Model | Features | Accuracy | Macro F1 | Within 1 bucket |
| --- | --- | --- | --- | --- |
| unweighted baseline | 11 | 0.783 | 0.140 | — |
| class-weighted | 11 | 0.523 | 0.193 | 0.787 |
| + genres/tags/categories/publisher | 117 | 0.613 | 0.248 | 0.855 |
| − leaky counts | 113 | 0.598 | 0.230 | 0.843 |
| − all tag features (clean) | 53 | 0.565 | 0.193 | 0.796 |

- The unweighted baseline was **worse than a `DummyClassifier`** (0.783 vs 0.784) — the section's centrepiece demonstration that accuracy lies on imbalanced data.
- **Leakage found after the fact**: `tags_count` was the top feature by 4.6x but is community-generated and accumulates post-launch; `dlc_count` and the language counts fail the same test. ~22% of the best model's macro F1 came from leakage.
- Dropping `tags_count` alone barely helped because summed tag indicators correlate with it at **0.936** — the leak survives through the indicators.
- **Train/serve mismatch** identified: the model learned "many tag indicators → popular" from mature catalog entries, so a pre-release game with 4-5 tags would be systematically under-predicted. This is why both models ship.
