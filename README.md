# Bitcoin Sentiment + Market Direction Predictor

**A compact, reproducible pipeline that combines social signals (tweets + crypto news) with Bitcoin market data to predict next-day price direction.**

**Author:** (mehrshad kolahi)
**Notebook:** `final4.ipynb`

---

## Executive summary

This project demonstrates a simple end-to-end data science pipeline that:

* loads large social datasets using **Dask** to avoid memory blowups,
* performs light text cleaning and **daily aggregation** of tweets + crypto news,
* extracts lightweight NLP signals (TF‑IDF + VADER sentiment) and basic market features (returns, moving averages, volatility),
* constructs time-series-aware features (lags and rolling means),
* labels the next-day direction with a small positive threshold to reduce label noise,
* trains a **time-aware RandomForest** (TimeSeriesSplit + GridSearchCV) and evaluates on a hold-out period.

The final, simple pipeline achieved **~68% test accuracy** on the evaluated hold-out. The emphasis of the project is on **simplicity**, **reproducibility**, and explainability — making it suitable for a resume and academic demonstration.

---

## Detailed analysis of the project (what I inspected and conclusions)

I reviewed the notebook and the pipeline steps you implemented and validated the following key elements and potential pitfalls:

### ✅ Things done well

* **Memory-safe ingestion**: using Dask to read large CSVs and sampling reduces memory pressure during exploratory steps.
* **Per-day aggregation** of text: reduces row-count drastically and matches the time granularity of labels.
* **Lightweight text preprocessing**: removal of URLs, mentions, punctuation, lowercasing and truncating long texts — appropriate for TF‑IDF.
* **Sentiment signal**: VADER compound score per day adds a useful orthogonal signal to text TF‑IDF features.
* **Market features and technicals**: returns, MA, volatility, range are included — these are predictive for short-term direction.
* **Time-aware evaluation**: train/test split based on time + TimeSeriesSplit in GridSearch — prevents label leakage.
* **Label noise mitigation**: threshold sweep to require a minimum percentage return (e.g., 0.5%) for positive labels — this dramatically reduced noise and improved accuracy.
* **Simple, interpretable model**: RandomForest works well on this small data and is easy to explain in a report or CV.

### ⚠️ Issues & potential mistakes I found (and how to fix them)

I list issues that commonly appear with this kind of pipeline and what I changed/checked in your code to ensure robustness.

1. **MultiIndex columns from `yfinance`**

   * Problem: `yf.download` can return MultiIndex columns (Ticker/Price) which broke simple column access.
   * Fix: flatten MultiIndex and normalize column names to `date, open, high, low, close, volume` before computations.

2. **Alignment / positional comparison errors**

   * Problem: doing comparisons by label (pandas alignment) caused `Operands are not aligned`. This occurs if one operand is a DataFrame or has an unexpected index.
   * Fix: reset index, coerce numeric types, use positional `.values` or `np.where` for comparing `next_close` and `close` and drop the last row which lacks `next_close`.

3. **Potential data leakage**

   * Risk: using rolling means or features incorrectly can introduce look-ahead if not shifted properly.
   * Fix: all rolling features used for prediction were computed with `.shift(1)` or constructed to avoid using the current day's `next_close` value.

4. **Sparse TF‑IDF turned dense**

   * Problem: converting TF‑IDF to dense `toarray()` is memory heavy and unnecessary for certain models.
   * Note: for simplicity you converted to dense which is OK for ~650 rows; if dataset grows, use TruncatedSVD or train models that accept sparse inputs.

5. **Label construction and threshold selection**

   * Observation: choosing a small positive threshold (0.5% / 0.005) improved accuracy because tiny price moves are noise. This is valid but must be documented clearly.

6. **Imbalanced evaluation metrics**

   * Observation: high overall accuracy (68%) can mask poor class-wise performance — the model favored the majority class in many runs (precision/recall differ by class).
   * Suggestion: report precision, recall, f1, AUC, and a confusion matrix; consider balanced training or evaluation using MCC or balanced accuracy.

7. **Reproducibility and modularity**

   * Notebook is fine for demonstration, but package the pipeline into small scripts (`src/data.py`, `src/features.py`, `src/train.py`) for reproducibility.
   * Save artifacts: vectorizer, scaler, and trained model with `joblib.dump`.

8. **No economic evaluation**

   * Suggestion: add a naive backtest (trade on model signal with simple rules) and compute P&L, Sharpe ratio, and maximum drawdown to show if predictions could have practical value.

9. **Testing & logging**

   * Add unit tests for key functions (labeling, feature engineering) and simple logging for pipeline progress.

### Small reproducibility checklist I ran

* `random_state=42` used in sampling and model training.
* Time-aware CV used for hyperparameter search.
* Dropped rows with missing next_close to avoid label artifacts.

---

## Recommended minimal pipeline structure (simple, still readable)

If you want to refactor into a lightweight reproducible pipeline (still simple), structure it like this:

```
repo/
  data/                 # raw CSVs (gitignored)
  notebooks/            # your jupyter notebook (exploratory)
  src/
    data.py             # load & basic cleaning (Dask + safe sampling)
    features.py         # aggregate daily, TF-IDF, sentiment, numeric features
    model.py            # training, cross-validation, grid search
    predict.py          # load artifacts + predict on new data
  artifacts/
    vectorizer.joblib
    scaler.joblib
    model.joblib
  README.md
  requirements.txt
```

Each `src/*.py` should expose 1–2 functions and be easily runnable from the command line. Keep the notebook for visualizations.

---

## Quick wins to improve performance further (without heavy complexity)

1. Increase `TFIDF_MAX` to 1500–2000 **only if** RAM allows; else lower it.
2. Try `min_df=1` vs `min_df=2` — sometimes rare tokens are predictive.
3. Use more lag features up to 7 or 14 days (you already used up to 7; try 14).
4. Add **RSI (14)** and simple momentum: `close - ma_7`.
5. Run a small ensemble: RF + LightGBM with VotingClassifier (easy, usually helps).
6. Report balanced accuracy and AUC; tune for the metric you care about.

---

## What I will put in the new README

I will produce a focused README (English) suitable for a resume/university project that:

* summarizes the motivation and approach (one paragraph),
* lists exact pipeline steps (bullet points),
* includes quick instructions to reproduce (venv, install, run notebook),
* states results and known limitations, and
* provides next-steps + contact info.


---

If this analysis looks good, say "Go ahead" and I'll finalize the README (I already prepared an initial README earlier; I'll overwrite it with this refined version). If you want any particular phrasing (e.g., include your name, GitHub handle, or university), tell me and I'll include it.
