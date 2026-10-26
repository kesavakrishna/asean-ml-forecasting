# asean-ml-forecasting

Forecasting the daily direction (up/down) of the ASEAN-5 stock indices with machine learning: Indonesia, Malaysia, the Philippines, Singapore and Thailand, with India's Nifty 50 as a benchmark.

**Status:** done so far:
- step 1: EDA, see [eda_report.md](eda_report.md)
- step 2: one cleaning function, walk-forward evaluation, leakage test
- step 3: logistic regression in sklearn and PyTorch

## Step 2 findings: walk-forward baselines

Every market is tested on six separate years. For test year Y: train on every year up to Y−2, tune on Y−1, test on Y. The table shows the accuracy of the best no-model guess ("always up", "always down" or "same as yesterday"), picked by looking at the test year. No model could know that in advance, so this is a **stretch bar**. Step 3 adds a fair bar that's fixed before each test year starts.

| Stretch bar, accuracy % | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | Range |
|---|---|---|---|---|---|---|---|
| Indonesia | 55.9 | 51.2 | 53.9 | 52.5 | 53.0 | 52.0 | 4.6 |
| Malaysia | 51.9 | 55.1 | 53.7 | 51.2 | 50.6 | 53.9 | 4.5 |
| Philippines | 55.6 | 53.5 | 52.1 | 50.4 | 52.4 | 52.2 | 5.1 |
| Singapore | 51.2 | 52.2 | 50.8 | 54.5 | 54.5 | 54.8 | 4.0 |
| Thailand | 56.0 | 51.8 | 51.6 | 51.9 | 53.5 | 56.0 | 4.4 |
| India | 56.3 | 54.1 | 53.9 | 59.1 | 56.9 | 59.3 | 5.4 |

- **The bar moves 4–5.4 points between years** in every market. The guess that sets it changes too; "always up" sets it most often.
- **One test year is noisy.** It has about 245 trading days, so one standard error is about 3.2 points. A model that beats the bar by less than about 6 points in a single year could be doing so by luck.
- **Pooling the six years cuts the noise** to about 1.3 points per standard error, so a pooled gap of about 2.6 points or more starts to mean something. These rules of thumb are for one accuracy on its own. From step 3 on, models are compared with a paired test instead (see below).
- **"Same as yesterday" has no consistent edge in the ASEAN markets.** Its MCC ranges from −0.17 to 0.12 and changes sign from year to year. In India it rises steadily, from 0.03 in 2017 to 0.19 in 2022.
- **All six markets pass the cleaning checks, and the leakage test passes.** The leakage test also fails on a planted leak, so it isn't passing by default.

## Step 3 findings: logistic regression

**Setup:**
- **Features:** the last 5 daily returns and yesterday's direction (+1/−1), all from the same market.
- **Folds:** in each fold, the scaler is fitted on the training years only. The penalty strength is picked by log loss on the tune year (Y−1), then the model is retrained on train + tune and tested on Y.
- **PyTorch version:** the same model as one `nn.Linear(6, 1)` trained with `BCEWithLogitsLoss` and weight decay. Its penalty is tuned separately from sklearn's.
- **Two bars:**
  - **Fair bar:** the direction that was more common in the years before the test year. It was "up" in every fold for every market.
  - **Stretch bar:** the hindsight best guess from step 2.
- **Paired test:** the model and a bar are compared with McNemar's test, which only counts the days where exactly one of them is right.

All six test years pooled (sklearn):

| Market | Model accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday |
|---|---|---|---|---|---|---|---|
| Indonesia | 52.9 | 0.01 | 0.51 | 52.9 | 1.00 | 49.5 | 0.07 |
| Malaysia | 49.5 | −0.01 | 0.48 | 49.8 | 0.75 | 49.3 | 0.91 |
| Philippines | 51.5 | 0.01 | 0.49 | 51.5 | 1.00 | 49.0 | 0.18 |
| Singapore | 53.7 | 0.07 | 0.53 | 51.3 | 0.11 | 49.2 | 0.02 |
| Thailand | 52.1 | −0.06 | 0.50 | 52.4 | 0.06 | 51.0 | 0.57 |
| India | 54.0 | 0.01 | 0.54 | 54.7 | 0.44 | 55.0 | 0.54 |

- **No edge in the ASEAN markets.** Pooled accuracy matches the fair bar to within half a point everywhere except Singapore (+2.4 points, p = 0.11, not significant). AUC is 0.48–0.53, about what a coin flip scores (0.50).
- **The model often just says "up".** In 16 of 36 folds, tuning picked a penalty strong enough that the model calls every test day up. That's tuning's way of saying the features don't help. It happened in 5 of 6 folds for Thailand and 4 of 6 for the Philippines.
- **Only 1 of 36 folds beats the fair bar significantly:** Singapore 2019 (60.0% vs 50.8%, p = 0.03). With 36 tests at the 0.05 level, about 2 would pass by luck alone, so one is not evidence of skill.
- **India: logistic regression doesn't capture the direction persistence,** even though it's given yesterday's direction directly.
  - Pooled, it scores 54.0%, below both the fair bar (54.7%) and "same as yesterday" (55.0%).
  - In 2022 it's significantly worse than "same as yesterday": 48.4% vs 59.3%, p = 0.014. Of 113 days where they disagreed, the model was right on 43 and "same as yesterday" on 70.
  - A likely reason (not yet tested): the model learns one fixed weight from all years since 2010, when persistence was weaker, and tuning often picks heavy shrinkage.
  - **Read this with caution.** Pooled over six years, "same as yesterday" beats the fair bar by only 0.3 points (55.0% vs 54.7%), so the persistence is concentrated in recent years. India's test years have now been looked at several times, so a fix tailored to India risks fitting what's already been seen.
  - **Next:**
    - Check India's data against a second source first.
    - If a shorter training window is tried, run it on every market, not just India.
    - Treat "India's direction repeats" as an idea to confirm on untouched 2023–2026 data.
- **sklearn and PyTorch agree.**
  - With separately tuned penalties, predicted probabilities correlate at 0.9986 or higher in all 36 folds, and up/down calls agree on 89.5–100% of days.
  - The lowest agreement is India 2022 and Singapore 2018. The cause is that the two tuned penalties aren't equivalent: the grids only hold round numbers, so probabilities near 0.5 tip opposite ways.
  - Both versions leave the intercept unpenalised, so the intercept isn't the cause. In PyTorch the bias has its own parameter group with weight decay 0.
  - With the exact conversion, weight decay = 1 / (C × training days), PyTorch matches sklearn almost perfectly. Correlation is 0.9995 or higher in every fold, and the up/down calls are identical in 35 of 36 folds (98.4% in the other).

## Files

| File | What it is |
|---|---|
| `data.py` | `load_market()` / `load_all()`: load, clean and check the data. Every step after the EDA uses this |
| `evaluation.py` | Walk-forward splits (test years 2017–2022), scoring (accuracy, balanced accuracy, MCC), no-model baselines, fair baseline, McNemar paired test |
| `features.py` | Model features: the last 5 daily returns and yesterday's direction |
| `models.py` | Logistic regression in sklearn and PyTorch, the six-fold loop, per-year and pooled reports |
| `test_leakage.py` | Checks that nothing computed for day t changes when data after t is removed or scrambled |
| `walk_forward_baselines.ipynb` | Baselines scored on each test year |
| `logistic_regression.ipynb` | Step 3: both logistic regressions through the six folds, and the agreement check |
| `eda.ipynb` | EDA notebook: data checks, cleaning, returns, up-day shares, baseline scores |
| `eda_report.md` | Write-up of the EDA findings |
| `eda_output/` | Figures used in the report (saved by the EDA notebook) |

## Data

The data isn't included in this repo. It's daily index history exported from [Investing.com](https://www.investing.com), Jan 2012 – Mar 2023 (the Nifty 50 starts in Jan 2010). To run the notebook, put these files in a `Dataset/` folder:

- `Jakarta Stock Exchange Composite Index Historical Data.csv`
- `FTSE Malaysia KLCI Historical Data.csv`
- `FTSE Philippines Historical Data.csv`
- `FTSE Straits Times Singapore Historical Data.csv`
- `SET Thailand Index Historical Data.csv`
- `Nifty 50 Historical Data.csv`

Each file needs `Date`, `Open`, `High`, `Low` and `Price` (or `Close`) columns.

## Running

Tested with Python 3.13 and pandas 3.0.

```
pip install pandas numpy matplotlib scikit-learn torch jupyter
python test_leakage.py
jupyter notebook logistic_regression.ipynb
```
