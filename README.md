# asean-ml-forecasting

Forecasting the daily direction (up/down) of the ASEAN-5 stock indices with machine learning: Indonesia, Malaysia, the Philippines, Singapore and Thailand, with India's Nifty 50 as a benchmark.

**Status:** done so far:
- step 1: EDA on the original data, see [eda_report.md](eda_report.md)
- step 2: one cleaning function, walk-forward evaluation, leakage test
- step 3: logistic regression in sklearn and PyTorch
- data updated to Oct 2026, with an untouched holdout; steps 2–3 re-run on it

## Data

| Market | Index | Source | File in `Dataset/` |
|---|---|---|---|
| Indonesia | IDX Composite | Yahoo `^JKSE` | `Jakarta Composite (Yahoo).csv` |
| Malaysia | FTSE Bursa Malaysia KLCI | Yahoo `^KLSE` | `FTSE Bursa Malaysia KLCI (Yahoo).csv` |
| Philippines | PSEi Composite | Investing.com | `PSEi Composite Historical Data.csv` |
| Singapore | Straits Times Index | Yahoo `^STI` | `Straits Times Index (Yahoo).csv` |
| Thailand | SET Index | Investing.com | `SET Index Historical Data.csv` |
| India | Nifty 50 | Yahoo `^NSEI` | `Nifty 50 (Yahoo).csv` |

- **Coverage:** every market runs from 1 Jan 2012 to 23 Oct 2026. Thailand's last day is 22 Oct, because 23 Oct is a Thai holiday.
- **Two periods:**
  - **Development:** 1 Jan 2012 to 16 Mar 2023. Everything so far uses only this.
  - **Holdout:** 17 Mar 2023 to 23 Oct 2026. It stays untouched until the final check. `load_market()` leaves it out unless called with `include_holdout=True`.
- **Checks made when switching sources:**
  - The original files were all from Investing.com and ended in Mar 2023. Over 2012 to Mar 2023, Yahoo matches them: the up/down direction agrees on 99.8–99.9% of days.
  - Yahoo also fills Singapore's missing May–Dec 2020.
  - The new SET file is identical to the original one.
  - The PSEi replaces the FTSE Philippines index used before. It's a different index, with real Open/High/Low data. Its daily returns correlate 0.976 with the old file's.
- **India second-source check:** "same as yesterday" scores the same in both sources (56.9% in 2021, 59.3% in 2022), so the persistence isn't a data error.

**Getting the data.** The data files aren't in this repo, for licensing reasons.
1. Run `python download_data.py`. It downloads the four Yahoo markets up to a fixed end date, so every run gets the same data.
2. From Investing.com (free account), open the historical data pages for **PSEi Composite** and **SET Index**. Choose Daily, set 01/01/2012 to 10/23/2026, click Download, and save the files in `Dataset/` under the names above.

Step 1 (the EDA) was done on the original Investing.com files, and `eda.ipynb` still reads those.

## Step 2 findings: walk-forward baselines

Every market is tested on six separate years. For test year Y: train on every year up to Y−2, tune on Y−1, test on Y. The table shows the accuracy of the best no-model guess ("always up", "always down" or "same as yesterday"), picked by looking at the test year. No model could know that in advance, so this is a **stretch bar**. Step 3 adds a fair bar that's fixed before each test year starts.

| Stretch bar, accuracy % | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | Range |
|---|---|---|---|---|---|---|---|
| Indonesia | 55.9 | 51.2 | 53.9 | 52.7 | 53.0 | 52.0 | 4.6 |
| Malaysia | 51.7 | 55.8 | 53.7 | 51.0 | 50.6 | 53.9 | 5.2 |
| Philippines | 56.4 | 53.5 | 51.0 | 52.3 | 51.4 | 53.1 | 5.3 |
| Singapore | 51.2 | 52.2 | 50.8 | 53.6 | 54.5 | 54.8 | 4.0 |
| Thailand | 56.0 | 51.8 | 51.6 | 51.9 | 53.5 | 56.0 | 4.4 |
| India | 56.3 | 53.9 | 53.1 | 58.8 | 56.9 | 59.3 | 6.2 |

- **The bar moves 4–6.2 points between years** in every market. The guess that sets it changes too; "always up" sets it most often.
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
  - **Fair bar:** the direction that was more common in the years before the test year. It was "up" in every fold except Singapore 2021, where the training years had slightly more down days.
  - **Stretch bar:** the hindsight best guess from step 2.
- **Paired test:** the model and a bar are compared with McNemar's test, which only counts the days where exactly one of them is right.

All six test years pooled (sklearn):

| Market | Model accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday |
|---|---|---|---|---|---|---|---|
| Indonesia | 52.9 | 0.01 | 0.51 | 52.9 | 1.00 | 49.5 | 0.06 |
| Malaysia | 49.6 | −0.01 | 0.48 | 49.8 | 0.91 | 49.3 | 0.88 |
| Philippines | 51.4 | −0.04 | 0.51 | 51.8 | 0.15 | 48.4 | 0.11 |
| Singapore | 52.1 | 0.04 | 0.53 | 49.4 | 0.07 | 49.6 | 0.16 |
| Thailand | 52.1 | −0.06 | 0.50 | 52.4 | 0.06 | 51.0 | 0.57 |
| India | 53.8 | −0.01 | 0.54 | 54.6 | 0.28 | 54.8 | 0.59 |

- **No edge in the ASEAN markets.** Pooled accuracy matches the fair bar to within half a point everywhere except Singapore (+2.7 points, p = 0.07, not significant). AUC is 0.48–0.53, about what a coin flip scores (0.50).
- **The model often just says "up".** In 17 of 36 folds, tuning picked a penalty strong enough that the model calls every test day up. That's tuning's way of saying the features don't help. It happened in 5 of 6 folds for Thailand, and 4 of 6 for Malaysia and the Philippines.
- **No fold beats the fair bar significantly** (p < 0.05 in none of the 36). The model scores above the fair bar in 8 folds, below it in 11, and exactly matches it in 17.
- **India: logistic regression doesn't capture the direction persistence,** even though it's given yesterday's direction directly.
  - Pooled, it scores 53.8%, below both the fair bar (54.6%) and "same as yesterday" (54.8%).
  - In 2022 it's significantly worse than "same as yesterday": 49.2% vs 59.3%, p = 0.026. Of 117 days where they disagreed, the model was right on 46 and "same as yesterday" on 71.
  - A likely reason (not yet tested): the model learns one fixed weight from all years since 2012, when persistence was weaker, and tuning often picks heavy shrinkage.
  - **Read this with caution.** Pooled over six years, "same as yesterday" beats the fair bar by only 0.2 points (54.8% vs 54.6%), so the persistence is concentrated in recent years. India's test years have now been looked at several times, so a fix tailored to India risks fitting what's already been seen.
  - **Next:**
    - The second-source check is done; Yahoo agrees (see Data).
    - If a shorter training window is tried, run it on every market, not just India.
    - Treat "India's direction repeats" as an idea to confirm on the untouched holdout.
- **sklearn and PyTorch agree.**
  - With separately tuned penalties, predicted probabilities correlate at 0.993 or higher in all 36 folds, and up/down calls agree on 86.7–100% of days.
  - The lowest agreement is India 2022 (86.7%), Singapore 2018 (93.6%) and India 2020 (94.8%). The cause is that the two tuned penalties aren't equivalent: the grids only hold round numbers, so probabilities near 0.5 tip opposite ways.
  - Both versions leave the intercept unpenalised, so the intercept isn't the cause. In PyTorch the bias has its own parameter group with weight decay 0.
  - With the exact conversion, weight decay = 1 / (C × training days), PyTorch matches sklearn almost perfectly. Correlation is 0.99999 or higher, and the up/down calls agree on at least 99.6% of days in every fold.

## Files

| File | What it is |
|---|---|
| `download_data.py` | Downloads the four Yahoo markets into `Dataset/`, up to a fixed end date |
| `data.py` | `load_market()` / `load_all()`: load, clean and check the data, development period only unless asked. Every step after the EDA uses this |
| `evaluation.py` | Walk-forward splits (test years 2017–2022), scoring (accuracy, balanced accuracy, MCC), no-model baselines, fair baseline, McNemar paired test |
| `features.py` | Model features: the last 5 daily returns and yesterday's direction |
| `models.py` | Logistic regression in sklearn and PyTorch, the six-fold loop, per-year and pooled reports |
| `test_leakage.py` | Checks that nothing computed for day t changes when data after t is removed or scrambled |
| `walk_forward_baselines.ipynb` | Baselines scored on each test year |
| `logistic_regression.ipynb` | Step 3: both logistic regressions through the six folds, and the agreement check |
| `eda.ipynb` | EDA notebook on the original data: data checks, cleaning, returns, up-day shares, baseline scores |
| `eda_report.md` | Write-up of the EDA findings |
| `eda_output/` | Figures used in the report (saved by the EDA notebook) |

## Running

Tested with Python 3.13 and pandas 3.1.

```
pip install pandas numpy matplotlib scikit-learn torch yfinance jupyter
python download_data.py
python test_leakage.py
jupyter notebook logistic_regression.ipynb
```
