# asean-ml-forecasting

Forecasting the daily direction (up/down) of the ASEAN-5 stock indices with machine learning: Indonesia, Malaysia, the Philippines, Singapore and Thailand, with India's Nifty 50 as a benchmark.

**Status:** done so far:
- step 1: EDA on the original data, see [eda_report.md](eda_report.md)
- step 2: one cleaning function, walk-forward evaluation, leakage test
- step 3: logistic regression in sklearn and PyTorch
- data updated to Oct 2026, with an untouched holdout; steps 2–3 re-run on it
- step 4: the paper's 12 indicators and seven models, tuned and tested properly
- step 5: the US overnight move as a feature, close-to-close vs open-to-close
- step 5b: the two open-to-close exceptions checked; a draft holdout plan
- step 6: a small GRU in PyTorch on the same task
- step 7: predicting volatility: HAR, random forest, GRU, GARCH, and HAR with the US move
- step 8: the one-shot holdout, 17 Mar 2023 – 23 Oct 2026
- interpretability: does the GRU rediscover HAR?

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
  - **Holdout:** 17 Mar 2023 to 23 Oct 2026. It stays untouched until the final check. `load_market()` leaves it out unless called with `include_holdout=True`, and no code in the repo does that.
  - No model has used 2023 at all: the last test year is 2022.
- **US data (step 5):** S&P 500 from Yahoo `^GSPC`, 2010–2022 only.
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

## Step 4 findings: the paper's setup

**Setup:**
- **Features:** the 12 technical indicators from Kim (2003), as used by Kumar & Thenmozhi (2006): stochastic %K, %D and slow %D, momentum, rate of change, Williams %R, A/D oscillator, 5- and 10-day disparity, price oscillator, CCI and RSI. Look-back is 10 days (the papers don't say).
- **Models:** LDA, logistic regression, KNN, SVM, random forest and AdaBoost in sklearn, plus a neural network with one hidden layer in PyTorch.
- **Tuning:** each model tries a few settings and keeps the best on the tune year. The network picks its hidden nodes (2–12) and its number of epochs the same way, using early stopping.
- **Fixes the paper's mistake:** Kumar & Thenmozhi picked the SVM's settings and the network's hidden nodes by their score on the test period.
- **Skipped:** Kernel Factory, an R package with no standard Python version.
- **Data notes:**
  - The Philippines keeps all 12 indicators, because the PSEi file has a real High and Low. The old FTSE Philippines file had High = Low every day, so the A/D oscillator (which divides by High − Low) couldn't be computed.
  - Yahoo has 11 days with High = Low (Indonesia 5, Malaysia 6). The A/D oscillator is blank on those days, so the following day is left out.
- **Leakage test:** now also checks that the features for day t don't change when day t's own prices change. It passes, and it fails on a planted leak.

All 36 folds per model (6 markets × 6 test years):

| Model | Mean AUC | Folds above fair bar | Folds below | Significantly above | Significantly below | One-answer folds |
|---|---|---|---|---|---|---|
| LDA | 0.51 | 15 | 18 | 0 | 0 | 0 |
| Logistic regression | 0.52 | 11 | 7 | 0 | 0 | 16 |
| KNN | 0.51 | 17 | 14 | 2 | 1 | 4 |
| SVM | 0.51 | 12 | 10 | 0 | 0 | 7 |
| Random forest | 0.51 | 18 | 13 | 1 | 0 | 0 |
| AdaBoost | 0.52 | 16 | 18 | 0 | 0 | 2 |
| Neural network | 0.51 | 18 | 15 | 0 | 1 | 0 |

Pooled accuracy minus the fair bar, in points (the full table for each model is in `technical_indicators.ipynb`):

| Model | Indonesia | Malaysia | Philippines | Singapore | Thailand | India |
|---|---|---|---|---|---|---|
| LDA | 0.1 | −0.3 | 0.0 | 1.7 | −1.0 | −1.4 |
| Logistic regression | 0.2 | −0.3 | 0.3 | 1.0 | −0.1 | −0.1 |
| KNN | −0.1 | 1.8 | −1.2 | 3.3 | −0.1 | 0.3 |
| SVM | 0.0 | −0.1 | −0.5 | 0.7 | 0.0 | −0.1 |
| Random forest | −0.6 | −0.8 | 0.1 | 3.4 | 0.8 | −1.2 |
| AdaBoost | −1.5 | 0.8 | 0.3 | 1.5 | −1.6 | −0.1 |
| Neural network | −0.9 | 0.5 | −1.2 | −0.2 | 1.0 | −0.4 |

- **No model has an edge.** No pooled gap is significant; the smallest p-value is 0.06. Mean AUC is 0.51–0.52, where a coin flip scores 0.50.
- **The few significant folds look like luck.** Out of 252 model-folds, 3 beat the fair bar and 2 lost to it at p < 0.05. Testing that many folds, you'd expect a handful either way by chance.
- **Why Singapore looks best: its fair bar is weak, not its models strong.** Random forest is +3.4 points (p = 0.07) and KNN +3.3 (p = 0.06).
  - Most of the gain is in 2021 (+10 points), when the fair bar said "down" and the market rose on 54.5% of days. That "down" call was a near-tie: the training years had 1,120 up days and 1,122 down. With a few more early-2012 rows (as in step 5) it flips to "up".
  - Smaller gains come from 2018, a down year where "always up" scored badly.
  - MCC and AUC, which this can't fool, stay near a coin flip.
- **India:** the paper reported 56–68% on the Nifty for 2004–05. Here the same models score 53.1–54.8% for 2017–2022, while the fair bar scores 54.6% and "same as yesterday" 54.8%.
- **Giving one answer every day** (one-answer folds: all up, or all down): logistic regression did this in 16 of 36 folds, SVM in 7, KNN in 4 and AdaBoost in 2. LDA, random forest and the network never did. Only AdaBoost had an all-down year; the rest were all up.
- **AdaBoost doesn't win by saying "up".** It said up on every day in only 1 of 36 folds, and down on every day in 1 (both Thailand). Its accuracy was never significantly different from the fair bar (p ≥ 0.13), and its AUC ranged from 0.46 to 0.61. It simply has no edge.
- **AdaBoost tuning check:** with learning rates 0.1 and 1.0 only, 1.0 was picked in 33 of 36 folds, the top of the grid. With 2.0 and 4.0 added (and 500 estimators), 1.0 is picked 19 times, 2.0 9 times and 4.0 5 times. The scores barely move, and there's still no edge.
- **In one sentence:** once settings are no longer picked on the test period, and models are compared against a fair baseline, the paper's results disappear.

## Step 5 findings: the US overnight move

**Setup:**
- **Feature:** the S&P 500's daily return (Yahoo `^GSPC`, 2010–2022 only).
- **Timing:** the US session dated t ends after Asia has closed on day t. So Asian day t uses the last US session dated *before* t (Monday uses Friday). Joining on the same date would leak.
- **Leakage test:** removing or scrambling US data from day t onward doesn't change the feature for Asian days up to t. A same-date join fails the test, as it should.
- **Two targets:**
  - **Close-to-close**, as before. Not tradeable: the US move happens after Asia's previous close.
  - **Open-to-close.** Tradeable: see the US move, buy at the Asian open, sell at the close.
- **Models:** logistic regression and random forest, kept simple so the result is about the feature.
- **Fake opens:** on some days the open is just yesterday's close: Indonesia 2020 (45% of days) and 2022 (26%), Singapore 2012 (89%), the Philippines 2012–15 (about 6%). Those days are left out of the open-to-close target. Other years' opens are real.

How closely the US move lines up with each part of the Asian day (correlation):

| | Indonesia | Malaysia | Philippines | Singapore | Thailand | India |
|---|---|---|---|---|---|---|
| Close-to-close | 0.27 | 0.31 | 0.32 | 0.35 | 0.24 | 0.27 |
| Overnight gap (close → open) | 0.56 | 0.52 | 0.47 | 0.60 | 0.50 | 0.49 |
| Open-to-close | 0.06 | 0.05 | 0.12 | −0.03 | −0.09 | −0.03 |

Logistic regression, all six test years pooled:

| Market | Close-to-close % | Fair bar % | AUC | Open-to-close % | Fair bar % | p vs fair | AUC |
|---|---|---|---|---|---|---|---|
| Indonesia | 58.0 | 52.9 | 0.62 | 48.3 | 49.0 | 0.16 | 0.50 |
| Malaysia | 59.9 | 49.8 | 0.65 | 49.8 | 48.9 | 0.43 | 0.55 |
| Philippines | 60.5 | 51.8 | 0.64 | 55.9 | 50.2 | < 0.001 | 0.58 |
| Singapore | 61.4 | 50.9 | 0.66 | 49.4 | 49.8 | 0.42 | 0.48 |
| Thailand | 57.8 | 52.4 | 0.62 | 54.8 | 54.4 | 0.74 | 0.53 |
| India | 60.8 | 54.6 | 0.64 | 52.4 | 52.5 | 1.00 | 0.49 |

- **Close-to-close looks very predictable:** 58–61% accuracy and AUC 0.62–0.66. It beats the fair bar in every market (p < 0.001), and in 15 of 36 single years. But you couldn't trade it.
- **Open-to-close, the tradeable version, mostly loses it.** The open already absorbs the US move: it correlates about 0.5 with the overnight gap and about 0 with what follows. Four of six markets show no edge.
- **Two exceptions,** checked in step 5b below:
  - Philippines: 55.9% vs 50.2% (p < 0.001, AUC 0.58), with both models.
  - Malaysia: random forest only, 54.5% vs 48.9% (p = 0.002).
- Random forest gives the same picture (tables below).

**Full tables**, all six test years pooled. **One-answer folds** = test years (out of 6) where the model gave the same answer every day (all up, or all down).

**Close-to-close, logistic regression:**

| Market | Accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday | Called up % | One-answer folds |
|---|---|---|---|---|---|---|---|---|---|
| Indonesia | 58.0 | 0.15 | 0.62 | 52.9 | < 0.001 | 49.5 | < 0.001 | 76.8 | 0 |
| Malaysia | 59.9 | 0.21 | 0.65 | 49.8 | < 0.001 | 49.3 | < 0.001 | 62.1 | 0 |
| Philippines | 60.5 | 0.21 | 0.64 | 51.8 | < 0.001 | 48.4 | < 0.001 | 66.8 | 0 |
| Singapore | 61.4 | 0.23 | 0.66 | 50.9 | < 0.001 | 49.6 | < 0.001 | 53.7 | 0 |
| Thailand | 57.8 | 0.16 | 0.62 | 52.4 | < 0.001 | 51.0 | < 0.001 | 80.0 | 0 |
| India | 60.8 | 0.20 | 0.64 | 54.6 | < 0.001 | 54.8 | < 0.001 | 74.0 | 0 |

**Close-to-close, random forest:**

| Market | Accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday | Called up % | One-answer folds |
|---|---|---|---|---|---|---|---|---|---|
| Indonesia | 56.7 | 0.12 | 0.60 | 52.9 | 0.02 | 49.5 | < 0.001 | 63.3 | 0 |
| Malaysia | 61.0 | 0.22 | 0.64 | 49.8 | < 0.001 | 49.3 | < 0.001 | 54.7 | 0 |
| Philippines | 60.7 | 0.21 | 0.64 | 51.8 | < 0.001 | 48.4 | < 0.001 | 59.5 | 0 |
| Singapore | 61.4 | 0.23 | 0.65 | 50.9 | < 0.001 | 49.6 | < 0.001 | 48.2 | 0 |
| Thailand | 57.3 | 0.14 | 0.61 | 52.4 | 0.001 | 51.0 | < 0.001 | 68.8 | 0 |
| India | 60.6 | 0.20 | 0.63 | 54.6 | < 0.001 | 54.8 | 0.001 | 63.2 | 0 |

**Open-to-close, logistic regression:**

| Market | Accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday | Called up % | One-answer folds |
|---|---|---|---|---|---|---|---|---|---|
| Indonesia | 48.3 | −0.05 | 0.50 | 49.0 | 0.16 | 48.2 | 0.97 | 97.4 | 5 |
| Malaysia | 49.8 | 0.01 | 0.55 | 48.9 | 0.43 | 50.0 | 0.94 | 84.0 | 2 |
| Philippines | 55.9 | 0.13 | 0.58 | 50.2 | < 0.001 | 48.6 | < 0.001 | 70.3 | 1 |
| Singapore | 49.4 | −0.03 | 0.48 | 49.8 | 0.42 | 49.3 | 1.00 | 2.5 | 4 |
| Thailand | 54.8 | 0.05 | 0.53 | 54.4 | 0.74 | 48.0 | < 0.001 | 16.2 | 0 |
| India | 52.4 | −0.01 | 0.49 | 52.5 | 1.00 | 48.4 | 0.03 | 0.5 | 5 |

**Open-to-close, random forest:**

| Market | Accuracy % | MCC | AUC | Fair bar % | p vs fair | Same as yesterday % | p vs same as yesterday | Called up % | One-answer folds |
|---|---|---|---|---|---|---|---|---|---|
| Indonesia | 50.7 | 0.02 | 0.51 | 49.0 | 0.34 | 48.2 | 0.21 | 65.2 | 0 |
| Malaysia | 54.5 | 0.09 | 0.55 | 48.9 | 0.002 | 50.0 | 0.01 | 54.5 | 0 |
| Philippines | 56.0 | 0.12 | 0.57 | 50.2 | 0.001 | 48.6 | < 0.001 | 59.7 | 0 |
| Singapore | 49.2 | −0.02 | 0.49 | 49.8 | 0.75 | 49.3 | 0.97 | 41.7 | 1 |
| Thailand | 52.6 | 0.03 | 0.51 | 54.4 | 0.31 | 48.0 | 0.02 | 38.1 | 0 |
| India | 50.3 | −0.03 | 0.48 | 52.5 | 0.07 | 48.4 | 0.32 | 20.6 | 3 |

## Step 5b findings: the two exceptions

**The question:** do the Philippines and Malaysia open-to-close results come from something you could trade, or from how the index's opening number is recorded?

**The likely explanation:** the index's opening value only partly reflects the overnight US news. Some stocks haven't traded yet when it's recorded, so they still carry yesterday's price, and the rest of the news shows up after the open. The Philippines fits this: it has the weakest link between the US move and its open (0.47) and the strongest link after the open (0.12). The fake-open check can't catch this, because the open is only partly stale.

1. **Not a few huge days.** With 2020 removed from training, tuning and testing, the edge gets slightly *stronger*: Philippines 56.5% vs 50.1% (p < 0.001), Malaysia (random forest) 55.0% vs 49.1% (p = 0.003). The open-to-close correlations rise too (Philippines 0.12 → 0.19, Malaysia 0.05 → 0.11). Thailand's −0.09 barely moves (−0.08), so 2020 doesn't explain that one either.
2. **It's the US news arriving after the open.** With the market's own past returns instead of the US move, the edge disappears: 48.7–51.3% against fair bars of 48.9–50.2%.
3. **No tradeable price to test.** The Philippines' FMETF fund isn't on Yahoo. Malaysia's KLCI ETF (`0820EA.KL`) had no trades on 41% of days, and its open equals its close on 82%. I found no KLCI futures on Yahoo: its search returned nothing relevant, and the tickers I tried (`FKLI=F`, `FKLI.KL`, `KLCI=F`, `FBMKLCI=F`) returned no data.
4. **Money test, on the index:** buy at the open and sell at the close on days the model says up; otherwise stay out. Total over the six test years, at different round-trip trading costs:

| | Days in | Average day in % | No cost | 0.1% cost | 0.25% cost |
|---|---|---|---|---|---|
| Philippines, logistic regression | 1,027 of 1,461 | 0.09 | +128% | −18% | −83% |
| Philippines, random forest | 872 of 1,461 | 0.12 | +163% | +10% | −70% |
| Philippines, buy every day | 1,461 | | +3% | −76% | −97% |
| Malaysia, random forest | 796 of 1,460 | 0.05 | +43% | −35% | −80% |
| Malaysia, buy every day | 1,460 | | −11% | −79% | −98% |

**Average day in %** is the break-even cost: any round-trip cost above it wipes out the edge. For scale, the Philippine stock transaction tax on shares is 0.6% of each sale ([Inquirer](https://business.inquirer.net/?p=246692)). I couldn't confirm how it applies to ETFs.

**Verdict:** the edge comes from US news showing up in the index after its opening number is recorded, as you'd expect if the open is partly stale. Without a tradeable price we can't prove that. But it doesn't change the answer: even taken at face value, the edge is 0.05–0.12% per trade, less than realistic trading costs. It's not something you could trade.

## Step 6 findings: does a deep model change anything?

**Setup:** a small GRU in PyTorch reads the last 20 days in order, oldest first. Hidden size (8, 16 or 32) and epochs are picked on the tune year with early stopping. It's compared with logistic regression on the same days.

**Own returns only, close-to-close** (logistic regression gets the same 20 returns):

| Market | GRU % | Logistic regression % | Fair bar % | p GRU vs fair | GRU AUC |
|---|---|---|---|---|---|
| Indonesia | 53.6 | 52.7 | 52.9 | 0.24 | 0.52 |
| Malaysia | 47.9 | 49.6 | 49.8 | 0.23 | 0.49 |
| Philippines | 50.4 | 51.6 | 51.8 | 0.35 | 0.49 |
| Singapore | 46.8 | 50.6 | 49.4 | 0.16 | 0.47 |
| Thailand | 51.7 | 52.4 | 52.4 | 0.35 | 0.52 |
| India | 54.5 | 53.4 | 54.6 | 1.00 | 0.54 |

**With US data** (the GRU reads 20 days of own and US returns; logistic regression gets only the last US move, as in step 5):

| Market | Close-to-close: GRU % | Logistic regression % | p GRU vs LR | Open-to-close: GRU % | Logistic regression % | p GRU vs LR |
|---|---|---|---|---|---|---|
| Indonesia | 59.4 | 58.0 | 0.14 | 53.1 | 48.3 | 0.007 |
| Malaysia | 60.8 | 59.9 | 0.32 | 51.8 | 49.8 | 0.14 |
| Philippines | 60.5 | 60.5 | 1.00 | 55.7 | 55.9 | 0.90 |
| Singapore | 61.2 | 61.4 | 0.82 | 47.0 | 49.4 | 0.17 |
| Thailand | 58.1 | 57.8 | 0.83 | 52.8 | 54.8 | 0.21 |
| India | 60.2 | 60.8 | 0.66 | 53.5 | 52.4 | 0.39 |

**Random seeds, open-to-close.** The same GRU with four different starting points. **Average day in %** is what a trade earns on the index before costs:

| Market | GRU % | Fair bar % | Seeds with p < 0.05 | Average day in % |
|---|---|---|---|---|
| Indonesia | 50.1–53.1 | 49.0 | 2 of 4 | −0.04 to 0.00 |
| Malaysia | 49.7–51.8 | 48.9 | 1 of 4 | 0.00 to 0.03 |
| Philippines | 55.1–56.3 | 50.2 | 4 of 4 | 0.11 to 0.13 |

- **No, the deep model doesn't change the answer.** With own returns only there's still no edge: mean AUC is 0.50.
- **Where there is signal, one number gets almost all of it.** On close-to-close, the GRU reads 40 inputs and doesn't significantly beat logistic regression on the last US move alone.
- **Indonesia's open-to-close win over logistic regression isn't reliable.** It's significant for only 2 of 4 seeds, and those trades earn about nothing. Accuracy can rise without earning money.
- **The Philippines holds up across all seeds,** but it's the same edge as step 5: logistic regression matches it, and it earns about 0.12% per trade, which is below realistic costs.
- **Random seeds matter.** One market's accuracy moved by 3 points from the starting point alone, so a single run of a neural network can mislead.

## Step 7 findings: volatility

**Setup:** predict how big day t's move will be, not its direction. The target is the log of log(High / Low). Same six folds.

- **HAR:** linear regression on the average range over the last 1, 5 and 22 days.
- **Random forest:** the same three inputs.
- **GRU:** the last 20 daily ranges, three random seeds.
- **R²:** 1 is perfect, and 0 is no better than guessing the average of all earlier days.

**Is there anything to learn?** Correlation with one day earlier:

| Market | Direction | Range |
|---|---|---|
| Indonesia | −0.01 | 0.51 |
| Malaysia | 0.03 | 0.48 |
| Philippines | 0.02 | 0.40 |
| Singapore | −0.01 | 0.45 |
| Thailand | 0.02 | 0.54 |
| India | 0.08 | 0.44 |

**All six test years pooled** (R²; GRU shows the range over three seeds):

| Market | Same as yesterday | HAR | Forest | GRU |
|---|---|---|---|---|
| Indonesia | 0.167 | **0.412** | 0.356 | 0.395–0.401 |
| Malaysia | 0.136 | **0.430** | 0.389 | 0.339–0.356 |
| Philippines | −0.099 | **0.274** | 0.191 | 0.233–0.256 |
| Singapore | −0.061 | **0.307** | 0.199 | 0.169–0.244 |
| Thailand | 0.242 | **0.486** | 0.395 | 0.419–0.443 |
| India | 0.137 | **0.408** | 0.311 | 0.354–0.393 |

**HAR's R² by year:**

| Market | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
|---|---|---|---|---|---|---|
| Indonesia | 0.46 | 0.21 | 0.20 | 0.63 | 0.32 | 0.24 |
| Malaysia | 0.39 | 0.25 | 0.03 | 0.69 | 0.32 | 0.32 |
| Philippines | 0.15 | 0.27 | 0.13 | 0.51 | 0.16 | 0.15 |
| Singapore | 0.12 | 0.16 | 0.07 | 0.62 | 0.05 | 0.10 |
| Thailand | 0.70 | 0.06 | 0.24 | 0.68 | 0.19 | 0.20 |
| India | 0.53 | 0.27 | 0.12 | 0.61 | 0.25 | 0.23 |

- **Volatility can be predicted; direction couldn't.** HAR beats both the average so far and "same as yesterday" in every market (p < 0.001 against "same as yesterday").
- **The simplest model wins.** HAR has the top R² in all six markets. The forest is significantly worse in all six (p ≤ 0.007) and beats HAR in only 1 of 36 market-years. The GRU is worse in all 18 market-and-seed runs, significantly in 15, and beats HAR in 11–14 of 36 market-years.
- **The pooled R² is flattered by unusual years.** 2020 is the best year in five of the six markets (0.51–0.69, Thailand's 0.68 just behind its 0.70 in 2017), but most other market-years are 0.05–0.32. HAR isn't much better in 2020; the average so far is just much worse. HAR's typical error is 0.46 that year against 0.39–0.42 in the others, while the average's is 0.76 against 0.42–0.53.
- **Weeks and months count more than yesterday.** HAR's weight on the 1-day average is 0.13–0.27; the 5- and 22-day averages get 0.27–0.40.
- **Indonesia's fake opens don't change the picture outside 2020.** Leaving them out moves R² only from 0.299 to 0.303 in the other five years. Across all years it falls from 0.412 to 0.320, because 2020's fake-open days score 0.734 against 0.418 for its real opens. So Indonesia's headline number leans on those 108 days, and 0.30 is the fairer figure.
- **This predicts the size of a move, not which way it goes,** so it isn't a trading signal on its own.

### Two more models and a second score

- **GARCH(1,1):** the classic volatility model, refitted before each test year. It forecasts the size of returns, so a straight line fitted on earlier years turns that into a range forecast.
- **HAR + US:** HAR plus the log size of the last US session's move, which is known before Asia opens.
- **QLIKE:** how far off a forecast is in relative terms, like "2× too high". Lower is better.

**R², all six years** (p is the paired test against HAR on squared errors):

| Market | HAR | GARCH | p | HAR + US | p | US weight |
|---|---|---|---|---|---|---|
| Indonesia | 0.412 | 0.314 | <0.001 | **0.420** | 0.018 | 0.046 |
| Malaysia | 0.430 | 0.386 | 0.004 | **0.434** | 0.004 | 0.030 |
| Philippines | 0.274 | 0.237 | 0.006 | **0.280** | 0.36 | 0.061 |
| Singapore | 0.307 | 0.294 | 0.53 | **0.325** | 0.001 | 0.053 |
| Thailand | 0.486 | 0.448 | 0.004 | **0.488** | 0.071 | 0.019 |
| India | **0.408** | 0.365 | 0.005 | **0.408** | 0.72 | 0.013 |

**R² without 2020** (GRU shows the range over three seeds):

| Market | HAR | HAR + US | GARCH | Forest | GRU |
|---|---|---|---|---|---|
| Indonesia | 0.299 | **0.303** | 0.175 | 0.261 | 0.273–0.293 |
| Malaysia | 0.278 | **0.282** | 0.232 | 0.243 | 0.265–0.277 |
| Philippines | 0.169 | **0.172** | 0.108 | 0.131 | 0.159–0.161 |
| Singapore | 0.098 | **0.115** | 0.091 | 0.067 | 0.059–0.081 |
| Thailand | 0.387 | **0.388** | 0.333 | 0.295 | 0.325–0.340 |
| India | **0.311** | **0.311** | 0.249 | 0.252 | 0.289–0.304 |

**QLIKE, all six years:**

| Market | HAR | HAR + US | GARCH | Forest | GRU |
|---|---|---|---|---|---|
| Indonesia | 0.477 | **0.463** | 0.644 | 0.551 | 0.489–0.501 |
| Malaysia | 0.575 | **0.568** | 0.615 | 0.650 | 0.741–0.771 |
| Philippines | 0.629 | **0.609** | 0.673 | 0.865 | 0.678–0.761 |
| Singapore | 0.378 | **0.354** | 0.362 | 0.521 | 0.452–0.560 |
| Thailand | 0.366 | **0.360** | 0.377 | 0.458 | 0.390–0.463 |
| India | 0.507 | 0.504 | 0.491 | 0.655 | **0.472**–0.583 |

- **HAR's win holds up without 2020.** Its R² still beats GARCH, the forest and every GRU seed in all six markets.
- **The US move adds a little.** HAR + US is never worse on R² and is best on QLIKE in five of six markets, but the R² gain is at most 0.018. Doubling the US move raises the predicted range by only about 1–4%. It's significant on squared errors in Indonesia, Malaysia and Singapore.
- **GARCH loses on R², but it's close on QLIKE.** It's worse on R² in all six markets, significantly in five. On QLIKE it beats HAR in Singapore and India, and only Indonesia's gap (HAR better) is significant.
- **India is the one QLIKE exception.** Two GRU seeds and GARCH beat HAR there, but none significantly, and the third seed is worse.
- **QLIKE isn't free of 2020 either.** It punishes forecasts that are too low much harder than ones that are too high, so 2020 costs some models a lot: the Philippines forest goes from 0.572 without 2020 to 0.865 with it.
- **R² here was already on the log of the range,** so a wild day never got extra weight for its size.

## Holdout plan (committed in `87aa8d5` before opening the holdout)

Each test runs once, on the code and settings as committed before opening. Same walk-forward rules as development: for test year Y, train up to Y−2, tune on Y−1, test on Y. Test years are 2023 (from 17 Mar), 2024, 2025 and 2026 (to 23 Oct). Nothing changes after looking, and every result gets reported, including anything that flips.

Direction tests are scored against the fair bar (the majority direction before each test year) with McNemar's test and AUC.

1. **The paper's setup:** all seven models on the 12 indicators, close-to-close, all six markets. Expect no edge.
2. **Close-to-close with the US feature:** logistic regression, all six markets. Expect it to beat the fair bar, though it can't be traded.
3. **Open-to-close with the US feature:** logistic regression for the Philippines, random forest for Malaysia, plus the money test at 0.1% and 0.25% costs. Expect any edge to stay below trading costs. These assume you could buy at the index's opening number, which may not be a real price.
4. **India, "same as yesterday" vs the fair bar:** close-to-close. Checks whether India's direction persistence is real.
5. **Volatility, all six markets:** HAR vs "same as yesterday", GARCH, the GRU (seeds 0, 1 and 2) and HAR + US. Each is scored on R² and QLIKE, with the paired test on both. Expect HAR to beat the first three on R², and HAR + US to add a little.

The US data for 2023–2026 is downloaded only when these tests run.

## Step 8 findings: the holdout

Each test ran once, as planned. After the plan was committed, the only code changes were to let the loops take other test years and to load the new US file. The development results still reproduced exactly afterwards. One check was added before looking: Indonesia's volatility without its fake-open days.

| Test | Expected | Result |
|---|---|---|
| 1. The paper's setup | No edge | **As expected:** no edge |
| 2. Close-to-close with the US move | Beats the fair bar | **Mostly:** above it in all six markets, significant in four, weaker than in development |
| 3. Open-to-close | Edge below costs | **As expected, and worse:** the Philippines trades lose money even before costs; Malaysia's edge is gone |
| 4. India, same as yesterday | Is it real? | **No** |
| 5. HAR vs same as yesterday and GARCH | HAR wins | **As expected** |
| 5. HAR vs the GRU | HAR wins | **Only just:** about a tie outside Thailand |
| 5. HAR + US vs HAR | Adds a little | **Not as expected:** no significant gain, and significantly worse in Indonesia |

### Test 1: the paper's setup

Over 24 market-years per model:

| Model | Mean AUC | Above fair bar | Below fair bar | Significantly above | Significantly below |
|---|---|---|---|---|---|
| LDA | 0.510 | 7 | 16 | 0 | 3 |
| Logistic regression | 0.515 | 5 | 11 | 0 | 1 |
| KNN | 0.510 | 9 | 14 | 0 | 1 |
| SVM | 0.509 | 6 | 10 | 0 | 1 |
| Random forest | 0.513 | 10 | 11 | 1 | 1 |
| AdaBoost | 0.516 | 12 | 11 | 0 | 1 |
| Neural network | 0.509 | 8 | 15 | 0 | 0 |

Accuracy minus the fair bar, in points, all test years pooled:

| Model | Indonesia | Malaysia | Philippines | Singapore | Thailand | India |
|---|---|---|---|---|---|---|
| LDA | −1.8 | 1.4 | 1.4 | −3.6 | −1.6 | −1.6 |
| Logistic regression | −0.8 | 2.0 | 1.5 | −3.6 | −0.8 | 0.2 |
| KNN | −0.5 | 0.7 | 0.2 | −2.6 | −0.9 | −0.1 |
| SVM | 0.0 | −0.5 | 0.3 | −3.1 | 0.2 | 0.3 |
| Random forest | −1.3 | 3.2 | 2.4 | −4.2 | −2.6 | 1.8 |
| AdaBoost | −4.7 | 1.8 | 0.7 | −2.4 | −1.1 | −0.2 |
| Neural network | −0.7 | 0.9 | 1.4 | −2.5 | −0.1 | 0.6 |

No model beats the fair bar significantly in any market over the whole holdout, and two fall significantly below it (AdaBoost in Indonesia, the random forest in Singapore).

### Test 2: close-to-close with the US move

| Market | Holdout % | Fair bar % | p vs fair | AUC | Development % | Development AUC |
|---|---|---|---|---|---|---|
| Indonesia | 56.7 | 52.2 | 0.008 | 0.58 | 58.0 | 0.62 |
| Malaysia | 57.7 | 50.3 | 0.001 | 0.62 | 59.9 | 0.65 |
| Philippines | 54.9 | 50.3 | 0.021 | 0.59 | 60.5 | 0.64 |
| Singapore | 57.7 | 54.4 | 0.14 | 0.62 | 61.4 | 0.66 |
| Thailand | 53.1 | 50.5 | 0.13 | 0.56 | 57.8 | 0.62 |
| India | 57.0 | 53.7 | 0.049 | 0.59 | 60.8 | 0.64 |

The link with the US move is real and it holds up on new data, but every market is weaker than in development. It still can't be traded.

### Test 3: open-to-close with the US move

| | Holdout % | Fair bar % | p vs fair | AUC | Average day in % | Development % | Development average day in % |
|---|---|---|---|---|---|---|---|
| Philippines, logistic regression | 53.4 | 49.0 | 0.044 | 0.56 | −0.03 | 55.9 | 0.09 |
| Malaysia, random forest | 51.6 | 50.1 | 0.53 | 0.54 | 0.03 | 54.5 | 0.05 |

Total return over the holdout:

| | Days in | 0.1% cost | 0.25% cost |
|---|---|---|---|
| Philippines, model | 503 of 876 | −48% | −75% |
| Philippines, buy every day | 876 | −83% | −95% |
| Malaysia, model | 459 of 883 | −28% | −64% |
| Malaysia, buy every day | 883 | −62% | −90% |

- **The Philippines is more accurate than the bar but makes no money.** Its 53.4% only just clears the significance line, and the days it calls up lose 0.03% on average before any costs. Accuracy doesn't guarantee profit: a few big moves can outweigh many small right calls.
- **Malaysia's edge is gone.**
- **The losses are smaller than buying every day mostly because the models trade less often.** Neither one makes money at either cost.

### Test 4: India, same as yesterday

Same as yesterday is right 52.9% of the time and the fair bar 53.7% (p = 0.77, over 886 days). India's direction persistence doesn't hold up.

### Test 5: volatility

**R²** (development HAR for comparison; GRU shows the range over three seeds):

| Market | Development HAR | Same as yesterday | HAR | HAR + US | GARCH | GRU |
|---|---|---|---|---|---|---|
| Indonesia | 0.412 | 0.222 | 0.471 | 0.462 | 0.367 | 0.462–0.474 |
| Malaysia | 0.430 | −0.345 | 0.115 | 0.113 | 0.040 | 0.094–0.116 |
| Philippines | 0.274 | −0.590 | 0.058 | 0.063 | 0.029 | 0.053–0.060 |
| Singapore | 0.307 | −0.251 | 0.190 | 0.195 | 0.137 | 0.170–0.184 |
| Thailand | 0.486 | −0.288 | 0.244 | 0.244 | 0.191 | 0.190–0.213 |
| India | 0.408 | −0.109 | 0.302 | 0.301 | 0.228 | 0.285–0.300 |

**p vs HAR on squared errors:**

| Market | Same as yesterday | HAR + US | GARCH | GRU seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|---|---|---|
| Indonesia | < 0.001 | 0.029 | < 0.001 | 0.37 | 0.66 | 0.57 |
| Malaysia | < 0.001 | 0.75 | 0.022 | 0.17 | 0.12 | 0.90 |
| Philippines | < 0.001 | 0.60 | 0.12 | 0.66 | 0.73 | 0.90 |
| Singapore | < 0.001 | 0.52 | 0.044 | 0.15 | 0.66 | 0.61 |
| Thailand | < 0.001 | 1.00 | 0.017 | 0.008 | 0.016 | 0.078 |
| India | < 0.001 | 0.64 | 0.014 | 0.062 | 0.71 | 0.31 |

**QLIKE** (lower is better):

| Market | Same as yesterday | HAR | HAR + US | GARCH | GRU |
|---|---|---|---|---|---|
| Indonesia | 0.626 | 0.487 | 0.501 | 0.670 | **0.483**–0.509 |
| Malaysia | 0.715 | 0.460 | **0.455** | 0.498 | 0.468–0.504 |
| Philippines | 0.802 | 0.420 | **0.416** | 0.464 | 0.430–0.436 |
| Singapore | 0.493 | 0.327 | 0.317 | 0.351 | **0.314**–0.320 |
| Thailand | 0.557 | **0.321** | 0.322 | 0.366 | 0.357–0.373 |
| India | 0.840 | 0.489 | 0.489 | **0.437** | 0.485–0.489 |

**HAR's R² by year:**

| Market | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| Indonesia | 0.22 | 0.12 | 0.34 | 0.67 |
| Malaysia | 0.18 | 0.07 | 0.13 | 0.09 |
| Philippines | 0.06 | −0.00 | 0.14 | 0.02 |
| Singapore | −0.04 | 0.14 | 0.31 | 0.18 |
| Thailand | 0.02 | −0.02 | 0.42 | 0.33 |
| India | 0.55 | 0.08 | 0.31 | 0.29 |

- **Volatility is still predictable, but less so.** HAR beats same as yesterday everywhere (p ≤ 0.001 on both scores). Its R² falls in five of six markets, though, and sharply in Malaysia (0.430 to 0.115) and the Philippines (0.274 to 0.058, close to no skill against the average so far).
- **Indonesia is the exception, and it leans on one year.** Its 0.471 is pulled up by 0.67 in 2026; the other years are 0.12–0.34. Fake opens aren't the cause here: without those 107 days it's 0.484.
- **HAR still beats GARCH.** It has the higher R² in all six markets, significantly in five. On QLIKE it wins in five of six, significantly in four, with India going the other way (p = 0.21).
- **The GRU is now about tied with HAR.** In development it was significantly worse in 15 of 18 runs; here it's 2 of 18, both in Thailand. Three GRU runs have a slightly higher R², and none is significant. HAR is far simpler for the same accuracy, but "HAR beats deep models" is too strong: "a GRU doesn't beat HAR" is what this supports.
- **The US move didn't hold up.** HAR + US gained significantly in three markets in development, but here it gains nowhere and is significantly worse in Indonesia. The development gain wasn't reliable.

### What survives

- **Direction from a market's own history:** no edge in development or in the holdout.
- **The US link:** real but not tradeable, because the open already absorbs it.
- **Tradeable direction:** nothing that makes money after costs.
- **Volatility:** predictable, with HAR hard to beat, but less so than the development years suggested.

## Interpretability: does the GRU rediscover HAR?

**Setup:** the step 7 GRUs, refitted with their saved settings, on the development test years 2017–2022 only. All 18 reproduce their saved forecasts. HAR's averages cover 1, 5 and 22 days, but the GRU reads only 20 days, so this uses 1, 5 and 20. Every line is fitted on earlier years and scored on the test year, except the weights table, which describes the test years. Results are averaged over three seeds. These are comparisons of scores, with no significance tests.

**1. Does the memory hold the averages?** R² of a straight line from the GRU's memory to each average. "Random" is the same GRU before any training:

| Market | Trained: 1 day | 5 days | 20 days | Random: 1 day | 5 days | 20 days |
|---|---|---|---|---|---|---|
| Indonesia | 0.991 | 0.978 | 0.902 | 0.995 | 0.984 | 0.889 |
| Malaysia | 0.986 | 0.976 | 0.920 | 0.994 | 0.980 | 0.895 |
| Philippines | 0.988 | 0.963 | 0.852 | 0.995 | 0.971 | 0.811 |
| Singapore | 0.965 | 0.952 | 0.782 | 0.994 | 0.967 | 0.832 |
| Thailand | 0.989 | 0.955 | 0.869 | 0.994 | 0.987 | 0.908 |
| India | 0.938 | 0.941 | 0.905 | 0.994 | 0.966 | 0.863 |

**2. Is the forecast a straight line on the averages?** Share of the GRU's forecast that the closest straight line matches (1 = all of it), by seed:

| Market | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| Indonesia | 0.969 | 0.964 | 0.969 |
| Malaysia | 0.972 | 0.964 | 0.949 |
| Philippines | 0.903 | 0.931 | 0.938 |
| Singapore | 0.853 | 0.856 | 0.896 |
| Thailand | 0.899 | 0.837 | 0.860 |
| India | 0.927 | 0.870 | 0.931 |

The weights on each average, for the GRU's forecasts and for a line fitted to the real target:

| Market | GRU: 1 day | 5 days | 20 days | Real target: 1 day | 5 days | 20 days |
|---|---|---|---|---|---|---|
| Indonesia | 0.20 | 0.35 | 0.19 | 0.28 | 0.33 | 0.25 |
| Malaysia | 0.12 | 0.36 | 0.14 | 0.20 | 0.24 | 0.45 |
| Philippines | 0.17 | 0.32 | 0.15 | 0.21 | 0.32 | 0.23 |
| Singapore | 0.07 | 0.31 | 0.12 | 0.18 | 0.36 | 0.28 |
| Thailand | 0.14 | 0.37 | 0.11 | 0.22 | 0.37 | 0.31 |
| India | 0.08 | 0.36 | 0.27 | 0.21 | 0.37 | 0.29 |

**3. Does training help?** Forecast R² against the average so far, as in step 7. "Imitation" keeps only the straight-line part of the GRU's forecast. "Untrained GRU + line" keeps random memory and fits only the last line to the target:

| Market | HAR | GRU | Imitation | Untrained GRU + line |
|---|---|---|---|---|
| Indonesia | **0.412** | 0.399 | 0.400 | 0.408 |
| Malaysia | **0.430** | 0.349 | 0.368 | 0.427 |
| Philippines | **0.274** | 0.245 | 0.265 | 0.262 |
| Singapore | **0.307** | 0.217 | 0.263 | 0.303 |
| Thailand | **0.486** | 0.433 | 0.478 | 0.443 |
| India | 0.408 | 0.379 | **0.409** | 0.363 |

**4. If we erase an average, does the forecast get worse?** Take out of the memory whatever moves in step with an average, then forecast from what's left. The control erases a single old day, 15 days ago. A fresh line can no longer read an erased average back (R² at or below 0):

| Market | Nothing erased | 1 day | 5 days | 20 days | All three | Control |
|---|---|---|---|---|---|---|
| Indonesia | 0.399 | 0.124 | 0.050 | 0.094 | −0.007 | 0.347 |
| Malaysia | 0.349 | 0.118 | −0.005 | 0.013 | −0.039 | 0.277 |
| Philippines | 0.245 | 0.062 | −0.014 | 0.038 | −0.047 | 0.208 |
| Singapore | 0.217 | −0.003 | −0.088 | −0.016 | −0.091 | 0.141 |
| Thailand | 0.433 | 0.048 | −0.114 | −0.071 | −0.173 | 0.319 |
| India | 0.379 | 0.147 | −0.040 | −0.019 | −0.096 | 0.311 |

- **In what it forecasts, mostly yes.** A straight line on the three averages matches 84–97% of the GRU's forecast, and erasing all three wipes out its skill in every market.
- **The averages in its memory aren't something it learned.** An untrained GRU holds them about as well, and usually slightly better. The probes alone would have suggested a discovery that isn't there.
- **Training adds little.** Random memory with only the last line fitted beats the fully trained GRU in five of six markets, all but India.
- **What the GRU adds beyond the straight line doesn't help.** Throwing it away improves the forecast in all six markets (barely in Indonesia). This is the likeliest reason it trails HAR in development.
- **It leans on the 5-day average.** That gets its largest weight in all six markets, and erasing it does the most damage in all six. It uses less of the 1- and 20-day averages than the real target does.
- **The windows can't be fully separated.** The three averages move together, so erasing one also removes much of what they share. The control's smaller drop shows that erasing isn't simply breaking the network, but it also falls, because old days carry some of that shared level too.

## Files

| File | What it is |
|---|---|
| `download_data.py` | Downloads the four Yahoo markets and the S&P 500 into `Dataset/`, up to fixed end dates |
| `data.py` | `load_market()` / `load_all()`: load, clean and check the data, development period only unless asked. `load_us()`: the S&P 500. Every step after the EDA uses this |
| `evaluation.py` | Walk-forward splits (test years 2017–2022), scoring (accuracy, balanced accuracy, MCC), no-model baselines, fair baseline, McNemar and Diebold-Mariano paired tests, money test |
| `features.py` | Model features: the last 5 daily returns and yesterday's direction (step 3), the 12 technical indicators (step 4), the US move (step 5), return sequences (step 6), daily ranges, GARCH and the size of the US move (step 7); the two labels (close-to-close, open-to-close) |
| `models.py` | Logistic regression in sklearn and PyTorch, the six-fold loop, per-year and pooled reports |
| `classifiers.py` | Step 4 models: the six sklearn classifiers, the PyTorch network, summary tables |
| `test_leakage.py` | Checks that nothing computed for day t changes when data after t is removed or scrambled, that features for day t don't use day t itself, and that the US feature doesn't use the US session dated t |
| `walk_forward_baselines.ipynb` | Baselines scored on each test year |
| `logistic_regression.ipynb` | Step 3: both logistic regressions through the six folds, and the agreement check |
| `technical_indicators.ipynb` | Step 4: all seven models on the 12 indicators, a table per model, and the AdaBoost check |
| `gru.py` | Step 6: the GRU network and its tuning |
| `sequence_model.ipynb` | Step 6: GRU vs logistic regression on three tasks, and the random-seed check |
| `volatility.py` | Step 7: the six-fold loop for predicting a number, the GRU for ranges, GARCH, R², QLIKE and the paired test |
| `volatility.ipynb` | Step 7: HAR vs forest, GRU and GARCH, HAR + US, year by year, scores without 2020, HAR's weights, and the Indonesia fake-open check |
| `holdout.ipynb` | Step 8: all five one-shot tests on the holdout |
| `gru_probes.py` | Interpretability: reads the GRU's memory, compares it with an untrained GRU, and erases averages from it |
| `gru_probes.ipynb` | Interpretability: the four questions about whether the GRU rediscovers HAR. Needs `volatility.ipynb`'s saved results |
| `results/` | Saved predictions, so a rerun skips finished work. Not in git. Delete the step 7 files after changing a model. Keep the `holdout_*` files: they're the one-shot results |
| `us_overnight.ipynb` | Steps 5 and 5b: the US move as a feature, close-to-close vs open-to-close, the open-price check, and the checks of the two exceptions |
| `eda.ipynb` | EDA notebook on the original data: data checks, cleaning, returns, up-day shares, baseline scores |
| `eda_report.md` | Write-up of the EDA findings |
| `eda_output/` | Figures used in the report (saved by the EDA notebook) |

## Running

Tested with Python 3.13 and pandas 3.1.

```
pip install pandas numpy matplotlib scikit-learn torch yfinance jupyter arch
python download_data.py
python test_leakage.py
jupyter notebook logistic_regression.ipynb
jupyter notebook technical_indicators.ipynb   # about 25 minutes
jupyter notebook us_overnight.ipynb
jupyter notebook sequence_model.ipynb         # about 35 minutes
jupyter notebook volatility.ipynb             # about 20 minutes
jupyter notebook holdout.ipynb                # about 50 minutes on a first run
jupyter notebook gru_probes.ipynb             # under 10 minutes
```
