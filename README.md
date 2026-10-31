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

## Holdout plan (draft: confirm before opening the holdout)

Each test runs once on 17 Mar 2023 – 23 Oct 2026. Same procedure as the folds: pick settings with 2022 as the tune year, retrain on all development data, then test. Each is scored against the fair bar (the majority direction in the development data) with McNemar's test and AUC.

1. **Close-to-close with the US feature**, logistic regression, all six markets. Expect it to beat the fair bar.
2. **Open-to-close with the US feature:** logistic regression for the Philippines, random forest for Malaysia, plus the money test at 0.1% and 0.25% costs. Expect any edge to stay below trading costs.
3. **India, "same as yesterday" vs the fair bar**, close-to-close. Checks whether India's direction persistence is real.

The US data for 2023–2026 is downloaded only when these tests run.

## Files

| File | What it is |
|---|---|
| `download_data.py` | Downloads the four Yahoo markets and the S&P 500 into `Dataset/`, up to fixed end dates |
| `data.py` | `load_market()` / `load_all()`: load, clean and check the data, development period only unless asked. `load_us()`: the S&P 500. Every step after the EDA uses this |
| `evaluation.py` | Walk-forward splits (test years 2017–2022), scoring (accuracy, balanced accuracy, MCC), no-model baselines, fair baseline, McNemar paired test, money test |
| `features.py` | Model features: the last 5 daily returns and yesterday's direction (step 3), the 12 technical indicators (step 4), the US move (step 5); the two labels (close-to-close, open-to-close) |
| `models.py` | Logistic regression in sklearn and PyTorch, the six-fold loop, per-year and pooled reports |
| `classifiers.py` | Step 4 models: the six sklearn classifiers, the PyTorch network, summary tables |
| `test_leakage.py` | Checks that nothing computed for day t changes when data after t is removed or scrambled, that features for day t don't use day t itself, and that the US feature doesn't use the US session dated t |
| `walk_forward_baselines.ipynb` | Baselines scored on each test year |
| `logistic_regression.ipynb` | Step 3: both logistic regressions through the six folds, and the agreement check |
| `technical_indicators.ipynb` | Step 4: all seven models on the 12 indicators, a table per model, and the AdaBoost check |
| `gru.py` | Step 6: the GRU network and its tuning |
| `sequence_model.ipynb` | Step 6: GRU vs logistic regression on three tasks, and the random-seed check |
| `us_overnight.ipynb` | Steps 5 and 5b: the US move as a feature, close-to-close vs open-to-close, the open-price check, and the checks of the two exceptions |
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
jupyter notebook technical_indicators.ipynb   # about 25 minutes
jupyter notebook us_overnight.ipynb
jupyter notebook sequence_model.ipynb         # about 35 minutes
```
