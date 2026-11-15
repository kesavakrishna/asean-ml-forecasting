# Plan: the forward test, and volatility on the new markets

Written before any of these results exist. Both checks were chosen after seeing steps 1–9 and the check on new markets, so they are labelled "added after looking" in the write-up, but neither has seen its own data.

## 1. The forward test: new days

**The idea:** freeze step 9's models now and score them on days that did not exist when they were trained. No choice in this project can have seen these days.

**Freezing (now):**
- Every run in step 9 (close to close: sets A, B, C, D; open to close: A, D), the same 7 models, in all 10 markets (the 6 of step 9 and the 4 new ones): 420 models.
- Each is trained on every day up to **2026-10-23**. As in step 9's folds, its setting is picked on the last year (2026, January to 23 October) after training on the years before, then it is refitted on everything.
- Set C's tsfresh selection uses the same rows as each fit, so the tune year never touches it.
- The fair bar for each model is the direction that was more common in its training rows, fixed now.
- `python forward_test.py freeze` saves each model in `results/forward/` (not in git) and writes `forward_test_models.csv` (in git): each model's setting, fair bar and SHA-256 hash. Scoring checks every hash first, so a changed model can't be scored.

**Scoring (once, in late 2027):**
- **Days:** 2026-10-24 to **2027-11-26**, about 260 trading days per market. The end date is fixed in `download_data.py`, so nobody chooses it after looking.
- **Data:** a fresh download into `Dataset/forward/` (the fixed files in `Dataset/` stay as they are), plus the two Investing.com files to the same end date. Run with the same package versions (`requirements.txt`).
- Same days for every set, as in step 9. Features use only data up to day t−1, as always.
- `forward_test.ipynb` is written and committed now, and run once.

**Measures:** the same as step 9's.

| Measure | How |
|---|---|
| **Main:** better than A? | For each model, B, C and D against A, close to close: McNemar's test, all markets together. Holm over the 21 comparisons |
| Open to close | D against A, Holm over 7 |
| How much? | AUC difference against A, with a weekly-block bootstrap range |
| Any edge at all? | Accuracy minus the frozen fair bar, per model and market, Holm within each set (70 tests) |

**What I expect:**

| Set | Expected | Why |
|---|---|---|
| A, B, C | No edge over the fair bar; B and C no better than A | Steps 4 and 9, and the new markets |
| D, close to close | Beats A for most of the 7 models after Holm, and beats the fair bar in most markets. With about 260 days per market, some single markets may not pass | The US move, seen in every check so far |
| D, open to close | No gain over A in most markets | The open already contains the US move |

## 2. Volatility on the new markets

**The idea:** step 7's other positive result was that HAR, a three-number linear model, predicts tomorrow's range at least as well as GARCH, a forest and a GRU. The holdout mostly agreed (step 8). Korea, Taiwan, Hong Kong and Japan have not been used for volatility at all: none of their ranges has been looked at.

**Setup:** step 7's code and settings, unchanged: same as yesterday, HAR, HAR + US, GARCH, random forest and the GRU (seeds 0, 1, 2). Target: the log of log(High / Low). Test years 2017–2026 (train up to Y−2, tune on Y−1). Run once.

**Measures:** R² and QLIKE, all test years together, per market. Each model against HAR with the paired test on squared errors and on QLIKE. Holm over the 24 comparisons for each score (6 rivals × 4 markets).

**What I expect:**

| Comparison | Expected |
|---|---|
| HAR against same as yesterday | HAR better in all 4 markets, significant after Holm |
| HAR against GARCH | HAR's R² higher in all 4, significant in at least 3 |
| HAR against the forest | HAR's R² higher in all 4 |
| HAR against the GRU | A tie: no seed significantly better than HAR in any market |
| HAR + US against HAR | A gain of at most 0.03 in R². It may be significant here: set D showed these markets follow the US closely |

## Files

| File | What it holds |
|---|---|
| `forward_test.py` | `freeze`: trains and saves the 420 models. `score`: checks the hashes and predicts the new days |
| `forward_test_models.csv` | Each frozen model's setting, fair bar and hash |
| `forward_test.ipynb` | The forward test's scoring, written now, run in late 2027 |
| `download_data.py` | `forward`: the new days, to the fixed end date, into `Dataset/forward/` |
| `new_markets_volatility.ipynb` | Volatility on the new markets |
