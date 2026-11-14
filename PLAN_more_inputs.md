# Plan: more inputs (steps 9 and 10)

Written before any of these results exist. Added after looking at steps 1–8 and LLM tests 1–3, so everything here is labelled that way in the write-up.

**The question:** the paper's 12 indicators gave no edge (step 4). Do more indicators, automatic features (tsfresh), or data from other markets do better? And does an LLM do better when it is given the indicators instead of raw prices?

## Ground rules

- **Data:** 2012 to 2026-10-23. The old holdout (2023-03-17 onward) was spent in step 8, so it now counts as development data, not as a clean check. It adds test years 2023–2026, about 65% more test days. New downloads (set D) stop at 2026-10-23.
- **Same setup as step 4:** the six markets, test years 2017–2026 (train up to Y−2, tune on Y−1, test on Y), the same 7 models (LDA, logistic regression, KNN, SVM, random forest, AdaBoost, a one-hidden-layer network) and the same tuning grids. Target: up or down, close to close.
- **Same days for every set:** each set is scored on the days where all four sets have values, so differences come from the inputs, not the rows.
- **The leakage test covers every new feature:** the features for day t must not change when day t's prices change. It must also fail on a planted leak.
- **A pilot comes first:** one market, one test year, to check the code and the run time. It is not scored or looked at beyond "it runs".
- **The plan is frozen when committed.** Every result is reported, whichever way it comes out.

## Step 9: four sets of inputs for the ML models

| Set | Inputs | About how many |
|---|---|---|
| A. Kim's 12 | What step 4 used (`make_indicators`) | 12 |
| B. Standard extras | A, plus RSI, stochastic %K, Williams %R, rate of change and CCI at 5 and 20 days (A has 10), and disparity at 20 days (A has 5 and 10); MACD line, signal and histogram (12, 26, 9), divided by the close; Bollinger %B and band width (20 days, 2 standard deviations); ATR (14) divided by the close; ADX, +DI and −DI (14). Volume indicators (OBV change, MFI) only if all six markets have volume on at least 95% of days; this check looks at the volume column only | 32 |

**Volume check (done while building, volume column only):** Singapore has volume on 84.1% of days and India on 92.2%, so set B has no volume indicators.
| C. tsfresh | Features of the last 20 daily returns, from tsfresh's `EfficientFCParameters` (version 0.21.2) | 777 |
| D. Other markets | A, plus the last move before day t of: the S&P 500, VIX, the market's currency against the US dollar, and Brent oil | 16 |

**Set B:** computed in our own code with pandas, so every formula can be read in `features.py`.

**Set C, the risky one:** tsfresh's feature selection uses the target, so it must run on the training rows of each fold only:
- Select with tsfresh's `select_features` (its default false-discovery rate, 0.05) on the training rows. If it keeps none, keep the 10 with the smallest p-values.
- Missing values are filled with the median of earlier days only (0 if there are none), so the fill never looks ahead.
- The tune and test years never touch the selection.

**Set D, bad prints:** Yahoo's currency data has closes that are plainly wrong (the rupiah stuck at 9,612 for days in 2013 while it traded near 11,800). A currency close more than 10× its typical daily move away from the median of the 11 closes around it is blanked, and so are the moves into and out of it: 85 closes in all. That rule looks at later closes, but only to spot errors in the currency series. VIX and oil are left as they are, because their extremes are real (Feb 2018, Apr 2020). Days with a blank move drop out of all four sets.

**Set D, the timing:** these markets close after Asia does, so "the last move before day t" means the session dated strictly before t (as in step 5). That is known before day t's close but after day t−1's close, so, as in step 5, it is scored both close to close and open to close.

**Measures:**

| Measure | How |
|---|---|
| **Main:** better than A? | For each model, B, C and D against A on the same days: McNemar's test on right/wrong, pooled over the 36 folds. Holm adjustment over the 21 comparisons (3 sets × 7 models). Test years 2017–2026 |
| How much? | AUC difference against A, with a weekly-block bootstrap range |
| Any edge at all? | Pooled accuracy minus the fair bar, per model and market, Holm-adjusted within each set (as in step 8) |
| What tsfresh kept | Features selected per fold, and how often each was chosen |

**What I expect:**

| Set | Against A | Why |
|---|---|---|
| B | No difference: AUC within ±0.01, no comparison significant after Holm | Same past prices, rearranged |
| C | No difference, or slightly worse. Most folds select no features and fall back to the top 10 | Hundreds of features of noise |
| D, close to close | Better: a gain in most markets, like step 5's US move (53–58% on the holdout) | New information |
| Any set, 2017–2022 against 2023–2026 | The same pattern in both | The old holdout years are just more years |
| D, open to close | No difference in most markets | The open already contains it (step 5) |

## Step 10: an LLM given the indicators (`llm_harness`)

**Model:** `gemma-3-4b-it`, as in tests 2 and 3. **Days:** the same 1,296 days as test 3 (36 months × 36 questions, 2017–2019).

**The change:** below the 20-day table, the question adds two lines, computed from the 20 days shown only (so the wall is unchanged):
- *Moves:* the direction of the 1-, 5- and 20-day moves to day −1 (e.g. "1-day move: up"), which is exactly what each rule on the menu needs.
- *Indicators on day −1:* Kim's 12, to 2 decimals. The price oscillator is shown in %, so it doesn't round to 0.

Everything else (menu, JSON shape, notebook, scorecard, tester) is as in test 3. The new wording is frozen with this plan, in a new file. The test 3 files don't change.

**Three runs:**

| Run | Compared with |
|---|---|
| planted, tester, with the new lines | its twin (below), and the old planted tester run |
| planted, none, with the new lines | the twin |
| real, none, with the new lines | the old real none run (AUC 0.511), and set A's models on the same days |

**Measures and expectations:**

| Question | Measure | Test 3 got | What I expect |
|---|---|---|---|
| Was reading the table the problem? | Answer agrees with its own rule, months 25–36 | 47–55% | 65–80%. Higher, but the "up" habit stays |
| Does it now use the planted rule? | Planted tester minus its twin, McNemar | +1.2 points | +3 to +8 points |
| What happens to the twin? | Planted none accuracy | 49.2% | Lower, toward 40%, if it now obeys its default "follow the 1-day move" (right 36% of the time there) |
| Do indicators give it skill on real data? | AUC, real none run | 0.511 | 0.48–0.53, no better than set A's models |

If agreement stays near 50%, the problem is not reading the numbers: it does not act on its own stated rule at all.

## Clean checks

The holdout is spent, and deleting its results wouldn't unspend it: these steps were chosen *after* seeing it. A clean check needs data that none of our choices has seen. Two sources:

- **Markets we never touched:** Korea (KOSPI, `^KS11`), Taiwan (`^TWII`), Hong Kong (Hang Seng, `^HSI`) and Japan (Nikkei 225, `^N225`). Downloaded only after steps 9 and 10 are written up, then run once with all four sets, the same code and the same settings, test years 2017–2026. Weaker than truly new days, because the years and world events are the same ones, but no choice here was made using these markets.
- **New days (a forward test):** once steps 9 and 10 are written up, each set's choices are frozen. Each model is then retrained on all data up to 2026-10-23, its settings picked on the last year as usual, and saved. It predicts only days from 2026-10-24 onward, scored once a year of them exists (late 2027). This is what a model would face in real use.

**What I expect on the new markets:** the same pattern as step 9: no gain from B or C, a close-to-close gain from D.

## Files

| File | What it will hold |
|---|---|
| `features.py` | `make_indicators_extended` (B), `make_tsfresh` (C), `make_other_markets` (D) |
| `data.py` | `load_other_moves`, with the bad-print rule |
| `more_inputs.py` | Step 9's runs: the four sets on the same days, tsfresh's selection on training years only |
| `download_data.py` | VIX, currencies and Brent, to 2023-03-16; the new markets later, in their own files |
| `test_leakage.py` | Checks for B, C and D |
| `more_inputs.ipynb` | Step 9 |
| `llm_harness/run_with_indicators.py` | Step 10, reusing test 3's loop with the new question |
| `llm_harness/with_indicators.ipynb` | Step 10's scores |
| `new_markets.ipynb` | The check on new markets |
| `forward_test.py` | Retrains the frozen models on all data and saves them; later scores the new days |
