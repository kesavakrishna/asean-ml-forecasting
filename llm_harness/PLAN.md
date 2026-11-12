# Test 1 plan: does the model predict, or remember?

Written before the real run. The wording below was tried on 2012–2016 days only, which are never scored.

## What is asked

For a market and a day t, the model answers one word: **up** or **down** (did day t close above day t−1?). The same day is asked three ways:

| Way of asking | What the model sees |
|---|---|
| **Named** | The index name, the real dates, and the last 20 days of prices |
| **Hidden** | The same 20 days with the name and dates removed, rescaled so the first close is 100 |
| **Date only** | The index name and the date. No prices |

- **Model:** `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA NIM, thinking off, temperature 0.
- **Its answer:** the word it gives. **P(up)** comes from the probabilities it puts on "up" and "down".
- **The wall:** a question about day t never contains anything from day t or later (`questions.days_before`).

## Which days

The model's card says its main training data ends in **Sep 2025** and its fine-tuning data in **May 2026**.

| Period | Dates | Days per market | Could the model have read about it? |
|---|---|---|---|
| Before | Jan 2017 – Aug 2025 | 300, picked at random | Yes |
| In between | Sep 2025 – May 2026 | all, about 180 | Only through fine-tuning data |
| After | Jun 2026 – 23 Oct 2026 | all, about 100 | No |

Two extras:
- **Fake data:** 300 hidden questions on made-up random prices. Nothing can be predicted there, so this checks the harness itself.
- **Thinking on:** 600 of the date-only questions are asked again with thinking allowed, to see if thinking first helps it remember.

In all: 11,373 questions.

## How it is scored

- **Main score: AUC.** Does P(up) rank the up days above the down days? 0.50 is a coin flip. AUC isn't fooled by a model that mostly says one word.
- **Second score:** accuracy against the fair bar (the direction that was more common before each test year), with McNemar's paired test.
- **Ranges:** whole weeks resampled 2,000 times, middle 95%, six markets together.
- **Many tests:** Holm's adjustment within each comparison (three periods each).

The three comparisons, each in every period:
1. **Hidden** against a coin flip: can it predict from prices alone?
2. **Named minus hidden,** on the same days: do the name and dates add anything?
3. **Date only** against a coin flip: pure memory.

**What would count as memory:** comparison 2 or 3 above zero in the "before" period, and not in the "after" period.

## What I expect

| Comparison | Before | In between | After |
|---|---|---|---|
| 1. Hidden, AUC | 0.50 | 0.50 | 0.50 |
| 2. Named minus hidden, AUC | 0 to 0.02 | about 0 | 0 |
| 3. Date only, AUC | 0.50 to 0.53 | about 0.50 | 0.50 |

- **Prices alone:** no skill, as with every model in steps 3–6.
- **Memory:** small. I expect it to remember famous stretches such as March 2020, but not ordinary days.
- **Accuracy:** at or below the fair bar in every case. In the pilot the model said "down" on 85% of the date-only questions, and markets rise slightly more often than they fall.
- **Fake data:** AUC 0.50 and accuracy near 50%. Anything else means the harness has a fault.
- **Thinking on:** a little better at remembering "before", no different "after".

I'm not confident in the size of the memory effect. That's the number this test is for.

## Rules

- The wording in `questions.py` is frozen from this commit.
- Every result is reported, whichever way it comes out.
- Anything added after looking is labelled as such.
