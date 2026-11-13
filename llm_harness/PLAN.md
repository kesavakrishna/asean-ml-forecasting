# Test 1 plan: does the model predict, or remember?

Written before the real run. The wording below was tried on 2012–2016 days only, which are never scored.

## Change before running: Gemma 3 4B on this laptop

Written after the Nemotron run stopped and before any Gemma answer for 2017 onward existed.

- **Why:** NIM's free tier refused most requests after about 1,000. The Nemotron run stopped at 829 of 11,373 answers, all from Indonesia before Sep 2025. Those answers are kept but not scored as test 1.
- **New model:** `gemma-3-4b-it`, 4-bit, run locally with llama.cpp on the graphics card. Its training data ends in **Aug 2024**.
- **Periods:** *before* = Jan 2017 – Aug 2024, *after* = Sep 2024 – 23 Oct 2026. 300 random days per market in each. There is no "in between".
- **Dropped:** the thinking side check. Gemma 3 has no thinking mode.
- **Unchanged:** the questions, the three ways of asking, the fake data, the scoring and the rules below.
- **One technical change:** the 20 likeliest first words are read instead of 5, because a small model's second choice can sit further down the list.
- **From the pilot (2012–2016 only):** Gemma said "up" to 189 of 190 questions, usually with P(up) close to 1. Its probabilities still differ between days, so AUC can still rank them. Accuracy will mostly equal "always up".

What I expect for Gemma:

| Comparison | Before | After |
|---|---|---|
| 1. Hidden, AUC | 0.50 | 0.50 |
| 2. Named minus hidden, AUC | 0 to 0.02 | 0 |
| 3. Date only, AUC | 0.50 to 0.52 | 0.50 |

- **Memory:** smaller than I'd have expected from Nemotron. A 4B model holds far less detail, so I expect little or none.
- **Accuracy:** about the fair bar everywhere, because it nearly always says "up".
- **Fake data:** AUC 0.50.

## Second model: Llama 2 7B on this laptop

Written after Gemma's results and before any Llama 2 answer for 2017 onward existed.

- **Model:** `llama-2-7b-chat`, 4-bit, run locally with llama.cpp. Its pretraining data ends in **Sep 2022**, and some fine-tuning data in **Jul 2023**.
- **Periods:** *before* = Jan 2017 – Aug 2022, *in between* = Sep 2022 – Jul 2023, *after* = Aug 2023 – 23 Oct 2026. 300 random days per market in *before* and *after*, and every day *in between* (about 225 per market). 15,126 questions in all.
- **Unchanged:** the questions, the three ways of asking, the fake data and the scoring.
- **Two technical fixes, found in the pilot (2012–2016 only):**
  - The model file carries no chat format, so the server was falling back to another model family's and Llama 2 replied with nonsense. `llama2_chat_template.jinja` gives it Llama 2's own format.
  - Llama 2 starts its reply with a space. The answer reader now reads P(up) where "up" or "down" is the most likely word, instead of the first position that lists either. This changes none of the 11,100 Gemma or 779 Nemotron readings.
- **From the pilot:** with prices it says "Up" almost always (P(up) 0.97–1.00). With only a date it is more mixed (P(up) 0.07–0.84, mean 0.48).

What I expect for Llama 2:

| Comparison | Before | In between | After |
|---|---|---|---|
| 1. Hidden, AUC | 0.50 | 0.50 | 0.50 |
| 2. Named minus hidden, AUC | 0 to 0.02 | 0 | 0 |
| 3. Date only, AUC | 0.50 to 0.52 | 0.50 | 0.50 |

- **Memory:** little or none, as with Gemma. A 7B model is only a little bigger.
- **Accuracy:** about the fair bar with prices, since it nearly always says "Up". With only a date it says "Down" more often, so its accuracy there may fall below the fair bar.
- **Fake data:** AUC 0.50.

The rest of this file is the original plan, written for Nemotron. Where it names Nemotron's dates or periods, read Gemma's from the list above.

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
