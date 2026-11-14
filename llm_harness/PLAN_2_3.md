# Tests 2 and 3 plan: is the model's explanation true, and does feedback help?

Written before the real runs. The wording was tried on 2012–2016 days only, which are never scored.

**Model:** `gemma-3-4b-it`, 4-bit, on this laptop with llama.cpp. It follows instructions better than Llama 2, and test 1 found no memory in it.

## How the model must answer

Every question shows the 20 days before day t with the name and dates hidden (as in test 1). The server forces the reply into this JSON shape (`explained.py`):

| Part | What it is | How it's checked |
|---|---|---|
| `evidence` | 1–3 numbers from the table, each with its day and field | Against the table it was shown |
| `rule` | One rule from a menu: follow or reverse the 1-, 5- or 20-day move, or "none of these" | The harness can apply the rule itself (`rules.py`) |
| `answer` | up or down | Against what happened |
| `confidence` | 50 to 100 | Turned into P(up) for AUC |

The menu is what makes a reason checkable: "follow the 5-day move" can be applied to any table, scored on past data, and tested by turning that move upside down.

## Test 2: is the explanation true?

300 days, 50 per market, Jan 2017 – Dec 2022, development data only. Each is asked four ways (`run_reasoning.py`):

- **base:** the table, first close 100.
- **rescaled:** the same table, first close 1000. Same information.
- **mirrored:** the move named by the model's own rule turned upside down (a 1% rise becomes a 1% fall). Earlier days unchanged.
- **control:** a move it did *not* name turned upside down instead.

| Measure | Question it answers | What I expect |
|---|---|---|
| Wrong numbers | Does it copy numbers correctly? | Under 5% wrong |
| Answer agrees with its own rule | Does it do what it says? | 60–70% (pilot: 63%) |
| Same answer when rescaled | Is it steady when nothing real changes? | 90% or more |
| Flips when mirrored, minus flips in control | Does the reason it gives drive the answer? | About 0. The pilot flipped 20% against 23% |
| AUC | Any skill? | 0.50 |

**Main result:** flips when mirrored minus flips in control, on the same days, with McNemar's paired test. A reason that really drives the answer would flip it almost every time when mirrored, and rarely in the control.

## Test 3: does feedback make it better?

Each month for 36 months (Jan 2017 – Dec 2019) the model answers 36 questions (6 days × 6 markets) with its notebook in front of it. At month end it gets a scorecard and rewrites the notebook, in at most 150 words (`run_feedback.py`). The same days are used in every run.

**Three kinds of feedback**, each its own run:
- **none:** the twin. Its notebook stays empty.
- **scorecard:** this month's right/wrong count, each rule's record when used, confidence against results, and wrong numbers.
- **tester:** the scorecard, plus how every rule on the menu did on all earlier days.

**Three kinds of data:**
- **real:** the six markets, name and dates hidden.
- **random:** six made-up markets where nothing can be learned. Any "improvement" here is noise.
- **planted:** six made-up markets where "reverse the 1-day move" is right 60% of the time and every other rule is near 50%.

That's 9 runs and about 12,000 calls, roughly 4.5 hours.

| Measure | How |
|---|---|
| Better than its twin? | Accuracy and AUC, scorecard and tester runs against the none run on the same days. McNemar's test |
| Learning over time? | Months 25–36 against months 1–12 |
| Found the planted rule? | Share of answers using "reverse the 1-day move" in months 25–36 |
| Made-up numbers in its notebook | Percentages in a notebook that match no number (within 1 point) in any scorecard it has seen |
| Notebook churn | Share of months where the notebook changed |

What I expect:

| Data | scorecard vs none | tester vs none |
|---|---|---|
| real | no difference | no difference |
| random | no difference | no difference |
| planted | little or none: it mostly uses one rule, so the scorecard only reports that rule | switches to "reverse the 1-day move" within 6 months and beats its twin by 5 points or more |

- **Made-up numbers:** common. In the pilot most percentages in its notebook matched nothing in the scorecards.
- **Stubbornness:** in the pilot it used "follow the 1-day move" for 107 of 108 answers. I expect it to stick to one rule unless the tester shows a clearly better one.

## The OpenWorker ideas, as built here

| Idea | Here |
|---|---|
| Scoped tools | `questions.days_before()` only returns days before t, and `checks.check_wall()` stops the run if a question would show more |
| Hooks | `check_wall` before each call; `read_reply` and `wrong_citations` after |
| Turn traces | Every call is saved with its request, reply, token counts and `seconds` |
| Durable sessions | Each test 3 run saves its state after every month (`results/llm/feedback_*_state.json`) and carries on from there |
| Approvals (our input kept apart) | No hints from us in these runs. Feedback is built only from outcomes |

## Rules

- The wording in `explained.py`, `rules.py` and `run_feedback.py` is frozen from this commit.
- Every result is reported, whichever way it comes out.
- Anything added after looking is labelled as such.
