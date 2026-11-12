# llm_harness

Can a language model predict tomorrow's market direction? And when it seems to, is it predicting, remembering, or making things up?

This folder is a separate experiment from the main project. It reuses the main project's data loader and scoring.

## The tests

| Test | Question | Status |
|---|---|---|
| 1. Accuracy | Does it predict, or remember? Asked three ways, before and after its training cutoff | Plan written: [PLAN.md](PLAN.md) |
| 2. Reasoning | Are the numbers it cites right, and do its reasons really drive its answers? | Not built yet |
| 3. Feedback | Given a scorecard each month, does its notebook of rules get better? | Not built yet |
| 4. Old local model | The same tests on a model whose training data ends in 2022–23 | Not built yet |

## Files

| File | What it is |
|---|---|
| `client.py` | Sends a question to NVIDIA NIM, stays under the rate limit, and saves every question and reply |
| `questions.py` | The three ways of asking. `days_before()` is the wall: day t is never shown |
| `answers.py` | Reads the model's reply: the word, and P(up) |
| `run_accuracy.py` | Test 1: builds the questions and collects the answers |
| `PLAN.md` | Test 1's plan and expected results, written before the run |

## Running

Put your NVIDIA key in `.env` in the project folder as `NVIDIA_API_KEY=...`. That file is ignored by git.

```
python -m llm_harness.run_accuracy pilot   # 2012-2016 days only, about 6 minutes
python -m llm_harness.run_accuracy         # the real test, about 6 hours
```

Replies are saved in `results/llm/`, which is not in git. A stopped run carries on where it left off.

**What leaves your computer:** each question sends 20 days of index prices to NVIDIA. Nothing else is sent.
