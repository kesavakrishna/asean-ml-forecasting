# llm_harness

Can a language model predict tomorrow's market direction? And when it seems to, is it predicting, remembering, or making things up?

This folder is a separate experiment from the main project. It reuses the main project's data loader and scoring.

## The tests

| Test | Question | Status |
|---|---|---|
| 1. Accuracy | Does it predict, or remember? Asked three ways, before and after its training cutoff | **Gemma 3 4B: done.** No skill and no memory: AUC 0.49–0.51 everywhere, and it says "up" to almost everything. See `accuracy.ipynb`. Nemotron on NIM stopped at 829 answers (rate limits) |
| 2. Reasoning | Are the numbers it cites right, and do its reasons really drive its answers? | **Gemma 3 4B: done.** It copies numbers correctly, but its stated rule doesn't drive its answer: flipping that move changes the answer no more often than flipping another (20.0% vs 19.7%). See `reasoning.ipynb` |
| 3. Feedback | Given a scorecard each month, does its notebook of rules get better? | **Gemma 3 4B: done.** No run beats its twin (all within 1.3 points). Told which rule works on planted data, it names that rule every time but its answers don't follow it. See `feedback.ipynb` |
| 4. Old local model | Test 1 on Llama 2 7B, whose training data ends in 2022–23 | **Done.** Like Gemma: no skill, no clear memory. See `accuracy_llama2.ipynb` |
| 5. With indicators (step 10) | Test 3 again, with the direction of each move and Kim's 12 indicators written below the table. Was reading the numbers the problem? | **Gemma 3 4B: done.** No. It now copies the 1-day move's direction (61–65% of answers), whatever rule it names: in the planted tester run it named "reverse" every time and agreed with it only 35% of the time. No run beats its twin; real-data AUC 0.50, the same as the ML models. See `with_indicators.ipynb` and [PLAN_more_inputs.md](../PLAN_more_inputs.md) |

## Files

| File | What it is |
|---|---|
| `client.py` | Sends a question to NVIDIA NIM or to a model on this laptop, stays under NIM's rate limit, and saves every question and reply |
| `questions.py` | The three ways of asking. `days_before()` is the wall: day t is never shown |
| `answers.py` | Reads the model's reply: the word, and P(up) |
| `run_accuracy.py` | Test 1: builds the questions and collects the answers |
| `explained.py` | Tests 2 and 3: questions whose reply must give evidence, a rule from the menu, an answer and a confidence |
| `rules.py` | The menu of checkable rules: what each rule says, turning a move upside down, and each rule's past hit rate |
| `checks.py` | Checks around every call: the wall before, reading the reply and checking cited numbers after |
| `fake_markets.py` | Made-up markets: random ones, and ones with a planted rule |
| `run_reasoning.py` | Test 2: each question asked four ways (base, rescaled, mirrored, control) |
| `run_feedback.py` | Test 3: the monthly notebook loop, with its state saved after every month |
| `run_with_indicators.py` | Test 5 (step 10): test 3's loop, with the moves and indicators added below the table |
| `PLAN_2_3.md` | Tests 2 and 3's plan and expected results, written before the runs |
| `llama2_chat_template.jinja` | Llama 2's chat format. Its model file doesn't include one, so the server is given this |
| `PLAN.md` | Test 1's plan and expected results, written before the run |
| `PLAN_11.md` | Step 11's plan: probes inside Gemma, steering vectors, a small fine-tune and direct weight edits, with their controls and expected results |
| `inside.py`, `inside.ipynb` | Step 11, stages 0–1: does `transformers` match llama.cpp (no: 83.5% in 4-bit, 86.0% in 8-bit), and probes. In the none runs Gemma works out its rule's answer and still answers against it on 41% of questions |
| `PLAN_11b.md` | Step 11b's plan: steering, a fine-tune and weight edits on 8-bit Gemma (added after looking) |
| `accuracy.ipynb` | Test 1's scores for Gemma: the three comparisons, the fake-data check, and a year-by-year look |
| `accuracy_llama2.ipynb` | The same scores for Llama 2 |
| `reasoning.ipynb` | Test 2's scores: the checks on each explanation, and the mirrored-against-control result |
| `feedback.ipynb` | Test 3's scores: each run against its twin, month by month, and the notebooks it wrote |
| `with_indicators.ipynb` | Test 5's scores: does it use its rule now, against its twin and test 3, and what it copies |

## Running

**A local model** (Gemma 3 4B) runs with [llama.cpp](https://github.com/ggml-org/llama.cpp). Download a Windows CUDA build and `gemma-3-4b-it-Q4_K_M.gguf` (2.5 GB, from `ggml-org/gemma-3-4b-it-GGUF` on Hugging Face), then start the server and leave it running:

```
llama-server -m gemma-3-4b-it-Q4_K_M.gguf --alias gemma-3-4b-it -ngl 99 -c 8192 -np 4 --cache-ram 0 --port 8080
```

`-ngl 99` puts the whole model on the graphics card. `--cache-ram 0` stops llama.cpp keeping up to 8 GB of past questions in normal memory: every question here is new, so that store is never used.

For **Llama 2 7B** (`llama-2-7b-chat.Q4_K_M.gguf`, 4.1 GB, from `TheBloke/Llama-2-7B-Chat-GGUF`):

```
llama-server -m llama-2-7b-chat.Q4_K_M.gguf --alias llama-2-7b-chat --chat-template-file llm_harness/llama2_chat_template.jinja -ngl 99 -c 2560 -np 2 --cache-ram 0 --port 8080
```

Llama 2 needs four times Gemma's memory per token (a token is a word or part of one), so it gets two slots of 1,280 tokens instead of four of 2,048. The longest question is 1,063 tokens. With Gemma's settings it overflowed the graphics card and ran three times slower.

**A NIM model** (Nemotron) needs your NVIDIA key in `.env` in the project folder as `NVIDIA_API_KEY=...`. That file is ignored by git.

Then, from the project folder:

```
python -m llm_harness.run_accuracy gemma pilot   # 2012-2016 days only, under a minute
python -m llm_harness.run_accuracy gemma         # the real test
python -m llm_harness.run_reasoning               # test 2, about 25 minutes
python -m llm_harness.run_feedback real tester    # test 3: one run (data: real, random, planted; feedback: none, scorecard, tester)
python -m llm_harness.run_with_indicators         # test 5: three runs, about 2 hours
```

Replies are saved in `results/llm/`, which is not in git. A stopped run carries on where it left off.

**What leaves your computer:** nothing, for a local model. For a NIM model, each question sends 20 days of index prices to NVIDIA.
