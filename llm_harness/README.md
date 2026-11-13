# llm_harness

Can a language model predict tomorrow's market direction? And when it seems to, is it predicting, remembering, or making things up?

This folder is a separate experiment from the main project. It reuses the main project's data loader and scoring.

## The tests

| Test | Question | Status |
|---|---|---|
| 1. Accuracy | Does it predict, or remember? Asked three ways, before and after its training cutoff | Gemma 3 4B: plan written, see [PLAN.md](PLAN.md). Nemotron on NIM stopped at 829 answers (rate limits) |
| 2. Reasoning | Are the numbers it cites right, and do its reasons really drive its answers? | Not built yet |
| 3. Feedback | Given a scorecard each month, does its notebook of rules get better? | Not built yet |
| 4. Old local model | The same tests on a model whose training data ends in 2022–23 | Not built yet |

## Files

| File | What it is |
|---|---|
| `client.py` | Sends a question to NVIDIA NIM or to a model on this laptop, stays under NIM's rate limit, and saves every question and reply |
| `questions.py` | The three ways of asking. `days_before()` is the wall: day t is never shown |
| `answers.py` | Reads the model's reply: the word, and P(up) |
| `run_accuracy.py` | Test 1: builds the questions and collects the answers |
| `PLAN.md` | Test 1's plan and expected results, written before the run |

## Running

**A local model** (Gemma 3 4B) runs with [llama.cpp](https://github.com/ggml-org/llama.cpp). Download a Windows CUDA build and `gemma-3-4b-it-Q4_K_M.gguf` (2.5 GB, from `ggml-org/gemma-3-4b-it-GGUF` on Hugging Face), then start the server and leave it running:

```
llama-server -m gemma-3-4b-it-Q4_K_M.gguf --alias gemma-3-4b-it -ngl 99 -c 8192 -np 4 --cache-ram 0 --port 8080
```

`-ngl 99` puts the whole model on the graphics card. `--cache-ram 0` stops llama.cpp keeping up to 8 GB of past questions in normal memory: every question here is new, so that store is never used.

**A NIM model** (Nemotron) needs your NVIDIA key in `.env` in the project folder as `NVIDIA_API_KEY=...`. That file is ignored by git.

Then, from the project folder:

```
python -m llm_harness.run_accuracy gemma pilot   # 2012-2016 days only, under a minute
python -m llm_harness.run_accuracy gemma         # the real test
```

Replies are saved in `results/llm/`, which is not in git. A stopped run carries on where it left off.

**What leaves your computer:** nothing, for a local model. For a NIM model, each question sends 20 days of index prices to NVIDIA.
