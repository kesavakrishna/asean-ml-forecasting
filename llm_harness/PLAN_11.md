# Step 11 plan: looking inside Gemma, steering it, and editing its weights

Written before any of this is built or run. Chosen after seeing tests 1–5, so it is labelled "added after looking".

**The question:** in step 10, Gemma 3 4B named the right rule ("reverse the 1-day move") in every late answer of the planted tester run, and then agreed with it only 35% of the time. Why? Three possible explanations:

1. **It never works out what the rule says.** The rule's answer can't be read from its activations when it answers.
2. **It works it out, but an "up" habit or copying wins.** The rule's answer can be read, and it is overridden.
3. **It applies the rule to the wrong thing.** Step 10 found it copies the 1-day move's direction (61–65% of answers).

Then: can the gap between its rule and its answer be closed by steering it, by a small fine-tune, or by editing its weights directly? None of these can add market skill that isn't in the prices, so on real data the expectation stays at about 50%.

## Background

- **Features and circuits.** Interpretability work has found readable features in networks: curve detectors in an image model ([Zoom In, Distill 2020](https://distill.pub/2020/circuits/zoom-in/)), and a "Golden Gate Bridge" feature in Claude 3 Sonnet, found with a sparse autoencoder rather than in one neuron, because most neurons in language models mix many meanings (Anthropic, *Scaling Monosemanticity*, 2024). Turning that feature up made the model mention the bridge everywhere: steering.
- **Steering vectors.** Add a fixed direction to a layer's activations while the model answers. The direction is the difference between two average activations, such as answers that say "down" minus answers that say "up".
- **Weight-space programming.** A frontier model reads another network's weights and edits them directly, without gradient descent. MIT CSAIL's technical report *Sparks of Weight-Space Programming* (Takehi, Teoh, Zhang, Khattab, 2 Nov 2026) reports GPT-6 Astra changing about 0.016% of Qwen3.5-0.8B's weights and raising MATH-500 from 7.9% to 48.1%, mostly by ending thinking loops. It is unreviewed, uses one editor model, and the editor used code and forward passes, so it is partly a guided search. Here the editor is Claude, working through code in this repo.
- **The lesson from our own GRU probes:** a probe can find structure that comes from the input, not from learning. Every probe here has controls.

## Setup

- **Model:** `gemma-3-4b-it`, as in tests 2, 3 and 5. Run in Hugging Face `transformers`, 4-bit (NF4), on the RTX 4060 (8 GB). Full precision (8.6 GB) doesn't fit. One engine for every stage, so the stages can be compared.
- **Questions:** the saved questions and replies of step 10's three runs (planted tester, planted none, real none), 36 months × 36 questions each. Each question is replayed with its saved notebook text, so the feedback loop is not rerun.
- **Two halves:**
  - **Build half:** months 1–24 (2017–2018), 2,592 questions. Probes are trained, vectors are made, and settings and edits are chosen here only.
  - **Score half:** months 25–36 (2019), 1,296 questions. Scored once per stage, at the end of that stage.
- **Agreement with its own rule:** the answer matches what its named rule says for that table (`rules.py`), the measure from tests 3 and 5.
- **The confound:** in the planted tester run the named rule is always "reverse the 1-day move", so the rule's answer is just the 1-day move flipped. Probes for the rule's answer are therefore trained on the two "none" runs, where the named rule varies, and judged on **conflict questions**: those where the rule's answer differs from the 1-day move's direction.

## Stage 0: does `transformers` behave like llama.cpp?

Replay 200 questions from the build half, sampled evenly across the three runs, with the saved reply up to `"answer": "`. Compare its next word ("up" or "down") with the saved answer from llama.cpp.

**Gate:** at least 90% the same. If not, report it, and run only stage 1 (probes on `transformers`). No steering or edit result is claimed for the llama.cpp model.

## Stage 1: probes

At the position just before the answer word, for every layer, a logistic regression reads:

| Target | What it tells us |
|---|---|
| T1: its final answer | Sanity check: should be readable at late layers |
| T2: the 1-day move's direction | Whether the input it copies is represented |
| T3: its named rule's answer, judged on conflict questions | **Explanation 1 vs 2:** whether it works out what its rule says |

**Controls:** the same probe trained on shuffled labels; the same probe at the embedding layer (before any attention, so it sees one token only). Trained on build months, scored on score months.

**Expected:**

| Target | Best layer, score months | Controls |
|---|---|---|
| T1 | ≥ 95% | about 50% |
| T2 | ≥ 90% | about 50% |
| T3, conflict questions | 60–75% | at most 55% |

Explanation 2 is supported if T3 reaches at least 65% at some layer while both controls stay at or below 55%. Explanation 1 if T3 stays at or below 55% everywhere.

## Stage 2: steering

Two vectors, made from build months at the layer where T1 is first readable at 90%:

- **S1, "down minus up":** the average activation when it answers "down" minus when it answers "up". It tests the up habit.
- **S2, "agrees minus disagrees":** the average when its answer agrees with its named rule minus when it doesn't. It tests whether there is a general "follow your rule" direction.
- **Control:** a random direction with the same length as S2.

Each is added at strengths −2, −1, −0.5, 0.5, 1 and 2 times the vector's length. The strength is chosen on build months (the best agreement with valid JSON in at least 95% of replies), then scored once on score months, generating freely.

**Expected:**

| Vector | Expected on score months |
|---|---|
| S1 | The share of "up" answers falls steadily as the strength rises. Agreement changes by less than 5 points at the chosen strength |
| S2 | Agreement in the planted tester run rises by 5–15 points (from about 35%) |
| Random | Agreement changes by less than 2 points |

## Stage 3: a small fine-tune (the gradient baseline)

A LoRA adapter (rank 8, 4-bit, one epoch), trained on build months to write its named rule's answer after its rule. Scored on score months, generating freely.

**Expected:** agreement at least 80%. This is the ceiling a gradient method reaches with the same data, so stage 4 can be compared with it.

## Stage 4: weight-space programming

Claude, working through code, inspects Gemma's weights and activations on build months and proposes direct edits.

- **An edit:** a sparse change to specific weight entries, applied in fp16 next to the 4-bit weights. At most 0.1% of the model's weights.
- **Budget:** 10 edit attempts, each one logged with its reasoning and its build-month score. Only the last chosen edit is scored on score months, once.
- **Control:** a random edit with the same number of entries and the same sizes, in the same layers.

**Expected:** it reaches more than stage 2's S2 but less than stage 3's LoRA. It counts as working if it beats S2 on agreement by at least 5 points (McNemar, Holm-adjusted), and its random edit doesn't.

## Measures across the stages

| Measure | How |
|---|---|
| **Main: agreement with its own rule** | Planted tester run, score months. Each stage's best against the unsteered model: McNemar's test, Holm over the 4 comparisons (S1, S2, LoRA, edit) |
| Accuracy, planted | Planted runs, score months. Should rise with agreement: the planted rule is right 63.7% of the time |
| Accuracy and AUC, real | Real none run, score months. **Expected 48–52% and 0.48–0.53 for every stage**, no better than step 9's models |
| Valid JSON | Share of replies that parse. Below 95% counts as breaking the model |

## Stage 5, if time allows: Llama 2 7B

Llama 2 7B (4-bit, about 4 GB) was never run on the rule questions. First run step 10's planted tester loop on it, then repeat stages 0–2 with the same code and thresholds. Expected: the same pattern as Gemma.

## Gates and order

1. Stage 0's gate must pass before stages 2–4.
2. Stage 4 runs only if stage 1 finds T3 readable (explanation 2) or stage 2's S2 works: otherwise the editor has nothing to aim at, and that is reported.
3. Memory: everything runs one stage at a time. Stage 4's edits stay in fp16 next to the 4-bit weights, so the full-precision model is never loaded.

## Files

| File | What it will hold |
|---|---|
| `inside.py` | Loading Gemma in `transformers`, replaying saved questions, reading activations, the probes, steering and sparse edits |
| `inside.ipynb` | Stages 0–3: the reproduction check, the probe curves, steering by strength, the LoRA baseline |
| `weight_edits.md` | Stage 4's log: every edit attempt, its reasoning and its build-month score |
| `weight_edits.ipynb` | Stage 4's scores and its random-edit control |
