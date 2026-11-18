# Step 11b plan: steering, a fine-tune and weight edits on 8-bit Gemma

Written after stages 0 and 1 of [PLAN_11.md](PLAN_11.md), before any of this is built or run. **Added after looking.**

**Why a second plan:** stage 0's gate failed (8-bit `transformers` gives the same answer as llama.cpp on 86% of questions, not 90%), so PLAN_11 stops after stage 1. But 8-bit behaves like step 10's model overall (it agrees with its rule 32–62% of the time, llama.cpp 35–64%), and stage 1 found something to aim at: in the none runs, the rule's answer is readable at layers 20–30 even when it answers against it. So here **8-bit Gemma is the model**. Nothing below is claimed for llama.cpp's model.

**The question:** can the gap between its rule and its answer be closed by steering, a small fine-tune, or direct weight edits? None can add market skill, so on real data accuracy should stay about 50%.

## Setup

- **Model:** `gemma-3-4b-it` in 8-bit `transformers`, as in stage 1.
- **Questions:** step 10's saved questions. Each is replayed with llama.cpp's saved reply up to `"answer": "`, so the rule it named is fixed, and only the next word ("up" or "down") is read. Whether a change would make it name a different rule is **not** tested.
- **Halves:** as in PLAN_11. Build: 2017–2018. Score: 2019 (1,296 questions), each stage scored once, at its end.
- **Build sample:** choosing strengths and edits on all 2,592 build questions is too slow (about 3 s each in 8-bit), so choices use a fixed sample of 300 (100 per run, seed 0).
- **Agreement:** its next word equals what its named rule says (`rule_answer` in stage 1).

**Unchanged model, 2019, from stage 1:** agreement 32.2% (planted tester), 56.7% (planted none), 62.0% (real none).

## Stage 2: steering

- **Where:** layer 17, where T1 first reaches 90% (PLAN_11's rule), and **only at the answer position.** Earlier tokens are run once and cached, so each strength costs one token. This differs from steering every position, which is too slow here.
- **S1, "down minus up":** the average layer-17 activation when it answers "down" minus when it answers "up", build months, all runs.
- **S2, "agrees minus disagrees":** the average when its answer agrees with its rule minus when it doesn't, build months, **the two none runs** (in planted tester, agreeing just means "not the 1-day move").
- **Random:** a random direction as long as S2 (seed 0).
- **Strengths:** −4, −2, −1, −0.5, 0.5, 1, 2, 4 times the vector's length. Chosen on the build sample: the best agreement in planted tester, among strengths where "up" or "down" is still the most likely next token in at least 95% of questions.

**Expected (2019, planted tester):**

| Vector | Expected |
|---|---|
| S1 | The share of "up" falls steadily as the strength rises. Agreement changes by less than 5 points |
| S2 | Agreement rises by 5–15 points |
| Random | Agreement changes by less than 2 points |

**Change before any steering was scored (15 Nov 2026):** the cached shortcut doesn't work in 8-bit. bitsandbytes' 8-bit multiply keeps a column at full precision when any token in the same call has a large value there, so a token run alone is rounded differently from the same token in a full pass. On 4 build questions the cached gap differed by up to 26 (and flipped sign twice); in 4-bit it matched within 0.25. So:

- Every condition is one **full pass**, with the vector added at the answer position only, as before.
- The strengths are chosen on the build sample as planned (25 conditions × 300 questions, about 6 hours).
- 2019 is scored **only at the 3 chosen strengths** (about 4 hours), not at every strength. So S1's "falls steadily with strength" is judged on the build sample.

**A fair bar, added after the build sample and before 2019 is scored:** on the build sample the unchanged model agrees with its rule less often (38%, planted tester) than answering "up" to everything would (57%), and the large strengths mostly make it say "up". So beating the unchanged model can just mean saying "up" more. Every stage is therefore also compared with **the fair bar: the same word every time, whichever agrees more often**. In 2019's planted tester run that is "up", at 52.1%, worked out from the labels alone. A stage closes the gap only if it beats the fair bar too (McNemar, Holm over the 4 stages). Chosen strengths (by the rule above; S1's tie between 2 and 4 went to the first, 2): **S1 at 2, S2 at −4, random at 2.**

## Stage 3: a small fine-tune (the gradient baseline)

A LoRA adapter (rank 8, on the attention's query and value weights, learning rate 2e-4, one epoch) on 600 build questions (200 per run, seed 0), trained to give **its rule's answer** at the answer position. Loss on that one token only.

**Expected:** agreement at least 80% in planted tester, 2019.

## Stage 4: weight-space programming

Claude, working through code, inspects the 8-bit model's weights and activations on build months and proposes direct edits.

- **An edit:** a sparse change to specific weight entries, applied as a bf16 change next to the 8-bit weights (the 8-bit weights are never changed). At most 0.1% of the language model's weights. It acts at every position, unlike stage 2's steering.
- **Budget:** 10 attempts, each logged in `weight_edits.md` with its reasoning and its build-sample score. Only the last chosen edit is scored on 2019, once.
- **Control:** a random edit with the same number of entries, the same sizes and the same weight matrices.

**Expected:** better than S2, worse than LoRA. It counts as working if it beats S2 on agreement by at least 5 points (McNemar, Holm-adjusted) and the random edit doesn't.

## Measures

| Measure | How |
|---|---|
| **Main: agreement with its own rule** | Planted tester, 2019 (432 questions). Each stage's choice against the unchanged model: McNemar's test, Holm over 4 (S1, S2, LoRA, edit). **And against the fair bar** (the same word every time: "up", 52.1%), the same way |
| Agreement, none runs | Also reported, same test |
| Accuracy and AUC, real | Real none run, 2019. **Expected 48–52% and 0.48–0.53 for every stage** |
| Still answers | Share of questions where "up" or "down" is the most likely next token. Below 95% counts as breaking the model |

## Order

1. Stage 2, then 3, then 4. Each stage's 2019 score is run once, after its choice is fixed and logged.
2. Stage 4's first attempt is written down before stage 3's score is seen, so the editor isn't aiming at LoRA's number.

## Files

| File | What it will hold |
|---|---|
| `inside.py` | Steering, the LoRA, sparse edits, the replay scorer |
| `steering.ipynb` | Stages 2 and 3: steering by strength, the LoRA, the tests |
| `weight_edits.md` | Stage 4's log: every attempt, its reasoning and its build-sample score |
| `weight_edits.ipynb` | Stage 4's 2019 scores and its random-edit control |
