"""
Step 11: looking inside Gemma (see PLAN_11.md). Runs in venv_gpu, which has GPU PyTorch.

Every question and reply from step 10 is saved, so a question can be replayed exactly:
the model is fed the question and its own reply up to the answer word, and we read what it does there.

Run:  venv_gpu/Scripts/python llm_harness/inside.py reproduce     # stage 0: does transformers agree with llama.cpp?
      venv_gpu/Scripts/python llm_harness/inside.py reproduce 8   # the same in 8-bit (added after looking)
      venv_gpu/Scripts/python llm_harness/inside.py activations   # stage 1: save what 8-bit Gemma holds before each answer
"""
import json
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from llm_harness.checks import read_reply

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "results" / "llm"            # step 10's saved questions and replies; not in git
OUT = ROOT / "results" / "inside"           # step 11's results; not in git
MODEL = "google/gemma-3-4b-it"
RUNS = ["planted_tester", "planted_none", "real_none"]  # step 10's three runs
BUILD_MONTHS = pd.period_range("2017-01", "2018-12", freq="M")  # months 1-24: build here
SCORE_MONTHS = pd.period_range("2019-01", "2019-12", freq="M")  # months 25-36: score once
ANSWER_START = '"answer": "'


# ---------- the saved questions ----------

def load_run(run):
    """Step 10's saved records for one run, one row per question, with the reply's parts.

    Replies that can't be read are left out, as step 10 left them out.
    """
    rows = []
    for line in open(SAVED / f"feedback_{run}_indicators.jsonl", encoding="utf-8"):
        record = json.loads(line)
        if "market" not in record:  # the monthly notebook rewrites are saved here too
            continue
        parsed = read_reply(record["reply"])  # the harness's own reader
        if parsed is None:
            continue
        rows.append({"run": run, "market": record["market"], "day": pd.Timestamp(record["day"]),
                     "messages": record["request"]["messages"], "reply": record["reply"]["choices"][0]["message"]["content"],
                     "rule": parsed["rule"], "answer": parsed["answer"]})
    df = pd.DataFrame(rows)
    df["month"] = df["day"].dt.to_period("M")
    df["half"] = np.where(df["month"].isin(BUILD_MONTHS), "build", np.where(df["month"].isin(SCORE_MONTHS), "score", "other"))
    return df


def load_all_runs():
    return pd.concat([load_run(run) for run in RUNS], ignore_index=True)


# ---------- the model ----------

def load_gemma(bits=4):
    """Gemma 3 4B on the GPU, in 4-bit (NF4) or 8-bit. Full precision (8.6 GB) doesn't fit in 8 GB."""
    from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    if bits == 4:
        squeeze = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        squeeze = BitsAndBytesConfig(load_in_8bit=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, quantization_config=squeeze, dtype=torch.bfloat16,
                                                           device_map="cuda:0")
    model.eval()
    return model, AutoTokenizer.from_pretrained(MODEL)  # text only, so no image processor


def up_to_answer(tokenizer, row):
    """The question in Gemma's chat format, then its own reply up to (not including) the answer word."""
    prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    reply = row["reply"]
    return prompt + reply[:reply.index(ANSWER_START) + len(ANSWER_START)]


@torch.no_grad()
def next_word(model, tokenizer, row):
    """Which of "up" and "down" the model makes more likely next, and the gap between them (log-odds)."""
    ids = tokenizer(up_to_answer(tokenizer, row), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    logits = model(input_ids=ids).logits[0, -1].float()
    up, down = (tokenizer.encode(word, add_special_tokens=False)[0] for word in ("up", "down"))
    gap = (logits[up] - logits[down]).item()
    return ("up" if gap > 0 else "down"), gap


# ---------- stage 0 ----------

def reproduce(n=200, seed=0, bits=4):
    """Stage 0: replay n build-half questions, spread evenly over the three runs, and compare answers with llama.cpp's."""
    questions = load_all_runs()
    build = questions[questions["half"] == "build"]
    sample = pd.concat([g.sample(n // len(RUNS) + 1, random_state=seed) for _, g in build.groupby("run")]).head(n)
    model, tokenizer = load_gemma(bits)
    words = [next_word(model, tokenizer, row) for _, row in sample.iterrows()]
    sample = sample.assign(transformers=[w for w, _ in words], gap=[g for _, g in words])
    sample["same"] = sample["transformers"] == sample["answer"]
    OUT.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["messages", "reply"]).to_csv(OUT / ("stage0_reproduce.csv" if bits == 4 else f"stage0_reproduce_{bits}bit.csv"), index=False)
    print(f"same answer as llama.cpp: {100 * sample['same'].mean():.1f}% of {len(sample)} (gate: at least 90%)")
    print(sample.groupby("run")["same"].mean().mul(100).round(1).to_string())


# ---------- stage 1: what each layer holds just before the answer word ----------

STATED_MOVE = re.compile(r"(\d+)-day move \(day -\d+ to day -1\): (up|down)")  # "1-day move (day -2 to day -1): up"
FLIP = {"up": "down", "down": "up"}


def stated_moves(row):
    """The moves the question states, e.g. {1: "up", 5: "down", 20: "up"}."""
    return {int(n): word for n, word in STATED_MOVE.findall(row["messages"][-1]["content"])}


def rule_answer(row):
    """What its named rule says for this question. None for "none of these"."""
    direction, _, window = row["rule"].partition(" the ")  # "reverse the 1-day move" -> "reverse", "1-day move"
    if not window:
        return None
    move = stated_moves(row)[int(window.split("-")[0])]
    return move if direction == "follow" else FLIP[move]


@torch.no_grad()
def look_inside(model, tokenizer, row):
    """Every layer's activation at the last position, the average input embedding, and the model's own next word."""
    ids = tokenizer(up_to_answer(tokenizer, row), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    out = model(input_ids=ids, output_hidden_states=True, logits_to_keep=1)
    layers = torch.stack([h[0, -1] for h in out.hidden_states])  # layer 0 is the embedding layer
    words = out.hidden_states[0][0].mean(0)                      # knows which words appear, not their order
    up, down = (tokenizer.encode(word, add_special_tokens=False)[0] for word in ("up", "down"))
    gap = (out.logits[0, -1, up] - out.logits[0, -1, down]).item()
    return layers.half().cpu().numpy(), words.half().cpu().numpy(), ("up" if gap > 0 else "down"), gap


def save_activations():
    """Run all 3,888 questions through 8-bit Gemma once and save what it holds. About an hour."""
    questions = load_all_runs()
    model, tokenizer = load_gemma(bits=8)
    layers, words, own, gaps = [], [], [], []
    start = time.time()
    for i, (_, row) in enumerate(questions.iterrows()):
        a, w, word, gap = look_inside(model, tokenizer, row)
        layers.append(a), words.append(w), own.append(word), gaps.append(gap)
        if (i + 1) % 200 == 0:
            print(f"{i + 1} of {len(questions)}, {(time.time() - start) / 60:.0f} min")
    table = questions.drop(columns=["messages", "reply"]).assign(
        own=own, gap=gaps,
        move_1=[stated_moves(row)[1] for _, row in questions.iterrows()],
        rule_answer=[rule_answer(row) for _, row in questions.iterrows()])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "activations_8bit.csv", index=False)
    np.savez(OUT / "activations_8bit.npz", layers=np.stack(layers), words=np.stack(words))
    print(f"saved {len(table)} questions -> {OUT / 'activations_8bit.npz'}")


def fit_probe(X, y, months):
    """Standardize, then logistic regression with its strength picked by 5-fold cross-validation, folds split by month."""
    folds = list(GroupKFold(5).split(X, y, months))
    probe = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1], cv=folds,
                                                                 scoring="balanced_accuracy", max_iter=2000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # scikit-learn's notices about its next version
        return probe.fit(X, y)


# Each target: the column it reads, the runs it's trained on, and which score questions judge it.
PROBE_TARGETS = {
    "T1 its answer": ("own", RUNS, lambda t: t["half"] == "score"),
    "T2 the 1-day move": ("move_1", RUNS, lambda t: t["half"] == "score"),
    "T3 its rule's answer": ("rule_answer", ["planted_none", "real_none"],
                             lambda t: (t["half"] == "score") & t["rule_answer"].notna() & (t["rule_answer"] != t["move_1"])),
}


def probe_one(X, table, target, labels):
    """Train on build questions, return the balanced accuracy on the score questions that judge this target."""
    column, runs, judged = PROBE_TARGETS[target]
    build = (table["half"] == "build") & table["run"].isin(runs) & table[column].notna()
    y = table.loc[build, column].to_numpy()
    if labels == "shuffled":
        y = np.random.default_rng(0).permutation(y)
    probe = fit_probe(X[build.to_numpy()], y, table.loc[build, "month"].astype(str))
    score = judged(table).to_numpy()
    return balanced_accuracy_score(table.loc[score, column], probe.predict(X[score]))


def probes(jobs=4):
    """Every target, at every layer and on the average input embedding, with real and shuffled labels."""
    table = pd.read_csv(OUT / "activations_8bit.csv")
    saved = np.load(OUT / "activations_8bit.npz")
    inputs = {f"layer {i}": saved["layers"][:, i].astype(np.float32) for i in range(saved["layers"].shape[1])}
    inputs["average input embedding"] = saved["words"].astype(np.float32)
    keys = [(target, name, labels) for target in PROBE_TARGETS for name in inputs for labels in ("real", "shuffled")]
    scores = Parallel(n_jobs=jobs)(delayed(probe_one)(inputs[name], table, target, labels) for target, name, labels in keys)
    result = pd.DataFrame(keys, columns=["target", "input", "labels"]).assign(balanced_accuracy=scores)
    result.to_csv(OUT / "stage1_probes.csv", index=False)
    return result


if __name__ == "__main__":
    if "reproduce" in sys.argv:
        reproduce(bits=8 if "8" in sys.argv else 4)
    elif "activations" in sys.argv:
        save_activations()
    else:
        print(__doc__)
