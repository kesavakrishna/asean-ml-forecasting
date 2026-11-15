"""
Step 11: looking inside Gemma (see PLAN_11.md). Runs in venv_gpu, which has GPU PyTorch.

Every question and reply from step 10 is saved, so a question can be replayed exactly:
the model is fed the question and its own reply up to the answer word, and we read what it does there.

Run:  venv_gpu/Scripts/python llm_harness/inside.py reproduce   # stage 0: does transformers agree with llama.cpp?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

def load_gemma():
    """Gemma 3 4B in 4-bit (NF4), on the GPU. Full precision (8.6 GB) doesn't fit in 8 GB."""
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    four_bit = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = Gemma3ForConditionalGeneration.from_pretrained(MODEL, quantization_config=four_bit, device_map="cuda:0")
    model.eval()
    return model, AutoProcessor.from_pretrained(MODEL).tokenizer


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

def reproduce(n=200, seed=0):
    """Stage 0: replay n build-half questions, spread evenly over the three runs, and compare answers with llama.cpp's."""
    questions = load_all_runs()
    build = questions[questions["half"] == "build"]
    sample = build.groupby("run", group_keys=False).apply(lambda g: g.sample(n // len(RUNS) + 1, random_state=seed)).head(n)
    model, tokenizer = load_gemma()
    words = [next_word(model, tokenizer, row) for _, row in sample.iterrows()]
    sample = sample.assign(transformers=[w for w, _ in words], gap=[g for _, g in words])
    sample["same"] = sample["transformers"] == sample["answer"]
    OUT.mkdir(parents=True, exist_ok=True)
    sample.drop(columns=["messages", "reply"]).to_csv(OUT / "stage0_reproduce.csv", index=False)
    print(f"same answer as llama.cpp: {100 * sample['same'].mean():.1f}% of {len(sample)} (gate: at least 90%)")
    print(sample.groupby("run")["same"].mean().mul(100).round(1).to_string())


if __name__ == "__main__":
    if "reproduce" in sys.argv:
        reproduce()
    else:
        print(__doc__)
