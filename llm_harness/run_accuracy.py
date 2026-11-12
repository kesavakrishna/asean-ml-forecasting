"""Test 1: ask the model for the next day's direction, three ways.

Run from the project folder:
    python -m llm_harness.run_accuracy pilot   # 2012-2016 days only, to settle the wording
    python -m llm_harness.run_accuracy         # the real test (about 5 hours)

Every reply is saved in results/llm/, so a stopped run carries on where it left off.
"""
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from data import MARKETS, clean, load_market
from llm_harness import questions
from llm_harness.client import Saved
from make_fake_data import fake_prices

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
# From its model card: the main training data ends in Sep 2025, and the fine-tuning data in May 2026.
PERIODS = {"before": ("2017-01-01", "2025-08-31"), "in between": ("2025-09-01", "2026-05-31"), "after": ("2026-06-01", "2026-10-23")}
PILOT = {"pilot": ("2012-03-01", "2016-12-31")}
FAKE = {"fake data": ("2012-03-01", "2022-12-31")}
MODES = {"named": questions.named, "hidden": questions.hidden, "date only": questions.date_only}


def request(question, thinking=False):
    """The request for one question. With thinking off, the first word is the answer and comes with its probability."""
    settings = {"max_tokens": 4000} if thinking else {"max_tokens": 8, "logprobs": True, "top_logprobs": 5}
    return {
        "model": MODEL,
        "messages": [{"role": "system", "content": questions.SYSTEM}, {"role": "user", "content": question}],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
        **settings,
    }


def pick_days(df, start, end, most, seed):
    """Every trading day in the period, or a random `most` of them if there are more."""
    period = df.loc[start:end]
    days = period.index[period["ret"].notna()]
    if len(days) > most:
        days = days[np.sort(np.random.default_rng(seed).choice(len(days), size=most, replace=False))]
    return list(days)


def build_tasks(pilot):
    """One task per market, day and way of asking."""
    tasks = []
    periods, most, most_thinking = (PILOT, 10, 2) if pilot else (PERIODS, 300, 50)
    for seed, market in enumerate(MARKETS):
        df = load_market(market, include_holdout=True)
        for period, (start, end) in periods.items():
            for number, day in enumerate(pick_days(df, start, end, most, seed)):
                for mode, ask in MODES.items():
                    tasks.append({"market": market, "day": f"{day:%Y-%m-%d}", "period": period, "mode": mode,
                                  "request": request(ask(market, df, day))})
                if number < most_thinking and period != "in between":  # a side check: does thinking first help it remember?
                    tasks.append({"market": market, "day": f"{day:%Y-%m-%d}", "period": period, "mode": "date only, thinking",
                                  "request": request(questions.date_only(market, df, day), thinking=True)})

    # Made-up random prices: nothing can be predicted, so the score here must be about 50%.
    fake = clean(fake_prices(seed=99), "fake")
    for period, (start, end) in FAKE.items():
        for day in pick_days(fake, start, end, 10 if pilot else 300, seed=99):
            tasks.append({"market": "fake", "day": f"{day:%Y-%m-%d}", "period": period, "mode": "hidden",
                          "request": request(questions.hidden("fake", fake, day))})
    return tasks


def main(pilot):
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))  # no idle sleep while this runs
    saved = Saved("accuracy_pilot" if pilot else "accuracy")
    tasks = build_tasks(pilot)
    print(f"{len(tasks)} questions, {len(saved.records)} already answered", flush=True)

    def answer(task):
        """True if the question now has a saved reply. A failed question is left for the next run."""
        labels = {name: value for name, value in task.items() if name != "request"}
        try:
            saved.ask(task["request"], **labels)
            return True
        except RuntimeError as error:
            print("failed:", task["market"], task["day"], task["mode"], "|", error, flush=True)
            return False

    answered = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for count, ok in enumerate(pool.map(answer, tasks), start=1):
            answered += ok
            if count % 250 == 0:
                print(f"{count} of {len(tasks)}", flush=True)
    print(f"done: {answered} of {len(tasks)} answered. Run again to retry any that failed.", flush=True)


if __name__ == "__main__":
    main(pilot="pilot" in sys.argv)
