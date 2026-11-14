"""Test 3: does a notebook of rules, rewritten after a monthly scorecard, make the model better?

Each month the model answers questions with its notebook in front of it. At month end it gets a scorecard
built only from outcomes now known, and rewrites the notebook. The next month uses the new notebook.

Three kinds of feedback, each its own run:
- none:      the twin. The notebook stays empty and is never rewritten.
- scorecard: the scorecard only.
- tester:    the scorecard, plus how each rule on the menu did on every earlier day (the rule tester).

Three kinds of data: the real markets (name and dates hidden), random made-up markets where nothing can be
learned, and made-up markets with a planted rule ("reverse the 1-day move" is right about 60% of the time).

Run from the project folder, with the model's server running (see README.md):
    python -m llm_harness.run_feedback pilot          # 3 months of 2016 data
    python -m llm_harness.run_feedback real tester    # one run; any dataset and feedback kind

Each run saves its state after every month, so a stopped run carries on where it left off.
"""
import ctypes
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from data import MARKETS, load_market
from llm_harness import checks, explained, fake_markets, rules
from llm_harness.client import LOCAL_URL, SAVED, Saved
from llm_harness.questions import days_before

MODEL = "gemma-3-4b-it"
MONTHS = pd.period_range("2017-01", "2019-12", freq="M")
PILOT_MONTHS = pd.period_range("2016-01", "2016-03", freq="M")
DAYS_PER_MARKET = 6  # each month: 6 days x 6 markets = 36 questions
EMPTY_NOTEBOOK = "(empty: no rules written yet)"
MOST_WORDS = 150
REWRITE_SYSTEM = "You are taking part in a forecasting experiment."


def load(dataset):
    """Six markets as {name: DataFrame}: the real ones (development period only) or made-up ones."""
    if dataset == "real":
        return {market: load_market(market) for market in MARKETS}
    return fake_markets.markets(dataset)


def days_in(month, markets):
    """Six random trading days per market in this month, the same for every run."""
    seed = int(month.strftime("%Y%m"))
    chosen = []
    for number, (market, df) in enumerate(markets.items()):
        in_month = df.index[(df.index.to_period("M") == month) & df["ret"].notna()]
        picks = np.random.default_rng(seed + number).choice(len(in_month), size=min(DAYS_PER_MARKET, len(in_month)), replace=False)
        chosen += [(market, day) for day in in_month[np.sort(picks)]]
    return chosen


def predict(saved, markets, market, day, notebook, run):
    """One question, with the notebook in front of the model. Returns what happened."""
    rows = days_before(markets[market], day)
    checks.check_wall(rows, day)  # before anything is sent
    table = explained.rescaled(rows)
    request = explained.request(MODEL, explained.question(table, notebook))
    answer = checks.read_reply(saved.ask(request, run=run, market=market, day=f"{day:%Y-%m-%d}")["reply"])
    if answer is None:
        return None
    return {"market": market, "day": f"{day:%Y-%m-%d}", "answer": answer["answer"], "p_up": explained.p_up(answer),
            "rule": answer["rule"], "up": bool(markets[market].loc[day, "ret"] > 0),
            "numbers wrong": checks.wrong_citations(answer["evidence"], table)}


def scorecard(month, results, markets, kind):
    """The month's scorecard, in plain words. Built only from outcomes known by the end of the month."""
    right = [r["answer"] == ("up" if r["up"] else "down") for r in results]
    always_up = np.mean([r["up"] for r in results])
    lines = [f"Results for month {month}: {sum(right)} right out of {len(results)} ({100 * np.mean(right):.0f}%). "
             f"Saying \"up\" every time would have got {100 * always_up:.0f}%."]
    lines.append("How each rule did when you used it this month:")
    for rule in rules.RULES:
        used = [ok for r, ok in zip(results, right) if r["rule"] == rule]
        if used:
            lines.append(f"- {rule}: used {len(used)} times, right {sum(used)}.")
    sure = [ok for r, ok in zip(results, right) if abs(r["p_up"] - 0.5) >= 0.3]
    unsure = [ok for r, ok in zip(results, right) if abs(r["p_up"] - 0.5) < 0.3]
    if sure:
        lines.append(f"When your confidence was 80 or more you were right {100 * np.mean(sure):.0f}% of the time ({len(sure)} answers).")
    if unsure:
        lines.append(f"When it was below 80 you were right {100 * np.mean(unsure):.0f}% of the time ({len(unsure)} answers).")
    wrong_numbers = sum(r["numbers wrong"] for r in results)
    lines.append(f"Numbers you cited that didn't match the table: {wrong_numbers}.")
    if kind == "tester":
        month_end = (month + 1).to_timestamp()
        lines.append("Rule tester: how each rule would have done on every earlier day in these markets:")
        for rule in rules.RULES[:-1]:
            rate = np.mean([rules.hit_rate(df, rule, month_end) for df in markets.values()])
            lines.append(f"- {rule}: right {100 * rate:.1f}% of the time.")
    return "\n".join(lines)


def rewrite(saved, notebook, card, run, month):
    """The model's new notebook, at most MOST_WORDS words."""
    text = (f"You keep a notebook of rules for forecasting whether a stock index closes up or down the next day.\n\n"
            f"Your notebook so far:\n{notebook}\n\n{card}\n\n"
            f"Write your new notebook: the rules you now believe, in at most {MOST_WORDS} words. "
            "Keep what worked and change what didn't. Reply with the notebook only.")
    request = {"model": MODEL, "temperature": 0, "max_tokens": 400,
               "messages": [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": text}]}
    reply = saved.ask(request, run=run, month=str(month), step="rewrite")["reply"]
    words = (reply["choices"][0]["message"]["content"] or "").split()
    return " ".join(words[:MOST_WORDS]) or EMPTY_NOTEBOOK


def run(dataset, kind, months):
    """One run of the loop. Its state is saved after every month."""
    name = f"feedback_{dataset}_{kind}" + ("_pilot" if months is PILOT_MONTHS else "")
    saved = Saved(name, LOCAL_URL)
    state_path = SAVED / f"{name}_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"months": []}
    markets = load(dataset)
    done = {m["month"] for m in state["months"]}

    for month in months:
        if str(month) in done:
            continue
        notebook = state["months"][-1]["notebook after"] if state["months"] and kind != "none" else EMPTY_NOTEBOOK
        jobs = days_in(month, markets)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = [r for r in pool.map(lambda job: predict(saved, markets, *job, notebook, name), jobs) if r]
        card = scorecard(month, results, markets, kind)
        new_notebook = notebook if kind == "none" else rewrite(saved, notebook, card, name, month)
        state["months"].append({"month": str(month), "notebook used": notebook, "results": results,
                                "scorecard": card, "notebook after": new_notebook})
        state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")  # durable: a restart continues from here
        right = np.mean([r["answer"] == ("up" if r["up"] else "down") for r in results])
        print(f"{name} {month}: {len(results)} answers, {100 * right:.0f}% right", flush=True)
    return state


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))  # no idle sleep while this runs
    if "pilot" in sys.argv:
        run("real", "tester", PILOT_MONTHS)
    else:
        run(sys.argv[1], sys.argv[2], MONTHS)
