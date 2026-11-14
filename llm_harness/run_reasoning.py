"""Test 2: when the model explains itself, is the explanation true?

Each question is asked four ways, all built from the same 20 days:
- base:     the table, rescaled so the first close is 100
- rescaled: the same table with the first close at 1000. Same information, so the answer should not change
- mirrored: the move the model's own rule names is turned upside down. If it really uses that rule, its answer should flip
- control:  a move it did not name is turned upside down instead. Its answer should flip less often than when mirrored

Run from the project folder, with the model's server running (see README.md):
    python -m llm_harness.run_reasoning pilot   # 2012-2016 days only
    python -m llm_harness.run_reasoning         # the real test
"""
import ctypes
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from data import MARKETS, load_market
from llm_harness import checks, explained, rules
from llm_harness.client import LOCAL_URL, Saved
from llm_harness.questions import days_before

MODEL = "gemma-3-4b-it"
PERIOD = ("2017-01-01", "2022-12-31")
PILOT = ("2012-03-01", "2016-12-31")
DAYS_PER_MARKET = 50
OTHER_WINDOW = {1: 5, 5: 19, 19: 1}  # for the control: a different move from the one the rule names


def pick_days(df, start, end, most, seed):
    """`most` random trading days in the period."""
    period = df.loc[start:end]
    days = period.index[period["ret"].notna()]
    chosen = np.sort(np.random.default_rng(seed).choice(len(days), size=min(most, len(days)), replace=False))
    return list(days[chosen])


def ask(saved, table, **labels):
    """Ask one explained question about this table. Returns the model's answer as a dict, or None if unreadable."""
    record = saved.ask(explained.request(MODEL, explained.question(table)), **labels)
    return checks.read_reply(record["reply"])


def one_day(saved, market, df, day):
    """All four versions of one question, and what the checks found."""
    rows = days_before(df, day)
    checks.check_wall(rows, day)  # before anything is sent
    labels = {"market": market, "day": f"{day:%Y-%m-%d}"}
    base_table = explained.rescaled(rows)
    base = ask(saved, base_table, **labels, version="base")
    if base is None:
        return {**labels, "readable": False}

    window = rules.parse(base["rule"])[1] if rules.parse(base["rule"]) else 1
    mirrored_table = explained.rescaled(rules.mirror(rows, window))
    control_table = explained.rescaled(rules.mirror(rows, OTHER_WINDOW[window]))
    rescaled = ask(saved, explained.rescaled(rows, base=1000), **labels, version="rescaled")
    mirrored = ask(saved, mirrored_table, **labels, version="mirrored")
    control = ask(saved, control_table, **labels, version="control")

    return {
        **labels,
        "readable": True,
        "up": bool(df.loc[day, "ret"] > 0),
        "answer": base["answer"],
        "p_up": explained.p_up(base),
        "rule": base["rule"],
        "rule's answer": rules.implied_answer(base_table, base["rule"]),
        "numbers cited": len(base["evidence"]),
        "numbers wrong": checks.wrong_citations(base["evidence"], base_table),
        "same when rescaled": None if rescaled is None else rescaled["answer"] == base["answer"],
        "flips when mirrored": None if mirrored is None else mirrored["answer"] != base["answer"],
        "flips in control": None if control is None else control["answer"] != base["answer"],
    }


def main(pilot):
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))  # no idle sleep while this runs
    name = "reasoning_pilot" if pilot else "reasoning"
    saved = Saved(name, LOCAL_URL)
    start, end = PILOT if pilot else PERIOD
    jobs = []
    for seed, market in enumerate(MARKETS):
        df = load_market(market)  # development period only
        jobs += [(market, df, day) for day in pick_days(df, start, end, 5 if pilot else DAYS_PER_MARKET, seed)]
    print(f"{len(jobs)} questions, asked up to 4 ways each", flush=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda job: one_day(saved, *job), jobs))
    table = pd.DataFrame(results)
    table.to_csv(saved.path.with_suffix(".csv"), index=False)
    print(f"done: {table['readable'].sum()} of {len(table)} readable. Table saved to {saved.path.with_suffix('.csv')}", flush=True)


if __name__ == "__main__":
    main(pilot="pilot" in sys.argv)
