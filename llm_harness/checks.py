"""Checks run around every call (the "hooks"): one before a question is sent, one after the reply comes back."""
import json

import pandas as pd

from llm_harness.questions import DAYS_SHOWN

FIELDS = ["open", "high", "low", "close"]


def check_wall(rows, day):
    """Before sending: the model may only see the 20 trading days before `day`. Stops the run if not."""
    assert len(rows) == DAYS_SHOWN, f"expected {DAYS_SHOWN} days, got {len(rows)}"
    assert rows.index.max() < pd.Timestamp(day), f"a question about {day} would show {rows.index.max():%Y-%m-%d}"


def read_reply(reply):
    """After the reply: the model's JSON as a dict, or None if it can't be read."""
    try:
        answer = json.loads(reply["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, TypeError, KeyError):
        return None
    if answer.get("answer") not in ("up", "down"):
        return None
    return answer


def wrong_citations(evidence, table):
    """How many of the numbers the model cited don't match the table it was shown (to the 2 decimals shown)."""
    wrong = 0
    for item in evidence:
        position = DAYS_SHOWN + item["day"]  # day -1 is the last row
        shown = round(float(table[item["field"].capitalize()].iloc[position]), 2)
        if abs(shown - float(item["value"])) > 0.011:
            wrong += 1
    return wrong
