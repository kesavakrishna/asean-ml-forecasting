"""Step 10: test 3 again, with the moves and Kim's 12 indicators written below the table. See PLAN_more_inputs.md.

The new lines are computed from the 20 days shown only, so the wall doesn't change. The wording here is frozen
with the plan. Test 3's files and saved answers don't change.

Run from the project folder, with the Gemma server running (see README.md):
    python -m llm_harness.run_with_indicators pilot   # 3 months of 2016 data, not scored
    python -m llm_harness.run_with_indicators         # the three runs
"""
import ctypes
import math
import os
import sys

from features import indicator_values
from llm_harness.rules import WINDOWS
from llm_harness.run_feedback import MONTHS, PILOT_MONTHS, run

SUFFIX = "_indicators"
RUNS = [("planted", "tester"), ("planted", "none"), ("real", "none")]
NAMES = {  # column in features.py -> how the model sees it
    "stoch_k": "stochastic %K", "stoch_d": "stochastic %D", "slow_d": "slow %D",
    "momentum": "momentum (4 days)", "roc": "rate of change (10 days)", "williams_r": "Williams %R",
    "ad_osc": "A/D oscillator", "disparity5": "5-day disparity", "disparity10": "10-day disparity",
    "oscp": "price oscillator (%)", "cci": "CCI", "rsi": "RSI (10 days)",
}


def extra_lines(table):
    """The two new lines: the direction of each move a rule can use, and the indicators on day -1."""
    close = table["Close"]
    moves = ", ".join(f"{name} (day {-1 - n} to day -1): {'up' if close.iloc[-1] > close.iloc[-1 - n] else 'down'}"
                      for name, n in WINDOWS.items())
    values = indicator_values(table).iloc[-1]
    values["oscp"] *= 100  # a small fraction; in % it survives rounding
    shown = ", ".join(f"{label} {'n/a' if math.isnan(values[c]) else f'{values[c]:.2f}'}" for c, label in NAMES.items())
    return f"Moves to day -1: {moves}.\nIndicators on day -1, computed from the table: {shown}."


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(ctypes.c_uint(0x80000001))  # no idle sleep while this runs
    if "pilot" in sys.argv:
        run("real", "tester", PILOT_MONTHS, extra_lines, SUFFIX)
    else:
        for dataset, kind in RUNS:
            run(dataset, kind, MONTHS, extra_lines, SUFFIX)
