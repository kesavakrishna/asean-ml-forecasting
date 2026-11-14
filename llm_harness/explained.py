"""Questions where the model must explain itself (tests 2 and 3): an answer, numbers from the table, and a rule.

The server forces the reply into JSON with this shape, so every part can be checked.
"""
from llm_harness.questions import DAYS_SHOWN, as_text
from llm_harness.rules import RULES

SYSTEM = "You are taking part in a forecasting experiment. Reply only with JSON in the format asked for."

# Order matters: the model writes its evidence and rule first, then its answer.
SCHEMA = {
    "type": "object",
    "properties": {
        "evidence": {
            "type": "array", "minItems": 1, "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer", "minimum": -DAYS_SHOWN, "maximum": -1},
                    "field": {"enum": ["open", "high", "low", "close"]},
                    "value": {"type": "number"},
                },
                "required": ["day", "field", "value"],
            },
        },
        "rule": {"enum": RULES},
        "answer": {"enum": ["up", "down"]},
        "confidence": {"type": "integer", "minimum": 50, "maximum": 100},
    },
    "required": ["evidence", "rule", "answer", "confidence"],
}

INSTRUCTIONS = f"""Will it close higher or lower on the next trading day than on day -1?
Reply with JSON:
- "evidence": the one to three numbers from the table that matter most to your answer, each with its day, field and value, copied exactly.
- "rule": the rule you used, one of: {", ".join(f'"{rule}"' for rule in RULES)}. "Follow" means you expect the move to continue; "reverse" means you expect it to turn.
- "answer": "up" or "down".
- "confidence": how sure you are, from 50 (a guess) to 100 (certain)."""


def rescaled(rows, base=100):
    """The rows rescaled so the first close is `base`. This is the table the model sees."""
    return rows * base / rows["Close"].iloc[0]


def question(table, notebook=None, extra=None):
    """The question text: the model's notebook if it has one, the table, any extra lines (step 10), and the instructions."""
    parts = []
    if notebook is not None:
        parts.append(f"Your notebook of rules, written by you after earlier months:\n{notebook}\n")
    base = round(float(table["Close"].iloc[0]))
    parts.append("Market: a stock index (name and dates hidden)\n"
                 f"Its last {DAYS_SHOWN} trading days (day, open, high, low, close), rescaled so the first close is {base}:\n"
                 f"{as_text(table, [f'day {n}' for n in range(-DAYS_SHOWN, 0)])}\n")
    if extra is not None:
        parts.append(extra + "\n")
    parts.append(INSTRUCTIONS)
    return "\n".join(parts)


def request(model_name, text):
    """A request whose reply must follow SCHEMA."""
    return {
        "model": model_name,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_schema", "json_schema": {"name": "forecast", "schema": SCHEMA}},
    }


def p_up(answer):
    """The model's own P(up), from its answer and confidence: up at 70 -> 0.70, down at 70 -> 0.30."""
    sure = answer["confidence"] / 100
    return sure if answer["answer"] == "up" else 1 - sure
