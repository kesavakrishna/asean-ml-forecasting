"""Reads the model's replies."""
import math


def up_probability(reply):
    """P(up), from the probabilities the model gave to the words "up" and "down". None if it used neither.

    With thinking off, the first word of the reply is the answer, and NIM returns the five likeliest words there.
    """
    for position in reply["choices"][0]["logprobs"]["content"]:
        chance = {"up": 0.0, "down": 0.0}
        for option in position["top_logprobs"]:
            word = option["token"].strip().lower()
            if word in chance:
                chance[word] += math.exp(option["logprob"])
        if chance["up"] + chance["down"] > 0:
            return chance["up"] / (chance["up"] + chance["down"])
    return None


def last_word(reply):
    """The answer when the model was allowed to think first: the last "up" or "down" in its reply. None if neither."""
    text = (reply["choices"][0]["message"].get("content") or "").lower()
    found = max(text.rfind("up"), text.rfind("down"))
    if found < 0:
        return None
    return "up" if text.rfind("up") > text.rfind("down") else "down"
