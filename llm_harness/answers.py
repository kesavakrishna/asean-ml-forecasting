"""Reads the model's replies."""
import math


def up_probability(reply):
    """P(up), from the probabilities the model gave to the words "up" and "down". None if it used neither.

    The answer is read at the first position where "up" or "down" is the model's most likely word,
    so a leading space (Llama 2 starts with one) is skipped. "Up", "UP" and " up" all count as "up".
    """
    for position in reply["choices"][0]["logprobs"]["content"]:
        options = position["top_logprobs"]
        if options[0]["token"].strip().lower() not in ("up", "down"):
            continue
        chance = {"up": 0.0, "down": 0.0}
        for option in options:
            word = option["token"].strip().lower()
            if word in chance:
                chance[word] += math.exp(option["logprob"])
        return chance["up"] / (chance["up"] + chance["down"])
    return None


def last_word(reply):
    """The answer when the model was allowed to think first: the last "up" or "down" in its reply. None if neither."""
    text = (reply["choices"][0]["message"].get("content") or "").lower()
    found = max(text.rfind("up"), text.rfind("down"))
    if found < 0:
        return None
    return "up" if text.rfind("up") > text.rfind("down") else "down"
