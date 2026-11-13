"""Talks to a model, on NVIDIA NIM or on this laptop. Every question and reply is saved, so nothing is asked twice."""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LOCAL_URL = "http://127.0.0.1:8080/v1/chat/completions"  # llama.cpp's llama-server, see README.md
SAVED = ROOT / "results" / "llm"   # not in git
SECONDS_BETWEEN_REQUESTS = 2.4     # NIM only: 25 a minute. At 35 a minute NIM blocked us for 5 minutes
MAX_TRIES = 6
GIVE_UP_AFTER = 10                 # NIM only: "too many requests" this many times in a row, stop asking altogether

_lock = threading.Lock()
_next_turn = 0.0
_refusals_in_a_row = 0


class QuotaError(RuntimeError):
    """NIM keeps answering "too many requests". Carrying on would only make it worse."""


@cache
def read_key():
    """The NIM API key from .env. It is never printed, and never saved anywhere else."""
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "NVIDIA_API_KEY":
            return value.strip().strip("\"'")
    raise SystemExit("Put NVIDIA_API_KEY=... in the .env file")


def wait_for_turn():
    """Space NIM requests out, so several threads together stay under the rate limit."""
    global _next_turn
    with _lock:
        turn = max(time.time(), _next_turn)
        _next_turn = turn + SECONDS_BETWEEN_REQUESTS
    time.sleep(max(0.0, turn - time.time()))


def pause_everyone(seconds):
    """Hold back every thread's next NIM request."""
    global _next_turn
    with _lock:
        _next_turn = max(_next_turn, time.time() + seconds)


def send(request, url):
    """One request. Server errors and timeouts are retried.

    On NIM, "too many requests" pauses every thread for a minute. If it keeps happening, QuotaError stops the run.
    """
    global _refusals_in_a_row
    on_nim = url == NIM_URL
    body = json.dumps(request).encode()
    headers = {"Content-Type": "application/json"}
    if on_nim:
        headers["Authorization"] = f"Bearer {read_key()}"
    for attempt in range(MAX_TRIES):
        if on_nim:
            if _refusals_in_a_row >= GIVE_UP_AFTER:
                raise QuotaError("NIM keeps saying 'too many requests': the free quota may be used up")
            wait_for_turn()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=120) as response:
                _refusals_in_a_row = 0
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if on_nim and error.code == 429:
                _refusals_in_a_row += 1
                pause_everyone(60)
                continue
            if error.code not in (500, 502, 503, 504):
                raise RuntimeError(f"The model refused the request: {error.code} {error.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"No answer from {url} after several tries")


class Saved:
    """The replies of one run, kept in a file with one line per question."""

    def __init__(self, run_name, url):
        self.url = url
        self.path = SAVED / f"{run_name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.records[record["key"]] = record

    def ask(self, request, **labels):
        """Return the saved reply to this exact request, or send it and save the reply with its labels."""
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
        if key not in self.records:
            record = {"key": key, **labels, "request": request, "reply": send(request, self.url)}
            with _lock:
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                self.records[key] = record
        return self.records[key]
