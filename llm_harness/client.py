"""Talks to a model on NVIDIA NIM. Every question and reply is saved, so nothing is asked twice."""
import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
URL = "https://integrate.api.nvidia.com/v1/chat/completions"
SAVED = ROOT / "results" / "llm"   # not in git
SECONDS_BETWEEN_REQUESTS = 1.7     # about 35 a minute; NIM's free limit is reported as about 40
MAX_TRIES = 6

_lock = threading.Lock()
_next_turn = 0.0


@cache
def read_key():
    """The API key from .env. It is never printed, and never saved anywhere else."""
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "NVIDIA_API_KEY":
            return value.strip().strip("\"'")
    raise SystemExit("Put NVIDIA_API_KEY=... in the .env file")


def wait_for_turn():
    """Space the requests out, so several threads together stay under the rate limit."""
    global _next_turn
    with _lock:
        turn = max(time.time(), _next_turn)
        _next_turn = turn + SECONDS_BETWEEN_REQUESTS
    time.sleep(max(0.0, turn - time.time()))


def send(request):
    """One request to NIM. Rate limits, server errors and timeouts are retried."""
    body = json.dumps(request).encode()
    headers = {"Authorization": f"Bearer {read_key()}", "Content-Type": "application/json"}
    for attempt in range(MAX_TRIES):
        wait_for_turn()
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=body, headers=headers), timeout=120) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"NIM refused the request: {error.code} {error.read().decode()[:200]}")
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(10 * (attempt + 1))
    raise RuntimeError("NIM did not answer after several tries")


class Saved:
    """The replies of one run, kept in a file with one line per question."""

    def __init__(self, run_name):
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
            record = {"key": key, **labels, "request": request, "reply": send(request)}
            with _lock:
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record) + "\n")
                self.records[key] = record
        return self.records[key]
