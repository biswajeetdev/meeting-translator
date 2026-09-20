#!/usr/bin/env python3
"""Translate a live transcript stream and serve it to a teleprompter.

Reads whisper-stream's stdout on our stdin, one finalized utterance per line,
translates each into the target language, and pushes it to the teleprompter.

    whisper-stream ... | translator.py --target English --port 8765

Design notes that are not obvious:

* Translation runs on ONE worker thread, not a pool. Utterances must reach the
  teleprompter in the order they were spoken, and a pool reorders them whenever
  one call is slower than the next. At ~0.4s per call against Groq, a single
  worker keeps up with human speech comfortably.
* Each call carries the last few utterances as context. Without it the model
  translates every sentence cold, and pronouns, callbacks and running jokes
  all flatten out -- the exact things a literal translation loses.
* The model returns a translation AND an optional note, for when meaning is
  carried by something other than the words: idiom, sarcasm, indirect refusal,
  politeness convention. Ordinary sentences get no note, or the notes become
  noise and the reader stops seeing them.
* Groq sits behind Cloudflare, which rejects Python's default urllib
  User-Agent with error 1010. The UA below is required, not decoration.
* Every utterance is appended to a JSONL transcript as it happens, so a crashed
  browser or a closed laptop lid never costs the meeting record.
"""
import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
FAST_MODEL = "qwen/qwen3.8-27b"          # ~0.4s, strong multilingual
REASONING_MODEL = "openai/gpt-oss-120b"  # slower, better on implication

# whisper-stream speaks two different dialects and we have to read both.
#
# Sliding-window mode (--step N) redraws one line using ANSI erase codes.
# VAD mode (--step 0) emits blocks instead:
#
#     ### Transcription 4 START | t0 = 0 ms | t1 = 5648 ms
#     [00:00:00.000 --> 00:00:29.980]   Buenos días a todos.
#     ### Transcription 4 END
#
# so the timestamp prefix has to come off or it gets translated as if spoken.
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[2K")
STAMP = re.compile(r"^\[\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}\]\s*")
NOISE = re.compile(r"^\s*[\[(*].*[\])*]\s*$")
SKIP_PREFIXES = ("init:", "main:", "whisper_", "ggml_", "load_backend",
                 "[Start speaking]", "###")

SYSTEM = """You are interpreting a live meeting into {target}.

You receive the recent conversation for context, then one new utterance. \
Translate ONLY that new utterance.

Return strict JSON, nothing else:
{{"text": "<the translation>", "note": "<short gloss>" or null}}

Rules for "text":
- Natural {target}, as a skilled interpreter would say it aloud. Not word-for-word.
- Speech recognition makes mistakes. If a word is garbled, translate the most \
likely intended word rather than the noise.
- If the utterance is already in {target}, return it cleaned up but unchanged in meaning.

Rules for "note" — this is what a literal translation loses. Set it ONLY when one \
of these is genuinely present, and keep it under 12 words:
- an idiom or figure of speech whose literal sense is misleading
- sarcasm, irony, or a joke
- indirectness: a soft "no", a hedge, a face-saving evasion, a veiled objection
- a politeness or seniority convention that carries weight in the source culture
- an implied request or commitment that is not stated outright
- a pun or wordplay that cannot survive translation
Otherwise "note" MUST be null. Most utterances get null. Do not explain ordinary \
sentences; if every line has a note, the notes are worthless."""

STATE = {"segments": [], "target": "English"}
SUBSCRIBERS: "list[queue.Queue]" = []
LOCK = threading.Lock()


def clean(line: str) -> str:
    """Strip whisper-stream's decoration; return '' for anything not speech."""
    line = ANSI.sub(" ", line).strip()
    if not line or line.startswith(SKIP_PREFIXES):
        return ""
    line = STAMP.sub("", line).strip()
    if not line or NOISE.match(line):
        return ""
    return line


SENTENCE_END = tuple(".!?…。！？")


class LocalAgreement:
    """Commit only the words two successive passes agree on.

    In VAD mode whisper-stream re-transcribes a *growing* buffer, so the same
    sentence arrives again every couple of seconds, each time a little longer,
    until the pause that flushes it. Dropping exact repeats (what this used to do)
    handles the stuck-hallucination case but not the growing one: "Buenos días"
    then "Buenos días a todos" are different strings, so both went through and the
    opening got translated twice.

    LocalAgreement is the standard fix, from Whisper-Streaming (Machácek et al.,
    2023) and used by WhisperLiveKit. A word is only emitted once two consecutive
    hypotheses put it in the same position. The newest, least certain tail is held
    back until the next pass confirms it, so output is append-only: nothing already
    shown is ever retracted or repeated.

    It absorbs the silence hallucination for free — ten identical "Thank you."
    passes agree, commit once, and then have nothing left to add.
    """

    def __init__(self) -> None:
        self.committed: "list[str]" = []
        self.prev: "list[str]" = []

    def reset(self) -> None:
        self.committed = []
        self.prev = []

    def insert(self, text: str) -> "list[str]":
        """Feed one hypothesis; get back only the words newly agreed upon."""
        words = text.split()
        # If the new hypothesis no longer starts with what we already emitted, the
        # decoder flushed its buffer and this is a different utterance.
        if words[:len(self.committed)] != self.committed:
            self.reset()
        i = len(self.committed)
        fresh: "list[str]" = []
        while i < len(words) and i < len(self.prev) and words[i] == self.prev[i]:
            fresh.append(words[i])
            i += 1
        self.committed = words[:i]
        self.prev = words
        return fresh


class Utterances:
    """Turn agreed words into whole sentences worth translating.

    LocalAgreement commits a few words at a time. Translating two-word fragments
    produces nonsense and burns a call each, so words accumulate here and flush on
    a sentence ending, on a buffer reset, or when the sentence runs too long to
    keep waiting.
    """

    def __init__(self, max_words: int = 28) -> None:
        self.pending: "list[str]" = []
        self.max_words = max_words

    def add(self, words: "list[str]") -> "list[str]":
        out = []
        for w in words:
            self.pending.append(w)
            if w.endswith(SENTENCE_END) or len(self.pending) >= self.max_words:
                out.append(" ".join(self.pending))
                self.pending = []
        return out

    def flush(self) -> "list[str]":
        if not self.pending:
            return []
        out = [" ".join(self.pending)]
        self.pending = []
        return out


def parse_reply(raw: str, fallback: str) -> "tuple[str, str | None]":
    """Pull {text, note} out of the model's reply, tolerating stray prose."""
    body = raw.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-z]*\n?|\n?```$", "", body).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", body, re.S)
        if not m:
            # Model ignored the format and just translated. Still useful.
            return (body or fallback), None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return (body or fallback), None
    if not isinstance(obj, dict):
        return (body or fallback), None
    text = str(obj.get("text") or "").strip() or fallback
    note = obj.get("note")
    note = str(note).strip() if note not in (None, "", "null") else None
    return text, note


def translate(text: str, context: "deque", args, key: str) -> "tuple[str, str | None]":
    messages = [{"role": "system", "content": SYSTEM.format(target=args.target)}]
    if context:
        recent = "\n".join(f"- {c}" for c in context)
        messages.append({
            "role": "user",
            "content": f"Recent conversation, for context only — do not translate it:\n{recent}",
        })
        messages.append({"role": "assistant", "content": "Understood. Send the new utterance."})
    messages.append({"role": "user", "content": text})

    body = {"model": args.model, "temperature": 0, "max_tokens": 600, "messages": messages}
    if args.reasoning:
        body["reasoning_effort"] = "low"
    req = urllib.request.Request(
        GROQ_URL,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Required: Cloudflare 1010-blocks the default urllib UA.
            "User-Agent": "meeting-translator/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        data = json.load(resp)
    return parse_reply(data["choices"][0]["message"]["content"], text)


# Target language -> locale codes to look for in `say -v '?'`. Only the ones
# worth guessing; anything else is handled by passing --voice explicitly.
VOICE_LOCALES = {
    "english": ("en_US", "en_GB"), "indonesian": ("id_ID",), "spanish": ("es_MX", "es_ES"),
    "french": ("fr_FR", "fr_CA"), "german": ("de_DE",), "italian": ("it_IT",),
    "portuguese": ("pt_BR", "pt_PT"), "dutch": ("nl_NL",), "japanese": ("ja_JP",),
    "korean": ("ko_KR",), "chinese": ("zh_CN", "zh_TW"), "mandarin": ("zh_CN",),
    "hindi": ("hi_IN",), "arabic": ("ar_001", "ar_SA"), "russian": ("ru_RU",),
    "turkish": ("tr_TR",), "thai": ("th_TH",), "vietnamese": ("vi_VN",),
    "polish": ("pl_PL",), "swedish": ("sv_SE",), "malay": ("ms_MY",),
}

# macOS ships novelty voices (Albert, Bad News, Bubbles, Boing) that sort ahead of
# the usable ones alphabetically, so scanning by locale alone hands you a cartoon
# reading your meeting. These are the natural-sounding default per locale.
PREFERRED = {
    "en_US": "Samantha", "en_GB": "Daniel", "id_ID": "Damayanti", "es_MX": "Paulina",
    "es_ES": "Monica", "fr_FR": "Thomas", "de_DE": "Anna", "it_IT": "Alice",
    "pt_BR": "Luciana", "nl_NL": "Xander", "ja_JP": "Kyoko", "ko_KR": "Yuna",
    "zh_CN": "Tingting", "hi_IN": "Lekha", "ru_RU": "Milena",
}

# "Eddy (Spanish (Spain))   es_ES    # ..." -- the name carries nested parentheses,
# so the locale column is the only reliable anchor.
VOICE_LINE = re.compile(r"^(?P<name>.+?)\s+(?P<loc>[a-z]{2}[-_][A-Z0-9]{2,3})\s+#")


def resolve_voice(target: str) -> "str | None":
    """Find a macOS voice that actually speaks the target language.

    Saying Indonesian text with an English voice produces confident gibberish, so
    a missing voice is reported rather than silently substituted. The full voice
    name matters: several voices exist once per locale ("Eddy (Japanese (Japan))"),
    and passing the bare "Eddy" gets you whichever one sorts first -- English.
    """
    locales = VOICE_LOCALES.get(target.strip().lower())
    if not locales:
        return None
    try:
        listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None

    by_locale: "dict[str, list[str]]" = {}
    for line in listing.splitlines():
        m = VOICE_LINE.match(line)
        if m:
            by_locale.setdefault(m.group("loc").replace("-", "_"), []).append(m.group("name").strip())

    for loc in locales:
        names = by_locale.get(loc)
        if not names:
            continue
        want = PREFERRED.get(loc)
        if want and want in names:
            return want
        # Otherwise prefer a voice dedicated to this locale (a plain name) over a
        # multi-locale one, whose parenthesised form is longer and less predictable.
        return sorted(names, key=lambda n: ("(" in n, len(n)))[0]
    return None


def speak_worker(q: "queue.Queue", args) -> None:
    """Speak translations aloud, skipping any that the meeting has outrun.

    Synthesis happens in real time, so a busy meeting queues faster than it can
    be spoken. An interpreter who is ninety seconds behind is worse than one who
    misses a line, so anything older than --speak-lag is dropped rather than
    played late.
    """
    while True:
        item = q.get()
        if item is None:
            return
        when, text = item
        if time.time() - when > args.speak_lag:
            continue
        cmd = ["say"]
        if args.voice:
            cmd += ["-v", args.voice]
        if args.speak_device:
            # Target the speakers directly. The system output is a Multi-Output
            # device feeding BlackHole, so speaking through it would put our own
            # voice back into the transcriber and translate it forever.
            cmd += ["-a", args.speak_device]
        if args.speak_rate:
            cmd += ["-r", str(args.speak_rate)]
        cmd.append(text)
        try:
            subprocess.run(cmd, timeout=90, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def publish(segment: dict) -> None:
    with LOCK:
        STATE["segments"].append(segment)
        dead = []
        for q in SUBSCRIBERS:
            try:
                q.put_nowait(segment)
            except queue.Full:
                dead.append(q)
        for q in dead:
            SUBSCRIBERS.remove(q)


def worker(jobs: "queue.Queue", args, key, log_path: Path, speech: "queue.Queue | None") -> None:
    context: deque = deque(maxlen=args.context)
    while True:
        item = jobs.get()
        if item is None:
            jobs.task_done()
            return
        idx, source = item
        note = None
        try:
            text, note = translate(source, context, args, key)
            err = None
        except urllib.error.HTTPError as e:
            text, err = source, f"groq http {e.code}"
        except Exception as e:  # network blip: show the source rather than nothing
            text, err = source, str(e)[:80]
        context.append(text if not err else source)
        seg = {
            "i": idx,
            "t": time.strftime("%H:%M:%S"),
            "source": source,
            "text": text,
            "note": note,
            "error": err,
        }
        publish(seg)
        if speech is not None and not err:
            speech.put((time.time(), text))
        with log_path.open("a") as fh:
            fh.write(json.dumps(seg, ensure_ascii=False) + "\n")
        marker = "!" if err else ">"
        tail = f"   [{note}]" if note else ""
        print(f"{marker} {seg['t']}  {text}{tail}", file=sys.stderr, flush=True)
        jobs.task_done()


class Handler(BaseHTTPRequestHandler):
    page = b""

    def log_message(self, *a):  # keep the console clean for the transcript
        pass

    def do_GET(self):
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=1000)
            with LOCK:
                backlog = list(STATE["segments"])
                SUBSCRIBERS.append(q)
            try:
                for seg in backlog:
                    self._send(seg)
                while True:
                    try:
                        self._send(q.get(timeout=15))
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with LOCK:
                    if q in SUBSCRIBERS:
                        SUBSCRIBERS.remove(q)
            return

        if self.path.startswith("/config"):
            self._json({"target": STATE["target"]})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(self.page)))
        self.end_headers()
        self.wfile.write(self.page)

    def _json(self, obj):
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send(self, seg: dict) -> None:
        self.wfile.write(f"data: {json.dumps(seg, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()


def open_window(url: str) -> None:
    """Open the teleprompter as its own frameless window, not a browser tab.

    Chrome's --app gives a chromeless window that sits beside the call like a
    tool rather than behind seventeen tabs. Falls back to the default browser.
    """
    for app in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    ):
        if Path(app).exists():
            try:
                subprocess.Popen(
                    [app, f"--app={url}", "--window-size=560,760", "--window-position=40,80"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                break
    subprocess.Popen(["open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="English", help="target language, e.g. English, Indonesian")
    p.add_argument("--model", default="", help="Groq model (default depends on --reasoning)")
    p.add_argument("--reasoning", action="store_true",
                   help="use a reasoning model: better on implication, slower")
    p.add_argument("--context", type=int, default=6,
                   help="utterances of conversation history sent with each line")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--log", default="", help="JSONL transcript path")
    p.add_argument("--no-window", action="store_true", help="do not open the teleprompter window")
    p.add_argument("--speak", action="store_true", help="also speak the translation aloud")
    p.add_argument("--voice", default="", help="macOS voice (default: picked from --target)")
    p.add_argument("--speak-device", default=os.environ.get("SPEAK_DEVICE", ""),
                   help="audio device to speak through; MUST NOT be the Multi-Output device")
    p.add_argument("--speak-rate", type=int, default=0, help="words per minute, e.g. 190")
    p.add_argument("--speak-lag", type=float, default=12.0,
                   help="drop speech older than this many seconds")
    args = p.parse_args()

    if not args.model:
        # Deliberately NOT GROQ_MODEL: that name is already used by other tools on
        # this machine and points at a reasoning model, which is both slow for live
        # translation and prone to returning an empty message. Opting in takes a
        # name of our own.
        args.model = os.environ.get("MT_GROQ_MODEL") or (
            REASONING_MODEL if args.reasoning else FAST_MODEL
        )

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("error: GROQ_API_KEY not set (put it in ~/class-notes/.env)", file=sys.stderr)
        return 2

    STATE["target"] = args.target
    here = Path(__file__).resolve().parent
    Handler.page = (here / "teleprompter.html").read_bytes()

    log_path = Path(args.log) if args.log else here.parent / "transcripts" / (
        time.strftime("meeting-%Y%m%d-%H%M%S") + ".jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}"
    print(f">> teleprompter: {url}  ({args.target}, {args.model})", file=sys.stderr)
    print(f">> transcript:   {log_path}", file=sys.stderr)
    if not args.no_window:
        open_window(url)

    speech: "queue.Queue | None" = None
    if args.speak:
        if not args.voice:
            args.voice = resolve_voice(args.target) or ""
        if not args.voice:
            print(f"error: no macOS voice found for {args.target}. Pass --voice NAME "
                  f"(see: say -v '?'). Speaking it with the wrong voice would be gibberish.",
                  file=sys.stderr)
            return 2
        if not args.speak_device:
            print(">> warning: no --speak-device. If system output is the Multi-Output "
                  "device, spoken translations will be captured and translated again.",
                  file=sys.stderr)
        where = f" via '{args.speak_device}'" if args.speak_device else ""
        print(f">> speaking:     {args.voice}{where}", file=sys.stderr)
        speech = queue.Queue()
        threading.Thread(target=speak_worker, args=(speech, args), daemon=True).start()

    jobs: queue.Queue = queue.Queue()
    threading.Thread(target=worker, args=(jobs, args, key, log_path, speech), daemon=True).start()

    idx = 0
    agree = LocalAgreement()
    utter = Utterances()
    for raw in sys.stdin:
        text = clean(raw)
        if not text:
            continue
        was = len(agree.committed)
        fresh = agree.insert(text)
        # A reset means the previous utterance ended; send whatever was still held.
        if len(agree.committed) < was:
            for sentence in utter.flush():
                idx += 1
                jobs.put((idx, sentence))
        for sentence in utter.add(fresh):
            idx += 1
            jobs.put((idx, sentence))

    for sentence in utter.flush():
        idx += 1
        jobs.put((idx, sentence))

    jobs.join()  # finish translating what was already said before exiting
    print(f">> {idx} utterances -> {log_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
