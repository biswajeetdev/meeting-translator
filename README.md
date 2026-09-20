# meeting-translator

Read a live meeting in your language, while it is happening.

It listens to the call's audio, transcribes each utterance locally, translates it,
and puts it on a teleprompter beside the call — with a note whenever the meaning
is carried by something other than the words.

```
 meeting audio ──► BlackHole ──► whisper-stream ──► translator ──► teleprompter
   (you still          (loopback)    (local, VAD)      (Groq)        (your browser)
    hear it)
```

Not a post-meeting transcript tool. The whole point is that the text arrives while
there is still time to respond to it.

## What the notes are for

A literal translation is often a wrong translation. The model returns a short gloss
when — and only when — something would otherwise be lost:

| Heard | Translated | Note |
|---|---|---|
| それはちょっと難しいですね | That's a bit difficult, isn't it? | soft refusal/hedge |
| Me estás tomando el pelo | You're pulling my leg | idiom meaning teasing |
| The budget is fifty thousand euros. | The budget is fifty thousand euros. | *(none)* |

That third row matters as much as the first two. If every line carries a note, the
notes are worthless and you stop reading them.

## Requirements

- macOS (the audio routing is CoreAudio-specific)
- [BlackHole 2ch](https://existential.audio/blackhole/) — `brew install blackhole-2ch`
- `brew install whisper-cpp switchaudio-osx`
- A whisper model, e.g. `ggml-large-v3-turbo.bin`
- A Groq API key in `GROQ_API_KEY`
- A **Multi-Output Device** containing your speakers *and* BlackHole, with
  "Multi-Output" in its name. This is what lets you hear the meeting and capture
  it at the same time.

## Use

```bash
export GROQ_API_KEY=gsk_...

scripts/meeting.sh start English              # translate whatever is spoken into English
scripts/meeting.sh start Indonesian           # or into anything else
scripts/meeting.sh start Indonesian --speak   # ...and say it out loud
scripts/meeting.sh devices                    # list capture devices whisper-stream sees
```

The teleprompter opens in its own frameless window. Ctrl-C stops everything and
restores your normal audio output.

Options worth knowing:

| Variable / flag | Effect |
|---|---|
| `SOURCE_LANG=es` | Skip auto-detect. Worth setting when you know the language. |
| `--reasoning` | Use a reasoning model. Better on implication, noticeably slower. |
| `--context N` | Utterances of history sent with each line (default 6). |
| `WHISPER_MODEL=...` | Point at a different ggml model. |
| `VAD_THOLD=0.6` | Raise if silence is being transcribed, lower if speech is missed. |

Every meeting is also written to `transcripts/meeting-<timestamp>.jsonl`, one
object per utterance, with source text, translation and note.

## Things learned the hard way

**`whisper-large-v3-turbo` cannot translate.** It accepts `--translate`, reports
`task = translate`, and returns the source language anyway — measured, not
assumed. The turbo model was fine-tuned for transcription only. This is why
translation is a separate stage and not a whisper flag.

**Whisper's translate task only ever goes *into* English.** Even on a model that
supports it, there is no English → Indonesian. Any tool that needs arbitrary
target languages needs a translation stage regardless.

**Groq rejects Python's default User-Agent** with Cloudflare error 1010 — a 403
that looks exactly like an auth failure and is not. Send a real UA.

**Translation runs on one thread, deliberately.** A thread pool reorders
utterances whenever one call outruns another, and a meeting transcript in the
wrong order is worse than a slow one.

**The audio device is always restored.** On Ctrl-C, on crash, on a closed
terminal. Leaving a Mac's output pointed at a loopback device means the next call
you take is silent and you do not know why.

## Speaking it aloud

`--speak` reads each translation out through macOS `say`, which ships usable
voices for ~40 languages offline and free. The voice is chosen from the target
language — Indonesian gets Damayanti, Japanese gets Kyoko — and a language with
no installed voice is an error, never a substitution, because Indonesian text
read by an English voice is confident nonsense.

| Flag | Effect |
|---|---|
| `--speak` | speak translations aloud |
| `--voice NAME` | override the chosen voice (`say -v '?'` lists them) |
| `--speak-rate 190` | words per minute |
| `--speak-lag 12` | drop speech older than this, in seconds |

Two things this has to get right:

**It must not hear itself.** System output goes to a Multi-Output Device that
feeds BlackHole, so speaking through it would put the translation back into the
transcriber and loop forever. This is not theoretical: whisper transcribes `say`'s
Indonesian back verbatim at p=0.998. Spoken audio is therefore sent straight to
the speakers with `say -a`, bypassing the loopback entirely. `meeting.sh` sets
this from whatever your output device was before it started.

**It must stay current.** Synthesis runs in real time, so a busy meeting queues
faster than it can be spoken. An interpreter ninety seconds behind is worse than
one who misses a line, so anything older than `--speak-lag` is dropped rather
than played late.

Wear headphones. Otherwise the spoken translation and the original voice
compete, and your microphone picks up both.

## Roadmap

- Re-processing the saved audio afterwards into a cleaner transcript than live
  decoding can produce
- Speaker labels (`whisper-stream` has `--tinydiarize`)
- Ducking the meeting's own volume while a translation is being spoken

## License

MIT
