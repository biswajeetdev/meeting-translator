# What's upstream — GitHub survey, 21 Sep 2026

Searched for work that solves the parts of this project that are still weak. The short
version: **one project has already solved the hard engine problems, and it is worth adopting
rather than re-deriving.** What this repo does that it does not, is worth keeping.

## WhisperLiveKit — the one that matters

[`QuentinFuxa/WhisperLiveKit`](https://github.com/QuentinFuxa/WhisperLiveKit) · 11,073★ ·
Apache-2.0 · last push 2026-09-19 · `pip install whisperlivekit`

It bundles four things we either did badly or listed as roadmap:

| Our problem | What they use |
|---|---|
| VAD re-emits a growing buffer; we could only drop exact repeats | **LocalAgreement** ([whisper_streaming](https://github.com/ufal/whisper_streaming)) and **AlignAtt** (Simul-Whisper, SOTA 2025) — commit only what successive passes agree on, giving **append-only transcripts** |
| Groq dependency; target languages limited by what the account serves | **NLLB-200 distilled**, local, **200 languages** in and out |
| No speaker labels (roadmap) | **Streaming Sortformer** (SOTA 2025) real-time diarization |
| large-v3-turbo on an M1 | **MLX Whisper backend** for Apple Silicon |

Their README makes the point we learned the hard way: *"Whisper is designed for complete
utterances, not real-time chunks. Processing small segments loses context, cuts off words
mid-syllable."* That is exactly the growing-buffer behaviour documented in our commit
`41ecbee`, and `is_repeat()` is the crude version of what LocalAgreement does properly.

Per-session it takes `?language=fr` and `?target_language=de`, so one server handles
mixed-language meetings.

**Recommended move: keep our layer, swap our engine.** Run WLK as the ASR + translation
backend over its WebSocket, and keep what is ours on top.

## What this repo has that WhisperLiveKit does not

Worth being clear about, because it is the reason not to just delete this project:

1. **The intent/innuendo layer.** Nothing found in the survey does this. Returning
   `{text, note}` and flagging only idiom, sarcasm, indirect refusal or implied asks — with
   the restraint to return null on ordinary sentences — appears to be genuinely novel.
2. **macOS audio routing that works.** BlackHole + a Multi-Output device so the meeting is
   heard *and* captured, with output restored on every exit path.
3. **Spoken output that does not feed itself.** `say -a` straight to the speakers, bypassing
   the loopback. Verified necessary: whisper transcribes `say`'s Indonesian back at p=0.998.
4. **The teleprompter**, and the decision to render notes as a quiet annotation.

## Integrations worth having

**[`Vexa-ai/vexa`](https://github.com/Vexa-ai/vexa)** · 2,804★ — "Open-source meeting bots and
real-time transcription, cloud or fully self-hosted." A bot *joins* Google Meet, Teams and
Zoom and streams speaker-attributed transcripts.

This sidesteps our single biggest untested risk. Right now we capture system audio through
BlackHole, and the live path has never been run against real speech. A bot in the meeting
needs no loopback, no Multi-Output device, no audio routing at all — and it gets speaker
attribution for free. Worth having as a second input source rather than a replacement:
BlackHole still wins for in-person audio and anything not on a supported platform.

## TTS — checked, and we are already right

| Option | Verdict |
|---|---|
| `rhasspy/piper` · 11,286★ | Fast local neural TTS. Voice list shows Spanish, Hindi and many others — **no Indonesian found**. |
| `hexgrad/kokoro` · 8,912★ | High quality, limited language set. |
| macOS `say` | 184 voices, **Damayanti (id_ID)** verified: whisper reads her Indonesian back verbatim at p=0.998. |

**Keep `say` for now.** Piper would sound better where it has the language, but it does not
appear to cover Indonesian, which is the target we actually built for. Revisit only if a
target language turns up that `say` lacks.

## Also surveyed, not adopted

| Repo | ★ | Why not |
|---|---|---|
| `ggml-org/whisper.cpp` | 53,808 | already our engine |
| `SYSTRAN/faster-whisper` | 25,483 | CTranslate2 backend; WLK can use it, no reason to wire it directly |
| `m-bain/whisperX` | 24,151 | batch, not streaming — great for the post-meeting re-process idea |
| `KoljaB/RealtimeSTT` | 10,137 | good streaming STT, but no translation or diarization |
| `pyannote/pyannote-audio` | 10,573 | diarization building blocks; WLK's Sortformer is the streaming answer |
| `collabora/WhisperLive` | 4,288 | overlaps WLK, less capable |
| `PolyTalkIO/polytalk` | 75 | same idea, much smaller, no intent layer |

## Suggested order of work

1. **Vexa as a second input** — removes the untested audio path for online meetings and
   brings speaker labels with it. Biggest risk reduction per hour.
2. **WLK as the engine** — append-only output kills the duplicate problem properly, and
   local NLLB removes the Groq dependency and its model-availability surprises.
3. **Keep the intent layer on top of both.** It is the part worth having.
4. `whisperX` for the post-meeting clean re-transcribe, if that v2 reading is ever wanted.
