# Prior art, and what we take from it

A survey of the streaming-speech ecosystem, 21 Sep 2026, and a record of which ideas this
project adopts. Real-time ASR is an active research area; the sensible thing is to use what
has been worked out rather than re-derive it, and to be explicit about where it came from.

## WhisperLiveKit, and the policies behind it

[`QuentinFuxa/WhisperLiveKit`](https://github.com/QuentinFuxa/WhisperLiveKit) · 11,073★ ·
Apache-2.0 · last push 2026-09-19 · `pip install whisperlivekit`

It packages several pieces of research that bear directly on this project:

| Problem | Their approach |
|---|---|
| VAD re-emits a growing buffer | **LocalAgreement** ([whisper_streaming](https://github.com/ufal/whisper_streaming), Machácek et al. 2023) and **AlignAtt** (Simul-Whisper, 2025) — commit only what successive passes agree on, giving **append-only transcripts**. **Adopted — see `src/translator.py`.** |
| Groq dependency; target languages limited by what the account serves | **NLLB-200 distilled**, local, **200 languages** in and out |
| No speaker labels (roadmap) | **Streaming Sortformer** (SOTA 2025) real-time diarization |
| large-v3-turbo on an M1 | **MLX Whisper backend** for Apple Silicon |

Their README states the constraint plainly: *"Whisper is designed for complete utterances,
not real-time chunks. Processing small segments loses context, cuts off words mid-syllable."*
That matches the growing-buffer behaviour measured here in commit `41ecbee`.

**LocalAgreement is now implemented in this repo.** A word is emitted only once two
consecutive hypotheses agree on its position, so output is append-only: nothing shown is
retracted or repeated. It replaced an exact-repeat filter that handled the stuck-hallucination
case but not the growing one — "Buenos días" and "Buenos días a todos" are different strings,
so the opening was translated twice. Verified: four passes over a growing buffer now yield one
translation, and ten identical hallucination passes still yield one line.

Per-session it takes `?language=fr` and `?target_language=de`, so one server handles
mixed-language meetings.

**Next step: WLK as an optional engine.** Running it as the ASR + translation backend over
its WebSocket would bring local NLLB translation and Sortformer diarization, with this
project's layer on top.

## What this project adds

The pieces that are specific to this repo, and the reason it stays a separate thing:

1. **The intent layer.** Nothing in this survey does it. Returning `{text, note}` and
   flagging only idiom, sarcasm, indirect refusal or implied asks — staying silent on ordinary
   sentences — is what separates an interpreter from a transcriber.
2. **macOS audio routing that works.** BlackHole + a Multi-Output device so the meeting is
   heard *and* captured, with output restored on every exit path.
3. **Spoken output that does not feed itself.** `say -a` straight to the speakers, bypassing
   the loopback. Verified necessary: whisper transcribes `say`'s Indonesian back at p=0.998.
4. **The teleprompter**, and the decision to render notes as a quiet annotation.

## Integrations worth having

**[`Vexa-ai/vexa`](https://github.com/Vexa-ai/vexa)** · 2,804★ — "Open-source meeting bots and
real-time transcription, cloud or fully self-hosted." A bot *joins* Google Meet, Teams and
Zoom and streams speaker-attributed transcripts.

Worth adding as a second input source rather than a replacement. Today this project captures
system audio through BlackHole, which works for anything the machine can play but needs the
loopback configured. A bot in the meeting needs no loopback, no Multi-Output device and no
audio routing at all, and it brings speaker attribution with it. BlackHole still wins for
in-person audio and for anything not on a supported platform.

## TTS — surveyed, staying with `say`

| Option | Verdict |
|---|---|
| `rhasspy/piper` · 11,286★ | Fast local neural TTS. Voice list shows Spanish, Hindi and many others — **no Indonesian found**. |
| `hexgrad/kokoro` · 8,912★ | High quality, limited language set. |
| macOS `say` | 184 voices, **Damayanti (id_ID)** verified: whisper reads her Indonesian back verbatim at p=0.998. |

**Staying with `say`.** Piper would likely sound better where it has the language, but its
voice list does not appear to cover Indonesian, which is the target this was built against.
Worth revisiting if a target language comes up that `say` lacks.

## Also surveyed

| Repo | ★ | Why not |
|---|---|---|
| `ggml-org/whisper.cpp` | 53,808 | already the engine here |
| `SYSTRAN/faster-whisper` | 25,483 | CTranslate2 backend; WLK can use it, no reason to wire it directly |
| `m-bain/whisperX` | 24,151 | batch, not streaming — great for the post-meeting re-process idea |
| `KoljaB/RealtimeSTT` | 10,137 | good streaming STT, but no translation or diarization |
| `pyannote/pyannote-audio` | 10,573 | diarization building blocks; WLK's Sortformer is the streaming answer |
| `collabora/WhisperLive` | 4,288 | overlaps WLK, less capable |
| `PolyTalkIO/polytalk` | 75 | closest in spirit — self-hosted real-time speech-to-speech |

## Suggested order of work

1. **Vexa as a second input** — removes the untested audio path for online meetings and
   brings speaker labels with it. Biggest risk reduction per hour.
2. **WLK as the engine** — append-only output kills the duplicate problem properly, and
   local NLLB removes the Groq dependency and its model-availability surprises.
3. **Keep the intent layer on top of both.** It is the part that is ours.
4. `whisperX` for the post-meeting clean re-transcribe, if that v2 reading is ever wanted.
