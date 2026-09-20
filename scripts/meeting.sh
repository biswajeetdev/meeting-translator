#!/usr/bin/env bash
# Live meeting translator: hear the meeting, read it in your language.
#
#   meeting.sh start [target-language]     # e.g. meeting.sh start English
#   meeting.sh start Indonesian --speak    # ...and say it out loud
#   meeting.sh devices                     # list capture devices whisper-stream can see
#
# What it does: routes system output through a Multi-Output Device (your speakers
# AND BlackHole) so you still hear the call, points whisper-stream at BlackHole,
# and pipes each finalized utterance into the translator, which serves a
# teleprompter page.
#
# It captures SYSTEM audio, which is what the other participants say. Your own
# microphone is not in that stream -- you do not need your own words translated.
#
# Ctrl-C, a crash, or a closed terminal all restore your normal output device.
# Audio routing is never left switched: that is the one failure that ruins the
# next call you take.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${WHISPER_MODEL:-$HOME/class-notes/models/ggml-large-v3-turbo.bin}"
BLACKHOLE_NAME="${BLACKHOLE_NAME:-BlackHole 2ch}"
PORT="${PORT:-8765}"
STATE="$ROOT/.state"
mkdir -p "$STATE"

die() { echo "error: $*" >&2; exit 1; }

# --- capture-device discovery ------------------------------------------------
# whisper-stream enumerates through SDL, whose indices differ from ffmpeg's.
# Probing costs a model load, so the answer is cached until the device list changes.
probe_devices() {
  ( "$1" -m "$MODEL" -c 999 2>&1 & local p=$!; sleep 8; kill $p 2>/dev/null ) \
    | grep -E "^init:    - Capture device" || true
}

capture_id() {
  local list
  if [ -s "$STATE/devices" ]; then list=$(cat "$STATE/devices"); else
    list=$(probe_devices whisper-stream); printf '%s\n' "$list" > "$STATE/devices"
  fi
  printf '%s\n' "$list" | sed -n "s/.*Capture device #\([0-9]*\): '$BLACKHOLE_NAME'.*/\1/p" | head -1
}

case "${1:-start}" in
devices)
  rm -f "$STATE/devices"
  command -v whisper-stream >/dev/null || die "whisper-stream missing (brew install whisper-cpp)"
  probe_devices whisper-stream
  exit 0
  ;;
start) ;;
*) die "unknown command '${1}' (start | devices)" ;;
esac

TARGET="${2:-English}"
shift $(( $# > 2 ? 2 : $# ))
EXTRA=("$@")   # anything else goes straight to translator.py, e.g. --speak

# --- preflight ---------------------------------------------------------------
command -v whisper-stream    >/dev/null || die "whisper-stream missing (brew install whisper-cpp)"
command -v SwitchAudioSource >/dev/null || die "SwitchAudioSource missing (brew install switchaudio-osx)"
[ -f "$MODEL" ] || die "model not found: $MODEL  (set WHISPER_MODEL)"

if [ -z "${GROQ_API_KEY:-}" ] && [ -f "$HOME/class-notes/.env" ]; then
  set -a; . "$HOME/class-notes/.env"; set +a
fi
[ -n "${GROQ_API_KEY:-}" ] || die "GROQ_API_KEY not set"

# --- find the Multi-Output Device (speakers + BlackHole) ---------------------
MO=$(SwitchAudioSource -a -t output 2>/dev/null | grep -i "multi-output" | head -1 || true)
if [ -z "$MO" ]; then
  HELPER="$HOME/classnotes-agent/scripts/make-multiout.py"
  if [ -f "$HELPER" ]; then
    echo ">> no Multi-Output Device found — creating one"
    python3 "$HELPER" >&2 || true
    MO=$(SwitchAudioSource -a -t output 2>/dev/null | grep -i "multi-output" | head -1 || true)
  fi
fi
[ -n "$MO" ] || die "no Multi-Output Device. Create one in Audio MIDI Setup containing your
       speakers AND '$BLACKHOLE_NAME', name it with 'Multi-Output', then re-run.
       Without it you would capture the meeting but not hear it."

CAP=$(capture_id)
[ -n "$CAP" ] || die "'$BLACKHOLE_NAME' is not in whisper-stream's capture list.
       Run: $0 devices"

# --- switch output, and guarantee we switch it back --------------------------
PREV=$(SwitchAudioSource -c -t output 2>/dev/null || true)
restore() {
  [ -n "${PREV:-}" ] && SwitchAudioSource -s "$PREV" >/dev/null 2>&1 \
    && echo ">> output restored to '$PREV'" >&2
}
trap restore EXIT INT TERM

SwitchAudioSource -s "$MO" >/dev/null 2>&1 || die "could not select '$MO'"

# Spoken translations must go straight to the speakers, NOT through the
# Multi-Output device -- that one feeds BlackHole, so our own voice would be
# captured, transcribed, translated and spoken again, forever.
export SPEAK_DEVICE="${SPEAK_DEVICE:-$PREV}"
echo ">> output: '$MO'  (you will still hear the meeting)" >&2
echo ">> capture: #$CAP '$BLACKHOLE_NAME'  ->  translating to $TARGET" >&2
echo ">> open http://127.0.0.1:$PORT   —   Ctrl-C to stop" >&2

# --step 0 puts whisper-stream in VAD mode: it waits for a pause and emits one
# finalized utterance, instead of re-emitting a revised sliding window every few
# seconds. Utterances are what we want -- each is translated exactly once.
exec whisper-stream \
  -m "$MODEL" \
  -c "$CAP" \
  --step 0 \
  --length 30000 \
  -vth "${VAD_THOLD:-0.60}" \
  -l "${SOURCE_LANG:-auto}" \
  -kc \
  2>/dev/null \
  | python3 "$ROOT/src/translator.py" --target "$TARGET" --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"}
