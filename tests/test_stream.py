"""The stream-parsing layer, which is where every bug so far has lived.

Two real failures came out of this code and both were silent:

  1. The parser was written against whisper-stream's sliding-window output and
     then pointed at its VAD output, which is a different format entirely --
     `### Transcription N START` markers and `[00:00:00.000 --> ...]` prefixes.
     Timestamps were being translated as though somebody had said them.

  2. Exact-repeat filtering looked like it handled re-emission, but a VAD buffer
     grows: "Buenos días" and "Buenos días a todos" are different strings, so the
     opening was translated twice.

Neither raised anything. Both are covered below.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from translator import (  # noqa: E402
    LocalAgreement, Utterances, VOICE_LOCALES, clean, parse_reply,
)


# --- clean(): what reaches the translator at all ---------------------------

@pytest.mark.parametrize("line", [
    "### Transcription 4 START | t0 = 0 ms | t1 = 5648 ms",
    "### Transcription 4 END",
    "init: found 5 capture devices:",
    "main: processing 48000 samples",
    "[Start speaking]",
    "[BLANK_AUDIO]",
    "*sighs*",
    "   ",
])
def test_non_speech_is_dropped(line):
    assert clean(line) == ""


def test_timestamp_prefix_is_stripped_not_translated():
    """The bug: the timestamp was reaching the model as if it were speech."""
    assert clean("[00:00:00.000 --> 00:00:29.980]   Buenos días a todos.") \
        == "Buenos días a todos."


def test_ansi_erase_codes_are_stripped():
    assert clean("\x1b[2K Hello there") == "Hello there"


def test_ordinary_speech_survives():
    assert clean("  Le chiffre d'affaires a augmenté.  ") == "Le chiffre d'affaires a augmenté."


def test_bracketed_text_mid_sentence_is_kept():
    """Only a line that is ENTIRELY bracketed is noise."""
    assert clean("the figure [see slide] rose") == "the figure [see slide] rose"


# --- LocalAgreement: commit only what two passes agree on -------------------

def test_nothing_commits_on_a_single_pass():
    """One hypothesis is not agreement; the tail must wait for confirmation."""
    assert LocalAgreement().insert("Buenos días a todos") == []


def test_growing_buffer_emits_each_word_once():
    a = LocalAgreement()
    out = []
    for hyp in ["Buenos días",
                "Buenos días a todos",
                "Buenos días a todos, vamos a revisar",
                "Buenos días a todos, vamos a revisar el presupuesto."]:
        out += a.insert(hyp)
    assert out == ["Buenos", "días", "a", "todos,", "vamos", "a", "revisar"]
    # The real property: everything emitted, in order, is exactly the agreed
    # prefix of the final hypothesis — so no word was sent twice and none
    # arrived out of order.
    final = "Buenos días a todos, vamos a revisar el presupuesto."
    assert final.startswith(" ".join(out))
    assert " ".join(out) == " ".join(a.committed)


def test_identical_repeats_commit_once():
    """Ten silence hallucinations must yield one line, not ten."""
    a = LocalAgreement()
    out = []
    for _ in range(10):
        out += a.insert("Thank you.")
    assert out == ["Thank", "you."]


def test_a_flush_resets_and_the_new_utterance_survives():
    a = LocalAgreement()
    for _ in range(2):
        a.insert("El presupuesto es alto.")
    before = len(a.committed)
    fresh = []
    for _ in range(2):
        fresh += a.insert("Completely different sentence now.")
    assert len(a.committed) < before or fresh, "a new utterance must not be swallowed"
    assert "Completely" in " ".join(fresh)


def test_divergence_mid_utterance_resets_rather_than_appending():
    a = LocalAgreement()
    a.insert("one two three")
    a.insert("one two three")
    out = a.insert("totally other words")
    assert out == [], "a diverged hypothesis starts over, it does not append"


# --- Utterances: whole sentences, not fragments -----------------------------

def test_words_accumulate_until_a_sentence_ends():
    u = Utterances()
    assert u.add(["The", "budget"]) == []
    assert u.add(["is", "high."]) == ["The budget is high."]


def test_each_terminator_closes_a_sentence():
    u = Utterances()
    assert u.add(["Really?"]) == ["Really?"]
    assert u.add(["Yes!"]) == ["Yes!"]
    assert u.add(["終わり。"]) == ["終わり。"]


def test_an_endless_sentence_still_flushes():
    """Without a cap, a speaker who never pauses is never translated."""
    u = Utterances(max_words=5)
    assert u.add(["a", "b", "c", "d"]) == []
    assert u.add(["e"]) == ["a b c d e"]


def test_flush_returns_the_remainder_then_empties():
    u = Utterances()
    u.add(["trailing", "words"])
    assert u.flush() == ["trailing words"]
    assert u.flush() == []


# --- parse_reply(): the model does not always obey ---------------------------

def test_well_formed_json_is_parsed():
    text, note = parse_reply('{"text": "Good morning", "note": "idiom"}', "src")
    assert (text, note) == ("Good morning", "idiom")


def test_null_note_becomes_none():
    assert parse_reply('{"text": "Plain sentence", "note": null}', "src")[1] is None


def test_json_wrapped_in_a_code_fence_is_recovered():
    text, _ = parse_reply('```json\n{"text": "Hola"}\n```', "src")
    assert text == "Hola"


def test_bare_prose_falls_back_to_the_prose():
    """If the model ignores the format, a translation is better than nothing."""
    assert parse_reply("Good morning everyone", "src")[0] == "Good morning everyone"


def test_unparseable_empty_reply_falls_back_to_the_source():
    assert parse_reply("", "the source line")[0] == "the source line"


# --- voice selection --------------------------------------------------------

def test_target_languages_map_to_plausible_locales():
    assert VOICE_LOCALES["indonesian"] == ("id_ID",)
    assert "en_US" in VOICE_LOCALES["english"]


def test_every_locale_entry_is_well_formed():
    for lang, locales in VOICE_LOCALES.items():
        assert isinstance(locales, tuple) and locales, lang
        for loc in locales:
            assert "_" in loc and loc[:2].islower(), f"{lang}: {loc}"
