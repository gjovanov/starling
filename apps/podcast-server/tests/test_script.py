import pytest

from podcast_server.config import Settings
from podcast_server.script import ScriptError, Turn, check_generation, parse_script, plan

VOICES = {"michael", "emma", "alex", "sam", "kim"}


def test_parse_basic_and_case_insensitive_speaker():
    assert parse_script("michael: Hello there.\nEmma: Hi!\n") == [
        Turn("michael", "Hello there."), Turn("emma", "Hi!")]


def test_parse_skips_blank_comment_lines_and_bom():
    src = "﻿# a comment\n\nmichael: One.\n   \n# another\nemma: Two.\n"
    assert [t.speaker for t in parse_script(src)] == ["michael", "emma"]


def test_parse_splits_on_first_colon_only():
    (turn,) = parse_script("emma: Note: it starts at 10:30.")
    assert turn.text == "Note: it starts at 10:30."


def test_parse_collapses_whitespace():
    (turn,) = parse_script("michael:   lots   of\tspace  ")
    assert turn.text == "lots of space"


@pytest.mark.parametrize("src, line", [
    ("just text", "line 1"),
    ("michael:", "line 1"),
    (": orphan text", "line 1"),
    ("michael: fine\nno speaker here", "line 2"),
])
def test_parse_errors_name_the_line(src, line):
    with pytest.raises(ScriptError, match=line):
        parse_script(src)


def test_parse_rejects_an_empty_script():
    with pytest.raises(ScriptError, match="no turns"):
        parse_script("# only a comment\n\n")


def test_plan_assigns_roles_by_first_appearance():
    # michael first: first-appearance order differs from alphabetical order
    p = plan([Turn("michael", "a"), Turn("emma", "b"), Turn("michael", "c")], VOICES, Settings())
    assert p.speakers == ("michael", "emma")
    assert p.roles == (0, 1, 0)


def test_plan_normalizes_speaker_and_text():
    p = plan([Turn(" Michael ", "  two\nlines  ")], VOICES, Settings())
    assert p.turns == (Turn("michael", "two lines"),)


def test_plan_unknown_speaker_lists_the_voices():
    with pytest.raises(ScriptError, match="unknown speaker 'bob'.*emma"):
        plan([Turn("bob", "hi")], VOICES, Settings())


def test_plan_allows_four_speakers_and_rejects_a_fifth():
    four = [Turn(n, "hi") for n in ("michael", "emma", "alex", "sam")]
    assert len(plan(four, VOICES, Settings()).speakers) == 4
    with pytest.raises(ScriptError, match="at most 4"):
        plan(four + [Turn("kim", "hi")], VOICES, Settings())


def test_plan_rejects_empty_text():
    with pytest.raises(ScriptError, match="turn 2: empty text"):
        plan([Turn("michael", "ok"), Turn("emma", "  \n ")], VOICES, Settings())


def test_plan_limits():
    with pytest.raises(ScriptError, match="3 turns exceeds"):
        plan([Turn("michael", "a")] * 3, VOICES, Settings(max_turns=2))
    with pytest.raises(ScriptError, match="per-turn limit"):
        plan([Turn("michael", "x" * 11)], VOICES, Settings(max_chars_per_turn=10))
    with pytest.raises(ScriptError, match="per-script limit"):
        plan([Turn("michael", "x" * 8)] * 2, VOICES, Settings(max_total_chars=15))


def test_max_new_tokens_budget_and_cap():
    s = Settings()
    p = plan([Turn("michael", "x" * 100)], VOICES, s)
    assert p.max_new_tokens == int(100 * s.tokens_per_char) + s.token_slack
    capped = plan([Turn("michael", "x" * 1000)], VOICES, Settings(max_new_tokens_cap=200))
    assert capped.max_new_tokens == 200


def test_budget_leaves_headroom_over_natural_speech():
    # ~14 chars/s of speech at 7.5 audio tokens/s; the guard must not clip it.
    n = 2000
    p = plan([Turn("michael", "x" * n)], VOICES, Settings())
    assert p.max_new_tokens >= 1.5 * (n / 14) * 7.5


@pytest.mark.parametrize("seed, cfg, steps, ok", [
    (42, 1.3, 10, True), (0, 1.0, 1, True), (2**32 - 1, 3.0, 50, True),
    (-1, 1.3, 10, False), (2**32, 1.3, 10, False), (42, 0.9, 10, False),
    (42, 3.1, 10, False), (42, 1.3, 0, False), (42, 1.3, 51, False),
])
def test_check_generation_bounds(seed, cfg, steps, ok):
    if ok:
        check_generation(seed, cfg, steps)
    else:
        with pytest.raises(ScriptError):
            check_generation(seed, cfg, steps)
