"""Offline tests for the Phase B MCP-roll single-step runner.

No CAN, no SDK, no motion. These exercise the preflight contract only, for all
four long fingers, since one table-driven runner serves all of them.
"""

from __future__ import annotations

import pytest

from tools.run_g20_mcp_roll_candidate_step import (
    ROLL_FINGERS,
    build_roll_command,
    parse_args,
)

# Measured 2026-08-03 read-only snapshot, used as the realistic starting state.
BASE_STATE20 = [254, 252, 254, 254, 254, 255, 142, 130, 116, 101, 252,
                0, 0, 0, 0, 254, 255, 255, 255, 255]
ALL_FINGERS = tuple(ROLL_FINGERS)


def _frame_for(finger: str, state20: list[int]) -> list[int]:
    spec = ROLL_FINGERS[finger]
    return [state20[spec.roll_slot], 0, state20[spec.pitch_slot], 0, 0,
            state20[spec.pip_slot]]


def _others(finger: str, state20: list[int]) -> dict[int, int]:
    return {
        spec.roll_slot: state20[spec.roll_slot]
        for name, spec in ROLL_FINGERS.items()
        if name != finger
    }


def _call(finger: str, target_raw: int, state20: list[int] | None = None,
          frame6: list[int] | None = None, **overrides):
    state20 = list(state20 or BASE_STATE20)
    spec = ROLL_FINGERS[finger]
    kwargs = {
        "finger": finger,
        "expected_start_raw": state20[spec.roll_slot],
        "target_raw": target_raw,
        "pitch_hold_raw": state20[spec.pitch_slot],
        "pip_hold_raw": state20[spec.pip_slot],
        "other_roll_holds": _others(finger, state20),
    }
    kwargs.update(overrides)
    return build_roll_command(
        state20, frame6 if frame6 is not None else _frame_for(finger, state20),
        **kwargs,
    )


@pytest.mark.parametrize("finger", ALL_FINGERS)
def test_only_the_target_roll_slot_changes(finger):
    spec = ROLL_FINGERS[finger]
    start = BASE_STATE20[spec.roll_slot]
    after, frame6, diff = _call(finger, start - 10)

    assert [item["slot"] for item in diff] == [spec.roll_slot]
    assert after[spec.roll_slot] == start - 10
    assert frame6[0] == start - 10
    assert frame6[2] == BASE_STATE20[spec.pitch_slot]
    assert frame6[5] == BASE_STATE20[spec.pip_slot]
    assert [frame6[i] for i in (1, 3, 4)] == [0, 0, 0]


@pytest.mark.parametrize("finger", ALL_FINGERS)
def test_other_roll_slots_are_never_written(finger):
    spec = ROLL_FINGERS[finger]
    after, _, diff = _call(finger, BASE_STATE20[spec.roll_slot] + 8)
    for name, other in ROLL_FINGERS.items():
        if name == finger:
            continue
        assert after[other.roll_slot] == BASE_STATE20[other.roll_slot]
        assert other.roll_slot not in {item["slot"] for item in diff}


@pytest.mark.parametrize("finger", ALL_FINGERS)
def test_step_larger_than_17_raw_is_refused(finger):
    spec = ROLL_FINGERS[finger]
    with pytest.raises(ValueError, match="must be 1..17 raw"):
        _call(finger, BASE_STATE20[spec.roll_slot] - 18)


@pytest.mark.parametrize("finger", ALL_FINGERS)
def test_zero_length_step_is_refused(finger):
    spec = ROLL_FINGERS[finger]
    with pytest.raises(ValueError, match="must be 1..17 raw"):
        _call(finger, BASE_STATE20[spec.roll_slot])


def test_drifted_non_target_roll_slot_blocks_the_step():
    state20 = list(BASE_STATE20)
    holds = _others("index", state20)
    state20[ROLL_FINGERS["ring"].roll_slot] += 5
    with pytest.raises(ValueError, match="non-target roll slot 8"):
        _call("index", 132, state20=state20, other_roll_holds=holds)


def test_other_roll_holds_must_cover_exactly_the_three_other_slots():
    with pytest.raises(ValueError, match="must cover exactly slots"):
        _call("index", 132, other_roll_holds={7: 130, 8: 116})
    with pytest.raises(ValueError, match="must cover exactly slots"):
        _call("index", 132, other_roll_holds={6: 142, 7: 130, 8: 116, 9: 101})


def test_expected_start_must_match_measured_slot():
    with pytest.raises(ValueError, match="must start within"):
        _call("index", 132, expected_start_raw=120)


def test_frame_disagreeing_with_raw20_blocks_the_step():
    frame6 = _frame_for("index", BASE_STATE20)
    frame6[0] += 6
    with pytest.raises(ValueError, match="disagrees with raw20 at element 0"):
        _call("index", 132, frame6=frame6)


def test_nonzero_frame_reserved_byte_blocks_the_step():
    frame6 = _frame_for("middle", BASE_STATE20)
    frame6[3] = 7
    with pytest.raises(ValueError, match="reserved byte 3 must be zero"):
        _call("middle", 122, frame6=frame6)


def test_nonzero_reserved_raw20_slot_blocks_the_step():
    state20 = list(BASE_STATE20)
    state20[13] = 4
    with pytest.raises(ValueError, match="reserved slots 11..14 are not zero"):
        _call("index", 132, state20=state20)


def test_pitch_or_pip_off_its_hold_blocks_the_step():
    state20 = list(BASE_STATE20)
    state20[ROLL_FINGERS["ring"].pip_slot] = 200
    with pytest.raises(ValueError, match="hold slot 18 must be within"):
        _call("ring", 108, state20=state20, pip_hold_raw=255)


def test_target_outside_raw_range_is_refused():
    with pytest.raises(ValueError, match="target raw must be in 0..255"):
        _call("index", 300)


def test_unknown_finger_is_refused():
    # Called directly: the helper would key-error on the table before the
    # runner got a chance to reject the finger.
    with pytest.raises(ValueError, match="unsupported finger"):
        build_roll_command(
            BASE_STATE20, [255, 0, 254, 0, 0, 254],
            finger="thumb", expected_start_raw=255, target_raw=250,
            pitch_hold_raw=254, pip_hold_raw=254, other_roll_holds={},
        )


@pytest.mark.parametrize("finger", ALL_FINGERS)
def test_slot_table_matches_the_phase_a_runners(finger):
    # Guards against a transcription slip in ROLL_FINGERS.
    expected = {"index": (6, 1, 16), "middle": (7, 2, 17),
                "ring": (8, 3, 18), "pinky": (9, 4, 19)}[finger]
    spec = ROLL_FINGERS[finger]
    assert (spec.roll_slot, spec.pitch_slot, spec.pip_slot) == expected


def test_cli_requires_execute_to_send_and_parses_other_roll_holds():
    args = parse_args([
        "--sdk-root", "/tmp/sdk", "--calib", "/tmp/c.json",
        "--finger", "index", "--expected-start-raw", "142",
        "--target-raw", "132", "--pitch-hold-raw", "252",
        "--pip-hold-raw", "255", "--other-roll-hold", "7=130",
        "--other-roll-hold", "8=116", "--other-roll-hold", "9=101",
        "--out", "/tmp/out.json",
    ])
    assert args.execute is False
    assert dict(args.other_roll_hold) == {7: 130, 8: 116, 9: 101}
