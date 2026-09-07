"""Pins for the S10 shared-node measurement.

The metrics and the decision rule are written before the run, so they are worth
a test for the same reason: a rule that can be adjusted after seeing the numbers
is not a rule. These check the arithmetic, not the models — no inference, no
network drive.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    """Import the tool without importing the whole tools directory as a package."""
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "_s10", TOOLS / "s10_shared_node_20k.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment, not logic
        pytest.skip(f"s10 not importable here: {exc}")
    return module


s10 = _load()


# --------------------------------------------------------------------------
# Frame selection must be deterministic and contiguous
# --------------------------------------------------------------------------
def test_block_starts_are_deterministic_and_evenly_spaced():
    starts = s10.block_starts(348600, blocks=5, block_len=1000)
    assert starts == s10.block_starts(348600, blocks=5, block_len=1000)
    assert starts == [34360, 104080, 173800, 243520, 313240]
    gaps = {b - a for a, b in itertools.pairwise(starts)}
    assert len(gaps) == 1, "blocks must be evenly spaced"


def test_blocks_stay_inside_the_recording_and_off_frame_zero():
    n = 348600
    starts = s10.block_starts(n, blocks=5, block_len=1000)
    assert starts[0] > 0, "frame 0 is handling, not behaviour"
    assert starts[-1] + 1000 <= n


def test_blocks_are_contiguous_because_tail_jump_reads_adjacent_frames():
    blocks = s10.block_frames([10, 100], block_len=4)
    assert blocks == [[10, 11, 12, 13], [100, 101, 102, 103]]


def test_block_len_larger_than_the_segment_is_an_error_not_an_overlap():
    with pytest.raises(ValueError):
        s10.block_starts(1000, blocks=5, block_len=500)


# --------------------------------------------------------------------------
# Thresholds are derived, and reproduce the brief's table
# --------------------------------------------------------------------------
@pytest.mark.parametrize(("median", "expected"), [(112.78, 60.0), (94.05, 50.0)])
def test_tail_jump_threshold_is_half_a_body_length_rounded_up_to_ten(median, expected):
    assert s10.tail_jump_threshold_px(median) == expected


def test_side_threshold_is_stricter_than_top():
    assert s10.tail_jump_threshold_px(94.05) < s10.tail_jump_threshold_px(112.78)


# --------------------------------------------------------------------------
# The shared-node metric
# --------------------------------------------------------------------------
def _instance(mouth, tail):
    return {
        "mouth": None if mouth is None else np.asarray(mouth, dtype=float),
        "tail": None if tail is None else np.asarray(tail, dtype=float),
    }


def test_shared_tail_is_counted_and_attributed_to_the_tail():
    per_frame = {0: [_instance((0, 0), (100, 100)), _instance((300, 300), (102, 100))]}
    c = s10.score_blocks([[0]], per_frame, tail_jump_px=60.0)
    assert c["pairs_tail"] == 1
    assert c["frames_shared_tail"] == 1
    assert c["pairs_mouth"] == 0
    assert c["frames_shared_any"] == 1


def test_a_pair_just_outside_five_pixels_is_not_shared():
    """A shared node reported off a montage turned out to be 36.3 px. Compute it."""
    per_frame = {0: [_instance((0, 0), (100, 100)), _instance((300, 300), (106, 100))]}
    c = s10.score_blocks([[0]], per_frame, tail_jump_px=60.0)
    assert c["pairs_tail"] == 0
    assert c["frames_shared_any"] == 0


def test_three_coincident_tails_are_three_pairs_but_one_frame():
    per_frame = {0: [_instance((0, 0), (10, 10)) for _ in range(3)]}
    c = s10.score_blocks([[0]], per_frame, tail_jump_px=60.0)
    assert c["pairs_tail"] == 3
    assert c["frames_shared_tail"] == 1


def test_missing_nodes_are_skipped_rather_than_treated_as_the_origin():
    per_frame = {0: [_instance((0, 0), None), _instance((500, 500), None)]}
    c = s10.score_blocks([[0]], per_frame, tail_jump_px=60.0)
    assert c["pairs_tail"] == 0


def test_pairs_at_risk_counts_the_pairs_that_could_have_been_confused():
    per_frame = {
        0: [_instance((0, 0), (i * 100, 0)) for i in range(3)],  # 3 instances -> 3 pairs
        1: [_instance((0, 0), (i * 100, 0)) for i in range(2)],  # 2 instances -> 1 pair
        2: [_instance((0, 0), (0, 0))],  # 1 instance  -> 0 pairs
    }
    c = s10.score_blocks([[0, 1, 2]], per_frame, tail_jump_px=60.0)
    assert c["pairs_at_risk"] == 4


def test_a_model_that_misses_an_animal_cannot_look_clean_on_the_normalised_rate():
    """The trap this denominator exists for: fewer detections, fewer chances to mix."""
    detects_all = {
        i: [_instance((j * 300, 0), (j * 300, 100)) for j in range(3)] for i in range(10)
    }
    misses_one = {
        i: [_instance((j * 300, 0), (j * 300, 100)) for j in range(2)] for i in range(10)
    }
    both = s10.rates(s10.score_blocks([list(range(10))], detects_all, 60.0))
    fewer = s10.rates(s10.score_blocks([list(range(10))], misses_one, 60.0))
    assert both["pairs_at_risk_per_frame"] == 3.0
    assert fewer["pairs_at_risk_per_frame"] == 1.0
    # Neither shares a node here, so the normalised rate is 0 for both — the
    # point is that the denominator differs by 3x and is now visible.
    assert both["shared_per_1k_pairs_at_risk"] == fewer["shared_per_1k_pairs_at_risk"] == 0.0


def test_cardinality_is_context_and_counts_every_frame():
    per_frame = {
        0: [_instance((0, 0), (0, 0))] * 2,
        1: [_instance((0, 0), (0, 0))] * 3,
        2: [_instance((0, 0), (0, 0))] * 4,
    }
    c = s10.score_blocks([[0, 1, 2]], per_frame, tail_jump_px=60.0)
    assert (c["frames_lt3"], c["frames_eq3"], c["frames_gt3"]) == (1, 1, 1)
    assert c["frames"] == 3


# --------------------------------------------------------------------------
# tail_jump — the metric that catches what shared-node cannot
# --------------------------------------------------------------------------
def test_tail_jump_counts_only_moves_beyond_the_threshold():
    per_frame = {
        0: [_instance((0, 0), (100, 100))],
        1: [_instance((0, 0), (140, 100))],  # 40 px, under 60
        2: [_instance((0, 0), (400, 100))],  # 260 px, over
    }
    c = s10.score_blocks([[0, 1, 2]], per_frame, tail_jump_px=60.0)
    assert c["tail_jump_n"] == 2
    assert c["tail_jump_over"] == 1


def test_tail_jump_does_not_bridge_two_blocks():
    """Adjacent indices only exist inside a block; across blocks they are hours apart."""
    per_frame = {
        0: [_instance((0, 0), (0, 0))],
        1: [_instance((0, 0), (0, 0))],
        500: [_instance((0, 0), (900, 900))],
        501: [_instance((0, 0), (900, 900))],
    }
    c = s10.score_blocks([[0, 1], [500, 501]], per_frame, tail_jump_px=60.0)
    assert c["tail_jump_n"] == 2
    assert c["tail_jump_over"] == 0


def test_the_nearest_tail_metric_is_blind_to_a_pure_permutation():
    """The defect this tool documents, pinned so it cannot be quietly forgotten.

    ``tail_jump`` was introduced to catch permutation chimeras. It cannot. A
    permutation moves the instance-to-tail *assignment* and leaves the set of
    tail positions untouched, so every tail still finds a tail 0 px away in the
    next frame.
    """
    a, b = (100.0, 100.0), (600.0, 600.0)
    per_frame = {
        0: [_instance((100, 90), a), _instance((600, 590), b)],
        1: [_instance((100, 90), b), _instance((600, 590), a)],  # tails swapped
    }
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["pairs_tail"] == 0, "a permutation shares no node"
    assert c["frames_shared_any"] == 0
    assert c["tail_jump_n"] == 2
    assert c["tail_jump_over"] == 0, "the pre-registered metric reads this as clean"


def test_the_mouth_linked_metric_does_see_the_permutation():
    """Same frames as above. Mouths are stationary, tails have exchanged."""
    a, b = (100.0, 100.0), (600.0, 600.0)
    per_frame = {
        0: [_instance((100, 90), a), _instance((600, 590), b)],
        1: [_instance((100, 90), b), _instance((600, 590), a)],
    }
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["tail_swap_n"] == 2
    assert c["tail_swap_over"] == 2, "both instances' tails crossed the tank"


def test_the_mouth_linked_metric_is_quiet_when_nothing_swapped():
    per_frame = {
        0: [_instance((100, 90), (100, 100)), _instance((600, 590), (600, 600))],
        1: [_instance((103, 91), (104, 101)), _instance((602, 592), (603, 601))],
    }
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["tail_swap_n"] == 2
    assert c["tail_swap_over"] == 0
    assert c["link_rejected"] == 0


def test_an_untrustworthy_mouth_link_is_rejected_rather_than_counted():
    """A mouth cannot cross half a body length in 6 ms; that is a linking error,
    and counting its tail displacement would manufacture the very events the
    metric exists to find."""
    per_frame = {
        0: [_instance((100, 100), (150, 150))],
        1: [_instance((900, 900), (950, 950))],
    }
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["link_rejected"] == 1
    assert c["tail_swap_n"] == 0


def test_an_instance_missing_a_node_takes_no_part_in_the_mouth_link():
    per_frame = {
        0: [_instance((100, 100), None), _instance((600, 600), (620, 620))],
        1: [_instance((101, 101), (130, 130)), _instance((601, 601), (621, 621))],
    }
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["tail_swap_n"] == 1, "only the instance carrying both nodes can be linked"


def test_a_frame_with_no_successor_detection_is_not_scored_as_a_jump():
    per_frame = {0: [_instance((0, 0), (100, 100))], 1: []}
    c = s10.score_blocks([[0, 1]], per_frame, tail_jump_px=60.0)
    assert c["tail_jump_n"] == 0


# --------------------------------------------------------------------------
# Pooling must be a sum, so group-level aggregation cannot drift from the total
# --------------------------------------------------------------------------
def test_counts_are_additive_so_pooling_is_a_sum():
    per_frame = {0: [_instance((0, 0), (10, 10)), _instance((5, 5), (12, 10))]}
    one = s10.score_blocks([[0]], per_frame, tail_jump_px=60.0)
    pooled = Counter()
    pooled.update(one)
    pooled.update(one)
    assert pooled["frames"] == 2 * one["frames"]
    assert s10.rates(pooled)["shared_frame_rate"] == s10.rates(one)["shared_frame_rate"]


# --------------------------------------------------------------------------
# The decision rule, applied as written
# --------------------------------------------------------------------------
def _result(shared_frames, frames, tail_jump_rate, tail_swap_rate=0.0):
    return {
        "frames": frames,
        "frames_shared_any": shared_frames,
        "shared_frame_rate": shared_frames / frames,
        "tail_jump_rate": tail_jump_rate,
        "tail_swap_rate": tail_swap_rate,
    }


def test_a_big_significant_reduction_with_no_tail_cost_is_a_win():
    outcome = s10.judge(_result(140, 20000, 0.01), _result(200, 20000, 0.01))
    assert outcome["relative_reduction"] == pytest.approx(0.30)
    assert outcome["p"] < 0.05
    assert outcome["verdict"] == "challenger wins"


def test_a_significant_but_small_reduction_is_not_a_win():
    outcome = s10.judge(_result(180, 20000, 0.01), _result(200, 20000, 0.01))
    assert outcome["relative_reduction"] < 0.20
    assert outcome["verdict"] == "no win"


def test_winning_on_shared_node_and_losing_on_tail_jump_is_the_permutation_trade():
    outcome = s10.judge(_result(140, 20000, 0.02), _result(200, 20000, 0.01))
    assert outcome["shared_node_criterion_met"]
    assert not outcome["tail_jump_not_higher"]
    assert outcome["verdict"] == "PERMUTATION TRADE - not a win"


def test_tail_swap_is_reported_but_never_changes_the_pre_registered_verdict():
    """Widening a rule after seeing the numbers is how a rule stops being one."""
    outcome = s10.judge(
        _result(140, 20000, 0.01, tail_swap_rate=0.09),
        _result(200, 20000, 0.01, tail_swap_rate=0.01),
    )
    assert outcome["verdict"] == "challenger wins"
    assert outcome["tail_swap_not_higher"] is False


def test_rates_under_the_power_floor_are_reported_as_indistinguishable():
    outcome = s10.judge(_result(10, 20000, 0.01), _result(20, 20000, 0.01))
    assert outcome["underpowered"]
    assert outcome["verdict"] == "indistinguishable (underpowered)"


def test_the_z_test_matches_a_hand_worked_case():
    # p1 = 0.0070, p2 = 0.0100, pooled = 0.0085, se = 9.1803e-4.
    z, p = s10.two_proportion_z(140, 20000, 200, 20000)
    assert z == pytest.approx(-3.2679, abs=1e-3)
    assert p == pytest.approx(0.0010836, abs=1e-6)


def test_identical_arms_give_p_of_one():
    z, p = s10.two_proportion_z(200, 20000, 200, 20000)
    assert z == 0.0
    assert p == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Trap 1: side1 and side2 share every frame_idx
# --------------------------------------------------------------------------
def test_every_view_maps_to_its_own_camera_folder():
    assert s10.VIEW_CAM == {"top": "top_cam", "side1": "side1_cam", "side2": "side2_cam"}
    assert len(set(s10.VIEW_CAM.values())) == 3


def test_both_side_views_are_scored_with_the_side_models():
    assert s10.VIEW_MODELS["side1"] == s10.VIEW_MODELS["side2"] == "side"
    assert s10.VIEW_MODELS["top"] == "top"


def test_every_comparison_names_models_that_exist_in_both_view_sets():
    for view_models in s10.MODELS_BY_VIEW.values():
        for challenger, baseline in s10.COMPARISONS:
            assert s10.match_model(view_models, challenger) is not None, challenger
            assert s10.match_model(view_models, baseline) is not None, baseline
