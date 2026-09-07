"""Behavioural tests for the mouth-anchored assignment lab.

The lab's whole claim is *structural*: after assignment, no tail can appear in
two instances — not "rarely", not "after filtering", but unrepresentable. A claim
like that is worth a test, because the day it silently stops holding is the day
every downstream identity chain starts corrupting without a symptom.

Skipped, not failed, if the lab has been retired. `labs/README.md` says a lab
ends by being deleted, and a test that forbids deleting it would invert the rule
these tests exist to enforce.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs" / "20260803_mouth_anchored_assignment"


def _load():
    module_path = LAB / "assignment.py"
    if not module_path.is_file():
        pytest.skip("lab 20260803_mouth_anchored_assignment has been retired")
    spec = importlib.util.spec_from_file_location("_lab_assignment", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def lab():
    return _load()


@pytest.fixture(scope="module")
def span(lab):
    """A span model with the shape of the real one: median ~113 px, tails at 35/153."""
    rng = np.random.default_rng(0)
    return lab.SpanModel.from_spans(np.clip(rng.normal(113.0, 22.0, 4000), 20.0, 210.0))


SHAPE = (1200, 1200)


def _tails(result):
    return [i[1] for i in result.instances if not np.isnan(i[1]).any()]


class TestSpanModel:
    def test_q_peaks_at_the_median(self, span):
        assert span.q(np.array([span.median]))[0] > 0.95
        assert span.q(np.array([span.median * 3]))[0] == pytest.approx(0.0)

    def test_bounds_bracket_the_median(self, span):
        low, high = span.bounds(0.01)
        assert low < span.median < high

    def test_refuses_a_sample_too_small_to_be_a_distribution(self, lab):
        with pytest.raises(ValueError, match="real sample"):
            lab.SpanModel.from_spans([100.0] * 10)


class TestStructuralGuarantee:
    """The one property the method exists to provide."""

    def test_two_mouths_cannot_keep_one_tail(self, lab, span):
        shared = np.array([500.0, 400.0])
        instances = [
            np.array([[400.0, 400.0], shared]),
            np.array([[600.0, 480.0], shared]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        tails = _tails(result)
        assert len(result.instances) == 2, "both anchors must survive; suppression is the bug"
        assert len(tails) <= 1
        assert result.n_mouth_only >= 1

    def test_the_loser_abstains_rather_than_taking_a_phantom(self, lab, span):
        """Frame 274709's shape: 3 mouths, 3 tails, one of the tails not real.

        The phantom is placed outside the plausible span of every free mouth, so
        the correct answer is a mouth-only instance and not a fabricated axis.
        """
        instances = [
            np.array([[400.0, 400.0], [500.0, 400.0]]),
            np.array([[420.0, 470.0], [500.0, 400.0]]),
            np.array([[900.0, 900.0], [905.0, 903.0]]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert result.n_mouth_only >= 1
        for instance in result.instances:
            if not np.isnan(instance[1]).any():
                distance = float(np.hypot(*(instance[0] - instance[1])))
                low, high = span.bounds(0.01)
                assert low <= distance <= high

    @pytest.mark.parametrize("seed", range(25))
    def test_no_shared_tail_on_random_frames(self, lab, span, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 7))
        instances = []
        for _ in range(n):
            mouth = rng.uniform(100, 1100, 2)
            angle = rng.uniform(0, 2 * np.pi)
            tail = mouth + rng.uniform(30, 170) * np.array([np.cos(angle), np.sin(angle)])
            instances.append(np.array([mouth, tail]))
        # Force collisions: half the frames reuse instance 0's tail everywhere.
        if seed % 2 and len(instances) > 1:
            for instance in instances[1:]:
                instance[1] = instances[0][1]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        tails = _tails(result)
        for a in range(len(tails)):
            for b in range(a + 1, len(tails)):
                assert float(np.hypot(*(tails[a] - tails[b]))) >= 5.0


class TestOriginPreference:
    def test_a_mouth_keeps_the_tail_the_model_gave_it(self, lab, span):
        """Two well-separated animals must not swap, even when swapping is feasible.

        This is the failure that rejected the first specification: a span-only
        cost moved 29 correct tails to wrong ones against 2 in the other
        direction, because a foreign tail nearer the *median* span outranks a
        true tail that is merely typical.
        """
        instances = [
            np.array([[400.0, 400.0], [400.0, 480.0]]),
            np.array([[400.0, 620.0], [400.0, 760.0]]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert len(result.instances) == 2
        by_mouth = {round(float(i[0][1])): i[1] for i in result.instances}
        assert by_mouth[400][1] == pytest.approx(480.0)
        assert by_mouth[620][1] == pytest.approx(760.0)

    def test_an_implausible_own_tail_is_refused_not_kept(self, lab, span):
        instances = [np.array([[600.0, 600.0], [600.0, 604.0]])]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert result.n_mouth_only == 1
        assert np.isnan(result.instances[0][1]).all()


class TestGates:
    def test_edge_rule_uses_the_mouth_bound(self, lab):
        assert pytest.approx(28.8) == lab.EDGE_MIN_MOUTH_PX
        assert lab.EDGE_MIN_MOUTH_PX < lab.EDGE_MIN_PX, (
            "the mouth bound must be looser than the centroid bound it was "
            "mistakenly given; a mouth sits closer to the edge than its centroid"
        )

    def test_a_mouth_in_the_corner_is_dropped(self, lab, span):
        instances = [
            np.array([[5.0, 5.0], [80.0, 80.0]]),
            np.array([[600.0, 600.0], [600.0, 710.0]]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert result.n_dropped_edge == 1
        assert len(result.instances) == 1

    def test_coincident_mouths_are_one_animal(self, lab, span):
        instances = [
            np.array([[600.0, 600.0], [600.0, 710.0]]),
            np.array([[603.0, 601.0], [700.0, 640.0]]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert result.n_dropped_dedup_mouth == 1
        assert len(result.instances) == 1

    def test_a_tail_only_prediction_is_a_candidate_not_an_instance(self, lab, span):
        instances = [
            np.array([[np.nan, np.nan], [500.0, 400.0]]),
            np.array([[400.0, 400.0], [np.nan, np.nan]]),
        ]
        result = lab.assign_frame_origin(instances, SHAPE, span)
        assert len(result.instances) == 1
        assert result.instances[0][1] == pytest.approx(np.array([500.0, 400.0]))

    def test_an_empty_frame_is_not_an_error(self, lab, span):
        result = lab.assign_frame_origin([], SHAPE, span)
        assert result.instances == []
