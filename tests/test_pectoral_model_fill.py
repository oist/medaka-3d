"""Pins for the model-filled 3-node dataset (`s02 --pectoral-mode model`).

Two properties carry the whole design and are worth failing loudly if they
change:

* a filled pectoral is anchored on the **human** mouth and tail, so a
  prediction that grouped parts from two animals is rejected rather than
  contributing its pectoral;
* an untouched seed is **not** a human label, so round-2's partially reviewed
  side pass can be used without interpolation re-entering dressed as
  annotation.

No network drive, no models — arithmetic only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("_s02", TOOLS / "s02_merge_pectoral.py")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # pragma: no cover - environment, not logic
        pytest.skip(f"s02 not importable here: {exc}")
    return module


s02 = _load()


def _human(mouth, tail):
    return {
        "mouth": (np.array(mouth, dtype=float), True),
        "tail": (np.array(tail, dtype=float), True),
    }


def _prediction(mouth, tail, pectoral):
    return {
        "mouth": None if mouth is None else np.array(mouth, dtype=float),
        "tail": None if tail is None else np.array(tail, dtype=float),
        "pectoral": None if pectoral is None else np.array(pectoral, dtype=float),
    }


# --------------------------------------------------------------------------
# Matching a prediction to a human instance
# --------------------------------------------------------------------------
def test_the_pectoral_comes_from_the_prediction_that_matches_this_animal():
    human = _human((100, 100), (200, 200))
    predictions = [
        _prediction((900, 900), (990, 990), (930, 930)),
        _prediction((101, 101), (199, 199), (130, 130)),
    ]
    got = s02.match_prediction(human, predictions)
    assert got is not None
    assert np.allclose(got, [130, 130])


def test_a_prediction_beyond_the_gate_is_rejected_rather_than_used():
    """A mis-grouped prediction must not donate its pectoral. This is the
    reason a model with mediocre grouping is still safe to seed from."""
    human = _human((100, 100), (200, 200))
    far = [_prediction((600, 600), (700, 700), (630, 630))]
    assert s02.match_prediction(human, far) is None


def test_the_gate_is_the_mean_of_mouth_and_tail_error():
    human = _human((0, 0), (100, 0))
    # mouth exact, tail 30 px out -> mean 15, inside the 20 px gate.
    inside = [_prediction((0, 0), (130, 0), (25, 5))]
    assert s02.match_prediction(human, inside) is not None
    # mouth exact, tail 50 px out -> mean 25, outside.
    outside = [_prediction((0, 0), (150, 0), (25, 5))]
    assert s02.match_prediction(human, outside) is None


def test_a_matched_prediction_without_a_pectoral_yields_nothing():
    human = _human((100, 100), (200, 200))
    assert s02.match_prediction(human, [_prediction((100, 100), (200, 200), None)]) is None


def test_an_instance_missing_mouth_or_tail_cannot_be_matched():
    """It has no anchor, so there is nothing to gate against."""
    half = {"mouth": (np.array([100.0, 100.0]), True), "tail": (None, False)}
    assert s02.match_prediction(half, [_prediction((100, 100), (200, 200), (130, 130))]) is None


def test_a_nan_anchor_is_treated_as_absent():
    nan = {
        "mouth": (np.array([np.nan, np.nan]), True),
        "tail": (np.array([200.0, 200.0]), True),
    }
    assert s02.match_prediction(nan, [_prediction((100, 100), (200, 200), (130, 130))]) is None


def test_no_predictions_at_all_is_not_an_error():
    assert s02.match_prediction(_human((1, 1), (2, 2)), []) is None


# --------------------------------------------------------------------------
# Interpolation fallback must not invent a point it cannot place
# --------------------------------------------------------------------------
def test_interpolation_places_the_pectoral_along_the_chord():
    out = s02.interpolate_pectoral(_human((0, 0), (100, 0)), 0.3)
    assert np.allclose(out["pectoral"][0], [30, 0])


def test_interpolation_declines_when_there_is_no_chord():
    """Every pectoral still NaN in the built dataset is one of these — an
    instance round 8 itself left partial. Verified 137/137 top and 716/716
    side at build time."""
    half = {"mouth": (np.array([1.0, 2.0]), True), "tail": (None, False)}
    assert "pectoral" not in s02.interpolate_pectoral(half, 0.3)


def test_interpolation_declines_on_a_nan_endpoint():
    nan = {
        "mouth": (np.array([np.nan, np.nan]), True),
        "tail": (np.array([100.0, 0.0]), True),
    }
    assert "pectoral" not in s02.interpolate_pectoral(nan, 0.3)


# --------------------------------------------------------------------------
# The thing the round-2 side exclusion was protecting against
# --------------------------------------------------------------------------
def test_the_moved_threshold_is_tight_enough_to_catch_a_real_edit():
    """`MOVED_PX` separates "the annotator placed this" from "the seed did".
    It must be far below any deliberate correction."""
    assert pytest.approx(0.01) == s02.MOVED_PX
    assert s02.MOVED_PX < 0.5


def test_the_match_gate_is_stated_and_finite():
    assert 0 < s02.MATCH_GATE_PX <= 25


def test_model_is_an_explicit_mode_and_there_is_no_default():
    """The tool refuses to choose how unlabelled frames are filled; three
    named modes, none of them implicit."""
    source = (TOOLS / "s02_merge_pectoral.py").read_text(encoding="utf-8")
    assert '"annotated-only", "interpolate", "model"' in source
    assert "required=True" in source


def test_the_fill_model_is_named_not_inferred():
    """Which model placed 70% of the pectorals is a fact the dataset has to
    carry, so it is a constant here and recorded in the sidecar."""
    assert s02.FILL_MODEL_KEY == "topdown 3-node"
