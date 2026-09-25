"""Independent baseline controls: row readout, only-scale OKS, frozen P parity."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
LAB = ROOT / "labs/20260917_variable_count_tracklets"


def load(name):
    spec = importlib.util.spec_from_file_location("variable_baseline_" + name, LAB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASELINE = load("baseline")
FROZEN = load("frozen_p0")


def test_frozen_p0_is_byte_identical_to_prior_source():
    assert (LAB / "frozen_p0.py").read_bytes() == (
        ROOT / "labs/20260917_clean2d_video_comparison/tracking.py"
    ).read_bytes()


def test_s1_uses_instance_identity_instead_of_return_order():
    frames = np.array([0, 0, 1, 1, 1])
    points = np.array([[[x, 1.0], [x + 4, 6.0], [x - 10, 10.0]] for x in [0, 100, 500, 0.1, 100.1]])
    labels, legacy, metadata = BASELINE.replay(frames, points, np.ones(5), 0, 2, view="top_cam")
    assert metadata["reordered_returns"][0]["returned_input_row_order"] == [1, 2, 0]
    assert labels[0] == labels[3] and labels[1] == labels[4]
    assert labels[2] not in labels[:2]
    assert np.count_nonzero(legacy != labels) == 3


def test_s2_is_positive_and_rotation_independent_with_missing_mouth():
    import sleap_nn.tracking.tracker as installed
    from sleap_nn.evaluation import compute_oks

    before = np.array([[np.nan, np.nan], [100.0, 50.0]])
    after = np.array([[np.nan, np.nan], [105.0, 50.0]])
    original = installed.compute_oks
    assert float(compute_oks(before, after)[0, 0]) == 0
    with BASELINE.oks_area_override(BASELINE.fixed_area("top_cam")):
        score = float(installed.compute_oks(before, after)[0, 0])
        rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
        assert np.isclose(installed.compute_oks(before @ rotation, after @ rotation)[0, 0], score)
        assert np.isclose(score, np.exp(-25 / 72))
    assert installed.compute_oks is original


def test_s2_keeps_all_non_scale_tracking_settings():
    frames = np.arange(4)
    points = np.array([[[i + 2.0, 1.0], [i + 4.0, 6.0], [i - 10.0, 10.0]] for i in frames])
    _, _, s1 = BASELINE.replay(frames, points, np.ones(4), 0, 4, view="side1_cam")
    _, _, s2 = BASELINE.replay(frames, points, np.ones(4), 0, 4, view="side1_cam", fixed_scale=True)
    for key in (
        "window_size",
        "min_match_points",
        "min_new_track_points",
        "max_tracks",
        "scoring_method",
        "scoring_reduction",
        "oks_stddev",
        "use_flow",
    ):
        assert s1[key] == s2[key]
    assert s1["fixed_oks_area_px2"] is None
    assert s2["fixed_oks_area_px2"] == 10000


def test_predecessors_do_not_create_zero_frame_or_unassigned_edges():
    frames = np.array([0, 0, 1, 3, 3])
    labels = np.array([0, -1, 0, 0, -1])
    previous, gap = BASELINE.predecessors(frames, labels)
    np.testing.assert_array_equal(previous, [-1, -1, 0, 2, -1])
    np.testing.assert_array_equal(gap, [0, 0, 1, 2, 0])


def test_frozen_p0_reproduces_prior_context_labels_and_edges():
    path = ROOT / "work/clean2d_video_comparison_v1/known_323819/tracking_P_side1_cam.npz"
    with np.load(path, allow_pickle=False) as prior:
        result = FROZEN.build_tracklets(
            prior["frame_idx"], prior["points"], view="side1_cam", feature_mode="pectoral"
        )
        np.testing.assert_array_equal(result.labels, prior["labels"])
        np.testing.assert_array_equal(result.clean_edge, prior["clean_edge"])


@pytest.mark.skipif(
    not (ROOT / "work/variable_count_tracklets_v1/summary.json").exists(),
    reason="requires completed local numerical run; synthetic controls run independently",
)
def test_completed_run_preserves_every_original_observation_and_one_to_one_identity():
    output = ROOT / "work/variable_count_tracklets_v1"
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["records"]) == 12 * 3 * 8
    for view in ("top_cam", "side1_cam", "side2_cam"):
        source = (
            ROOT
            / "work/generative_tracking_inputs/work/c01_s16_predictions/bottomup"
            / f"20251225_K_1_2_FCMC__{view}.npz"
        )
        with np.load(source, allow_pickle=False) as z:
            raw = {key: z[key] for key in ("frame_idx", "points", "score")}
        for case in protocol["cases"]:
            lo, hi = np.searchsorted(raw["frame_idx"], [case["start"] - 332, case["stop"] + 332])
            expected_rows = np.arange(lo, hi)
            for condition in protocol["conditions"]:
                with np.load(
                    output / case["name"] / f"tracking_{condition}_{view}.npz",
                    allow_pickle=False,
                ) as actual:
                    for key in ("frame_idx", "points", "score"):
                        np.testing.assert_array_equal(actual[key], raw[key][lo:hi])
                        assert actual[key].dtype == raw[key].dtype
                    np.testing.assert_array_equal(actual["global_row"], expected_rows)
                    np.testing.assert_array_equal(
                        actual["observed_mask"], np.isfinite(raw["points"][lo:hi]).all(axis=-1)
                    )
                    assert set(actual["interpretation"]) == {"unclassified"}
                    assert "point_scores" not in actual.files
                    previous, gaps = BASELINE.predecessors(actual["frame_idx"], actual["labels"])
                    np.testing.assert_array_equal(actual["previous_row"], previous)
                    np.testing.assert_array_equal(actual["clean_edge"], gaps == 1)
                    if condition.startswith("P"):
                        assert np.all(actual["labels"] >= 0)
                        assert not np.any(gaps > 1)
