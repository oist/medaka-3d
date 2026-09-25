"""Small fixed-cost adapter contracts; no old renderer/science suite rerun."""

from pathlib import Path

import compare_saved_new_top_background_image_terms as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path / "allowed")
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "OLD_ROOT", tmp_path / "old")
    monkeypatch.setattr(m, "BACKGROUND_ROOT", tmp_path / "background")

    def forbidden(*args, **kwargs):
        pytest.fail("real input/renderer entry reached by synthetic contract test")

    for module, names in (
        (m, ("load_headers", "source_code")),
        (m.old, ("load_headers", "load_frame", "image", "image_terms")),
        (m.saved, ("capture", "load_bank", "load_rig")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.resolve().is_relative_to(tmp_path), "real file reached"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)


def fixture_terms():
    n, count = 3, 4
    values = dict(
        silhouette=np.array([0.2, 0.4, 0.4]),
        per_view_silhouette=np.array([[0.0, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.1, 0.1]]),
        per_view_positive_near_global=np.zeros((n, 3, 3)),
        sampled_pixels=np.zeros((3, count, 2)),
        sampled_target=np.zeros((3, count), np.float32),
        sampled_weights=np.ones((3, count), np.float32),
        sampled_foreground=np.zeros((3, count), bool),
        ray_valid=np.ones((3, count), bool),
        ray_origins=np.zeros((3, count, 3)),
        ray_directions=np.ones((3, count, 3)),
    )
    bank = {k: np.zeros(n) for k in m.saved.FIELDS}
    bank.update(
        candidate_index=np.arange(n),
        candidate_pose_xyz=np.zeros((n, 3, 3, 3)),
        base_local_cost=np.array([10.0, 9.0, 9.0]),
        sixth_instance_term=np.array([1.0, 2.0, 2.0]),
        adjusted_local_cost=np.array([11.0, 11.0, 11.0]),
        parent_path_cost=np.array([3.0, 2.0, 2.0]),
    )
    bank["adjusted_path_cost"] = bank["parent_path_cost"] + bank["adjusted_local_cost"]
    for mode in m.old.MODES:
        bank.update({mode + "__" + k: v.copy() for k, v in values.items()})
    bank["detectron2__silhouette"] = np.array([0.1, 0.1, 0.1])
    delta = 3 * (values["silhouette"] - bank["detectron2__silhouette"])
    bank["counterfactual_local_cost"] = bank["adjusted_local_cost"] + delta
    bank["counterfactual_path_cost"] = bank["parent_path_cost"] + bank["counterfactual_local_cost"]
    return bank, values


def test_same_background_retains_old_ranks_and_all_fixed_costs():
    bank, new = fixture_terms()
    before = {k: v.copy() for k, v in bank.items()}
    arrays, report = m.compare_terms(bank, new, 3.0)
    assert m.byte_equal(arrays["new_background_local_cost"], bank["counterfactual_local_cost"])
    assert m.byte_equal(arrays["new_background_path_cost"], bank["counterfactual_path_cost"])
    assert report["rankings"]["new_background"]["fixed_parent_path"]["exact_minimum_indices"] == [
        1,
        2,
    ]
    assert report["top_old_to_new_sample_positions_changed"] == 0
    assert all(m.byte_equal(v, before[k]) for k, v in bank.items())
    assert arrays["sixth_instance_term"] is bank["sixth_instance_term"]


@pytest.mark.parametrize("field", ["sampled_pixels", "ray_origins", "per_view_silhouette"])
def test_side_change_refused_even_when_full_image_cost_is_same(field):
    bank, new = fixture_terms()
    index = (0, 1) if field == "per_view_silhouette" else (1, 0, 0)
    new[field][index] += 1
    with pytest.raises(ValueError, match="side sample/ray/cost"):
        m.compare_terms(bank, new, 3.0)


def test_top_cost_uses_full_reduction_and_parent_after_local_delta():
    bank, new = fixture_terms()
    new["silhouette"] = np.array([0.3, 0.2, 0.4])
    new["per_view_silhouette"][:, 0] = new["silhouette"] - 0.2
    # Artificial last-bit reduction discrepancy is reported, never substituted.
    new["per_view_silhouette"][0, 0] = np.nextafter(new["per_view_silhouette"][0, 0], np.inf)
    new["sampled_pixels"][0, 0] = [5.0, 6.0]
    arrays, report = m.compare_terms(bank, new, 3.0)
    expected = bank["parent_path_cost"] + (
        bank["adjusted_local_cost"] + 3 * (new["silhouette"] - bank["detectron2__silhouette"])
    )
    assert m.byte_equal(arrays["new_background_path_cost"], expected)
    assert report["rankings"]["new_background"]["fixed_parent_path"]["first_index"] == 1
    assert report["top_old_to_new_sample_positions_changed"] == 1
    assert np.array_equal(
        arrays["new_background_per_view_reduction_minus_full"],
        new["per_view_silhouette"].sum(1) - new["silhouette"],
    )


def test_output_guard_stops_before_any_actual_input(tmp_path):
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(tmp_path / "outside")
