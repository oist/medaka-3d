"""Human encounter evaluation must abstain honestly and catch returning changes."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from medaka3d.eval.encounter_review import score_encounter_review

LAB = Path(__file__).parents[1] / "labs/20260909_three_body"
SPEC = importlib.util.spec_from_file_location(
    "three_body_evaluate_review", LAB / "evaluate_review.py"
)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)
TRUTH_SPEC = importlib.util.spec_from_file_location("review_truth_fixture", LAB / "review_truth.py")
assert TRUTH_SPEC and TRUTH_SPEC.loader
truth = importlib.util.module_from_spec(TRUTH_SPEC)
TRUTH_SPEC.loader.exec_module(truth)


def evidence():
    frames = np.array([10, 11, 12])
    pixels = np.zeros((3, 3, 3, 2), dtype=float)
    for frame in range(3):
        for animal in range(3):
            for view in range(3):
                pixels[frame, animal, view] = [100 + 100 * animal + frame + view * 3, 200 + view]
    return {
        "frame_idx": frames, "projected_mouth": pixels.copy(),
        "observed_mouth": np.ones((3, 3), dtype=bool),
        "review_mask": np.zeros((3, 3), dtype=bool), "anchor_frames": frames.copy(),
        "human_mouth": pixels.copy(), "continuity": np.full((3, 2), "confirmed", dtype="U12"),
    }


def human_input():
    data = evidence()
    review = truth.make_template([10, 11, 12], "a" * 64, [10, 11, 12])
    review.update(reviewer="synthetic_unit_fixture", status="human_draft_requires_adjudication")
    for probe, anchor in enumerate(review["anchors"]):
        anchor["frame_choice"] = "human_selected"
        for animal, name in enumerate(truth.ANIMALS):
            anchor["animals"][name] = {
                "visibility": "visible", "points": {
                    view: data["human_mouth"][probe, animal, vi].tolist()
                    for vi, view in enumerate(truth.VIEWS)
                },
            }
    for state in review["continuity"].values():
        state.update(entry_to_after="confirmed", after_to_exit="confirmed",
                     basis="direct_visual_continuity")
    sequence = {"frames": [10, 11, 12], "fps": 166, "image_shape_hw": [1200, 1200],
                "experiment": review["experiment"]}
    return review, sequence


def candidate():
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    return {"frame_idx": np.array([10, 11, 12]), "pose_xyz": np.ones((3, 3, 3, 3)),
            "observed": np.ones((3, 3, 3), dtype=bool), "sources": source,
            "identity_review": np.zeros((3, 3), dtype=bool)}


def test_returning_change_has_preserved_endpoints_but_two_changed_intervals_per_animal():
    data = evidence()
    data["projected_mouth"][1] = data["projected_mouth"][1, [1, 0, 2]]
    report = score_encounter_review(**data)
    for arm in ("raw", "accepted"):
        result = report[arm]
        assert result["interval_counts"] == {"preserved": 2, "changed": 4, "unscored": 0}
        assert result["entry_exit_only_counts"]["same_as_entry"] == 3
        assert result["returning_changes_seen"] == 2
        assert result["post_entry_checkpoint_counts"]["different_from_entry"] == 2
    json.dumps(report, allow_nan=False)


def test_global_slot_permutation_cannot_change_identity_scores():
    data = evidence()
    data["projected_mouth"][1] = data["projected_mouth"][1, [1, 0, 2]]
    original = score_encounter_review(**data)
    data["projected_mouth"] = data["projected_mouth"][:, [2, 0, 1]]
    changed = score_encounter_review(**data)
    assert changed["raw"]["entry_animal_to_slot"] == {"A": 1, "B": 2, "C": 0}
    assert original["raw"]["interval_counts"] == changed["raw"]["interval_counts"]
    assert original["raw"]["returning_change_probes"] == changed["raw"]["returning_change_probes"]


def test_ambiguous_and_unreviewed_links_stay_in_denominator_and_break_entry_chain():
    data = evidence()
    data["continuity"][0, 0] = "ambiguous"
    data["continuity"][1, 1] = "unreviewed"
    report = score_encounter_review(**data)
    assert report["human_edge_states"] == {
        "confirmed": 4, "ambiguous": 1, "unreviewed": 1, "not_visible": 0,
    }
    assert report["raw"]["requested_animal_intervals"] == 6
    assert report["raw"]["interval_counts"] == {"preserved": 3, "changed": 0, "unscored": 3}
    assert report["raw"]["returning_change_probes"][0]["outcome"] == "unscored"


def test_nonunique_slots_abstain_even_when_review_would_remove_competitor():
    data = evidence()
    data["projected_mouth"][1, 1] = data["projected_mouth"][1, 0]
    data["review_mask"][1, 1] = True
    report = score_encounter_review(**data)
    for arm in ("raw", "accepted"):
        match = report[arm]["anchors"][1]["animals"][0]
        assert match["slot"] is None
        assert match["reason"] == "nonunique one-to-one correspondence"
        assert report[arm]["interval_counts"]["unscored"] == 4


def test_unobserved_latent_competitor_does_not_create_unique_identity_by_disappearing():
    data = evidence()
    data["projected_mouth"][1, 1] = data["projected_mouth"][1, 0]
    data["observed_mouth"][1, 1] = False
    report = score_encounter_review(**data)
    assert report["raw"]["anchors"][1]["animals"][0]["slot"] is None
    assert report["raw"]["anchors"][1]["animals"][0]["eligible_slots"] == [0, 1]


def test_cross_view_disagreement_abstains_instead_of_choosing_nearest_view():
    data = evidence()
    data["projected_mouth"][1, :, 1] = data["projected_mouth"][1, [1, 0, 2], 1]
    report = score_encounter_review(**data)
    assert report["raw"]["interval_counts"] == {"preserved": 2, "changed": 0, "unscored": 4}
    assert report["raw"]["anchors"][1]["animals"][0]["eligible_slots"] == []


def test_joint_unique_matching_allows_partial_missing_without_punishing_other_animals():
    data = evidence()
    data["human_mouth"][:, 2] = np.nan
    data["continuity"][2] = "not_visible"
    report = score_encounter_review(**data)
    assert report["raw"]["interval_counts"] == {"preserved": 4, "changed": 0, "unscored": 2}
    assert report["raw"]["entry_animal_to_slot"]["C"] is None


def test_two_humans_competing_for_one_slot_both_abstain():
    data = evidence()
    data["human_mouth"][1, 1] = data["human_mouth"][1, 0]
    result = score_encounter_review(**data)["raw"]
    assert [row["slot"] for row in result["anchors"][1]["animals"]] == [None, None, 2]


def test_finite_predictions_are_not_measurements_and_reviewed_slots_are_not_accepted():
    data = evidence()
    data["observed_mouth"][1, 0] = False
    data["review_mask"][1, 1] = True
    report = score_encounter_review(**data)
    assert report["raw"]["interval_counts"]["unscored"] == 2
    assert report["accepted"]["interval_counts"]["unscored"] == 4
    assert report["accepted"]["anchors"][1]["animals"][1]["reason"] == "identity requires review"


def test_radius_is_fixed_five_pixels_and_missing_candidate_frame_is_unscored():
    data = evidence()
    data["projected_mouth"][1, 0, :, 0] += 5
    assert score_encounter_review(**data)["raw"]["interval_counts"]["preserved"] == 6
    data["projected_mouth"][1, 0, :, 0] += 0.0001
    assert score_encounter_review(**data)["raw"]["interval_counts"]["unscored"] == 2
    for key in ("frame_idx", "projected_mouth", "observed_mouth", "review_mask"):
        data[key] = data[key][[0, 2]]
    report = score_encounter_review(**data)
    assert report["raw"]["interval_counts"]["unscored"] == 6
    assert report["raw"]["anchors"][1]["animals"][0]["reason"] == "candidate frame unavailable"
    assert report["raw"]["entry_exit_only_counts"]["same_as_entry"] == 3
    assert report["raw"]["returning_change_probes"][0]["outcome"] == "unscored"


def test_template_cannot_be_truth_and_human_draft_does_not_require_second_approval(tmp_path):
    review, sequence = human_input()
    original = copy.deepcopy(review)
    evaluation.validate_inputs(review, sequence, "a" * 64)
    assert review == original
    blank = truth.make_template([10, 11, 12], "a" * 64, [10, 11, 12])
    with pytest.raises(ValueError, match="reviewer is missing"):
        evaluation.validate_inputs(blank, sequence, "a" * 64)
    blank["reviewer"] = "placeholder_is_not_a_review"
    with pytest.raises(ValueError, match="template cannot be ground truth"):
        evaluation.validate_inputs(blank, sequence, "a" * 64)


@pytest.mark.parametrize("change,match", [
    (lambda r, s: r.update(source_manifest_sha256="b" * 64), "checksum"),
    (lambda r, s: r.update(source_fps=30), "166 fps"),
    (lambda r, s: r.update(coordinate_system="resized pixels"), "native mouth"),
    (lambda r, s: r.update(source_frame_range=[10, 14]), "frame range"),
    (lambda r, s: s.update(image_shape_hw=[600, 600]), "1200"),
    (lambda r, s: r["anchors"][1].update(frame=11.5), "integer source"),
    (lambda r, s: r["anchors"][1].update(frame_choice="proposed_not_human_confirmed"),
     "human-selected"),
    (lambda r, s: r["anchors"][1]["animals"]["A"]["points"].update(top_cam=[1200, 30]),
     "outside"),
])
def test_manifest_and_native_coordinate_binding(change, match):
    review, sequence = human_input()
    change(review, sequence)
    with pytest.raises(ValueError, match=match):
        evaluation.validate_inputs(review, sequence, "a" * 64)


def test_candidate_requires_measured_mouth_sources_and_consistent_native_frames():
    arrays = candidate()
    evaluation.validate_candidate(arrays, np.array([10, 11, 12]))
    arrays["sources"][1, 0, :2, 0] = -1
    with pytest.raises(ValueError, match="two source views"):
        evaluation.validate_candidate(arrays, np.array([10, 11, 12]))
    arrays["observed"][1, 0, 0] = False
    evaluation.validate_candidate(arrays, np.array([10, 11, 12]))
    arrays["frame_idx"][1] = 12
    with pytest.raises(ValueError, match="consecutive subset"):
        evaluation.validate_candidate(arrays, np.array([10, 11, 12]))


def test_projects_all_latent_mouths_including_unsupported_without_pectoral_substitution():
    arrays = candidate()
    arrays["pose_xyz"][:, :, 0] = [100, 200, 50]
    arrays["pose_xyz"][:, :, 1] = [800, 900, 50]
    arrays["observed"][1, 0, 0] = False

    class Camera:
        def project(self, xyz):
            return xyz[..., :2]

    class Rig:
        def camera(self, view):
            return Camera()

    result = evaluation.project_mouth(Rig(), arrays, np.array([10, 11, 12]))
    np.testing.assert_array_equal(result[0, 0], [[100, 200]] * 3)
    np.testing.assert_array_equal(result[1, 0], [[100, 200]] * 3)


def test_completely_missing_candidate_probes_do_not_call_camera_on_empty_arrays():
    arrays = candidate()
    arrays["pose_xyz"][:] = np.nan
    arrays["observed"][:] = False

    class Rig:
        def camera(self, view):
            raise AssertionError("no finite latent mouths need projection")

    projected = evaluation.project_mouth(Rig(), arrays, np.array([10, 11, 12]))
    data = evidence()
    data["projected_mouth"] = projected
    data["observed_mouth"][:] = False
    report = score_encounter_review(**data)
    assert report["raw"]["interval_counts"]["unscored"] == 6
    assert report["accepted"]["interval_counts"]["unscored"] == 6


def test_result_is_json_safe_and_does_not_mutate_inputs():
    data = evidence()
    before = copy.deepcopy(data)
    json.dumps(score_encounter_review(**data), allow_nan=False)
    for key in data:
        np.testing.assert_array_equal(data[key], before[key])
    data["human_mouth"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="both coordinates NaN"):
        score_encounter_review(**data)


def completed_fixture(tmp_path, monkeypatch):
    """Synthetic I/O fixture only; no actual clip acquires invented human labels."""
    source, run, inputs = tmp_path / "sequence", tmp_path / "run", tmp_path / "inputs"
    for path in (source, run, inputs):
        path.mkdir()
    human, sequence = human_input()
    manifest = source / "manifest.json"
    manifest.write_text(json.dumps(sequence), encoding="utf-8")
    human["source_manifest_sha256"] = evaluation.sha256(manifest)
    review_path = tmp_path / "human_fixture.json"
    review_path.write_text(json.dumps(human), encoding="utf-8")
    calibration = inputs / "calibration_fixture.json"
    calibration.write_text("{}", encoding="utf-8")
    arrays = candidate()
    arrays["pose_xyz"][:, :, 0, :2] = evidence()["projected_mouth"][:, :, 0]
    candidate_path = run / "candidate.npz"
    np.savez(candidate_path, **arrays)
    registered = {str(manifest): evaluation.sha256(manifest),
                  str(calibration): evaluation.sha256(calibration)}
    summary = {
        "technical_completion": True, "experiment": sequence["experiment"], "native_fps": 166,
        "temporal_subsampling": False, "frames": [10, 13], "returned_frames": 3,
        "input_sha256_before": registered, "input_sha256_after": registered,
        "inputs_unchanged_during_run": True,
    }
    summary_path = run / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    record = {"status": "ok", "outputs": {
        "candidate": {"sha256": evaluation.sha256(candidate_path)},
        "summary": {"sha256": evaluation.sha256(summary_path)},
    }}
    (run / "experiment.provenance.json").write_text(json.dumps(record), encoding="utf-8")

    class Camera:
        def __init__(self, view):
            self.index = evaluation.VIEWS.index(view)

        def project(self, xyz):
            return xyz[..., :2] + [self.index * 3, self.index]

    class Rig:
        def camera(self, view):
            return Camera(view)

    monkeypatch.setattr(evaluation, "verified_raw", lambda *args: {})
    monkeypatch.setattr(evaluation, "verified_calibration", lambda root: (
        Rig(), {calibration: evaluation.sha256(calibration)}
    ))
    return review_path, source, candidate_path, inputs, tmp_path / "evaluation"


def test_copies_exact_human_draft_and_preserves_inputs_no_overwrite(tmp_path, monkeypatch):
    args = completed_fixture(tmp_path, monkeypatch)
    human_before, candidate_before = args[0].read_bytes(), args[2].read_bytes()
    result = evaluation.evaluate(*args)
    assert result["scoring"]["raw"]["interval_counts"] == {
        "preserved": 6, "changed": 0, "unscored": 0,
    }
    assert result["human_input_metadata"]["status"] == "human_draft_requires_adjudication"
    assert (args[-1] / "human_review.json").read_bytes() == human_before
    assert args[0].read_bytes() == human_before and args[2].read_bytes() == candidate_before
    record = evaluation.read_json(args[-1] / "evaluation.provenance.json")
    assert record["status"] == "ok"
    assert len(list(args[-1].iterdir())) == 3
    with pytest.raises(FileExistsError):
        evaluation.evaluate(*args)


def test_changed_candidate_bytes_are_rejected_before_creating_output(tmp_path, monkeypatch):
    args = completed_fixture(tmp_path, monkeypatch)
    with args[2].open("ab") as stream:
        stream.write(b"changed after completed inference")
    with pytest.raises(ValueError, match="candidate checksum mismatch"):
        evaluation.evaluate(*args)
    assert not args[-1].exists()


def test_unreviewed_template_rejected_before_candidate_or_output_io(tmp_path, monkeypatch):
    args = completed_fixture(tmp_path, monkeypatch)
    blank = truth.make_template([10, 11, 12], evaluation.sha256(args[1] / "manifest.json"),
                                [10, 11, 12])
    args[0].write_text(json.dumps(blank), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer is missing"):
        evaluation.evaluate(*args)
    assert not args[-1].exists()


def test_completed_run_must_be_bound_to_actual_source_bytes(tmp_path, monkeypatch):
    args = completed_fixture(tmp_path, monkeypatch)
    summary_path = args[2].parent / "summary.json"
    record_path = args[2].parent / "experiment.provenance.json"
    summary = evaluation.read_json(summary_path)
    summary["input_sha256_before"] = summary["input_sha256_after"] = {"other": "0" * 64}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    record = evaluation.read_json(record_path)
    record["outputs"]["summary"]["sha256"] = evaluation.sha256(summary_path)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound to these"):
        evaluation.evaluate(*args)
    assert not args[-1].exists()
