"""The post-inference comparison cannot silently modify fixed geometry or ID evidence."""

import copy
import importlib
import json
import sys
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {n: sys.modules.get(n) for n in names}, sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        for name in names:
            sys.modules.pop(name, None)
        return importlib.import_module("evaluate_identity_dp")
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_module()


@pytest.fixture
def solution():
    n = 168
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(3)[None, :, None] * 20
    pose[..., 1] = np.arange(3)[None, None] * 3
    pose[..., 2] = np.arange(n)[:, None, None] / 100
    observed = np.ones((n, 3, 3), bool)
    observed[165:167, 0, 1] = False
    observed[167, 2, 0] = False
    review = np.zeros((n, 3), bool)
    review[165:, 0] = True
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    for t, fish, node in np.argwhere(~observed):
        source[t, fish, 1:, node] = -1
    xyz = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    geometry = dict(
        frame_idx=np.arange(333519, 333519+n, dtype=np.int64), pose_xyz=pose,
        sources=source, observed=observed, identity_review=review,
        xyz=xyz, accepted_xyz=np.where(~review[..., None], xyz, np.nan),
    )
    scores = dict(
        frame_idx=geometry["frame_idx"].copy(), unary=np.zeros((n, 6)),
        per_view_costs=np.zeros((n, 6, 3)), enabled_views=np.zeros((n, 3), bool),
        sampled_pixels=np.zeros((n, 3), np.int64), valid_pixels=np.zeros((n, 3), np.int64),
        available_core_pixels=np.zeros((n, 3), np.int64), visible_mass=np.zeros((n, 3, 3)),
        rendering_performed=np.zeros(n, bool),
    )
    scores["unary"][166:] = np.arange(6)[None] * .03 + .1
    scores["per_view_costs"][166:, :, 0] = scores["unary"][166:]
    scores["enabled_views"][166:, 0] = True
    scores["sampled_pixels"][166:, 0] = 12
    scores["valid_pixels"][166:, 0] = 10
    scores["available_core_pixels"][166:, 0] = 90
    scores["visible_mass"][166:, 0] = [2., 3., 1.]
    scores["rendering_performed"][166:] = True
    config = copy.deepcopy(m.FIXED_CONFIG)
    motion = m.build_adjacent_motion_cost(
        geometry["frame_idx"], xyz, observed[:, :, 1], scale_mm=.8, weight=.2
    )
    for key in ("transitions", "supported_pairs", "supported_pair_count", "squared_distance_mm2"):
        scores[key] = getattr(motion, key)
    solved = m.solve_identity_dp(geometry["frame_idx"], scores["unary"], motion.transitions)
    dp, scalars = m.split_solution(solved)
    identity = m.build_identity_output(geometry, solved.permutation_by_id)
    summary = {"config": config, "dp": scalars, **m.solution_totals(solved, scores)}
    enrollment = {"enabled_views": np.array([True, False, False])}
    return geometry, enrollment, identity, dp, scores, summary


def test_valid_replay_preserves_raw_and_review_without_mutating_inputs(solution):
    before = [{key: value.tobytes() for key, value in item.items()}
              for item in solution[:5]]
    solved, counts = m.validate_saved_solution(*solution)
    assert solved.status == "optimal"
    assert counts["per_native_frame_counts_conserved"]
    assert counts["direct_pectoral_fish_frames"] == 502
    assert counts["conservative_accepted_fish_frames"] == 501
    assert before == [{key: value.tobytes() for key, value in item.items()}
                      for item in solution[:5]]


@pytest.mark.parametrize("problem", [
    "raw_pose", "cleared_review", "invented_observed", "accepted_latent", "mapping",
    "dp_path", "dp_gap", "dp_scalar", "unknown_identity_key", "unknown_dp_key",
    "wrong_motion", "missing_motion_nan", "unary", "nonfinite_unary", "negative_cost",
    "prefix_image", "disabled_view", "pixel_overflow", "visible_mass", "int32_count",
    "rendering_flag", "summary_total", "float_frames", "unknown_scores",
])
def test_corrupted_saved_objective_export_or_counts_are_rejected(solution, problem):
    geometry, enrollment, identity, dp, scores, summary = solution
    if problem == "raw_pose":
        identity["raw_pose_xyz"][0, 0, 0, 0] += 1
    elif problem == "cleared_review":
        identity["prior_review_for_proposed_identity"][165:] = False
    elif problem == "invented_observed":
        identity["proposed_observed"][165:] = True
    elif problem == "accepted_latent":
        identity["conservative_accepted_xyz"][165, 0] = identity["proposed_pose_xyz"][165, 0, 1]
    elif problem == "mapping":
        identity["identity_to_raw_slot"][-1] = [2, 0, 1]
    elif problem == "dp_path":
        dp["path_state"][-1] = 5
    elif problem == "dp_gap":
        dp["min_marginal_delta"][-1, 0] += .1
    elif problem == "dp_scalar":
        summary["dp"]["path_cost"] += 1
    elif problem == "unknown_identity_key":
        identity["accepted_ids"] = np.ones(1)
    elif problem == "unknown_dp_key":
        dp["confidence"] = np.ones(1)
    elif problem == "wrong_motion":
        scores["transitions"][0, 0, 0] += .1
    elif problem == "missing_motion_nan":
        scores["squared_distance_mm2"][~scores["supported_pairs"]] = 0
    elif problem == "unary":
        scores["unary"][-1, 0] += .1
    elif problem == "nonfinite_unary":
        scores["unary"][-1, 0] = np.nan
    elif problem == "negative_cost":
        scores["per_view_costs"][-1, 0, 0] = -.1
    elif problem == "prefix_image":
        scores["available_core_pixels"][0, 0] = 1
    elif problem == "disabled_view":
        scores["per_view_costs"][-1, 0, 1] = 1
    elif problem == "pixel_overflow":
        scores["valid_pixels"][-1, 0] = 13
    elif problem == "visible_mass":
        scores["visible_mass"][-1, 0, 0] = 100
    elif problem == "int32_count":
        scores["valid_pixels"] = scores["valid_pixels"].astype(np.int32)
    elif problem == "rendering_flag":
        scores["rendering_performed"][-1] = False
    elif problem == "summary_total":
        summary["mapping_changes"] += 1
    elif problem == "float_frames":
        scores["frame_idx"] = scores["frame_idx"].astype(float)
    else:
        scores["confidence"] = np.ones(1)
    with pytest.raises(ValueError):
        m.validate_saved_solution(geometry, enrollment, identity, dp, scores, summary)


def test_motion_only_uses_fixed_transitions_and_zero_color_cost_and_preserves_review(solution):
    geometry, _, _, _, scores, summary = solution
    reference, output = m.motion_only_solution(geometry, scores, summary["config"])
    scores["unary"][:] = np.arange(6)[None] * 1000
    repeat, repeat_output = m.motion_only_solution(geometry, scores, summary["config"])
    for key, value in m.split_solution(reference)[0].items():
        m.exact_array(value, getattr(repeat, key), key)
    for key, value in output.items():
        m.exact_array(value, repeat_output[key], key)
    expected = m.solve_identity_dp(
        geometry["frame_idx"], np.zeros_like(scores["unary"]), scores["transitions"]
    )
    assert reference.path_cost == expected.path_cost
    assert reference.path_state[0] == 0
    assert m.conservation_counts(geometry, output)["per_native_frame_counts_conserved"]


def female(identities):
    anchors = [{"frame": 10+i*10, "identity": identity,
                "reason": "matched" if identity is not None else "mouth observation unmatched"}
               for i, identity in enumerate(identities)]
    intervals = [{"start": a["frame"], "stop": b["frame"],
                  "outcome": "unscored" if None in (a["identity"], b["identity"])
                  else "preserved" if a["identity"] == b["identity"] else "swap"}
                 for a, b in pairwise(anchors)]
    counts = {kind: sum(row["outcome"] == kind for row in intervals)
              for kind in ("preserved", "swap", "unscored")}
    return dict(anchors=anchors, intervals=intervals, interval_counts=counts)


def test_net_swap_count_improvement_does_not_hide_a_new_bad_endpoint_interval():
    raw, proposed = female([0, 1, 1]), female([0, 0, 1])
    result = m.female_regression_gate(raw, proposed)
    assert result["raw_swap_intervals"] == result["proposed_swap_intervals"] == 1
    assert result["passed"] is False
    assert result["preserved_to_swap_intervals"] == [{"start": 20, "stop": 30}]


def test_relative_gate_allows_improved_endpoints_but_one_anchor_is_unscored():
    assert m.female_regression_gate(female([0, 1]), female([2, 2]))["status"] == "pass"
    result = m.female_regression_gate(female([1]), female([2]))
    assert result["status"] == "unscored" and result["passed"] is None


def test_changed_support_or_review_availability_is_not_hidden_as_unscored():
    with pytest.raises(ValueError, match="support/availability"):
        m.female_regression_gate(female([0, 0]), female([0, None]))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    repo, run = tmp_path / "repo", tmp_path / "repo/run"
    run.mkdir(parents=True)
    monkeypatch.setattr(m.base, "REPO", repo)
    required = {"labs/a.py", "src/b.py"}
    monkeypatch.setattr(m, "REQUIRED_SOURCE", required)
    code, snapshots = {}, {}
    for name in required:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixed source\n", encoding="utf-8")
        snapshot = run / "code_snapshot" / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
        code[name] = m.base.sha256(path)
        snapshots["code_snapshot/" + name] = {
            "path": str(snapshot), "sha256": code[name],
        }
    registered = repo / "registered_input.bin"
    registered.write_bytes(b"SL/raw/masks/catalog bound by the independent loader")
    inputs = {str(registered): m.base.sha256(registered)}
    context = SimpleNamespace(
        frame_idx=np.arange(333519, 333719, dtype=np.int64), verified_hashes=inputs,
        prefix_audit={"same_window": True},
    )
    monkeypatch.setattr(m, "load_identity_inputs", lambda *args: context)
    summary = dict(
        protocol=m.STAGE, technical_completion=True, native_fps=166,
        geometry_initialization_count=1, external_reset_count=0, identity_initial_state=0,
        geometry_fits_in_this_stage=0, temporal_subsampling=False,
        raw_geometry_and_review_preserved=True, evaluation_data_opened_during_inference=False,
        code_unchanged_during_run=True, inputs_unchanged_during_run=True,
        config=copy.deepcopy(m.FIXED_CONFIG), code_sha256_before=code, code_sha256_after=code,
        input_sha256_before=inputs, input_sha256_after=inputs,
        frames=[333519, 333719], requested_frames=200, returned_frames=200,
    )
    manifest = {key: {"sha256": value, "snapshot": "code_snapshot/" + key}
                for key, value in code.items()}
    for name in m.OUTPUTS.values():
        (run / name).write_bytes(b"sealed placeholder; verification does not decode this")
    write_json(run / "summary.json", summary)
    write_json(run / "source_manifest.json", manifest)
    write_json(run / "prefix_audit.json", context.prefix_audit)
    outputs = {role: {"path": str(run / name), "sha256": m.base.sha256(run / name)}
               for role, name in m.OUTPUTS.items()}
    outputs.update(snapshots)
    record = dict(
        status="ok", stage=m.STAGE,
        params=dict(output=str(run), config=copy.deepcopy(m.FIXED_CONFIG), code_hashes=code,
                    geometry_run=str(run), enrollment_run=str(run), enrollment_audit=str(run),
                    package_versions={"numpy": np.__version__},
                    evaluation_data_opened_during_inference=False,
                    code_unchanged_during_run=True, inputs_unchanged_during_run=True),
        inputs={key: {"path": key, "sha256": value} for key, value in inputs.items()},
        outputs=outputs,
    )
    write_json(run / "experiment.provenance.json", record)
    return run, record, summary, manifest, context


def test_completed_producer_accepts_sealed_windows_snapshot_roles(sealed):
    run, record, *_ = sealed
    record["outputs"] = {key.replace("/", "\\"): value
                         for key, value in record["outputs"].items()}
    write_json(run / "experiment.provenance.json", record)
    summary, context, hashes = m.verify_completed(run)
    assert summary["returned_frames"] == 200 and len(context.frame_idx) == 200
    assert str(run / "code_snapshot/labs/a.py") in hashes


@pytest.mark.parametrize("problem", [
    "incomplete", "missing_output", "changed_output", "unsealed_snapshot", "changed_snapshot",
    "changed_live_source", "omitted_dependency", "snapshot_path", "wrong_input_role",
    "changed_input_sha", "prefix_binding", "short_window", "config_bool", "numpy_version",
])
def test_broken_producer_source_input_or_completion_binding_fails_closed(sealed, problem):
    run, record, summary, manifest, _ = sealed
    if problem == "incomplete":
        record["status"] = "failed"
    elif problem == "missing_output":
        del record["outputs"]["scores"]
    elif problem == "changed_output":
        (run / "scores.npz").write_bytes(b"modified")
    elif problem == "unsealed_snapshot":
        del record["outputs"]["code_snapshot/labs/a.py"]
    elif problem == "changed_snapshot":
        (run / "code_snapshot/labs/a.py").write_bytes(b"modified")
    elif problem == "changed_live_source":
        (run.parent / "labs/a.py").write_bytes(b"modified")
    elif problem == "omitted_dependency":
        del record["params"]["code_hashes"]["labs/a.py"]
    elif problem == "snapshot_path":
        manifest["labs/a.py"]["snapshot"] = "../a.py"
    elif problem == "wrong_input_role":
        record["inputs"]["raw_top_cam"] = record["inputs"].pop(next(iter(record["inputs"])))
    elif problem == "changed_input_sha":
        record["inputs"][next(iter(record["inputs"]))]["sha256"] = "0" * 64
    elif problem == "prefix_binding":
        write_json(run / "prefix_audit.json", {"same_window": False})
        record["outputs"]["prefix_audit"]["sha256"] = m.base.sha256(run / "prefix_audit.json")
    elif problem == "short_window":
        summary["returned_frames"] = 199
    elif problem == "config_bool":
        record["params"]["config"]["initial_state"] = False
    else:
        record["params"]["package_versions"]["numpy"] = "0.0.0"
    write_json(run / "summary.json", summary)
    write_json(run / "source_manifest.json", manifest)
    for role in ("summary", "source_manifest"):
        record["outputs"][role]["sha256"] = m.base.sha256(run / m.OUTPUTS[role])
    write_json(run / "experiment.provenance.json", record)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.verify_completed(run)


def test_failed_verification_is_not_sealed_success_and_output_cannot_be_overwritten(sealed):
    run, record, *_ = sealed
    record["status"] = "failed"
    write_json(run / "experiment.provenance.json", record)
    output = run.parent / "evaluation"
    with pytest.raises(ValueError):
        m.main(["--run", str(run), "--output", str(output)])
    assert m.read_json(output / "evaluation.provenance.json")["status"] == "failed"
    assert not (output / "comparison.json").exists()
    with pytest.raises(FileExistsError):
        m.main(["--run", str(run), "--output", str(output)])


def test_report_retains_unscored_endpoint_limits_and_motion_ablation(tmp_path):
    metric = {"female_raw": female([1]), "female_accepted": female([1])}
    report = {"methods": {key: metric for key in (
        "frozen_geometry_ids", "proposed_ids", "motion_only_ids"
    )}, "raw_female_endpoint_regression_gate": {"status": "unscored"}}
    path = tmp_path / "RESULT.md"
    m.write_report(path, report)
    text = path.read_text(encoding="utf-8")
    assert "Motion-only" in text and "unscored" in text
    assert "not identity certification" in text and "Original review flags remain" in text
