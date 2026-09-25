"""Memory evaluation checks canonical P, variable nodes, history costs and immutable exports."""

import copy
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prior = load(
    Path(__file__).with_name("test_three_body_identity_dp_evaluation.py"),
    "_memory_evaluation_prior_fixtures",
)


def load_module():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {n: sys.modules.get(n) for n in names}, sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        for name in names:
            sys.modules.pop(name, None)
        return load(LAB / "evaluate_identity_memory.py", "_memory_identity_evaluator_test")
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
    geometry, enrollment, _identity, _dp, old_scores, _summary = prior.solution.__wrapped__()
    # Keep a terminal missing endpoint in addition to the earlier closed gap.
    geometry["observed"][-1, 2, 1] = False
    geometry["sources"][-1, 2, 1:, 1] = -1
    geometry["xyz"][-1, 2] = np.nan
    geometry["accepted_xyz"][-1, 2] = np.nan
    scores = {key: value for key, value in old_scores.items() if key in m.IMAGE_SHAPES}
    config = copy.deepcopy(m.FIXED_CONFIG)
    solved = m.solve_memory(geometry, scores["unary"], config)
    dp, scalars = m.common.split_solution(solved)
    identity = m.build_identity_output(geometry, solved.permutation_by_id)
    links = m.independent_links(geometry, solved, scores["unary"], config)
    summary = dict(config=config, dp=scalars, **m.solution_totals(solved, scores, links))
    return geometry, enrollment, identity, dp, scores, links, summary


def test_complete_memory_replay_preserves_raw_support_review_and_terminal_uncertainty(solution):
    before = [{k: v.tobytes() for k, v in item.items()} for item in solution[:6]]
    result, counts = m.validate_saved_solution(*solution)
    assert result.status == "optimal" and not result.approximation_used
    assert counts["per_native_frame_counts_conserved"]
    assert result.path_node[-1] > 6 and 0 <= result.path_state[-1] < 6
    assert result.terminal_unsettled_gap.any()
    assert before == [{k: v.tobytes() for k, v in item.items()} for item in solution[:6]]
    assert np.max(np.diff(result.layer_offsets)) > 6


@pytest.mark.parametrize(
    "bad",
    [
        "path_node_as_p",
        "path_p_as_node",
        "event",
        "layer_offsets",
        "edge_cost",
        "edge_source",
        "edge_gap",
        "gap",
        "dp_scalar",
        "approximation",
        "terminal_age",
        "link_cost",
        "link_gap",
        "link_previous",
        "link_missing_displacement",
        "link_terminal",
        "raw_pose",
        "cleared_review",
        "invented_support",
        "accepted_latent",
        "unary",
        "prefix",
        "disabled",
        "visible_mass",
        "count_dtype",
        "nan",
        "negative",
        "unknown_image",
        "unknown_dp",
        "unknown_link",
        "total",
    ],
)
def test_memory_objective_history_export_and_image_corruption_rejected(solution, bad):
    geometry, enrollment, identity, dp, scores, links, summary = solution
    if bad == "path_node_as_p":
        dp["path_node"][:] = dp["path_state"]
    elif bad == "path_p_as_node":
        dp["path_state"][:] = dp["path_node"]
    elif bad == "event":
        dp["state_last_event"][-1, 0] += 1
    elif bad == "layer_offsets":
        dp["layer_offsets"][-1] += 1
    elif bad == "edge_cost":
        dp["edge_cost"][-1] += 0.1
    elif bad == "edge_source":
        dp["edge_source"][-1] -= 1
    elif bad == "edge_gap":
        dp["edge_delta_frames"][-1, 0] += 1
    elif bad == "gap":
        dp["min_marginal_delta"][-1, 0] += 0.1
    elif bad == "dp_scalar":
        summary["dp"]["path_cost"] += 1
    elif bad == "approximation":
        summary["dp"]["approximation_used"] = True
    elif bad == "terminal_age":
        dp["terminal_age_frames"][0] += 1
    elif bad == "link_cost":
        links["brownian_motion_cost_by_id"][-1, 0] += 0.1
    elif bad == "link_gap":
        links["gap_frames_when_reobserved"][-1, 0] += 1
    elif bad == "link_previous":
        links["previous_direct_frame"][-1, 0] += 1
    elif bad == "link_missing_displacement":
        links["displacement_mm_when_reobserved"][~links["motion_cost_applied"]] = 0
    elif bad == "link_terminal":
        links["terminal_unsettled_gap"][:] = False
    elif bad == "raw_pose":
        identity["raw_pose_xyz"][0, 0, 0, 0] += 1
    elif bad == "cleared_review":
        identity["prior_review_for_proposed_identity"][:] = False
    elif bad == "invented_support":
        identity["proposed_observed"][:] = True
    elif bad == "accepted_latent":
        identity["conservative_accepted_xyz"][165, 0] = 0
    elif bad == "unary":
        scores["unary"][-1, 0] += 0.1
    elif bad == "prefix":
        scores["available_core_pixels"][0, 0] = 1
    elif bad == "disabled":
        scores["per_view_costs"][-1, 0, 1] = 0.1
    elif bad == "visible_mass":
        scores["visible_mass"][-1, 0, 0] = 1000
    elif bad == "count_dtype":
        scores["valid_pixels"] = scores["valid_pixels"].astype(np.int32)
    elif bad == "nan":
        scores["unary"][-1, 0] = np.nan
    elif bad == "negative":
        scores["per_view_costs"][-1, 0, 0] = -0.1
    elif bad == "unknown_image":
        scores["transitions"] = np.zeros(1)
    elif bad == "unknown_dp":
        dp["confidence"] = np.zeros(1)
    elif bad == "unknown_link":
        links["confidence"] = np.zeros(1)
    else:
        summary["reobservation_links"] += 1
    with pytest.raises(ValueError):
        m.validate_saved_solution(geometry, enrollment, identity, dp, scores, links, summary)


def test_explicit_path_audit_rejects_wrong_layer_and_memory_even_if_mapping_is_legal(solution):
    geometry, _enrollment, _identity, _dp, scores, _links, summary = solution
    solved = m.solve_memory(geometry, scores["unary"], summary["config"])
    with pytest.raises(ValueError, match="layer"):
        m.validate_path_events(geometry, replace(solved, path_node=solved.path_state.copy()))
    events = solved.state_last_event.copy()
    events[solved.path_node[-1], 0] = 0
    with pytest.raises(ValueError, match="last direct observation"):
        m.validate_path_events(geometry, replace(solved, state_last_event=events))


def test_motion_only_uses_same_full_memory_graph_and_is_invariant_to_saved_color_scores(solution):
    geometry, _enrollment, _identity, _dp, scores, _links, summary = solution
    color = m.solve_memory(geometry, scores["unary"], summary["config"])
    solved, identity, links = m.motion_only_solution(geometry, scores, summary["config"])
    scores["unary"][:] = np.arange(6)[None] * 1000
    other, other_identity, other_links = m.motion_only_solution(geometry, scores, summary["config"])
    for name in ("layer_offsets", "state_last_event", "edge_source", "edge_target", "edge_cost"):
        m.exact_array(getattr(solved, name), getattr(color, name), name)
    for name, value in m.common.split_solution(solved)[0].items():
        m.exact_array(value, getattr(other, name), name)
    for actual, expected in ((identity, other_identity), (links, other_links)):
        for name in actual:
            m.exact_array(actual[name], expected[name], name)
    assert np.isclose(
        solved.path_cost, links["brownian_motion_cost_by_id"].sum(), rtol=1e-11, atol=1e-10
    )


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    run, record, summary, manifest, context = prior.sealed.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(m, "common", prior.m)
    monkeypatch.setattr(m, "base", prior.m.base)
    monkeypatch.setattr(m, "REQUIRED_MEMORY_SOURCE", {"labs/a.py"})
    monkeypatch.setattr(m, "load_identity_inputs", lambda *_args: context)
    summary.update(protocol=m.STAGE, config=copy.deepcopy(m.FIXED_CONFIG))
    record["stage"] = m.STAGE
    record["params"]["config"] = copy.deepcopy(m.FIXED_CONFIG)
    path = run / "memory_links.npz"
    path.write_bytes(b"sealed placeholder for memory links")
    record["outputs"]["memory_links"] = dict(path=str(path), sha256=m.base.sha256(path))

    def seal():
        prior.write_json(run / "summary.json", summary)
        prior.write_json(run / "source_manifest.json", manifest)
        for role in ("summary", "source_manifest"):
            record["outputs"][role]["sha256"] = m.base.sha256(run / m.OUTPUTS[role])
        prior.write_json(run / "experiment.provenance.json", record)

    seal()
    return run, record, summary, manifest, context, seal


def test_new_stage_verifier_binds_memory_output_and_uses_no_gt(sealed):
    run, _record, _summary, _manifest, context, _seal = sealed
    result, actual, hashes = m.verify_completed(run)
    assert result["protocol"] == m.STAGE and actual is context
    assert str(run / "memory_links.npz") in hashes


@pytest.mark.parametrize(
    "bad",
    [
        "old_stage",
        "not_complete",
        "missing_links",
        "modified_links",
        "memory_model",
        "caps",
        "missing_dependency",
        "input",
        "source_role",
        "prefix",
        "short_window",
        "review_flag",
    ],
)
def test_new_memory_evidence_cannot_reuse_wrong_stage_or_lose_seals(sealed, bad):
    run, record, summary, _manifest, _context, seal = sealed
    if bad == "old_stage":
        record["stage"] = prior.m.STAGE
    elif bad == "not_complete":
        record["status"] = "running"
    elif bad == "missing_links":
        del record["outputs"]["memory_links"]
    elif bad == "modified_links":
        (run / "memory_links.npz").write_bytes(b"changed")
    elif bad == "memory_model":
        record["params"]["config"]["memory"]["model"] = "adjacent_only"
    elif bad == "caps":
        record["params"]["config"]["memory"]["max_layer_states"] = 10
    elif bad == "missing_dependency":
        del record["params"]["code_hashes"]["labs/a.py"]
    elif bad == "input":
        record["inputs"][next(iter(record["inputs"]))]["sha256"] = "0" * 64
    elif bad == "source_role":
        del record["outputs"]["code_snapshot/labs/a.py"]
    elif bad == "prefix":
        prior.write_json(run / "prefix_audit.json", {"same_window": False})
        record["outputs"]["prefix_audit"]["sha256"] = m.base.sha256(run / "prefix_audit.json")
    elif bad == "short_window":
        summary["returned_frames"] = 199
    else:
        summary["raw_geometry_and_review_preserved"] = False
    seal()
    with pytest.raises(ValueError):
        m.verify_completed(run)


def test_failed_memory_verification_writes_failed_seal_without_reading_annotations(
    sealed, monkeypatch
):
    run, record, _summary, _manifest, _context, seal = sealed
    record["status"] = "failed"
    seal()
    monkeypatch.setattr(m, "code_paths", lambda: [])

    def must_not_read(*_args):
        raise AssertionError("annotation boundary reached before verified inference")

    monkeypatch.setattr(m.base, "staging_catalog", must_not_read)
    output = run.parent / "comparison"
    with pytest.raises(ValueError):
        m.main(["--run", str(run), "--output", str(output)])
    assert json.loads((output / "evaluation.provenance.json").read_text())["status"] == "failed"
    assert not (output / "comparison.json").exists()


def test_both_exact_objectives_and_exports_precede_annotation_boundary(
    solution,
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    geometry, enrollment, identity, dp, scores, links, summary = solution
    run, output = tmp_path / "run", tmp_path / "comparison"
    run.mkdir()
    summary["frames"] = [int(geometry["frame_idx"][0]), int(geometry["frame_idx"][-1] + 1)]
    for role, values in (
        ("identity", identity),
        ("dp", dp),
        ("scores", scores),
        ("memory_links", links),
    ):
        np.savez(run / f"{role}.npz", **values)
    context = SimpleNamespace(
        geometry=geometry,
        enrollment=enrollment,
        frame_idx=geometry["frame_idx"],
        input_root=tmp_path,
    )
    monkeypatch.setattr(m.base, "REPO", tmp_path)
    monkeypatch.setattr(m, "code_paths", lambda: [])
    monkeypatch.setattr(m, "verify_completed", lambda _run: (summary, context, {}))
    trace = []
    old_full, old_motion = m.validate_saved_solution, m.motion_only_solution

    def full(*args):
        result = old_full(*args)
        trace.append("verified_full_memory_objective")
        return result

    def motion(*args):
        result = old_motion(*args)
        trace.append("solved_zero_color_memory_objective")
        return result

    def label_boundary(_root):
        assert trace == ["verified_full_memory_objective", "solved_zero_color_memory_objective"]
        assert all(
            (output / f"motion_only_{name}.npz").is_file() for name in ("dp", "identity", "links")
        )
        raise ValueError("test stops at annotation boundary")

    monkeypatch.setattr(m, "validate_saved_solution", full)
    monkeypatch.setattr(m, "motion_only_solution", motion)
    monkeypatch.setattr(m.base, "staging_catalog", label_boundary)
    with pytest.raises(ValueError, match="annotation boundary"):
        m.main(["--run", str(run), "--output", str(output)])
