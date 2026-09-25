"""New success/evaluation/display wiring; no real payload or model execution."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import render_body_point200 as m  # noqa: E402

from test_render_mask_transport200 import Rig, candidate  # noqa: E402


def evaluation_fixture(root, digest):
    audit = dict(
        schema="body_point_read_only_state_audit_v1",
        passed=True,
        native_interval=[m.FIRST, m.STOP],
        core_arrays=39,
        retained_fields=19,
        history_frames=200,
        semantic_core_and_sidecar_verified=True,
        initial_geometry_links_verified=True,
        metadata_rewritten=False,
        inputs_unchanged=True,
        GPU_used=False,
    )
    plan = dict(
        schema=m.EVALUATION_STAGE,
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.QUANTITATIVE_ARMS),
        source_tracking_sha256=digest,
        evidence=["lk", "ct_first"],
        renderer="original_dynamic_all7",
        samples_sha256=m.previous.SAMPLES_SHA,
        rendering_device="cpu",
        rendering_dtype="float64",
        packet_preparation_sha256=m.producer.PACKETS_SHA,
        previous_six_evaluation_sha256=m.producer.SIX_EVALUATION_SHA,
        point_mass_domain="saved FIT1024 m/S; IN-SAMPLE",
        point_t0_saved_not_fitted=True,
    )
    summary = dict(
        schema=m.EVALUATION_STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.QUANTITATIVE_ARMS),
        evidence=["lk", "ct_first"],
        previous_six_common19_exact_frames=200,
        strict_candidate_support_rechecks=1400,
        tracking_CT_packet_exact_frames=199,
        point_evaluation_packets=400,
        point_metrics_are_IN_SAMPLE=True,
        t0_packet_used_in_fit=False,
        unused_pixel_draws=153600,
        GPU_initialized=False,
        trajectory_or_review_modified=False,
        independent_ID_truth=False,
        new_state_audit=audit,
    )
    source = {
        str(root / "tracking.provenance.json"): digest,
        str(m.producer.PREVIOUS / "tracking.provenance.json"): m.producer.PREVIOUS_SHA,
        str(m.OLD_GAP / "tracking.provenance.json"): m.OLD_GAP_SHA,
        str(m.producer.PACKETS / "preparation.provenance.json"): m.producer.PACKETS_SHA,
        str(
            m.producer.SIX_EVALUATION / "evaluation.provenance.json"
        ): m.producer.SIX_EVALUATION_SHA,
        str(m.WORK / "evaluate_body_point200.py"): m.EVALUATOR_SHA,
        str(m.WORK / "run_body_point200.py"): m.PRODUCER_SHA,
    }
    refs = []
    for i, arm in enumerate(m.QUANTITATIVE_ARMS):
        source_root = (
            m.producer.PREVIOUS
            if arm in m.previous.ARMS
            else m.OLD_GAP
            if arm == "ct_gap"
            else root
            if arm == m.producer.ARM
            else m.WORK / "synthetic_fixed"
        )
        row = dict(arm=arm, path=str(source_root / arm / "candidate.npz"), sha256=str(i + 1) * 64)
        refs.append(row)
        source[row["path"]] = row["sha256"]
    plan["candidate_references"] = refs
    return plan, summary, dict(inputs=source)


@pytest.mark.parametrize("fault", [None, "short_eval", "wrong_packet", "state", "source"])
def test_success_gate_separates_in_sample_points_and_complete_independent_evaluation(fault):
    root, digest = m.WORK / "synthetic_new", "a" * 64
    plan, summary, seal = evaluation_fixture(root, digest)
    before = copy.deepcopy((plan, summary, seal))
    if fault == "short_eval":
        summary["previous_six_common19_exact_frames"] = 199
    elif fault == "wrong_packet":
        plan["packet_preparation_sha256"] = "b" * 64
    elif fault == "state":
        summary["new_state_audit"]["initial_geometry_links_verified"] = False
    elif fault == "source":
        seal["inputs"][str(root / "tracking.provenance.json")] = "b" * 64
    if fault:
        with pytest.raises(ValueError):
            m.validate_evaluation(plan, summary, seal, root, digest)
    else:
        m.validate_evaluation(plan, summary, seal, root, digest)
        assert (plan, summary, seal) == before


@pytest.mark.parametrize("fault", [None, "evaluated_sha", "candidate21", "incomplete_eval"])
def test_loader_binds_both_seals_before_candidate_decode_and_uses_prepared_geometry(
    monkeypatch, fault
):
    root, eval_root = m.WORK / "synthetic_new", m.WORK / "synthetic_eval"
    digest, eval_digest = "a" * 64, "b" * 64
    monkeypatch.setattr(m, "EVALUATOR_SHA", "c" * 64)
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    plan, summary, evaluation = evaluation_fixture(root, digest)
    new_candidate_path = str(root / m.producer.ARM / "candidate.npz")
    members = {new_candidate_path: dict(sha256="7" * 64)}
    tracking_plan = dict(original_context=dict(same="synthetic science"))
    source = {str(m.REPO / k): v for k, v in m.producer.SOURCE_PINS.items()}
    source.update(
        {
            str(m.WORK / "run_body_point200.py"): m.PRODUCER_SHA,
            str(m.producer.PREVIOUS / "tracking.provenance.json"): m.producer.PREVIOUS_SHA,
            str(m.producer.PLAN): m.producer.PLAN_SHA,
            str(m.producer.PACKETS / "preparation.provenance.json"): m.producer.PACKETS_SHA,
            str(
                m.producer.QUALIFICATION / "qualification.provenance.json"
            ): m.producer.QUALIFICATION_SHA,
        }
    )
    tracking = dict(inputs=source)
    calls, reads = [], []
    # There is deliberately no direct tracking SL3 claim. The frozen lineage
    # helper is the authority for prepared SL3 / direct calibration4 semantics.
    assert not any(Path(p).suffix == ".slp" for p in source)

    def verified(path, filename, stage, expected, evidence):
        if path == root:
            assert filename == "tracking.provenance.json" and expected == digest
            calls.append("tracking_success")
            return tracking, members
        assert path == eval_root and expected == eval_digest
        assert stage == m.EVALUATION_STAGE
        calls.append("evaluation_success")
        return evaluation, {}

    def bound_json(path, _, name):
        return (
            dict(
                zip(
                    ("plan.json", "summary.json", "initial_control.json"),
                    (tracking_plan, {"synthetic": "summary"}, {"synthetic": "initial"}),
                    strict=True,
                )
            )[name]
            if path == root
            else {"plan.json": plan, "summary.json": summary}[name]
        )

    def validate_run(p, s, i):
        assert p is tracking_plan and i == {"synthetic": "initial"}
        calls.append("new_metadata_gate")

    evaluator = SimpleNamespace(validate_run=validate_run, SOURCE_PINS={})
    monkeypatch.setattr(m.importlib, "import_module", lambda name: evaluator)
    monkeypatch.setattr(m.base, "verified", verified)
    monkeypatch.setattr(m.base, "bound_json", bound_json)
    monkeypatch.setattr(m.prepared, "bound_path", lambda r, _, name: r / name)
    common = dict(
        original_context=tracking_plan["original_context"],
        context={},
        visual_geometry_lineage=dict(prepared_SL3="same", calibration4="same"),
    )

    def prior(path, expected, evidence):
        assert calls[:3] == ["tracking_success", "new_metadata_gate", "evaluation_success"]
        assert path == m.OLD_GAP and expected == m.OLD_GAP_SHA
        reads.append("old_candidates")
        return common, {arm: candidate() for arm in m.prior_visual.ARMS}

    def lineage(seal, context, evidence):
        assert seal is tracking or seal is evaluation
        assert context is common["context"]
        calls.append("prepared_geometry")
        return common["visual_geometry_lineage"]

    value = candidate()
    if fault == "candidate21":
        value.pop("sources")
    elif fault == "evaluated_sha":
        # Coherent eval reference+claim still must match tracking output digest.
        plan["candidate_references"][-1]["sha256"] = "8" * 64
        evaluation["inputs"][new_candidate_path] = "8" * 64
    elif fault == "incomplete_eval":
        summary["strict_candidate_support_rechecks"] = 1399
    monkeypatch.setattr(m.prior_visual, "load_inputs", prior)
    monkeypatch.setattr(m.drawing, "bind_geometry_lineage", lineage)
    monkeypatch.setattr(m.previous, "read_arrays", lambda *args: value)
    evidence = SimpleNamespace(add=lambda *args: None)
    if fault:
        with pytest.raises(ValueError):
            m.load_inputs(root, digest, eval_root, eval_digest, evidence)
        if fault != "candidate21":
            assert not reads
    else:
        loaded, values, refs = m.load_inputs(root, digest, eval_root, eval_digest, evidence)
        assert loaded is common and tuple(values) == m.ARMS
        assert values[m.producer.ARM] is value
        assert [r["arm"] for r in refs if r["displayed"]] == list(m.ARMS)
        assert calls.count("prepared_geometry") == 2


def test_four_columns_preserve_raw_and_old_projection_pixels_and_all_marker_flags():
    assert m.projection_record is m.drawing.projection_record
    assert m.panel is m.drawing.panel and m.save_png is m.drawing.save_png
    assert m.verify_video is m.drawing.verify_video
    assert tuple(m.FIRST + t for t in (0, 16, 18, 40, 80, 120, 160, 199)) == m.FRAMES
    raw = np.arange(1200 * 1200 * 3, dtype=np.uint8).reshape(1200, 1200, 3)
    before = raw.tobytes()
    c = candidate()
    c["sources"][0, 0, 0, 0] = -1
    c["observed"][0, 1, 1] = False
    projections = [m.projection_record(c, 0, Rig()) for _ in m.ARMS]
    new = m.composite(raw, m.VIEWS[0], m.FIRST, projections)
    old = m.prior_visual.composite(raw, m.VIEWS[0], m.FIRST, [projections[0]] * 4)
    assert new.shape == (744, 2400, 3)
    assert new[:, :1200].tobytes() == old[:, :1200].tobytes()
    assert new[:, 1200:1800].tobytes() == old[:, 2400:3000].tobytes()
    assert new[m.IMAGE_Y :, 1800:].tobytes() == old[m.IMAGE_Y :, 2400:].tobytes()
    assert set(np.unique(projections[-1]["marker_kind"])) == {0, 1, 2}
    assert projections[-1]["identity_review"].tolist() == [False, False, True]
    assert raw.tobytes() == before
    with pytest.raises(ValueError):
        m.composite(raw, m.VIEWS[0], m.FIRST, projections[:-1])


def test_unreviewed_source_refuses_before_output_creation(tmp_path, monkeypatch):
    destination = tmp_path / "never-created"
    monkeypatch.setattr(m, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="reviewed final"):
        m.run(tmp_path, "a" * 64, tmp_path, "b" * 64, destination)
    assert not destination.exists()


def test_failed_input_keeps_failed_receipt_without_raw_access(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    monkeypatch.setattr(m, "EVALUATOR_SHA", "c" * 64)
    monkeypatch.setattr(m, "SOURCE_PINS", {})

    def rejected(*args):
        raise ValueError("synthetic incompatible successful seals")

    monkeypatch.setattr(m, "load_inputs", rejected)
    destination = tmp_path / "preserved_failure"
    with pytest.raises(ValueError, match="incompatible"):
        m.run(tmp_path, "a" * 64, tmp_path, "b" * 64, destination)
    seal = m.base.read_json(destination / "review.provenance.json")
    assert seal["status"] == "failed" and seal["stage"] == m.STAGE
    assert seal["completed_natives"] == [] and seal["outputs"] == {}
