"""New five-column selection/success lineage; no actual result or raw reads."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import render_fixed_body_width200 as m  # noqa: E402

from test_evaluate_fixed_body_width200 import run_fixture  # noqa: E402
from test_render_mask_transport200 import Rig, candidate  # noqa: E402


def evaluation_fixture(root):
    plan = dict(
        schema=m.EVALUATION_STAGE,
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.QUANTITATIVE_ARMS),
        source_tracking_sha256=m.TRACKING_SHA,
        evidence=["lk", "ct_first"],
        primary_renderer="original_dynamic_all6",
        secondary_renderer="registered_fixed_radial_all6",
        samples_sha256=m.previous.SAMPLES_SHA,
        rendering_device="cpu",
        rendering_dtype="float64",
        previous_tracking_sha256=m.producer.prior_gap.PREVIOUS_SHA,
        previous_evaluation_sha256=m.OLD_EVALUATION_SHA,
    )
    summary = dict(
        schema=m.EVALUATION_STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.QUANTITATIVE_ARMS),
        evidence=["lk", "ct_first"],
        original_four_CPU_image_control_exact_frames=200,
        strict_candidate_support_rechecks=1200,
        unused_pixel_draws=153600,
        GPU_initialized=False,
        trajectory_or_review_modified=False,
        width_reference_audits={
            arm: dict(
                passed=True,
                native_interval=[m.FIRST, m.STOP],
                core_arrays=39,
                history_frames=200,
                semantic_core_and_sidecar_verified=True,
                initial_geometry_links_verified=True,
                metadata_rewritten=False,
                GPU_used=False,
            )
            for arm in m.producer.ARMS
        },
    )
    seal = dict(
        inputs={
            str(root / "tracking.provenance.json"): m.TRACKING_SHA,
            str(
                m.producer.prior_gap.PREVIOUS / "tracking.provenance.json"
            ): m.producer.prior_gap.PREVIOUS_SHA,
            str(m.OLD_GAP / "tracking.provenance.json"): m.OLD_GAP_SHA,
            str(m.OLD_EVALUATION / "evaluation.provenance.json"): m.OLD_EVALUATION_SHA,
            str(m.WORK / "evaluate_fixed_body_width200.py"): m.EVALUATOR_SHA,
            str(Path(m.producer.__file__).resolve()): m.PRODUCER_SHA,
        }
    )
    return plan, summary, seal


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failed",
        "wrong_tracking",
        "five_arm_eval",
        "short",
        "own_renderer_primary",
        "missing_state_audit",
        "old_gap_source",
    ],
)
def test_visual_requires_complete_independent_six_arm_result(fault):
    root = m.WORK / "synthetic_tracking"
    plan, summary, seal = evaluation_fixture(root)
    before = copy.deepcopy((plan, summary, seal))
    if fault == "failed":
        summary["status"] = "failed"
    elif fault == "wrong_tracking":
        plan["source_tracking_sha256"] = "0" * 64
    elif fault == "five_arm_eval":
        summary["arms"].remove("lk")
    elif fault == "short":
        summary["strict_candidate_support_rechecks"] = 1194
    elif fault == "own_renderer_primary":
        plan["primary_renderer"] = "registered_fixed_radial_all6"
    elif fault == "missing_state_audit":
        summary["width_reference_audits"]["static_base"]["passed"] = False
    elif fault == "old_gap_source":
        seal["inputs"][str(m.OLD_GAP / "tracking.provenance.json")] = "0" * 64
    if fault:
        with pytest.raises(ValueError):
            m.validate_evaluation(plan, summary, seal, root, m.TRACKING_SHA)
    else:
        m.validate_evaluation(plan, summary, seal, root, m.TRACKING_SHA)
        assert (plan, summary, seal) == before


def test_frozen_projection_panel_and_raw_original_CTgap_pixels_are_unchanged():
    assert m.projection_record is m.drawing.projection_record is m.prior_visual.projection_record
    assert m.panel is m.drawing.panel and m.save_png is m.drawing.save_png
    assert m.ARMS == ("original", "ct_gap", "static_base", "static_gap")
    assert tuple(m.FIRST + t for t in (0, 16, 18, 40, 80, 120, 160, 199)) == m.FRAMES
    raw = np.arange(1200 * 1200 * 3, dtype=np.uint8).reshape(1200, 1200, 3)
    original_raw = raw.tobytes()
    projections = [m.projection_record(candidate(), 0, Rig()) for _ in range(4)]
    new = m.composite(raw, m.VIEWS[0], m.FIRST, projections)
    old4 = m.drawing.composite(raw, m.VIEWS[0], m.FIRST, projections[:3])
    old5 = m.prior_visual.composite(raw, m.VIEWS[0], m.FIRST, projections)
    w = m.PANEL_W
    assert new.shape == (m.PANEL_H, 5 * w, 3)
    assert new[:, : 2 * w].tobytes() == old4[:, : 2 * w].tobytes()
    assert new[:, 2 * w : 3 * w].tobytes() == old5[:, 4 * w : 5 * w].tobytes()
    for i, title in enumerate(m.TITLES):
        assert (
            new[:, (i + 1) * w : (i + 2) * w].tobytes()
            == m.panel(raw, title + " | terminal200 backtrace", projections[i], 0).tobytes()
        )
    assert raw.tobytes() == original_raw


def setup_lineage(monkeypatch, fault=None):
    root, eval_root = m.WORK / "synthetic_tracking", m.WORK / "synthetic_eval"
    old_root = m.producer.prior_gap.PREVIOUS
    roots = (old_root, m.OLD_GAP, root)
    memberships = {r: {} for r in roots}
    plan, summary, initial = run_fixture()
    plan["input_binding"]["technical_retry_failed_sha256"] = m.producer.FAILED_SHA
    plan["original_context"] = {"calibrated": "same synthetic"}
    eval_plan, eval_summary, eval_seal = evaluation_fixture(root)
    metadata = {
        root: {
            "plan.json": plan,
            "summary.json": summary,
            **{f"{a}/initial_control.json": initial[a] for a in m.producer.ARMS},
        },
        eval_root: {"plan.json": eval_plan, "summary.json": eval_summary},
        old_root: {
            "plan.json": {"original_context": copy.deepcopy(plan["original_context"])},
            "summary.json": {},
            "original_control.json": {},
            "original_reuse_qualification.json": {},
        },
        m.OLD_GAP: {
            "plan.json": {"original_context": copy.deepcopy(plan["original_context"])},
            "summary.json": {},
            "initial_control.json": {},
        },
    }
    source = {
        str(m.REPO / k): v
        for k, v in {
            **m.producer.prior_gap.SOURCE_PINS,
            **m.producer.SOURCE_PINS,
            "work/three_body/run_fixed_body_width200_v2.py": m.PRODUCER_SHA,
        }.items()
    }
    source.update(
        {
            str(old_root / "tracking.provenance.json"): m.producer.prior_gap.PREVIOUS_SHA,
            str(m.producer.PLAN): m.producer.PLAN_SHA,
            str(m.producer.FAILED / "tracking.provenance.json"): m.producer.FAILED_SHA,
            str(
                m.producer.QUALIFICATION / "qualification.provenance.json"
            ): m.producer.QUALIFICATION_SHA,
        }
    )
    gap_sources = {
        str(m.REPO / k): v
        for k, v in {
            **m.producer.prior_gap.SOURCE_PINS,
            "work/three_body/run_mask_transport_gap200.py": m.prior_visual.PRODUCER_SHA,
        }.items()
    }
    seals = {
        root: {"inputs": source},
        old_root: {
            "inputs": {
                str(Path(m.previous.__file__).resolve()): m.producer.prior_gap.SOURCE_PINS[
                    "work/three_body/run_mask_transport200_v2.py"
                ]
            }
        },
        m.OLD_GAP: {"inputs": gap_sources},
        eval_root: eval_seal,
    }
    for i, arm in enumerate(m.QUANTITATIVE_ARMS):
        source_root = old_root if arm in m.previous.ARMS else m.OLD_GAP if arm == "ct_gap" else root
        path = source_root / f"{arm}/candidate.npz"
        digest = f"{i + 1:064x}"
        memberships[source_root][str(path)] = dict(sha256=digest)
        eval_seal["inputs"][str(path)] = digest
    if fault == "failed_eval":
        eval_summary["status"] = "failed"
    elif fault == "omitted_LK_not_bound":
        del eval_seal["inputs"][str(old_root / "lk/candidate.npz")]
    elif fault == "wrong_static_candidate":
        eval_seal["inputs"][str(root / "static_gap/candidate.npz")] = "0" * 64
    elif fault == "producer_source":
        source[str(m.REPO / "work/three_body/fixed_body_width_tracker_v2.py")] = "0" * 64
    elif fault == "context":
        metadata[m.OLD_GAP]["plan.json"]["original_context"] = {"other": "context"}
    events = []
    common = dict(
        original_context=copy.deepcopy(plan["original_context"]),
        context={"source": "synthetic"},
        visual_geometry_lineage={"shared": "preparedSL3/directcalib4"},
    )

    def verified(path, seal_name, stage, digest, evidence):
        events.append(("seal", path))
        return seals[path], memberships.get(path, {})

    def bound_path(path, members, relative, expected=None):
        value = path / relative
        m.require(str(value) in members, "synthetic candidate absent")
        return value

    def load_common(evidence):
        events.append(("common",))
        return common

    def arrays(path, members, relative):
        events.append(("array", relative.split("/")[0]))
        return candidate()

    def lineage(seal, context, evidence):
        assert context is common["context"]
        events.append(("lineage",))
        return (
            {"other": "rig"}
            if fault == "lineage" and seal is seals[root]
            else common["visual_geometry_lineage"]
        )

    monkeypatch.setattr(m.base, "verified", verified)
    monkeypatch.setattr(m.base, "bound_json", lambda path, members, name: metadata[path][name])
    monkeypatch.setattr(m.prepared, "bound_path", bound_path)
    monkeypatch.setattr(m.drawing, "validate_completed", lambda *args: None)
    monkeypatch.setattr(m.drawing, "validate_reuse", lambda *args: None)
    monkeypatch.setattr(m.prior_visual, "validate_gap", lambda *args: None)
    monkeypatch.setattr(m.inputs, "load_common", load_common)
    monkeypatch.setattr(m.previous, "read_arrays", arrays)
    monkeypatch.setattr(m.drawing, "bind_geometry_lineage", lineage)
    return root, eval_root, common, events


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "failed_eval",
        "omitted_LK_not_bound",
        "wrong_static_candidate",
        "producer_source",
        "context",
        "lineage",
    ],
)
def test_load_only_four_candidates_after_all_six_lineage_success(monkeypatch, fault):
    root, eval_root, common, events = setup_lineage(monkeypatch, fault)
    evidence = SimpleNamespace(add=lambda *args: None)
    if fault:
        with pytest.raises(ValueError):
            m.load_inputs(root, m.TRACKING_SHA, eval_root, m.EVALUATION_SHA, evidence)
        assert not any(row[0] == "array" for row in events)
        if fault in {
            "failed_eval",
            "omitted_LK_not_bound",
            "wrong_static_candidate",
            "producer_source",
        }:
            assert ("common",) not in events
    else:
        loaded, values, references = m.load_inputs(
            root, m.TRACKING_SHA, eval_root, m.EVALUATION_SHA, evidence
        )
        assert loaded is common and tuple(values) == m.ARMS
        assert [row for row in events if row[0] == "array"] == [("array", a) for a in m.ARMS]
        assert [r["arm"] for r in references] == list(m.QUANTITATIVE_ARMS)
        assert [r["arm"] for r in references if r["displayed"]] == list(m.ARMS)
        assert len([row for row in events if row[0] == "lineage"]) == 4


def test_wrong_tracking_SHA_stops_before_any_metadata_or_payload(monkeypatch):
    events = []
    evidence = SimpleNamespace(add=lambda *args: events.append("read"))
    with pytest.raises(ValueError, match="tracking SHA"):
        m.load_inputs(m.WORK / "synthetic", "0" * 64, m.WORK / "eval", "e" * 64, evidence)
    assert events == []


def test_wrong_evaluation_SHA_stops_before_any_metadata_or_payload():
    events = []
    evidence = SimpleNamespace(add=lambda *args: events.append("read"))
    with pytest.raises(ValueError, match="evaluation SHA"):
        m.load_inputs(m.WORK / "synthetic", m.TRACKING_SHA, m.WORK / "eval", "0" * 64, evidence)
    assert events == []


def test_prepared_SL3_lineage_does_not_claim_direct_trackingSL_consumption(monkeypatch):
    shared = {str(m.WORK / f"synthetic_SL_{i}.npz"): str(i + 1) * 64 for i in range(3)}
    calibration = {str(m.WORK / f"synthetic_calib_{i}.json"): str(i + 4) * 64 for i in range(4)}
    shared.update(calibration)
    context = dict(plan={"shared_SL_calibration_sha256": shared}, seal={"inputs": shared.copy()})
    seal = dict(
        inputs={
            str(m.inputs.PREPARATION / "preparation.provenance.json"): m.inputs.PREPARATION_SHA,
            **calibration,
        }
    )
    reads = []
    result = m.drawing.bind_geometry_lineage(
        seal, context, SimpleNamespace(add=lambda p, h: reads.append(p))
    )
    assert len(result["SL3_sha256"]) == 3 and len(result["calibration4_sha256"]) == 4
    assert set(reads) == set(calibration)
    assert all(Path(p).suffix == ".json" for p in reads)
    del seal["inputs"][next(iter(calibration))]
    with pytest.raises(ValueError, match="calibration"):
        m.drawing.bind_geometry_lineage(seal, context, SimpleNamespace(add=lambda *args: None))


def test_video_full200_decoder_contract_is_identical_function_behavior(monkeypatch):
    class Capture:
        def __init__(self):
            self.n, self.closed = 0, False

        def isOpened(self):
            return True

        def get(self, field):
            return 20.0

        def read(self):
            self.n += 1
            return (
                (True, SimpleNamespace(shape=(m.PANEL_H, 5 * m.PANEL_W, 3)))
                if self.n <= 200
                else (False, None)
            )

        def release(self):
            self.closed = True

    capture = Capture()
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda path: capture)
    assert m.verify_video("synthetic-only")["frames"] == 200
    assert capture.closed
