"""New five-column wiring only; frozen projection/display arithmetic is reused."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parents[1] / "work/three_body"
sys.path.insert(0, str(WORK))
import render_mask_transport_gap200 as m  # noqa: E402

from test_render_mask_transport200 import Rig, candidate  # noqa: E402


def metadata():
    binding = dict(
        prior_tracking_sha256=m.producer.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.EVALUATION_SHA,
        preparation_sha256=m.inputs.PREPARATION_SHA,
        point_sha256=m.previous.POINTS_SHA,
        samples_sha256=m.previous.SAMPLES_SHA,
        gap_plan_sha256=m.producer.PLAN_SHA,
        source_pins=copy.deepcopy(m.producer.SOURCE_PINS),
    )
    plan = dict(
        schema=m.producer.STAGE,
        native_interval=[m.FIRST, m.STOP],
        arm="ct_gap",
        input_binding=binding,
        original_context={"same": "synthetic"},
        objective="base + 3*(owner + ray_gap)",
        gap_relative_coefficient=1.0,
        mask_coefficient=3.0,
        point_evidence_arm="ct_first",
        extra_proposals=0,
        maximum_pool=24,
        opt_steps=14,
    )
    summary = dict(
        schema=m.producer.STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arm=dict(arm="ct_gap", initialize_count=1, restore_count=0, frames=200),
        prior_tracking_sha256=m.producer.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.EVALUATION_SHA,
        maximum_pool=24,
        steps=14,
        initialize_calls=1,
        restore_calls=0,
        advance_calls=200,
        model_forwards=0,
        query_refreshes=0,
        no_external_pose_reset=True,
        original_control_passed=True,
        causal_initial_control_passed=True,
    )
    initial = dict(
        passed=True,
        native=m.FIRST,
        initialized_once=True,
        prior_trajectory_used=False,
        all19_retained_records_exact=True,
        all_initial_anchors_facts_exact=True,
    )
    return plan, summary, initial


@pytest.mark.parametrize(
    "fault", [None, "failed", "short", "prior", "source", "objective", "initial"]
)
def test_fixed_success_and_causal_metadata_contract(fault):
    plan, summary, initial = metadata()
    before = copy.deepcopy((plan, summary, initial))
    if fault == "failed":
        summary["status"] = "failed"
    elif fault == "short":
        summary["arm"]["frames"] = 199
    elif fault == "prior":
        plan["input_binding"]["prior_tracking_sha256"] = "0" * 64
    elif fault == "source":
        plan["input_binding"]["source_pins"] = {}
    elif fault == "objective":
        plan["mask_coefficient"] = 2
    elif fault == "initial":
        initial["prior_trajectory_used"] = True
    if fault:
        with pytest.raises(ValueError):
            m.validate_gap(plan, summary, initial)
    else:
        m.validate_gap(plan, summary, initial)
        assert (plan, summary, initial) == before


def test_frozen_helpers_and_first_four_image_regions_remain_exact():
    assert m.projection_record is m.drawing.projection_record
    assert m.panel is m.drawing.panel and m.save_png is m.drawing.save_png
    assert tuple(m.FIRST + n for n in (0, 16, 18, 40, 80, 120, 160, 199)) == m.FRAMES
    raw = np.arange(1200 * 1200 * 3, dtype=np.uint8).reshape(1200, 1200, 3)
    before = raw.tobytes()
    value = candidate()
    projections = [m.projection_record(value, 0, Rig()) for _ in range(4)]
    new = m.composite(raw, m.VIEWS[0], m.FIRST, projections)
    old = m.drawing.composite(raw, m.VIEWS[0], m.FIRST, projections[:3])
    assert new.shape == (744, 3000, 3)
    assert new[:, :1800].tobytes() == old[:, :1800].tobytes()
    # CT-owner title changes; all image/support/legend pixels stay the same.
    assert new[m.IMAGE_Y :, 1800:2400].tobytes() == old[m.IMAGE_Y :, 1800:2400].tobytes()
    assert m.TITLES == ("Original", "LK transport", "CT-owner transport", "CT-gap transport")
    assert raw.tobytes() == before


@pytest.mark.parametrize("fault", [None, "short", "fps", "shape"])
def test_five_column_video_has_all200_and_releases_decoder(monkeypatch, fault):
    class Capture:
        count, closed = 0, False

        def isOpened(self):
            return True

        def get(self, field):
            return 25 if fault == "fps" else 20

        def read(self):
            self.count += 1
            if self.count > (199 if fault == "short" else 200):
                return False, None
            return True, SimpleNamespace(shape=(744, 2400 if fault == "shape" else 3000, 3))

        def release(self):
            self.closed = True

    cap = Capture()
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda _: cap)
    if fault:
        with pytest.raises(ValueError):
            m.verify_video("synthetic-only.mp4")
    else:
        assert m.verify_video("synthetic-only.mp4") == dict(
            frames=200, fps=20, width=3000, height=744
        )
    assert cap.closed


@pytest.mark.parametrize("fault", [None, "source", "context", "candidate"])
def test_loader_binds_new_gap_to_frozen_three_arm_and_prepared_geometry(monkeypatch, fault):
    plan, summary, initial = metadata()
    data = dict(
        zip(
            ("plan.json", "summary.json", "initial_control.json"),
            (plan, summary, initial),
            strict=True,
        )
    )
    source = {str(m.REPO / k): v for k, v in m.producer.SOURCE_PINS.items()}
    source.update(
        {
            str(m.REPO / "work/three_body/run_mask_transport_gap200.py"): m.PRODUCER_SHA,
            str(m.producer.PREVIOUS / "tracking.provenance.json"): m.producer.PREVIOUS_SHA,
            str(m.producer.PLAN): m.producer.PLAN_SHA,
        }
    )
    if fault == "source":
        source[str(m.REPO / "work/three_body/mask_transport_gap_fitter.py")] = "0" * 64
    monkeypatch.setattr(m.base, "verified", lambda *args: ({"inputs": source}, {}))
    monkeypatch.setattr(m.base, "bound_json", lambda root, members, name: data[name])
    common = dict(
        original_context=copy.deepcopy(plan["original_context"]),
        context={"prepared": "synthetic"},
        visual_geometry_lineage={"same": "SL7"},
    )
    if fault == "context":
        common["original_context"] = {}
    calls = []

    def prior(root, digest, evidence):
        assert root == m.producer.PREVIOUS and digest == m.producer.PREVIOUS_SHA
        calls.append("prior")
        return common, {a: candidate() for a in m.previous.ARMS}

    def lineage(seal, context, evidence):
        assert context is common["context"]
        return common["visual_geometry_lineage"]

    value = candidate()
    if fault == "candidate":
        value.pop("observed")
    monkeypatch.setattr(m.drawing, "load_inputs", prior)
    monkeypatch.setattr(m.drawing, "bind_geometry_lineage", lineage)
    monkeypatch.setattr(m.previous, "read_arrays", lambda *args: value)
    evidence = SimpleNamespace(add=lambda *args: None)
    if fault:
        with pytest.raises(ValueError):
            m.load_inputs(Path("synthetic-gap"), "a" * 64, evidence)
        if fault == "source":
            assert not calls
    else:
        loaded, values = m.load_inputs(Path("synthetic-gap"), "a" * 64, evidence)
        assert loaded is common and tuple(values) == m.ARMS and values["ct_gap"] is value
