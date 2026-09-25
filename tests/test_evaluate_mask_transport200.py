"""New CLI boundaries and synthetic full-packet CPU rendering; no experiment data."""

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import evaluate_mask_transport200 as m  # noqa: E402
import test_directional_chest_tracker as toy  # noqa: E402


def run_metadata():
    plan = dict(
        schema=m.producer.STAGE,
        native_interval=[m.FIRST, m.STOP],
        arms=list(m.metrics.ARMS),
        original_context={"tracker_config": {"width_ratio": 0.06}},
        input_binding=dict(
            preparation_sha256=m.inputs.PREPARATION_SHA,
            point_sha256=m.producer.POINTS_SHA,
            samples_sha256=m.producer.SAMPLES_SHA,
            implementation_plan_sha256=m.inputs.PLAN_SHA,
        ),
    )
    summary = dict(
        schema=m.producer.STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        original_control_passed=True,
        initialize_calls=3,
        restore_calls=0,
        advance_calls=600,
        maximum_pool=24,
        steps=14,
        arms=[dict(arm=a, frames=200, initialize_count=1, restore_count=0) for a in m.metrics.ARMS],
    )
    control = dict(
        passed=True,
        candidate={"byte_exact": True},
        history={"byte_exact": True},
        state={"byte_exact": True},
    )
    return plan, summary, control


def test_successful_producer_metadata_schema_is_consumed_without_numeric_payload():
    values = run_metadata()
    before = copy.deepcopy(values)
    assert m.validate_run(*values) is None
    assert values == before


@pytest.mark.parametrize(
    "fault",
    [
        "unfinished",
        "native",
        "point",
        "sample",
        "width",
        "control",
        "history",
        "reset",
        "short_arm",
        "arm_order",
    ],
)
def test_incomplete_or_different_experiment_is_rejected_before_payload(fault):
    plan, summary, control = run_metadata()
    if fault == "unfinished":
        summary["status"] = "running"
    elif fault == "native":
        plan["native_interval"][1] -= 1
    elif fault == "point":
        plan["input_binding"]["point_sha256"] = "0" * 64
    elif fault == "sample":
        plan["input_binding"]["samples_sha256"] = "0" * 64
    elif fault == "width":
        plan["original_context"]["tracker_config"]["width_ratio"] = 0.08
    elif fault == "control":
        summary["original_control_passed"] = False
    elif fault == "history":
        control["history"]["byte_exact"] = False
    elif fault == "reset":
        summary["arms"][2]["restore_count"] = 1
    elif fault == "short_arm":
        summary["arms"][1]["frames"] = 199
    else:
        summary["arms"] = summary["arms"][::-1]
    with pytest.raises(ValueError):
        m.validate_run(plan, summary, control)


def test_actual_tree_writer_reader_preserves_all_anchor_arrays(tmp_path):
    output = m.base.Output(tmp_path / "synthetic_tree")
    value = dict(
        anchors=[dict(anchor_id=0, links=[np.eye(3, dtype=bool)] * 3)],
        facts=[
            dict(
                all_nodes=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
                empty=np.empty((0, 3), np.int32),
                missing=np.array([np.nan]),
            )
        ],
    )
    m.producer.save_tree(output, "original/initial_anchors", value)
    members = {r["path"]: r for r in output.records.values()}
    recovered = m.load_tree(output.root, members, "original/initial_anchors")
    arrays, restored = {}, {}
    assert m.producer.tracking._pack(value, arrays) == m.producer.tracking._pack(
        recovered, restored
    )
    assert {k: m.producer.tracking.signature(v) for k, v in arrays.items()} == {
        k: m.producer.tracking.signature(v) for k, v in restored.items()
    }
    recovered["anchors"][0]["links"][0].fill(False)
    assert value["anchors"][0]["links"][0].trace() == 3


def test_tree_header_array_disagreement_is_rejected(tmp_path):
    output = m.base.Output(tmp_path / "synthetic_bad_tree")
    arrays = {"x": np.array([1], np.int64)}
    output.write("bank.npz", arrays)
    output.write(
        "bank.json",
        dict(
            tree={"__array__": "x"},
            array_signatures={"x": m.producer.tracking.signature(np.array([2], np.int64))},
        ),
    )
    with pytest.raises(ValueError, match="tree signatures"):
        m.load_tree(output.root, {r["path"]: r for r in output.records.values()}, "bank")


@pytest.fixture(scope="module")
def packet_fixture():
    rig, _, _ = toy.inputs.__wrapped__()
    centers = np.array([[12.0, 12.0, 12.0], [28.0, 28.0, 28.0], [44.0, 44.0, 44.0]])
    pose = centers[:, None] + np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-3.0, -2.0, -1.0]])[None]
    masks = np.zeros((3, 64, 64), bool)
    for row, center in enumerate((12, 28, 44)):
        masks[row, center - 5 : center + 5, center - 5 : center + 5] = True
    rows = np.arange(256) % 3
    xy = np.column_stack((centers[rows, 0], centers[rows, 0]))
    pixels = np.stack([xy] * 3).astype(np.float64)
    anchors = []
    for anchor_id, mapping in enumerate(((0, 1, 2), (1, 0, 2))):
        reference = dict(
            slots=[
                dict(slot=b, assigned_mask_row=row, geometry_eligible=True, reasons=[])
                for b, row in enumerate(mapping)
            ]
        )
        anchors.append(
            m.transport.build_initial_anchor(
                (reference,) * 3, (3, 3, 3), anchor_id=anchor_id, initial_native=m.FIRST
            )
        )
    group = np.repeat(np.arange(3), 4).astype(np.int64)
    packets = []
    for evidence in range(2):
        membership = np.eye(3, dtype=bool)[group]
        if evidence:
            membership[np.arange(12) % 4 >= 2] = False
        votes = tuple(
            m.transport.transport_votes(
                group,
                np.ones(12, bool),
                membership,
                group_count=3,
                mask_available=True,
                native=m.FIRST + 1,
                release_horizon=m.FIRST + 15,
                query_sha256="a" * 64,
            )
            for _ in range(3)
        )
        packets.append(
            m.transport.packet_for_pixels(
                votes, tuple(anchors), (masks,) * 3, pixels, native_shape_hw=(64, 64)
            )
        )
    renderer = m.metrics.RefractiveFishRenderer(
        rig,
        dict(zip(m.metrics.VIEWS, pixels, strict=True)),
        m.metrics.RendererConfig(width_ratio=0.06),
        device="cpu",
        dtype=torch.float64,
    )
    sample = dict(
        frame_idx=np.asarray(m.FIRST + 1, np.int64),
        unused_pixels=pixels,
        unused_target=np.ones((3, 256)),
        unused_weights=np.full((3, 256), 1 / 256),
        unused_fit_overlap=np.zeros((3, 256), bool),
        unused_ray_valid=renderer.ray_valid.cpu().numpy(),
    )
    poses = np.stack((pose, pose[[1, 0, 2]], pose + 0.4))
    return poses, sample, tuple(packets), rig


def test_real_cpu_renderer_and_full_target_packet_preserve_cross_axes_and_manual_score(
    packet_fixture,
):
    poses, sample, packets, rig = packet_fixture
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    before = m.transport.signature(poses), tuple(m.transport.signature(p.target) for p in packets)
    try:
        score = m.metrics.score_fixed_images(poses, sample, packets, rig)
    finally:
        torch.set_num_threads(old)
    assert score["owner_loss"].shape == (2, 3, 2)
    assert score["target_mass"].shape == (2, 2, 3, 3)
    for e, packet in enumerate(packets):
        for arm in range(3):
            for anchor in range(2):
                direct = (
                    packet.target[anchor]
                    * (1 - score["rendered_ownership"][arm]) ** 2
                    * (sample["unused_weights"] * sample["unused_target"] * score["ray_valid"])[
                        ..., None
                    ]
                ).sum() / 3
                np.testing.assert_allclose(
                    score["owner_loss"][e, arm, anchor], direct, rtol=0, atol=1e-15
                )
    np.testing.assert_allclose(
        score["owner_loss"][1], score["owner_loss"][0] / 2, rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(score["union_loss"][0], score["union_loss"][1], rtol=0, atol=1e-15)
    assert score["owner_loss"][0, 0, 0] < score["owner_loss"][0, 1, 0]
    assert (
        m.transport.signature(poses),
        tuple(m.transport.signature(p.target) for p in packets),
    ) == before


@pytest.mark.parametrize("fault", ["mask", "query", "horizon", "pixels", "overlap"])
def test_different_common_packet_evidence_is_rejected_before_render(
    packet_fixture, monkeypatch, fault
):
    poses, sample, packets, rig = packet_fixture
    sample, packets = copy.deepcopy(sample), list(packets)
    p = packets[1]
    if fault == "mask":
        packets[1] = replace(p, mask_signatures=(None,) * 3)
    elif fault == "query":
        packets[1] = replace(p, query_sha256=("b" * 64,) * 3)
    elif fault == "horizon":
        packets[1] = replace(p, release_horizons=p.release_horizons + 8)
    elif fault == "pixels":
        packets[1] = replace(p, pixels_signature=m.transport.signature(p.pixels + 1))
    else:
        sample["unused_fit_overlap"][0, 0] = True

    def forbidden(*args, **kwargs):
        raise AssertionError("render must follow all common-evidence guards")

    monkeypatch.setattr(m.metrics, "RefractiveFishRenderer", forbidden)
    with pytest.raises(ValueError):
        m.metrics.score_fixed_images(poses, sample, tuple(packets), rig)


def support_fixture():
    rig, pose, _ = toy.inputs.__wrapped__()
    detections = {v: rig.camera(v).project(pose) for v in m.metrics.VIEWS}
    plan = np.broadcast_to(np.arange(3, dtype=np.int32)[:, None, None], (3, 3, 3)).copy()
    frame = SimpleNamespace(detections=detections)
    rays = m.RayObservations.from_points(rig, detections)
    sources, observed, _ = m.supported_nodes(pose, plan, rays, rig)
    candidates = {
        arm: dict(
            pose_xyz=pose[None].copy(), sources=sources[None].copy(), observed=observed[None].copy()
        )
        for arm in m.metrics.ARMS
    }
    return candidates, frame, rig


def test_current_support_is_actually_reprojected_for_all_three_arms():
    candidates, frame, rig = support_fixture()
    actual = m.recheck_support(candidates, 0, frame, rig)
    assert set(actual) == set(m.metrics.ARMS)
    assert all(r["observed"].all() for r in actual.values())
    candidates["ct_first"]["observed"][0, 0, 0] = False
    with pytest.raises(ValueError, match="current strict support"):
        m.recheck_support(candidates, 0, frame, rig)


def test_source_exclusivity_is_rechecked_even_if_helper_returns_unchanged_rows(monkeypatch):
    candidates, frame, rig = support_fixture()
    for c in candidates.values():
        c["sources"][0, 1, 0, 0] = c["sources"][0, 0, 0, 0]
    monkeypatch.setattr(
        m,
        "supported_nodes",
        lambda pose, sources, *args: (sources.copy(), np.ones((3, 3), bool), np.zeros((3, 3, 3))),
    )
    with pytest.raises(ValueError, match="reused"):
        m.recheck_support(candidates, 0, frame, rig)


def test_cli_saves_failed_seal_without_loading_payload_after_incomplete_control(
    tmp_path, monkeypatch
):
    plan, summary, control = run_metadata()
    control["history"]["byte_exact"] = False
    monkeypatch.setattr(m, "WORK", tmp_path)

    class Evidence:
        def __init__(self):
            self.paths = {}

        def add(self, path, expected=None):
            self.paths[str(path)] = expected or "a" * 64

    monkeypatch.setattr(m.base, "Evidence", Evidence)
    monkeypatch.setattr(
        m.base,
        "verified",
        lambda *args: ({"inputs": {str(Path(m.producer.__file__).resolve()): "b" * 64}}, {}),
    )
    docs = dict(
        zip(
            ("plan.json", "summary.json", "original_control.json"),
            (plan, summary, control),
            strict=True,
        )
    )
    monkeypatch.setattr(m.base, "bound_json", lambda root, members, name: docs[name])
    monkeypatch.setattr(
        m.producer,
        "load_inputs",
        lambda *args: pytest.fail("numeric assets must not open after failed replay control"),
    )
    target = tmp_path / "new_failure"
    with pytest.raises(ValueError, match="numerical replay"):
        m.run(tmp_path / "completed", "a" * 64, target)
    saved = json.loads((target / "evaluation.provenance.json").read_text())
    assert saved["stage"] == m.STAGE and saved["status"] == "failed"
    assert saved["completed_frames"] == 0 and saved["outputs"] == {}
    assert not (target / "summary.json").exists()
