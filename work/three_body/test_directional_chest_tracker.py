"""Migration must preserve full history; extra proposals must respect old budget."""

import copy
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"))
import directional_chest_tracker as module
from continuous_tracker import AcceleratedContinuousPersistentNodeTracker
from observations import FrameObservation
from tracker import Proposal, ThreeBodyTracker, TrackerConfig

from medaka3d.types import VIEWS


@dataclass
class Camera:
    axes: tuple

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes]

    def rays(self, uv):
        uv = np.asarray(uv)
        origin = np.zeros((*uv.shape[:-1], 3))
        origin[..., self.axes] = uv
        direction = np.zeros_like(origin)
        direction[..., next(iter(set(range(3)) - set(self.axes)))] = 1
        origin[~np.isfinite(uv).all(-1)] = np.nan
        return origin, direction


@dataclass
class Rig:
    cameras: tuple
    views: tuple = VIEWS
    world_units: str = "mm"

    def camera(self, view):
        return self.cameras[self.views.index(view)]


@pytest.fixture
def inputs():
    rig = Rig(tuple(Camera(axes) for axes in ((0, 1), (0, 2), (1, 2))))
    centers = np.array([[40.0, 50.0, 60.0], [90.0, 90.0, 90.0], [140.0, 140.0, 140.0]])
    pose = (
        centers[:, None] + np.array([[7.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-12.0, -8.0, -4.0]])[None]
    )
    config = TrackerConfig(
        device="cpu",
        mode="detectron2",
        opt_steps=1,
        beam_size=2,
        max_hypotheses=4,
        positive_samples=4,
        near_samples=4,
        global_samples=4,
        flow_proposals=True,
    )
    return rig, pose, config


@pytest.fixture(autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def checkpoint(inputs):
    rig, pose, cfg = inputs
    source = AcceleratedContinuousPersistentNodeTracker(rig, pose, cfg)
    # Exercise genuine temporal steps on CPU with the original explicit fitter.
    source._fit = MethodType(ThreeBodyTracker._fit, source)
    source.begin()
    for native in range(5):
        detections = {v: rig.camera(v).project(pose) for v in VIEWS}
        for value in detections.values():
            value[2, 1] = np.nan
        images = {v: np.zeros((192, 192, 3), np.uint8) for v in VIEWS}
        masks = {v: np.empty((0, 192, 192), bool) for v in VIEWS}
        source.advance(FrameObservation(native, detections, images, masks, images))
    return source.export_state()


def test_explicit_import_preserves_history_buffers_ages_review_and_lifecycle(inputs):
    arrays, metadata = checkpoint(inputs)
    old = AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = module.DirectionalChestContinuousTracker(*inputs)
    report = module.import_original_state(target, old, arrays, metadata)
    assert report["all_arrays_byte_preserved"] and report["original_metadata_unchanged"]
    assert target.processed_frames == 5 and target.previous_frame == 4
    assert target.initialize_count == 1 and target.restore_count == 1
    assert old.initialize_count == 0
    assert target._states[0].node_missing[2, 1] == 5
    assert target._states[0].review[2]
    assert any(key.startswith("previous_images_bgr") for key in report["arrays"])
    # Further ordinary exports/restores keep the experimental context and buffers.
    saved, meta = target.export_state()
    resumed = module.DirectionalChestContinuousTracker(*inputs)
    resumed.restore_state(saved, meta)
    assert resumed.restore_count == 2 and resumed.processed_frames == 5
    assert module.array_signature(saved) == module.array_signature(resumed.export_state()[0])
    with pytest.raises(RuntimeError):
        resumed.begin()
    with pytest.raises(ValueError, match="context"):
        old.restore_state(saved, meta)


def test_import_rejects_configuration_drift_before_target_start(inputs):
    arrays, metadata = checkpoint(inputs)
    rig, pose, config = inputs
    old = AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = module.DirectionalChestContinuousTracker(rig, pose, replace(config, motion_weight=0.3))
    with pytest.raises(ValueError, match="cannot change"):
        module.import_original_state(target, old, arrays, metadata)
    assert target.initialize_count == 0


def test_original_history_validation_cannot_be_bypassed_by_context_migration(inputs):
    arrays, metadata = checkpoint(inputs)
    arrays = copy.deepcopy(arrays)
    arrays["state_node_missing"][0, 2, 1] = 0
    old = AcceleratedContinuousPersistentNodeTracker(*inputs)
    target = module.DirectionalChestContinuousTracker(*inputs)
    with pytest.raises(ValueError, match="active beam differs"):
        module.import_original_state(target, old, arrays, metadata)
    assert target.initialize_count == 0


def test_extra_chests_do_not_displace_original_proposals_when_budget_full(monkeypatch, inputs):
    rig, pose, config = inputs
    tracker = SimpleNamespace(
        rig=rig,
        config=config,
        lengths=np.linalg.norm(np.diff(pose, axis=1), axis=-1),
        lower=np.zeros(3),
        upper=np.full(3, 192.0),
    )
    plan = np.zeros((3, 3, 3), np.int32)
    original = [Proposal(pose, pose, plan, 0, "predict")] * (config.max_hypotheses + 6)
    state = SimpleNamespace(node_missing=np.full((3, 3), 4), missing=np.zeros(3, int))
    monkeypatch.setattr(module, "supported_nodes", lambda *a: (plan, np.zeros((3, 3), bool), None))
    monkeypatch.setattr(module, "node_pools", lambda *a: [])
    monkeypatch.setattr(module, "recovery_poses", lambda *a, **k: [(0.0, pose, 0)] * 3)
    monkeypatch.setattr(module, "assignment_plans", lambda *a, **k: [(0.0, plan)])
    extra, diagnostic = module.chest_proposals(original, [state], None, tracker)
    assert extra == []
    assert diagnostic["generated"] == diagnostic["excluded_by_budget"] == 3
    assert diagnostic["admitted"] == 0
    state.missing[0] = 1
    extra, diagnostic = module.chest_proposals(original, [state], None, tracker)
    assert extra == [] and not diagnostic["enabled"]
