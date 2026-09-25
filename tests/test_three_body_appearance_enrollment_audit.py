"""CPU causal-prefix audit rejects source and feature corruption without tracking."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from medaka3d.types import VIEWS


def modules():
    lab = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
    names = (
        "prepare_sequence", "observations", "run_experiment", "appearance_diagnostic",
        "node_geometry", "online_appearance", "audit_appearance_enrollment",
    )
    previous = {name: sys.modules.get(name) for name in names}
    old_path = sys.path.copy()
    loaded = {}
    try:
        sys.path.insert(0, str(lab))
        for name in names:
            spec = importlib.util.spec_from_file_location(
                f"enroll_audit_test_{name}", lab / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        sys.path[:] = old_path
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return loaded


M = modules()
A = M["audit_appearance_enrollment"]
ONLINE = M["online_appearance"]


class Rig:
    views = VIEWS

    def camera(self, view):
        assert view in VIEWS

        def rays(points):
            origin = np.concatenate((points, np.zeros((*points.shape[:-1], 1))), axis=-1)
            direction = np.zeros_like(origin)
            direction[..., 2] = 1
            return origin, direction

        return SimpleNamespace(project=lambda xyz: np.asarray(xyz)[..., :2].copy(), rays=rays)


def frame_state(index):
    pose = np.array([
        [[12 + 30*i, 15, 8], [17 + 30*i, 15, 8], [20 + 30*i, 15, 8]] for i in range(3)
    ], np.float64)
    masks = np.zeros((3, 30, 100), bool)
    image = np.zeros((30, 100, 3), np.uint8)
    for fish, color in enumerate(([30, 30, 100], [20, 100, 160], [80, 100, 100])):
        masks[fish, 6:25, 5+30*fish:25+30*fish] = True
        image[masks[fish]] = np.array(color) + (index - 40)*3
    frame = SimpleNamespace(
        frame_idx=index, masks={view: masks.copy() for view in VIEWS},
        images_bgr={view: image.copy() for view in VIEWS},
        detections={view: pose[..., :2].copy() for view in VIEWS},
    )
    sources = np.broadcast_to(np.arange(3)[:, None, None], (3, 3, 3)).copy()
    return frame, pose, sources, np.ones((3, 3), bool), np.zeros(3, bool)


def collect(states, minimum=2):
    accumulator = ONLINE.OnlineAppearanceAccumulator(
        Rig(), ONLINE.OnlineAppearanceConfig(len(states), minimum)
    )
    for state in states:
        accumulator.observe(*state)
    return {k: v for k, v in accumulator.snapshot().items() if isinstance(v, np.ndarray)}


def test_replay_matches_causal_accumulator_with_review_missing_merged_and_disabled_view():
    states = [frame_state(40 + i) for i in range(3)]
    # The first two top views cannot enroll S0/S1 because the independent mask is merged.
    for state in states[:2]:
        masks = state[0].masks[VIEWS[0]]
        state[0].masks[VIEWS[0]] = np.array([masks[0] | masks[1], masks[2]])
    states[0][-1][2] = True
    states[0][0].masks[VIEWS[1]] = None
    states[1][0].masks[VIEWS[2]] = np.zeros((0, 30, 100), bool)
    arrays = collect(states)
    A.validate_snapshot(arrays, np.arange(40, 45, dtype=np.int64), 3)
    recomputed, audit, comparisons = A.replay_prefix(
        (state[0] for state in states), arrays, Rig(), 2
    )
    assert len(audit) == 3 and all(v["bitwise_equal"] for v in comparisons.values())
    np.testing.assert_array_equal(recomputed["enabled_views"], [False, True, False])
    assert np.isnan(recomputed["templates"][[0, 2]]).all()
    assert np.isfinite(recomputed["templates"][1]).all()
    assert audit[0]["views"][VIEWS[1]]["mask_available"] is False
    assert audit[1]["views"][VIEWS[2]]["mask_available"] is True
    assert audit[1]["views"][VIEWS[2]]["mask_count"] == 0
    assert recomputed["reasons"][0, 1, 2] == "identity_review"


def test_replay_enabled_medians_match_and_does_not_consume_future_or_rewrite_online_labels():
    states = [frame_state(40 + i) for i in range(3)]
    arrays = collect(states)
    seen, closed = [], []

    def observations():
        try:
            for state in states:
                seen.append(state[0].frame_idx)
                yield state[0]
            raise AssertionError("must not request a post-prefix frame")
        finally:
            closed.append(True)

    recomputed, _, _ = A.replay_prefix(observations(), arrays, Rig(), 2)
    assert seen == [40, 41, 42] and closed == [True]
    assert recomputed["enabled_views"].all()
    expected = np.median(arrays["features"], axis=0)
    A.exact_array(recomputed["templates"], expected, "known median")


@pytest.mark.parametrize("change", ["offset", "gap", "missing_online", "wrong_dtype", "support"])
def test_snapshot_rejects_wrong_native_prefix_or_online_schema(change):
    arrays = collect([frame_state(40 + i) for i in range(3)])
    if change == "offset":
        arrays["frame_idx"] += 1
    elif change == "gap":
        arrays["frame_idx"][1] += 1
    elif change == "missing_online":
        del arrays["online_pose_xyz"]
    elif change == "wrong_dtype":
        arrays["online_sources"] = arrays["online_sources"].astype(np.int32)
    else:
        arrays["online_observed"][0, 0, 0] = False
    with pytest.raises(ValueError):
        A.validate_snapshot(arrays, np.arange(40, 45, dtype=np.int64), 3)


@pytest.mark.parametrize("change", ["row", "shared", "nonfinite", "geometry", "observed"])
def test_online_support_rejects_real_sleap_or_geometry_contradiction(change):
    frame, pose, sources, observed, _ = frame_state(40)
    if change == "row":
        sources[0, 0, 0] = 3
    elif change == "shared":
        sources[1, 0, 0] = 0
    elif change == "nonfinite":
        frame.detections[VIEWS[0]][0, 0, 0] = np.nan
    elif change == "geometry":
        pose[0, 0, 0] += 3
    else:
        observed[0, 0] = False
    with pytest.raises(ValueError):
        A.validate_online_support(frame, pose, sources, observed, Rig())


def test_native_pixel_gate_is_checked_separately_from_ray_distance():
    class OffsetProjectionRig(Rig):
        def camera(self, view):
            camera = super().camera(view)
            camera.project = lambda xyz: np.asarray(xyz)[..., :2] + 15
            return camera

    frame, pose, sources, observed, _ = frame_state(40)
    # Rays pass through each saved node, while image projection contradicts it.
    with pytest.raises(ValueError, match="ray/pixel support gates"):
        A.validate_online_support(frame, pose, sources, observed, OffsetProjectionRig())


@pytest.mark.parametrize("field", ["features", "reasons", "instance_rows", "core_pixels"])
def test_replay_rejects_tampered_enrollment_and_closes_decoder(field):
    state = frame_state(40)
    arrays = collect([state], minimum=1)
    closed = []
    if field == "features":
        arrays[field][0, 0, 0, 0] += .001
    elif field == "reasons":
        arrays[field][0, 0, 0] = "identity_review"
    else:
        arrays[field][0, 0, 0] += 1

    def observations():
        try:
            yield state[0]
        finally:
            closed.append(True)

    with pytest.raises(ValueError, match="bitwise equal"):
        A.replay_prefix(observations(), arrays, Rig(), 1)
    assert closed == [True]


@pytest.mark.parametrize("field", ["counts", "templates", "enabled_views"])
def test_final_aggregation_is_recomputed_not_trusted(field):
    state = frame_state(40)
    arrays = collect([state], minimum=1)
    arrays[field].flat[0] = 0
    with pytest.raises(ValueError, match="bitwise equal"):
        A.replay_prefix(iter([state[0]]), arrays, Rig(), 1)


def test_tiny_core_and_local_missing_sources_do_not_silently_enroll():
    state = frame_state(40)
    frame, pose, sources, observed, review = state
    mask = frame.masks[VIEWS[0]]
    mask[0] = False
    mask[0, 15, 12:18] = True
    sources[1, 1, :2] = -1
    arrays = collect([state], minimum=1)
    replay = A.describe_online_frame(frame, pose, sources, observed, review, Rig())
    assert replay["reasons"][0, 0] == "core_too_small"
    assert replay["core_pixels"][0, 0] == 0
    assert replay["reasons"][1, 1] == "no_direct_mouth_pect"
    for key, value in replay.items():
        A.exact_array(value, arrays[key][0], key)


def test_bitwise_comparison_distinguishes_signed_zero():
    with pytest.raises(ValueError, match="bitwise equal"):
        A.exact_array(np.array([0.]), np.array([-0.]), "signed zero")


@pytest.mark.parametrize("values", [[np.nan, 0, 0, 0, 0, 0], [np.inf]*6, [1.01]*6])
def test_aggregation_rejects_partly_missing_or_out_of_range_features(values):
    features = np.zeros((2, 3, 3, 6))
    features[0, 0, 0] = values
    with pytest.raises(ValueError):
        A.aggregate_descriptors(features, 1)


def test_premature_end_fails_instead_of_sealing_partial_prefix():
    states = [frame_state(40 + i) for i in range(3)]
    with pytest.raises(ValueError, match="ended before"):
        A.replay_prefix(iter([states[0][0]]), collect(states), Rig(), 2)


def test_changed_live_or_unbound_replay_source_is_rejected(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    relative = "source.py"
    current = repo / relative
    current.parent.mkdir()
    current.write_text("version1", encoding="utf-8")
    run = repo / "run"
    snapshot = run / "code_snapshot" / relative
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(current.read_bytes())
    digest = A.base.sha256(current)
    monkeypatch.setattr(A.base, "REPO", repo)
    monkeypatch.setattr(A, "REPLAY_SOURCES", (relative,))
    summary = {"code_sha256_before": {relative: digest}}
    assert A.check_replay_sources(summary, run) == {str(current): digest}
    with pytest.raises(ValueError, match="recorded run"):
        A.check_replay_sources({"code_sha256_before": {}}, run)
    current.write_text("version2", encoding="utf-8")
    with pytest.raises(ValueError, match="recorded run"):
        A.check_replay_sources(summary, run)


def test_cli_refuses_existing_output_without_importing_evaluator(tmp_path):
    with pytest.raises(FileExistsError, match="overwrite"):
        A.main(["--run", "absent-run", "--output", str(tmp_path)])


def test_failed_completed_run_verification_writes_failed_audit_not_success(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({
        "enrollment_config": {"training_frames": 166, "minimum_template_samples": 5},
    }), encoding="utf-8")

    def reject(*args, **kwargs):
        assert kwargs["include_post_inference"] is False
        raise ValueError("run is not sealed complete")

    monkeypatch.setitem(sys.modules, "evaluate_appearance_minute", SimpleNamespace(
        verify_appearance_evidence=reject, verify_enrollment=None, validate_appearance_arrays=None,
    ))
    monkeypatch.setitem(sys.modules, "evaluate_variant_minute", SimpleNamespace(
        load_json=lambda path: json.loads(Path(path).read_text(encoding="utf-8")),
        local_path=lambda path: Path(path).resolve(strict=True),
    ))
    output = tmp_path / "audit"
    with pytest.raises(ValueError, match="not sealed complete"):
        A.main(["--run", str(run), "--output", str(output)])
    record = json.loads((output / "audit.provenance.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert not (output / "summary.json").exists()
    assert not (output / "recomputed_enrollment.npz").exists()
