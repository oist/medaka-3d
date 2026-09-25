"""Synthetic retained-record semantics against the actual frozen advance loop.

Proposal/fit/observation callbacks are synthetic; no fit, raw data, or GPU runs.
"""

import copy
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path[:0] = [
    str(Path(__file__).parent),
    str(Path(__file__).resolve().parents[2] / "labs/20260909_three_body"),
]
import bounded_fit_record_check as module  # noqa: E402
import continuous_tracker as frozen  # noqa: E402
from chest_candidate_diagnostics import selected_indices  # noqa: E402
from tracker import ThreeBodyTracker  # noqa: E402


def fixture():
    scene = np.zeros((3, 3, 3), np.float64)
    scene[:, :, 0] = np.arange(3)[:, None] * 10 + [-1, 0, 2]
    scene[:, :, 2] = 20
    pose = np.stack([scene.copy() for _ in range(6)])
    pose[1] += 0.1
    pose[2] = scene[[1, 0, 2]]
    pose[4, 0, 0, 1] += 1
    packet = dict(
        frame_idx=np.array(7, np.int64),
        proposal_parent=np.array([0, 0, 0, 1, 0, 0], np.int64),
        proposal_permutation=np.array(
            [[0, 1, 2], [0, 1, 2], [1, 0, 2], [0, 1, 2], [0, 1, 2], [1, 0, 2]], np.int64
        ),
        proposal_kind=np.array(
            [
                "predict",
                "node_alternative",
                "body_permutation",
                "predict",
                "mouth_recovery",
                "body_permutation",
            ],
            dtype="U30",
        ),
        parent_pose=np.stack([scene, scene + 0.2]),
        parent_missing=np.array([[3, 3, 0], [0, 2, 1]], np.int64),
        parent_review=np.array([[False, False, True], [False, True, False]]),
        parent_cost=np.array([10.0, 10.15]),
        parent_node_missing=np.arange(18, dtype=np.int64).reshape(2, 3, 3),
        parent_direct_pectoral_age=np.array([[4, 0, 5], [2, 7, 0]], np.int64),
        flow_present=np.array(True),
        flow_node_valid=np.zeros((3, 3), bool),
    )
    sources = np.broadcast_to(
        np.arange(3, dtype=np.int32)[None, :, None, None], (6, 3, 3, 3)
    ).copy()
    sources[0, 0, :, 1] = -1
    sources[0, 0, 1:, 0] = -1
    sources[0, 1, :, :2] = -1
    observed = (sources >= 0).sum(axis=2) >= 2
    masks = np.zeros((6, 3), bool)
    masks[0, :2] = True
    local = np.array([0.0, 0.02, 0.2, 0.05, 0.1, 0.03])
    parts = np.arange(30, dtype=float).reshape(6, 5) / 100
    cfg = SimpleNamespace(
        beam_size=2,
        review_gap_frames=4,
        ambiguity_margin=0.05,
        mode="detectron2",
        positive_samples=1,
        near_samples=1,
        global_samples=1,
        flow_proposals=True,
    )
    return packet, (pose, local, parts, masks, sources, observed), cfg


def choose(packet, values, cfg, count):
    pose, local, *_ = values
    path = local[:count] + packet["parent_cost"][packet["proposal_parent"][:count]]
    return selected_indices(
        pose[:count],
        path,
        packet["proposal_parent"][:count],
        packet["proposal_permutation"][:count],
        cfg.beam_size,
    )


def reconstruct(packet, values, cfg, count=6):
    selected, order = choose(packet, values, cfg, count)
    return module.reconstruct_records(packet, *values, selected, order, cfg, proposal_count=count)


def advance_oracle(packet, values, cfg, count, monkeypatch):
    """Run frozen _advance_owned verbatim with supplied fit/geometry callbacks."""
    pose, local, parts, masks, sources, observed = values
    monkeypatch.setattr(frozen.RayObservations, "from_points", lambda *a: object())
    monkeypatch.setattr(frozen, "sample_pixels", lambda *a: object())
    flow = SimpleNamespace(node_valid=packet["flow_node_valid"].copy())
    monkeypatch.setattr(frozen.FlowObservations, "from_frames", lambda *a: flow)
    monkeypatch.setattr(
        frozen,
        "supported_nodes",
        lambda value, plan, *a: (sources[plan].copy(), observed[plan].copy(), None),
    )
    states = []
    for p in range(len(packet["parent_cost"])):
        states.append(
            SimpleNamespace(
                pose=packet["parent_pose"][p].copy(),
                velocity=np.zeros((3, 3)),
                missing=packet["parent_missing"][p].copy(),
                review=packet["parent_review"][p].copy(),
                cost=float(packet["parent_cost"][p]),
                node_missing=packet["parent_node_missing"][p].copy(),
                direct_pectoral_age=packet["parent_direct_pectoral_age"][p].copy(),
            )
        )
    proposals = [
        SimpleNamespace(
            parent=int(packet["proposal_parent"][i]),
            permutation=tuple(packet["proposal_permutation"][i]),
            kind=str(packet["proposal_kind"][i]),
            plan=i,
        )
        for i in range(count)
    ]
    tracker = SimpleNamespace(
        config=cfg,
        rig=object(),
        _states=states,
        _history=[],
        _previous_observation=object() if bool(packet["flow_present"]) else None,
        _proposals=lambda *a: proposals,
        _fit=lambda *a: (
            pose[:count].copy(),
            local[:count].copy(),
            parts[:count].copy(),
            masks[:count].copy(),
        ),
        _different_identity=ThreeBodyTracker._different_identity,
    )
    frozen.ContinuousPersistentNodeTracker._advance_owned(
        tracker, SimpleNamespace(frame_idx=int(packet["frame_idx"]), detections={})
    )
    return {key: np.asarray([r[key] for r in tracker._history[-1]]) for key in frozen.RECORD_FIELDS}


@pytest.mark.parametrize("count", [3, 4, 6])
@pytest.mark.parametrize("flow_present", [False, True])
def test_all19_against_real_frozen_advance(monkeypatch, count, flow_present):
    packet, values, cfg = fixture()
    packet["flow_present"] = np.array(flow_present)
    actual = reconstruct(packet, values, cfg, count)
    expected = advance_oracle(packet, values, cfg, count, monkeypatch)
    assert module.RECORD_FIELDS == frozen.RECORD_FIELDS
    report = module.compare_records(actual, expected)
    assert report["fields"] == 19 and report["all_discrete_values_exact"]
    assert all(v["maximum_absolute_difference"] == 0 for v in report["floating_fields"].values())
    assert report["unicode_width_differences"]


def test_body_bridge_direct_ages_sticky_review_and_flow_presence():
    packet, values, cfg = fixture()
    result = reconstruct(packet, values, cfg, 4)
    np.testing.assert_array_equal(result["body_supported"][0], [True, False, True])
    np.testing.assert_array_equal(result["missing"][0], [0, 4, 0])
    np.testing.assert_array_equal(result["identity_review"][0], [False, True, True])
    np.testing.assert_array_equal(result["direct_pectoral_age"][0], [0, 0, 6])
    assert result["node_missing"][0, 0, 1] == packet["parent_node_missing"][0, 0, 1] + 1
    assert not result["proposal_flow_nodes"].any()
    assert np.all(result["proposal_flow_parent"] == 0)  # Object exists, though no valid flow nodes.


def test_unselected_pool_drives_margin_and_control_ignores_treatment_suffix():
    packet, values, cfg = fixture()
    cfg.beam_size = 1
    control = reconstruct(packet, values, cfg, 4)
    treatment = reconstruct(packet, values, cfg, 6)
    assert control["score_margin"][0] == pytest.approx(0.2)
    assert treatment["score_margin"][0] == pytest.approx(0.03)
    assert not control["identity_review"][0, 0] and treatment["identity_review"][0, 0]
    assert control["hypotheses"].tolist() == [4] and treatment["hypotheses"].tolist() == [6]


def test_same_spatial_tie_permutation_is_retained_and_negative_margin_is_preserved():
    packet, values, cfg = fixture()
    result = reconstruct(packet, values, cfg)
    np.testing.assert_array_equal(result["pose_xyz"][0], result["pose_xyz"][1])
    assert result["proposal_kind"].tolist() == ["predict", "body_permutation"]
    assert result["score_margin"][1] == pytest.approx(-0.03)
    assert result["identity_review"][1, :2].all()


def test_dedup_requires_same_parent_same_permutation_and_all_nodes_distance():
    packet, values, cfg = fixture()
    cfg.beam_size = 6
    chosen, _ = choose(packet, values, cfg, 6)
    assert 1 not in chosen  # Same parent/permutation, all nodes closer than .5.
    assert {0, 2, 3, 4, 5} == set(
        chosen
    )  # Another parent, mouth displacement or explicit swap remains.
    result = reconstruct(packet, values, cfg)
    assert len(result["frame_idx"]) == 5


@pytest.mark.parametrize(
    "bad", ["order", "selected", "count", "permutation", "support", "collision", "age"]
)
def test_bad_packet_or_selection_is_rejected(bad):
    packet, values, cfg = fixture()
    selected, order = choose(packet, values, cfg, 6)
    count = 6
    if bad == "order":
        order = order[::-1]
    elif bad == "selected":
        selected = selected[::-1]
    elif bad == "count":
        count = 0
    elif bad == "permutation":
        packet["proposal_permutation"][0] = [0, 0, 2]
    elif bad == "support":
        values[5][1, 0, 0] = False
    elif bad == "collision":
        values[4][1, 1, 0, 0] = values[4][1, 0, 0, 0]
    else:
        packet["parent_node_missing"][0, 0, 0] = np.iinfo(np.int64).max
    with pytest.raises(ValueError):
        module.reconstruct_records(packet, *values, selected, order, cfg, proposal_count=count)


def test_no_input_mutation_or_output_alias():
    packet, values, cfg = fixture()
    originals = [*packet.values(), *values]
    arrays = [a for a in originals if isinstance(a, np.ndarray)]
    before = [(a.dtype, a.shape, a.tobytes()) for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    result = reconstruct(packet, values, cfg)
    for output in result.values():
        assert all(not np.shares_memory(output, a) for a in arrays)
        output.flat[0] = "changed" if output.dtype.kind == "U" else 0
    assert before == [(a.dtype, a.shape, a.tobytes()) for a in arrays]


def test_npz_scalar_round_trip_preserves_record_semantics():
    packet, values, cfg = fixture()
    expected = reconstruct(packet, values, cfg)
    buffer = io.BytesIO()
    np.savez(buffer, **packet)
    buffer.seek(0)
    with np.load(buffer, allow_pickle=False) as saved:
        restored = {name: saved[name].copy() for name in saved.files}
    assert restored["frame_idx"].shape == restored["flow_present"].shape == ()
    assert restored["frame_idx"].dtype == np.dtype("int64")
    assert restored["flow_present"].dtype == np.dtype(bool)
    actual = reconstruct(restored, values, cfg)
    assert all(actual[name].tobytes() == expected[name].tobytes() for name in actual)
    restored["frame_idx"] = np.asarray(7.0)
    with pytest.raises(ValueError, match="integer scalar"):
        reconstruct(restored, values, cfg)
    restored["frame_idx"] = np.asarray(True)
    with pytest.raises(ValueError, match="integer scalar"):
        reconstruct(restored, values, cfg)
    restored["frame_idx"] = np.asarray(7, np.int64)
    restored["flow_present"] = np.asarray(1)
    with pytest.raises(ValueError, match="bool scalar"):
        reconstruct(restored, values, cfg)


@pytest.mark.parametrize("field", module.RECORD_FIELDS)
def test_comparison_covers_every_retained_field(field):
    packet, values, cfg = fixture()
    expected = reconstruct(packet, values, cfg)
    actual = copy.deepcopy(expected)
    value = actual[field]
    if value.dtype.kind == "b":
        value.flat[0] = not value.flat[0]
    elif value.dtype.kind == "U":
        value.flat[0] = "bad_kind"
    else:
        value.flat[0] += 1
    with pytest.raises(ValueError, match=field):
        module.compare_records(actual, expected)


def test_compare_fixed_absolute_tolerance_no_relative_tolerance():
    packet, values, cfg = fixture()
    expected = reconstruct(packet, values, cfg)
    actual = copy.deepcopy(expected)
    actual["pose_xyz"][0, 0, 0, 0] += 5e-11
    assert (
        module.compare_records(actual, expected)["floating_fields"]["pose_xyz"][
            "maximum_absolute_difference"
        ]
        > 0
    )
    expected["path_costs"][:] = 1e8
    actual = copy.deepcopy(expected)
    actual["path_costs"][0] += 1e-6
    with pytest.raises(ValueError, match="path_costs"):
        module.compare_records(actual, expected)


@pytest.mark.parametrize("bad", [np.nan, -np.inf, 0.0])
def test_margin_infinity_sign_and_nan_are_not_hidden(bad):
    packet, values, cfg = fixture()
    expected = reconstruct(packet, values, cfg, 1)
    assert np.isposinf(expected["score_margin"]).all()
    assert module.compare_records(expected, copy.deepcopy(expected))["status"] == "ok"
    actual = copy.deepcopy(expected)
    actual["score_margin"][0] = bad
    with pytest.raises(ValueError):
        module.compare_records(actual, expected)


def test_compare_numeric_dtype_exact_but_unicode_width_reported():
    packet, values, cfg = fixture()
    expected = reconstruct(packet, values, cfg)
    actual = copy.deepcopy(expected)
    actual["proposal_kind"] = actual["proposal_kind"].astype("U80")
    assert module.compare_records(actual, expected)["unicode_width_differences"]
    actual["parent"] = actual["parent"].astype(np.int32)
    with pytest.raises(ValueError, match=r"dtype.*parent"):
        module.compare_records(actual, expected)
