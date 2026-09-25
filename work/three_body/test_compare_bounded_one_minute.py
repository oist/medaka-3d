"""Synthetic new-branch gates and all-node arithmetic; no real result is opened."""

import copy
import json
from types import SimpleNamespace

import compare_bounded_one_minute as m
import numpy as np
import pytest
import test_audit_bounded_first_chunk as imported


def complete():
    return dict(
        schema="bounded_mouth_one_minute_v1",
        status="ok",
        one_minute_completed=True,
        native_interval=[m.START, m.STOP],
        native_frames=9960,
        logical_initialization_count=1,
        restore_count=4,
        external_pose_resets=0,
        checkpoint_count=6,
        diagnostic_frames=321,
        periodic_checkpoints_are_not_reinitializations=True,
    )


def test_completed_header_accepts_current_contract():
    m.validate_completion(complete())


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "incomplete"),
        ("one_minute_completed", False),
        ("native_frames", 1660),
        ("native_interval", [66400, 68060]),
        ("restore_count", 5),
        ("logical_initialization_count", 2),
        ("external_pose_resets", 1),
        ("checkpoint_count", 5),
        ("diagnostic_frames", 166),
        ("periodic_checkpoints_are_not_reinitializations", False),
    ],
)
def test_completed_header_rejects_wrong_scope_or_lifecycle(key, value):
    value_dict = complete()
    value_dict[key] = value
    with pytest.raises(ValueError):
        m.validate_completion(value_dict)


def test_exit_gate_precedes_any_filesystem_or_candidate_access():
    args = SimpleNamespace(owned_session=81686, owned_exit_code=1)
    with pytest.raises(ValueError, match="CLOSED0"):
        m.main(args)


def test_unfinished_stage_never_opens_payload(tmp_path, monkeypatch):
    path = tmp_path / "tracking.provenance.json"
    path.write_text(json.dumps(dict(stage="wanted", status="failed", outputs={})))
    monkeypatch.setattr(m.checks, "REPO", tmp_path.parent)
    # The inherited guard requires a work namespace before reading a seal.
    root = tmp_path.parent / "work" / "unfinished"
    root.mkdir(parents=True, exist_ok=True)
    (root / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("payload opened"))
    with pytest.raises(ValueError, match="successful stage"):
        m.successful(root, path.name, "wanted", {})


def trace():
    frames = np.arange(100, 109)
    pose = np.zeros((9, 3, 3, 3))
    pose[:, :, :, 0] = np.arange(9)[:, None, None]
    return dict(
        frame_idx=frames,
        pose_xyz=pose,
        observed=np.ones((9, 3, 3), bool),
        identity_review=np.zeros((9, 3), bool),
    )


def test_missing_runs_keep_all_nodes_boundaries_and_elapsed_time():
    a = trace()
    a["observed"][1:4, 0, 0] = False
    a["observed"][-2:, 2, 2] = False
    a["identity_review"][:, 0] = True
    a["pose_xyz"][3, 0, 0, 0] = 1000
    result = m.node_metrics(a, 102, 109)
    mouth = result["by_slot_node"][0]
    assert mouth["longest_missing_frames_in_interval"] == 2
    assert mouth["directly_supported_adjacent_steps_gt10mm"] == 0
    assert mouth["maximum_latent_adjacent_step_mm"] == 998
    event = result["reacquisitions"][0]
    assert event["previous_direct_frame"] == 100
    assert event["missing_frames"] == 3
    assert event["direct_endpoint_elapsed_frames"] == 4
    assert event["direct_endpoint_distance_mm"] == 4
    assert event["direct_endpoint_average_mm_per_s"] == 166
    assert event["previous_latent_to_reacquired_step_mm"] == 996
    assert event["identity_review"] and not event["biological_identity_confirmed"]
    assert result["by_slot_node"][-1]["longest_missing_frames_in_interval"] == 2
    assert result["missing_runs"][-1]["terminal_open"]


def test_initial_missing_has_no_invented_endpoint_and_empty_steps_are_neutral():
    a = trace()
    a["observed"][:4, 0, 0] = False
    a["observed"][:, 1, 2] = False
    result = m.node_metrics(a, 100, 109)
    event = result["reacquisitions"][0]
    assert event["previous_direct_frame"] is None
    assert event["direct_endpoint_average_mm_per_s"] is None
    tail = result["by_slot_node"][5]
    assert tail["maximum_directly_supported_adjacent_step_mm"] is None
    assert tail["longest_missing_frames_in_interval"] == 9


def test_non_native_or_missing_interval_rejected():
    a = trace()
    with pytest.raises(ValueError, match="complete evaluation"):
        m.node_metrics(a, 99, 109)
    a["frame_idx"][2] += 1
    with pytest.raises(ValueError, match="native"):
        m.node_metrics(a, 100, 109)


def test_three_pair_support_delta_keeps_slots_and_no_mutation():
    a, b = trace(), trace()
    a["observed"][1, 0, 0] = False
    b["observed"][2, 2, 2] = False
    b["identity_review"][:, 0] = True
    before = {k: v.copy() for k, v in b.items()}
    value, changes = m.support_delta(a, b, 100, 109)
    assert value["gained_by_node"] == [1, 0, 0]
    assert value["lost_by_node"] == [0, 0, 1]
    assert value["review_changed_by_slot"] == [9, 0, 0]
    assert changes == [
        dict(frame_idx=101, slot=0, node="mouth", change="gain"),
        dict(frame_idx=102, slot=2, node="tail", change="loss"),
    ]
    assert all(m.prior.same(before[k], b[k]) for k in b)
    reverse, _ = m.support_delta(b, a, 100, 109)
    assert reverse["gained_by_node"] == value["lost_by_node"]


def test_reused_support_requires_same_counts_gates_and_full_interval(monkeypatch):
    monkeypatch.setattr(m, "START", 100)
    monkeypatch.setattr(m, "STOP", 109)
    monkeypatch.setattr(m, "FIRST", 100)
    a = trace()
    a["sources"] = np.full((9, 3, 3, 3), -1)
    a["observed"][:] = False
    monkeypatch.setattr(m.prior.short, "stats", lambda *args: {"fixed": 1})
    check = dict(
        native_frames=9,
        frame_range=[100, 109],
        blocks=6,
        block_frames=1660,
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        temporal_subsampling=False,
        source_rows_checked=0,
        observed_nodes_checked=0,
    )
    saved = dict(
        methods=dict(original=dict(full_minute={"fixed": 1}, all_native_source_geometry=check))
    )
    assert m.reused_support(a, "original", saved) == check
    for key, value in (
        ("source_rows_checked", 1),
        ("native_pixel_gate", 13),
        ("frame_range", [100, 108]),
        ("temporal_subsampling", True),
    ):
        broken = copy.deepcopy(saved)
        broken["methods"]["original"]["all_native_source_geometry"][key] = value
        with pytest.raises(ValueError):
            m.reused_support(a, "original", broken)


def test_bounded_checkpoint_contract_nested_migration_not_double_restore():
    migration, arrays, parent, meta = imported.import_fixture()
    m.interim.verify_bounded_import(migration, arrays, parent, meta)
    migration["imported_metadata"]["restore_count"] = 5
    with pytest.raises(ValueError):
        m.interim.verify_bounded_import(migration, arrays, parent, meta)


def test_classical_reuse_requires_all7_and_fixed11_not_selected_successes():
    score = dict(
        requested_anchors=11,
        requested_intervals=10,
        anchors=[dict(frame=f) for f in m.ANCHOR_FRAMES],
    )
    rows = [
        dict(
            name=n,
            native_frames=9960,
            fish_frame_denominator=29880,
            female=dict(
                strict_raw=copy.deepcopy(score),
                strict_with_original_acceptance=copy.deepcopy(score),
            ),
        )
        for n in ("original", "directional", *("c34_lambda_" + v for v in m.LAMBDAS))
    ]
    summary = dict(
        schema="directional_minute_all7_classical_comparison_v1",
        status="ok",
        native_interval=[m.START, m.STOP],
        native_frames=9960,
        requested_female_anchors=11,
        requested_female_intervals=10,
        all_seven_lambdas_retained=True,
        methods=rows,
    )
    assert len(m.classical_rows(summary, {}, {})) == 9
    for change in ("lambda", "anchor", "denominator"):
        broken = copy.deepcopy(summary)
        if change == "lambda":
            broken["methods"].pop()
        elif change == "anchor":
            broken["methods"][0]["female"]["strict_raw"]["anchors"].pop()
        else:
            broken["methods"][0]["fish_frame_denominator"] = 29700
        with pytest.raises(ValueError):
            m.classical_rows(broken, {}, {})


def test_terminal_semantic_adapter_calls_frozen_decoder_without_restore(monkeypatch):
    arrays = {"state_pose": np.zeros((6, 3, 3, 3)), **{str(k): np.array(k) for k in range(38)}}
    cfg = m.interim.TrackerConfig(device="cuda")
    from dataclasses import asdict

    context = dict(
        tracker_config=asdict(cfg),
        recovery_config=asdict(m.interim.NodeRecoveryConfig()),
        persistence_config=asdict(m.interim.PersistentNodeConfig()),
    )
    calls = []

    class Fake:
        def __init__(self, rig, seed, config, **kwargs):
            self.cfg = config
            self.initialize_count = self.processed_frames = 0

        def _context(self):
            return {**copy.deepcopy(context), "tracker_config": asdict(self.cfg)}

        def _decode_state(self, supplied, metadata):
            assert metadata["context"]["tracker_config"]["device"] == "cpu"
            assert supplied is arrays
            calls.append("decode")
            return (
                [[None] * 6] * (m.STOP - m.FIRST),
                [None] * 6,
                SimpleNamespace(frame_idx=m.STOP - 1),
                {},
            )

    metadata = dict(context=context)
    before = copy.deepcopy(metadata)
    result = m.semantic_decode(Fake, arrays, metadata, None, None)
    assert calls == ["decode"] and result["array_count"] == 39
    assert result["restore_calls"] == result["advance_calls"] == 0
    assert metadata == before
    metadata["context"]["unexpected"] = True
    # Independent actual context lacks an injected field.
    original_context = Fake._context

    def without_unknown(self):
        value = original_context(self)
        value.pop("unexpected", None)
        return value

    monkeypatch.setattr(Fake, "_context", without_unknown)
    with pytest.raises(ValueError, match="context differs"):
        m.semantic_decode(Fake, arrays, metadata, None, None)
