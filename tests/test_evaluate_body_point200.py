"""New seven-arm caller boundaries; synthetic arrays and completed small JSON only."""

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import body_point_transport as point_module
import evaluate_body_point200 as m

from test_body_point_transport import fixture as point_fixture


def run_fixture():
    binding = dict(
        schema=m.producer.STAGE,
        arm=m.producer.ARM,
        point_evidence_arm="ct_first",
        preparation_sha256=m.inputs.PREPARATION_SHA,
        point_sha256=m.previous.POINTS_SHA,
        samples_sha256=m.previous.SAMPLES_SHA,
        prior_tracking_sha256=m.producer.PREVIOUS_SHA,
        prior_evaluation_sha256=m.producer.EVALUATION_SHA,
        prior_six_evaluation_sha256=m.SIX_SHA,
        body_point_preparation_sha256=m.PACKET_SHA,
        cuda_qualification_sha256=m.producer.QUALIFICATION_SHA,
        body_point_plan_sha256=m.producer.PLAN_SHA,
        source_pins=copy.deepcopy(m.producer.SOURCE_PINS),
    )
    plan = dict(
        schema=m.producer.STAGE,
        native_interval=[m.FIRST, m.STOP],
        input_binding=binding,
        arm=m.producer.ARM,
        point_evidence_arm="ct_first",
        objective="base + 3*(body_point_owner + body_point_ray_gap)",
        gap_relative_coefficient=1.0,
        mask_coefficient=3.0,
        extra_proposals=0,
        maximum_pool=24,
        opt_steps=14,
        point_to_node_correspondence=False,
        D2_missing_fallback=False,
        first_release_horizons=[m.previous.horizon(n) for n in range(m.FIRST, m.STOP)],
    )
    summary = dict(
        schema=m.producer.STAGE,
        status="ok",
        native_interval=[m.FIRST, m.STOP],
        arm=dict(arm=m.producer.ARM, initialize_count=1, restore_count=0, frames=200),
        maximum_pool=24,
        steps=14,
        initialize_calls=1,
        restore_calls=0,
        advance_calls=200,
        query_refreshes=0,
        model_forwards=0,
        no_external_pose_reset=True,
        causal_initial_control_passed=True,
    )
    for k in (
        "prior_tracking_sha256",
        "prior_evaluation_sha256",
        "prior_six_evaluation_sha256",
        "body_point_preparation_sha256",
        "cuda_qualification_sha256",
    ):
        summary[k] = binding[k]
    return plan, summary, dict(passed=True, native=m.FIRST, prior_trajectory_used=False)


def test_completed_new_arm_metadata_is_read_only():
    args = run_fixture()
    before = copy.deepcopy(args)
    m.validate_run(*args)
    assert args == before


@pytest.mark.parametrize(
    "fault",
    [
        "packet",
        "qualification",
        "sources",
        "reseed",
        "short",
        "node_link",
        "fallback",
        "horizon",
        "objective",
        "future_t0",
        "summary_lineage",
    ],
)
def test_reject_changed_input_or_scientific_contract(fault):
    plan, summary, initial = run_fixture()
    if fault == "packet":
        plan["input_binding"]["body_point_preparation_sha256"] = "a" * 64
    elif fault == "qualification":
        plan["input_binding"]["cuda_qualification_sha256"] = "a" * 64
    elif fault == "sources":
        plan["input_binding"]["source_pins"]["work/three_body/body_point_tracker.py"] = "a" * 64
    elif fault == "reseed":
        summary["restore_calls"] = 1
    elif fault == "short":
        summary["advance_calls"] = 199
    elif fault == "node_link":
        plan["point_to_node_correspondence"] = True
    elif fault == "fallback":
        plan["D2_missing_fallback"] = True
    elif fault == "horizon":
        plan["first_release_horizons"][0] += 8
    elif fault == "objective":
        plan["objective"] = "whole masks"
    elif fault == "future_t0":
        initial["prior_trajectory_used"] = True
    else:
        summary["prior_six_evaluation_sha256"] = "a" * 64
    with pytest.raises(ValueError):
        m.validate_run(plan, summary, initial)


@pytest.mark.parametrize("fault", ["draft", "missing_pin", "wrong_packet"])
def test_cli_pending_or_wrong_source_fails_before_output_or_inputs(monkeypatch, tmp_path, fault):
    output = tmp_path / "must_not_exist"
    packet_sha = m.PACKET_SHA
    if fault == "draft":
        monkeypatch.setattr(m, "EXECUTION_READY", False)
    elif fault == "missing_pin":
        monkeypatch.setattr(m, "PRODUCER_SHA", None)
    else:
        packet_sha = "a" * 64
    monkeypatch.setattr(
        m.base, "verified", lambda *a: pytest.fail("inputs opened before pin guard")
    )
    with pytest.raises(ValueError, match="pins required"):
        m.run(tmp_path / "tracking", "b" * 64, tmp_path / "packets", packet_sha, output)
    assert not output.exists()


def packet_metadata():
    """Existing successful SMALL JSON only; no NPZ, model, pixel or full-output hashing."""
    root = m.producer.PACKETS
    return tuple(
        json.loads((root / k).read_text(encoding="utf-8"))
        for k in (
            "plan.json",
            "summary.json",
            "preparation.provenance.json",
        )
    )


def test_actual_completed_packet_small_json_contract_no_payload(monkeypatch):
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no actual array read"))
    args = packet_metadata()
    before = copy.deepcopy(args)
    m.validate_packet_preparation(*args)
    assert args == before
    assert args[1]["packet_count"] == 400
    assert args[0]["all_initial_anchor_ids"] == [0, 1]


@pytest.mark.parametrize("fault", ["wrong_seal_name", "unused_mass", "t0_fit", "no_readback"])
def test_packet_small_json_lineage_and_in_sample_domain_refuse_tamper(fault):
    plan, summary, seal = packet_metadata()
    if fault == "wrong_seal_name":
        path = m.previous.SAMPLES / "samples.provenance.json"
        seal["inputs"][str(path.with_name("preparation.provenance.json"))] = seal["inputs"].pop(
            str(path)
        )
    elif fault == "unused_mass":
        plan["point_mass_domain"] = "unused256-derived mass"
    elif fault == "t0_fit":
        plan["t0_packet_used_in_tracking"] = True
    else:
        summary["all400_query_ray_readbacks_exact"] = False
    with pytest.raises(ValueError):
        m.validate_packet_preparation(plan, summary, seal)


def image_arrays():
    rng = np.random.default_rng(81)
    scores = {}
    for key, shape in dict(
        rendered_union=(7, 3, 4),
        rendered_ownership=(7, 3, 4, 3),
        rendered_q=(7, 3, 4, 3),
        union_per_view=(7, 3),
        union_loss=(7,),
        target_mass=(2, 2, 3, 3),
    ).items():
        scores[key] = rng.random(shape)
    for prefix in ("owner", "gap", "combined"):
        for suffix, shape in (
            ("per_body_view", (2, 7, 2, 3, 3)),
            ("per_view", (2, 7, 2, 3)),
            ("loss", (2, 7, 2)),
        ):
            scores[prefix + "_" + suffix] = rng.random(shape)
    scores.update(
        positive_target_mass=scores["target_mass"] > 0,
        ray_valid=np.ones((3, 4), bool),
        anchor_ids=np.array([0, 1], np.int64),
        frame_idx=np.array(m.FIRST, np.int64),
    )
    old = {}
    for k, v in scores.items():
        if k.startswith(("owner_", "gap_", "combined_")):
            old[k] = v[:, :6].copy()
        elif k.startswith(("rendered_", "union_")):
            old[k] = v[:6].copy()
        else:
            old[k] = v.copy()
    old.update(
        renderer_policy=np.asarray("original_dynamic_all6"),
        fixed_radial_sigma_mm=np.empty(0),
        reference_pose_xyz=np.empty(0),
    )
    return scores, old


def test_all19_old_six_axes_exact_and_new_seventh_is_free():
    scores, old = image_arrays()
    m.previous_image_control(scores, old)
    scores["union_loss"][6] += 100
    scores["gap_loss"][:, 6] += 100
    m.previous_image_control(scores, old)
    scores["gap_loss"][1, 5, 1] += 1e-12
    with pytest.raises(ValueError, match="six-arm CPU image control"):
        m.previous_image_control(scores, old)


@pytest.mark.parametrize("fault", ["dtype", "shared_query", "missing_key", "rendered_last_old"])
def test_old_common_image_exactness_is_not_approximate(fault):
    scores, old = image_arrays()
    if fault == "dtype":
        scores["owner_loss"] = scores["owner_loss"].astype(np.float32)
    elif fault == "shared_query":
        scores["ray_valid"][2, -1] = False
    elif fault == "missing_key":
        del old["combined_loss"]
    else:
        scores["rendered_q"][5, 2, -1, 2] = np.nextafter(scores["rendered_q"][5, 2, -1, 2], np.inf)
    with pytest.raises(ValueError):
        m.previous_image_control(scores, old)


def test_new_state_requires_external_plan_before_sidecar_and_keeps_original_metadata(monkeypatch):
    plan = {"input_binding": {"source": "expected"}}
    meta = {"context": {"mask_transport": {"input_binding": {"source": "unrelated"}}}}
    core, side, side_meta = {"synthetic_core": np.zeros(1)}, {}, {}
    seen = []

    def arrays(root, members, relative):
        seen.append(relative)
        return side if "side_arrays" in relative else core

    monkeypatch.setattr(m.previous, "read_arrays", arrays)
    monkeypatch.setattr(
        m.base, "bound_json", lambda r, members, p: side_meta if "side_metadata" in p else meta
    )
    common = dict(rig=object(), initial=np.zeros((3, 3, 3)), original_context={"device": "cuda"})
    points = SimpleNamespace(group_counts=(3, 3, 4), query_sha256=("a" * 64,) * 3)
    audit = SimpleNamespace(
        validate_state=lambda *a, **k: pytest.fail("unbound state reached audit")
    )
    with pytest.raises(ValueError, match="binding differs"):
        m.validate_new_state(Path("unused"), {}, common, points, plan, audit)
    assert len(seen) == 1 and "side" not in seen[0]
    meta["context"]["mask_transport"]["input_binding"] = copy.deepcopy(plan["input_binding"])
    before = copy.deepcopy(meta)

    def verified(a, b, c, d, **kw):
        assert a is core and b is meta and c is side and d is side_meta
        assert kw["original_context"] is common["original_context"]
        assert kw["expected_frames"] == 200 and kw["coefficient"] == 3
        assert kw["input_binding"] is plan["input_binding"]
        return a, {"validated": True}, {"metadata_rewritten": False}

    audit.validate_state = verified
    assert m.validate_new_state(Path("unused"), {}, common, points, plan, audit)[2] == {
        "metadata_rewritten": False
    }
    assert meta == before


def frame_fixture():
    sample = dict(
        fit_pixels=np.zeros((3, 2, 2)),
        fit_target=np.ones((3, 2), np.float32),
        fit_weights=np.full((3, 2), 0.5, np.float32),
        fit_foreground=np.ones((3, 2), bool),
    )
    records = [{k: np.array(i, np.int64) for k in m.previous.RECORD_FIELDS} for i in range(2)]
    records[0]["proposal_kind"] = np.asarray("synthetic initial")
    records[1]["proposal_kind"] = np.asarray("synthetic second")
    fit = dict(
        **{k: v.copy() for k, v in sample.items()},
        selected=np.array([1, 0]),
        cost=np.array([1.0, 2.0, 3.0]),
        owner={"all_pool": np.array([4.0, 5.0, 6.0])},
    )
    tree = dict(fit=fit, records=records)
    content = dict(fits=[copy.deepcopy(fit)])
    core = {"history_offsets": np.array([0, 2], np.int64)}
    core.update(
        {"history_" + k: np.stack([r[k] for r in records]) for k in m.previous.RECORD_FIELDS}
    )
    return tree, content, core, sample


@pytest.mark.parametrize(
    "fault", [None, "unretained_fit", "second_record", "fit_target", "missing_record"]
)
def test_full_fit_all19_and_actual_fit4_control(fault):
    tree, content, core, sample = frame_fixture()
    if fault == "unretained_fit":
        tree["fit"]["owner"]["all_pool"][2] += 1
    elif fault == "second_record":
        tree["records"][1]["parent"] += 1
    elif fault == "fit_target":
        sample["fit_target"][0, 0] = 0
    elif fault == "missing_record":
        tree["records"].pop()
    if fault:
        with pytest.raises(ValueError):
            m.frame_control(tree, content, core, 0, sample)
    else:
        m.frame_control(tree, content, core, 0, sample)


def packet_pair(native):
    args = point_fixture(second_anchor=True)
    args["anchors"] = [dataclasses.replace(a, initial_native=m.FIRST) for a in args["anchors"]]
    args["point_evidence"] = [
        dataclasses.replace(
            e,
            vote=dataclasses.replace(
                e.vote, native=native, release_horizon=m.previous.horizon(native)
            ),
        )
        for e in args["point_evidence"]
    ]
    packet = point_module.build_packet(**args)
    sample = dict(
        fit_pixels=args["fit_pixels"],
        fit_target=args["fit_foreground"],
        fit_weights=args["fit_weights"],
    )
    frame = SimpleNamespace(masks=dict(zip(m.transport.VIEWS, args["masks_by_view"], strict=True)))
    return packet, sample, frame, tuple(args["anchors"]), args["rig"]


@pytest.mark.parametrize("offset", [0, 1])
def test_real_canonical_packet_all_anchors_and_t0_fit_exclusion(monkeypatch, offset):
    native = m.FIRST + offset
    packet, sample, frame, anchors, rig = packet_pair(native)
    raw = dataclasses.asdict(packet)
    monkeypatch.setattr(m.old_evaluation, "load_tree", lambda *a: raw)
    fit = {"packet": None if offset == 0 else copy.deepcopy(raw)}
    result = m.checked_packets(
        Path("unused"),
        {},
        native,
        sample,
        frame,
        anchors,
        rig,
        point_module,
        fit,
        query_sha256=("a" * 64,) * 3,
    )
    assert len(result) == 2 and all(len(p.anchor_ids) == 2 for p in result)
    if offset == 0:
        fit["packet"] = raw
    else:
        fit["packet"]["point_weight"][1, 2, 1, 0] += 1
    with pytest.raises(ValueError):
        m.checked_packets(
            Path("unused"),
            {},
            native,
            sample,
            frame,
            anchors,
            rig,
            point_module,
            fit,
            query_sha256=("a" * 64,) * 3,
        )


@pytest.mark.parametrize(
    "fault", ["query", "sample_target", "current_mask", "anchor_bank", "release"]
)
def test_prepared_packet_cannot_substitute_fixed_or_current_evidence(monkeypatch, fault):
    native = m.FIRST + 1
    packet, sample, frame, anchors, rig = packet_pair(native)
    query = ("a" * 64,) * 3
    if fault == "query":
        query = ("b" * 64,) * 3
    elif fault == "sample_target":
        sample["fit_target"] = sample["fit_target"].astype(np.float64)
    elif fault == "current_mask":
        frame.masks[m.transport.VIEWS[2]][0, 3, 0] = True
    elif fault == "anchor_bank":
        anchors = anchors[:1]
    else:
        packet = dataclasses.replace(
            packet, release_horizons=np.full(3, m.previous.horizon(native) + 8, np.int64)
        )
        # Isolate external horizon gate; canonical decoder already tests its own internal fields.
        monkeypatch.setattr(point_module, "validate_packet", lambda *a, **k: packet)
    monkeypatch.setattr(m.old_evaluation, "load_tree", lambda *a: dataclasses.asdict(packet))
    with pytest.raises(ValueError):
        m.checked_packets(
            Path("unused"),
            {},
            native,
            sample,
            frame,
            anchors,
            rig,
            point_module,
            {"packet": dataclasses.asdict(packet)},
            query_sha256=query,
        )


def test_paired_baselines_and_native_support_rows_are_preserved():
    rows = [
        dict(
            union_loss=np.arange(7, dtype=float),
            union_per_view=np.tile(np.arange(7.0)[:, None], (1, 3)),
        )
        for _ in range(2)
    ]
    rows[1]["union_loss"][-1] = -1
    paired = m.paired_scores(rows)
    assert paired["original"]["per_native_delta"].tolist() == [6, -1]
    assert paired["ct_gap"]["per_native_delta"].tolist() == [3, -4]
    assert paired["original"]["better"] == paired["original"]["worse"] == 1
    old = np.ones((2, 3, 3), bool)
    old[0, 2, 2] = False
    new = old.copy()
    new[0, 2, 2], new[1, 0, 0] = True, False
    result = m.support_pair(
        {"original": {"observed": old}, m.producer.ARM: {"observed": new}}, "original"
    )
    assert result["gains"] == result["losses"] == 1
    assert result["gain_rows"] == [dict(native=m.FIRST, body=2, node=2)]
    assert result["loss_rows"] == [dict(native=m.FIRST + 1, body=0, node=0)]
