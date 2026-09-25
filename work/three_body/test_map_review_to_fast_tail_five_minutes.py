"""Synthetic caller integration; no actual human labels, video, GPU or tracking result."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import map_review_to_fast_tail_five_minutes as m
import numpy as np
import pytest
from test_import_long_review import Camera, base_document, encode


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else encode(value))
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signatures(arrays):
    return {key: (v.dtype.str, v.shape, v.tobytes()) for key, v in arrays.items()}


def geometry(frames=(74945, 75163, 75245)):
    document, manifest = base_document()
    first = 64740 if frames[0] < 66400 else 74700
    last = max(first + 1660, frames[-1] + 1)
    manifest["frames"] = list(range(first, last))
    document["source_frame_range"] = [first, last]
    document["source_manifest_sha256"] = hashlib.sha256(encode(manifest)).hexdigest()
    document["review_clip_range"] = [frames[0] - 1, frames[-1] + 1]
    for anchor, frame in zip(document["anchors"], frames, strict=True):
        anchor["frame"] = frame
    parsed = m.review.parse_review(
        encode(document), encode(manifest), clip_range=document["review_clip_range"]
    )
    pose = np.zeros((3, 3, 3, 3), np.float64)
    for fish in range(3):
        pose[:, fish, :, :] = [30.0 + 30 * fish, 100.0, 20.0]
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], pose.shape).copy().astype(np.int64)
    arrays = dict(
        frame_idx=parsed.anchor_frames.copy(),
        pose_xyz=pose,
        sources=sources,
        observed=np.ones((3, 3, 3), bool),
        identity_review=np.zeros((3, 3), bool),
    )
    points = {
        int(frame): {view: pose[t, :, :, :2].copy() for view in m.review.VIEWS}
        for t, frame in enumerate(parsed.anchor_frames)
    }
    return SimpleNamespace(
        parsed=parsed,
        arrays=arrays,
        points=points,
        rig=SimpleNamespace(camera=lambda view: Camera()),
    )


def documents():
    references = {
        arm: dict(path=f"synthetic/{arm}.npz", sha256=c * 64)
        for arm, c in (("original", "a"), ("fast_tail", "b"))
    }
    summary = dict(
        schema=m.comparison.STAGE,
        status="ok",
        horizon_stop=m.STOP,
        native_interval=[m.START, m.STOP],
        native_frames=m.STOP - m.START,
        whole_five_minute_process_complete=True,
        final_tracking_seal_verified=True,
        same_parent_and_horizon=True,
        candidate_all21_and_source_arrays_unchanged=True,
        review_modified=False,
        all_three_ID_truth=False,
        external_pose_resets=0,
        candidate_references=references,
    )
    plan = dict(
        schema=m.comparison.STAGE,
        horizon_stop=m.STOP,
        candidate_references=references,
        milestone_sha256="c" * 64,
        final_tracking_sha256="d" * 64,
    )
    return plan, summary


@pytest.mark.parametrize(
    "fault",
    [None, "partial", "two_minutes", "not_final", "bool_count", "modified", "reference", "bad_sha"],
)
def test_only_complete_same_horizon_comparison_enters_mapping(fault):
    plan, summary = documents()
    if fault == "partial":
        summary["whole_five_minute_process_complete"] = False
    elif fault == "two_minutes":
        summary["horizon_stop"] = 86320
    elif fault == "not_final":
        summary["final_tracking_seal_verified"] = False
    elif fault == "bool_count":
        summary["external_pose_resets"] = False
    elif fault == "modified":
        summary["review_modified"] = True
    elif fault == "reference":
        plan["candidate_references"] = copy.deepcopy(plan["candidate_references"])
        plan["candidate_references"]["fast_tail"]["sha256"] = "e" * 64
    elif fault == "bad_sha":
        plan["final_tracking_sha256"] = "unsealed"
    if fault is None:
        m.validate_comparison(plan, summary)
    else:
        with pytest.raises(ValueError):
            m.validate_comparison(plan, summary)


@pytest.mark.parametrize(
    "frames,clip,inside,intersection",
    [
        ([65931, 66151, 66217], [65859, 66223], [False] * 3, None),
        ([66399, 66400, 66401], [66398, 66402], [False, True, True], [66400, 66402]),
        ([74945, 75163, 75245], [74867, 75246], [True] * 3, [74867, 75246]),
    ],
)
def test_review_scope_counts_only_actual_scientific_endpoints(frames, clip, inside, intersection):
    result = m.scope_record(np.array(frames, np.int64), clip)
    assert result["anchors_inside_scientific_interval"] == inside
    assert result["scientific_anchor_times"] == sum(inside)
    assert result["requested_scientific_fish_endpoints"] == 3 * sum(inside)
    assert result["displayed_clip_intersection"] == intersection
    assert result["interval_interior_certified"] is False
    assert result["independently_verified_dense_frames"] == 0
    assert result["pre_start_anchors_counted_as_five_minute_truth"] is False


@pytest.mark.parametrize(
    "frames,clip",
    [
        ([49800, 49800, 49802], [49800, 49803]),
        ([49799, 49800, 49801], [49799, 49802]),
        ([116197, 116198, 116200], [116197, 116201]),
        ([74945, 75163, 75245], [74946, 75246]),
        ([74945.0, 75163.0, 75245.0], [74867, 75246]),
    ],
)
def test_invalid_anchor_history_or_clip_is_rejected(frames, clip):
    with pytest.raises(ValueError):
        m.scope_record(np.asarray(frames), clip)


def test_document_registration_allows_only_explicit_external_json(tmp_path, monkeypatch):
    repo = tmp_path / "synthetic_repo"
    local = write(repo / "pack.json", b"synthetic local pack")
    explicit = write(tmp_path / "explicit_synthetic_review.json", b"synthetic external answer")
    other = write(tmp_path / "unrelated.json", b"not an authorized external input")
    monkeypatch.setattr(m, "REPO", repo)
    document = SimpleNamespace(hashes={local: sha(local), explicit: sha(explicit)})
    evidence = m.comparison.producer.original.phase.Digests()
    result = m.register_document(document, explicit, evidence)
    assert result == {str(explicit): sha(explicit)}
    assert evidence.paths == {str(local): sha(local)}
    document.hashes[other] = sha(other)
    with pytest.raises(ValueError, match="explicitly provided"):
        m.register_document(document, explicit, evidence)


def score(f):
    return m.score_rows(
        f.parsed, f.arrays, f.rig, f.points, "a" * 64, {"synthetic_calibration": "b" * 64}, "c" * 64
    )


def test_real_matcher_keeps_latent_competitor_and_original_review():
    f = geometry()
    f.arrays["pose_xyz"][:, 1, 0] = f.arrays["pose_xyz"][:, 0, 0]
    f.arrays["sources"][:, 1, :, 0] = -1
    f.arrays["observed"][:, 1, 0] = False
    f.arrays["identity_review"][1, 2] = True
    before = signatures(f.arrays)
    result = score(f)
    assert result["endpoints"][0]["spatial_match"]["eligible_slots"] == [0, 1]
    assert result["endpoints"][0]["raw_match"]["slot"] is None
    assert result["endpoints"][5]["raw_match"]["slot"] == 2
    assert result["endpoints"][5]["accepted_match"]["slot"] is None
    assert len(result["endpoints"]) == 9 and len(result["links"]) == 6
    assert all(row["interior_identity_certified"] is False for row in result["links"])
    assert signatures(f.arrays) == before


def test_real_matcher_preserves_unknown_sex_and_unconfirmed_fish_without_elimination():
    f = geometry()
    document = copy.deepcopy(f.parsed.review)
    for anchor in document["anchors"]:
        anchor["animals"]["C"] = dict(visibility="occluded", points={})
    document["continuity"]["C"].update(entry_to_after="ambiguous", after_to_exit="not_visible")
    f.parsed = m.review.parse_review(
        encode(document), f.parsed.manifest_bytes, clip_range=document["review_clip_range"]
    )
    result = score(f)
    assert all(result["endpoints"][i]["raw_match"]["slot"] is None for i in (2, 5, 8))
    assert result["biological_naming_preserved_not_applied"]["C"]["sex"] == "unknown"
    assert [row["human_status"] for row in result["links"][-2:]] == ["ambiguous", "not_visible"]


@pytest.fixture
def integration(tmp_path, monkeypatch):
    f = geometry()
    f.root = tmp_path
    f.review_path = write(tmp_path / "synthetic_human_document.json", f.parsed.review_bytes)
    f.pack = tmp_path / "pack"
    f.pack.mkdir()
    f.document_evidence = m.legacy.raw.Evidence()
    f.document_evidence.bind(f.review_path)
    f.plan, f.summary = documents()
    f.plan["shared_SL_calibration_sha256"] = {"synthetic_calibration": "b" * 64}
    f.context = dict(
        plan=f.plan,
        summary=f.summary,
        original=dict(bundle="synthetic bundle"),
        new=dict(metadata=dict(context="synthetic camera context")),
        seal=dict(
            inputs={row["path"]: row["sha256"] for row in f.plan["candidate_references"].values()}
        ),
    )
    f.candidates = {"fast_tail": copy.deepcopy(f.arrays), "original": copy.deepcopy(f.arrays)}
    for key in ("pose_xyz", "sources", "observed", "identity_review"):
        f.candidates["original"][key] = f.candidates["original"][key][:, [2, 0, 1]].copy()
    f.candidates["fast_tail"]["identity_review"][1, 0] = True
    f.before = {arm: signatures(rows) for arm, rows in f.candidates.items()}
    monkeypatch.setattr(m, "EXECUTION_READY", True)
    monkeypatch.setattr(m, "source_code", lambda: {})

    def context(*args):
        m.validate_comparison(f.plan, f.summary)
        return f.context

    monkeypatch.setattr(m, "load_comparison", context)
    monkeypatch.setattr(
        m.review,
        "load_document",
        lambda *args: (f.parsed, {"synthetic": True}, f.document_evidence),
    )
    monkeypatch.setattr(
        m.source_binding,
        "verify_source_binding",
        lambda *args: dict(
            shared=f.plan["shared_SL_calibration_sha256"],
            videos={},
            manifest={},
            displayed=[],
            proof=dict(
                milestone_candidate_sha256=f.plan["candidate_references"]["fast_tail"]["sha256"]
            ),
        ),
    )
    monkeypatch.setattr(
        m.comparison.previous,
        "load_candidate",
        lambda *args: (f.candidates["original"], f.plan["candidate_references"]["original"]),
    )
    monkeypatch.setattr(
        m.comparison,
        "load_milestone_candidate",
        lambda *args: (f.candidates["fast_tail"], f.plan["candidate_references"]["fast_tail"]),
    )
    monkeypatch.setattr(
        m.legacy,
        "load_anchor_geometry",
        lambda *args: (f.rig, f.points, f.plan["shared_SL_calibration_sha256"]),
    )
    monkeypatch.setattr(
        m.legacy,
        "decode_anchor_images",
        lambda *args: [
            dict(synthetic=True, frame=int(frame), view=view)
            for frame in f.parsed.anchor_frames
            for view in m.review.VIEWS
        ],
    )
    f.args = SimpleNamespace(
        execute=True,
        output=tmp_path / "synthetic_mapping",
        comparison_root=tmp_path,
        comparison_sha="e" * 64,
        review_pack=f.pack,
        review=f.review_path,
        identity_root=None,
        identity_sha=None,
    )
    return f


def strict_json(path):
    def invalid(value):
        pytest.fail("non-JSON numeric constant escaped NPZ: " + value)

    return json.loads(path.read_text(encoding="utf8"), parse_constant=invalid)


def check_outputs(f, expected_arms):
    output = f.args.output
    summary = strict_json(output / "summary.json")
    assert summary["arms"] == expected_arms
    for key in (
        "all_three_ID_truth",
        "biological_swap_zero_verified",
        "human_labels_created",
        "human_annotations_modified",
        "interval_interior_certified",
        "raw_trajectory_modified",
        "production_correction_applied",
        "GPU_used",
    ):
        assert summary[key] is False
    assert summary["native_anchor_BGR_images_redecoded"] == 9
    assert summary["independently_verified_dense_frames"] == 0
    assert (output / "original_review.json").read_bytes() == f.parsed.review_bytes
    assert (output / "original_source_manifest.json").read_bytes() == f.parsed.manifest_bytes
    for arm in ("original", "fast_tail"):
        with np.load(output / f"anchor_rows/{arm}.npz", allow_pickle=False) as z:
            assert set(z.files) == set(f.candidates[arm])
            assert signatures(dict(z)) == f.before[arm]
        assert signatures(f.candidates[arm]) == f.before[arm]
    original = strict_json(output / "mapping/original.json")
    tail = strict_json(output / "mapping/fast_tail.json")
    assert [row["raw_match"]["slot"] for row in original["endpoints"]] == [1, 2, 0] * 3
    assert [row["raw_match"]["slot"] for row in tail["endpoints"]] == [0, 1, 2] * 3
    assert tail["endpoints"][3]["accepted_match"]["slot"] is None
    seal = strict_json(output / "mapping.provenance.json")
    assert seal["status"] == "ok"
    for row in seal["outputs"].values():
        path = Path(row["path"])
        assert row["sha256"] == sha(path) and row["size_bytes"] == path.stat().st_size
    return summary


def test_full_synthetic_caller_without_identity_preserves_raw_results(integration, monkeypatch):
    f = integration
    monkeypatch.setattr(
        m.identity_binding,
        "load_identity_anchors",
        lambda *a, **k: pytest.fail("optional identity was loaded"),
    )
    m.run(f.args)
    check_outputs(f, ["original", "fast_tail"])
    result = strict_json(f.args.output / "identity_binding.json")
    assert result["available"] is False


def test_partial_comparison_fails_before_opening_human_or_source(integration, monkeypatch):
    f = integration
    f.summary["whole_five_minute_process_complete"] = False
    monkeypatch.setattr(
        m.review, "load_document", lambda *a: pytest.fail("human opened before gate")
    )
    monkeypatch.setattr(
        m.source_binding,
        "verify_source_binding",
        lambda *a: pytest.fail("source opened before gate"),
    )
    with pytest.raises(ValueError, match="complete five-minute"):
        m.run(f.args)
    seal = strict_json(f.args.output / "mapping.provenance.json")
    assert seal["status"] == "failed" and not (f.args.output / "summary.json").exists()


def test_optional_identity_unavailable_is_not_extrapolated(integration, monkeypatch):
    f = integration
    older = geometry((65931, 66151, 66217))
    f.parsed, f.rig, f.points = older.parsed, older.rig, older.points
    f.candidates = {
        "original": copy.deepcopy(older.arrays),
        "fast_tail": copy.deepcopy(older.arrays),
    }
    for key in ("pose_xyz", "sources", "observed", "identity_review"):
        f.candidates["original"][key] = f.candidates["original"][key][:, [2, 0, 1]].copy()
    f.candidates["fast_tail"]["identity_review"][1, 0] = True
    f.before = {arm: signatures(rows) for arm, rows in f.candidates.items()}
    write(f.review_path, f.parsed.review_bytes)
    f.document_evidence = m.legacy.raw.Evidence()
    f.document_evidence.bind(f.review_path)
    f.args.identity_root, f.args.identity_sha = f.root / "synthetic_ID", "f" * 64
    monkeypatch.setattr(
        m.identity_binding,
        "load_identity_anchors",
        lambda *a, **k: dict(
            available=False,
            reason="anchors_outside_identity_interval",
            outside_frames=[65931, 66151, 66217],
            permutation_by_id=None,
            ambiguity=None,
            binding={"synthetic": True},
        ),
    )
    monkeypatch.setattr(
        m.identity_binding,
        "reorder_anchor_rows",
        lambda *a: pytest.fail("unavailable identity was extrapolated"),
    )
    m.run(f.args)
    summary = check_outputs(f, ["original", "fast_tail"])
    assert summary["scope"]["scientific_anchor_times"] == 0
    result = strict_json(f.args.output / "identity_binding.json")
    assert result["available"] is False and result["outside_frames"] == [65931, 66151, 66217]
    assert not (f.args.output / "anchor_rows/fast_tail_automatic_ID.npz").exists()


def test_synthetic_identity_arrays_and_infinite_gap_survive_exactly(integration, monkeypatch):
    f = integration
    f.args.identity_root, f.args.identity_sha = f.root / "synthetic_ID", "f" * 64
    result = dict(
        available=True,
        reason=None,
        outside_frames=[],
        permutation_by_id=np.array([[2, 0, 1]] * 3, np.int64),
        ambiguity=dict(
            state_alternative_gap=np.array([np.inf, 0.0, 3.0], np.float64),
            identity_alternative_gap=np.array([[np.inf, 1.0, 2.0]] * 3, np.float64),
            state_alternative_status=np.array(["no_feasible_alternative", "tie", "finite"]),
            terminal_unsettled_gap=np.array([[False, True]] * 3),
        ),
        binding=dict(synthetic=True, all_three_ID_truth=False, identity_provenance_sha256="f" * 64),
    )
    before_perm = signatures({"permutation_by_id": result["permutation_by_id"]})
    before_ambiguity = signatures(result["ambiguity"])
    monkeypatch.setattr(m.identity_binding, "load_identity_anchors", lambda *a, **k: result)
    m.run(f.args)
    check_outputs(f, ["original", "fast_tail", "fast_tail_automatic_ID"])
    assert signatures({"permutation_by_id": result["permutation_by_id"]}) == before_perm
    assert signatures(result["ambiguity"]) == before_ambiguity
    document = strict_json(f.args.output / "identity_binding.json")
    assert document["permutation_by_id"]["archive_member"] == "permutation_by_id"
    with np.load(f.args.output / "identity_anchor_arrays.npz", allow_pickle=False) as z:
        expected = {
            "permutation_by_id": result["permutation_by_id"],
            **{"ambiguity__" + k: v for k, v in result["ambiguity"].items()},
        }
        assert set(z.files) == set(expected)
        assert signatures(dict(z)) == signatures(expected)
    with np.load(f.args.output / "anchor_rows/fast_tail_automatic_ID.npz", allow_pickle=False) as z:
        expected = m.identity_binding.reorder_anchor_rows(
            f.candidates["fast_tail"], result["permutation_by_id"]
        )
        assert signatures(dict(z)) == signatures(expected)
    auto = strict_json(f.args.output / "mapping/fast_tail_automatic_ID.json")
    assert [row["raw_match"]["slot"] for row in auto["endpoints"]] == [1, 2, 0] * 3
    assert auto["endpoints"][3]["accepted_match"]["slot"] is None
    meaning = auto["slot_index_space"]
    assert meaning["kind"] == "automatic_initial_slot_ID"
    assert meaning["legacy_slot_fields_are_raw_candidate_slots"] is False
    assert meaning["identity_provenance_sha256"] == "f" * 64
    assert meaning["identity_anchor_arrays"] == document["anchor_arrays_archive"]
    assert meaning["biological_identity_assigned"] is False


@pytest.mark.parametrize("fault", ["different_reference", "missing_anchor"])
def test_selected_candidates_reject_mismatched_reference_or_anchor(integration, fault):
    f = integration
    frames = f.parsed.anchor_frames.copy()
    if fault == "different_reference":
        f.plan["candidate_references"]["fast_tail"]["sha256"] = "0" * 64
    else:
        frames[-1] += 1
    with pytest.raises(ValueError):
        m.selected_candidates(f.context, frames, m.comparison.producer.original.phase.Digests())


def test_selected_five_fields_are_owned_copies(integration):
    f = integration
    selected, _ = m.selected_candidates(
        f.context, f.parsed.anchor_frames, m.comparison.producer.original.phase.Digests()
    )
    for arm, rows in selected.items():
        assert set(rows) == set(m.legacy.ANCHOR_KEYS)
        assert signatures(rows) == f.before[arm]
        assert all(not np.shares_memory(rows[key], f.candidates[arm][key]) for key in rows)


@pytest.mark.parametrize("fault", ["partial_permutation", "partial_ambiguity", "object_array"])
def test_identity_serialization_rejects_partial_or_object_payloads(tmp_path, fault):
    output = m.proof.Output(tmp_path / "synthetic_identity_serialization")
    result = dict(available=False, reason="outside", permutation_by_id=None, ambiguity=None)
    if fault == "partial_permutation":
        result["permutation_by_id"] = np.array([[0, 1, 2]], np.int64)
    elif fault == "partial_ambiguity":
        result["ambiguity"] = {"gap": np.array([np.inf])}
    else:
        result.update(
            available=True,
            permutation_by_id=np.array([[0, 1, 2]] * 3, np.int64),
            ambiguity={"untrusted": np.array([{"not": "numeric"}], dtype=object)},
        )
    with pytest.raises(ValueError):
        m.serialize_identity_result(result, output)


def test_automatic_slot_description_requires_available_identity():
    reference = dict(path="synthetic.npz", sha256="a" * 64)
    for arm in ("original", "fast_tail"):
        meaning = m.describe_slot_space(arm, reference)
        assert meaning["kind"] == "raw_candidate_slot"
        assert meaning["legacy_slot_fields_are_raw_candidate_slots"] is True
        assert meaning["biological_identity_assigned"] is False
    with pytest.raises(ValueError, match="saved identity mapping"):
        m.describe_slot_space("fast_tail_automatic_ID", reference, {"available": False})


@pytest.mark.parametrize("fault", ["not_ready", "not_execute", "identity_half_pair"])
def test_execution_or_identity_pair_gate_precedes_output_creation(integration, monkeypatch, fault):
    f = integration
    if fault == "not_ready":
        monkeypatch.setattr(m, "EXECUTION_READY", False)
    elif fault == "not_execute":
        f.args.execute = False
    else:
        f.args.identity_sha = "f" * 64
    with pytest.raises(ValueError):
        m.run(f.args)
    assert not f.args.output.exists()
