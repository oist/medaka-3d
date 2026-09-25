"""Synthetic-only endpoint/lineage/codec tests; never read actual human or fish data."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import compare_fast_tail_transition_review as module
import numpy as np
import pytest
import review_truth


@pytest.fixture(autouse=True)
def isolated_work(monkeypatch, tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    monkeypatch.setattr(module, "WORK", root)
    # Every invocation test is additionally guarded by failing input-entry spies.
    return root


def encoded(value):
    return json.dumps(value, allow_nan=False).encode()


@pytest.mark.parametrize("enabled", [False, True])
def test_future_helper_activation_preserves_semantics_and_records_actual_bytes(enabled):
    reviewed = b'"""Synthetic only."""\nEXECUTION_READY = False\nVALUE = 7\n'
    actual = reviewed.replace(b"= False", b"= True") if enabled else reviewed
    expected = hashlib.sha256(reviewed).hexdigest()
    assert module.flag_source_digest(actual, expected) == hashlib.sha256(actual).hexdigest()
    with pytest.raises(ValueError, match="beyond declared"):
        module.flag_source_digest(actual.replace(b"VALUE = 7", b"VALUE = 8"), expected)


def test_flag_validation_rejects_missing_or_duplicate_switches():
    data = b"EXECUTION_READY = False\n"
    expected = hashlib.sha256(data).hexdigest()
    for invalid in (b"", data + data, data + b"EXECUTION_READY = True\n"):
        with pytest.raises(ValueError, match="one execution flag"):
            module.flag_source_digest(invalid, expected)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_flag_examples_inside_helper_are_not_rewritten(newline):
    lines = [
        b"EXECUTION_READY = False",
        b'EXAMPLES = (b"EXECUTION_READY = False", b"EXECUTION_READY = True")',
    ]
    reviewed = newline.join(lines) + newline
    actual = reviewed.replace(b"EXECUTION_READY = False", b"EXECUTION_READY = True", 1)
    assert (
        module.flag_source_digest(actual, hashlib.sha256(reviewed).hexdigest())
        == hashlib.sha256(actual).hexdigest()
    )


def parsed_fixture(edit=None):
    manifest = dict(
        experiment="20251225_K_1_2_FCMC",
        frames=list(range(89640, 91300)),
        fps=166,
        image_shape_hw=[1200, 1200],
    )
    data = encoded(manifest)
    answer = review_truth.make_template(
        manifest["frames"], hashlib.sha256(data).hexdigest(), [90121, 90220, 90380]
    )
    answer.update(
        status="human_draft_requires_adjudication",
        reviewer="SYNTHETIC ONLY",
        reviewed_utc="2026-09-11T00:00:00Z",
        elapsed_review_seconds=3.0,
        review_clip_range=module.CLIP.copy(),
    )
    for anchor in answer["anchors"]:
        anchor["frame_choice"] = "human_selected"
        for i, animal in enumerate(module.human.ANIMALS):
            anchor["animals"][animal] = dict(
                visibility="visible", points={"top_cam": [20 + 30 * i, 30]}
            )
    for state in answer["continuity"].values():
        state.update(
            entry_to_after="confirmed", after_to_exit="confirmed", basis="direct_visual_continuity"
        )
    if edit is not None:
        edit(answer)
    return module.human.parse_review(
        b"\xef\xbb\xbf" + encoded(answer), data, clip_range=module.CLIP
    )


def rows_fixture():
    pose = np.zeros((3, 3, 3, 3), np.float64)
    pose[..., 0] = np.array([20, 50, 80])[None, :, None]
    pose[..., 1] = 30
    return dict(
        frame_idx=np.array([90121, 90220, 90380]),
        pose_xyz=pose,
        sources=np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy(),
        observed=np.ones((3, 3, 3), bool),
        identity_review=np.zeros((3, 3), bool),
    )


class Rig:
    def camera(self, view):
        return SimpleNamespace(project=lambda xyz: np.asarray(xyz)[..., :2].copy())


def signature_rows(rows):
    return {k: module.proof.signature(v) for k, v in rows.items()}


def test_latent_reviewed_mouth_is_human_evidence_without_clearing_support_or_review():
    parsed, rows = parsed_fixture(), rows_fixture()
    rows["observed"][1, 0, 0] = False
    rows["sources"][1, 0, :, 0] = -1
    rows["identity_review"][2, 1] = True
    rows["pose_xyz"][2, 2, 2] = np.nan  # A tail absence must not disqualify its mouth.
    rows["observed"][2, 2, 2] = False
    rows["sources"][2, 2, :, 2] = -1
    before = signature_rows(rows)
    result = module.score_rows(parsed, rows, Rig())
    assert result["human_spatial"]["matched_anchors"] == 9
    assert result["human_spatial"]["interval_counts"] == dict(preserved=6, changed=0, unscored=0)
    assert result["saved_observed"]["matched_anchors"] == 8
    assert result["saved_observed_and_review"]["matched_anchors"] == 7
    assert result["saved_node_observed"][1][0][0] is False
    assert result["saved_identity_review"][2][1] is True
    assert signature_rows(rows) == before


def test_unsupported_competitor_cannot_be_removed_to_create_unique_match():
    rows = rows_fixture()
    rows["pose_xyz"][0, 1, 0] = rows["pose_xyz"][0, 0, 0]
    rows["observed"][0, 1, 0] = False
    rows["sources"][0, 1, :, 0] = -1
    result = module.score_rows(parsed_fixture(), rows, Rig())
    for layer in ("human_spatial", "saved_observed", "saved_observed_and_review"):
        match = result[layer]["anchors"][0]["animals"][0]
        assert match["slot"] is None and match["reason"] == "nonunique one-to-one correspondence"


def test_one_to_one_global_matching_not_nearest_greedy():
    def edit(answer):
        answer["anchors"][0]["animals"]["A"]["points"] = {"top_cam": [22, 30]}
        answer["anchors"][0]["animals"]["B"]["points"] = {"top_cam": [19, 30]}

    rows = rows_fixture()
    rows["pose_xyz"][0, 1, 0, 0] = 26
    # A can use 0 or 1; B only 0. Both must be assigned: A->1, B->0.
    matches = module.score_rows(parsed_fixture(edit), rows, Rig())["human_spatial"]["anchors"][0][
        "animals"
    ]
    assert [m["slot"] for m in matches] == [1, 0, 2]


@pytest.mark.parametrize("offset,matched", [(5.0, True), (np.nextafter(5.0, np.inf), False)])
def test_exact_five_pixel_gate(offset, matched):
    rows = rows_fixture()

    # Near zero prevents offset rounding away after addition to x=20.
    def edit(answer):
        answer["anchors"][0]["animals"]["A"]["points"] = {"top_cam": [0, 30]}

    rows["pose_xyz"][0, 0, 0, 0] = offset
    match = module.score_rows(parsed_fixture(edit), rows, Rig())["human_spatial"]["anchors"][0][
        "animals"
    ][0]
    assert (match["slot"] == 0) is matched
    assert match["marked_views"] == [0]


def test_every_marked_view_must_agree_even_with_a_good_top_match():
    def edit(answer):
        answer["anchors"][1]["animals"]["A"]["points"]["side2_cam"] = [80, 30]

    result = module.score_rows(parsed_fixture(edit), rows_fixture(), Rig())
    match = result["human_spatial"]["anchors"][1]["animals"][0]
    assert match["slot"] is None and match["marked_views"] == [0, 2]
    assert result["human_spatial"]["requested_anchors"] == 9


def test_uncertain_points_preserved_but_not_used_and_local_confirmed_edge_survives():
    def edit(answer):
        answer["anchors"][0]["animals"]["A"]["visibility"] = "ambiguous"
        answer["continuity"]["A"]["entry_to_after"] = "ambiguous"
        answer["biological_naming"]["A"]["sex"] = "female"

    parsed = parsed_fixture(edit)
    result = module.score_rows(parsed, rows_fixture(), Rig())
    assert parsed.review["anchors"][0]["animals"]["A"]["points"] == {"top_cam": [20, 30]}
    assert np.isnan(parsed.human_points[0, 0]).all()
    assert result["human_spatial"]["interval_counts"] == dict(preserved=4, changed=0, unscored=2)
    local = result["local_human_endpoint_links"]
    assert local["counts"] == dict(same=5, different=0, unscored=1)
    assert local["links"][1]["outcome"] == "same"
    assert local["entry_identity_inferred"] is False
    assert parsed.review["biological_naming"]["A"]["sex"] == "female"


def test_missing_middle_is_not_bridged_into_six_link_primary():
    rows = rows_fixture()
    rows["pose_xyz"][1, 0, 0] = np.nan
    rows["observed"][1, 0, 0] = False
    rows["sources"][1, 0, :, 0] = -1
    result = module.score_rows(parsed_fixture(), rows, Rig())
    spatial = result["human_spatial"]
    assert spatial["matched_anchors"] == 8
    assert spatial["interval_counts"] == dict(preserved=4, changed=0, unscored=2)
    assert result["local_human_endpoint_links"]["counts"]["unscored"] == 2
    assert spatial["entry_exit_only_counts"]["same_as_entry"] == 3  # Endpoint-only, not interior.
    assert spatial["returning_change_probes"][0]["outcome"] == "unscored"


def mapping_fixture():
    state = np.array([0, 5, 0], dtype=np.int64)
    return dict(
        frame_idx=np.array([90121, 90220, 90380]),
        path_state=state,
        permutation_by_id=module.prior.PERMS[state].copy(),
        conditional_identity_alternative_gap=np.ones((3, 3)),
        conditional_identity_alternative_status=np.full((3, 3), "finite_gap", dtype="U32"),
    )


def test_whole_fish_permutation_returning_change_and_unknown_gap_remain_separate():
    rows, mapping = rows_fixture(), mapping_fixture()
    rows["identity_review"][1, 2] = True
    mapping["conditional_identity_alternative_status"][1, 0] = "not_retained_unknown"
    mapping["conditional_identity_alternative_gap"][1, 0] = np.inf
    mapping["conditional_identity_alternative_status"][1, 2] = "tie_within_tolerance"
    mapping["conditional_identity_alternative_gap"][1, 2] = 0
    before = signature_rows(rows), signature_rows(mapping)
    result = module.evaluate_arm(parsed_fixture(), rows, Rig(), mapping)
    assert result["raw_slots"]["human_spatial"]["interval_counts"]["changed"] == 0
    automatic = result["automatic_initial_slot_IDs"]
    assert automatic["human_spatial"]["returning_changes_seen"] == 2
    assert automatic["human_spatial"]["matched_anchors"] == 9
    assert automatic["saved_source_rows"][1][0] == rows["sources"][1, 2].tolist()
    assert automatic["saved_identity_review"][1] == [True, False, False]
    assert result["conditional_ID_alternative_gap"][1][0] == "+Infinity"
    assert result["conditional_ID_alternative_status"][1][2] == "tie_within_tolerance"
    assert result["biological_naming_applied"] is False
    assert (signature_rows(rows), signature_rows(mapping)) == before


def test_parsed_point_tamper_and_nonbijective_mapping_rejected():
    parsed = parsed_fixture()
    parsed.human_points[0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="preserved human bytes"):
        module.score_rows(parsed, rows_fixture(), Rig())
    mapping = mapping_fixture()
    mapping["permutation_by_id"][1] = [0, 0, 2]
    with pytest.raises(ValueError, match="bijective"):
        module.evaluate_arm(parsed_fixture(), rows_fixture(), Rig(), mapping)


class Evidence:
    def __init__(self, root):
        self.root = root
        self.paths = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        assert path.is_relative_to(self.root), "TEST MUST NEVER READ ACTUAL INPUTS"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("synthetic SHA mismatch")
        self.paths[str(path)] = digest


def write_npz(path, values):
    with path.open("xb") as f:
        np.savez_compressed(f, **values)
    return dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def test_candidate_copy_archive_and_selected_field_bindings(isolated_work, monkeypatch):
    # Tiny synthetic retained horizon; arbitrary human frames are not the fixed female57 list.
    monkeypatch.setattr(module, "FIRST", 90120)
    monkeypatch.setattr(module, "STOP", 90125)
    frames = np.array([90120, 90122, 90124])
    small = rows_fixture()
    arrays = {
        k: (np.arange(90120, 90125) if k == "frame_idx" else np.repeat(v[:1], 5, axis=0))
        for k, v in small.items()
    }
    arrays.update({f"unused{i}": np.array([i]) for i in range(16)})
    signatures = signature_rows(arrays)
    descriptor = write_npz(isolated_work / "comparison_copy.npz", arrays)
    # Original final archive SHA may differ; the consumed copy has its own descriptor.
    result = module.load_anchor_rows(descriptor, signatures, frames, Evidence(isolated_work))
    assert result["frame_idx"].tolist() == frames.tolist()
    assert all(not v.flags.writeable for v in result.values())
    bad = copy.deepcopy(signatures)
    bad["sources"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="field bytes"):
        module.load_anchor_rows(descriptor, bad, frames, Evidence(isolated_work))
    with pytest.raises(ValueError, match="SHA mismatch"):
        module.load_anchor_rows(
            {**descriptor, "sha256": "f" * 64}, signatures, frames, Evidence(isolated_work)
        )


def beam_arrays(first=module.START, n=5):
    state = np.array([0, 5, 5, 0, 0])[:n]
    gap = np.ones((n, 3))
    gap[0] = np.inf
    status = np.full((n, 3), "finite_gap", dtype="U32")
    status[0] = "fixed_initial_constraint"
    arrays = dict(
        frame_idx=np.arange(first, first + n),
        path_state=state,
        permutation_by_id=module.prior.PERMS[state],
        conditional_identity_alternative_gap=gap,
        conditional_identity_alternative_status=status,
    )
    arrays.update(
        {k: np.array([777]) for k in module.depth.frozen.MAPPING_FIELDS if k not in arrays}
    )
    return arrays


def test_actual_beam14_reads_only_five_frame_fields_not_terminal_arrays(isolated_work, monkeypatch):
    monkeypatch.setattr(module, "STOP", module.START + 5)
    # Validator's default n is frozen at49800; override only its explicit count for this fixture.
    original = module.prior.validate_mapping
    monkeypatch.setattr(
        module.prior, "validate_mapping", lambda a, tolerance: original(a, n=5, tolerance=tolerance)
    )
    path = isolated_work / "identity_mapping_and_ambiguity.npz"
    descriptor = write_npz(path, beam_arrays())
    bundle = dict(
        root=isolated_work,
        seal=dict(outputs={path.name: descriptor}),
        scalars=dict(numerical_tie_tolerance=1e-10),
    )
    frames = np.array([module.START, module.START + 2, module.START + 4])
    result = module.load_mapping(bundle, frames, Evidence(isolated_work))
    assert set(result) == set(module.MAPPING_FIELDS)
    assert result["path_state"].tolist() == [0, 5, 0]
    assert result["conditional_identity_alternative_status"][0, 0] == "fixed_initial_constraint"


@pytest.mark.parametrize("change", ["old15", "frame_gap", "initial_P", "unknown_status"])
def test_beam_schema_and_numeric_guards(isolated_work, monkeypatch, change):
    monkeypatch.setattr(module, "STOP", module.START + 5)
    original = module.prior.validate_mapping
    monkeypatch.setattr(
        module.prior, "validate_mapping", lambda a, tolerance: original(a, n=5, tolerance=tolerance)
    )
    arrays = beam_arrays()
    if change == "old15":
        arrays["old_streaming_extra"] = np.array([1])
    elif change == "frame_gap":
        arrays["frame_idx"][2] += 1
    elif change == "initial_P":
        arrays["path_state"][0] = 5
        arrays["permutation_by_id"][0] = module.prior.PERMS[5]
    else:
        arrays["conditional_identity_alternative_status"][1, 0] = "not_retained_unknown"
    path = isolated_work / "identity_mapping_and_ambiguity.npz"
    descriptor = write_npz(path, arrays)
    bundle = dict(
        root=isolated_work,
        seal=dict(outputs={path.name: descriptor}),
        scalars=dict(numerical_tie_tolerance=1e-10),
    )
    with pytest.raises(ValueError):
        module.load_mapping(
            bundle,
            np.array([module.START, module.START + 2, module.START + 4]),
            Evidence(isolated_work),
        )


@pytest.mark.parametrize(
    "frames", [[66399, 90220, 90380], [90120, 90220, 116200], [90120, 90120, 90380]]
)
def test_outside_or_reordered_anchors_never_extrapolated(frames):
    with pytest.raises(ValueError, match="no shifting or extrapolation"):
        module.selected_indices(np.array(frames))


def importer_fixture(parsed):
    proposal = dict(proposed_anchor_frames=[90120, 90220, 90380])
    endpoints, links, changes = [], [], []
    for t, anchor in enumerate(parsed.review["anchors"]):
        changes.append(
            dict(
                phase=anchor["phase"],
                proposed_frame=proposal["proposed_anchor_frames"][t],
                returned_frame=anchor["frame"],
                changed=t == 0,
                returned_frame_choice="human_selected",
            )
        )
        for animal, mark in anchor["animals"].items():
            endpoints.append(
                dict(
                    phase=anchor["phase"],
                    frame=anchor["frame"],
                    anonymous_animal=animal,
                    visibility=mark["visibility"],
                    original_points=mark["points"],
                    model_slot=None,
                    physical_identity_certified=False,
                )
            )
    for animal, state in parsed.review["continuity"].items():
        for edge, name in enumerate(module.human.EDGES):
            links.append(
                dict(
                    anonymous_animal=animal,
                    edge=name,
                    start_frame=int(parsed.anchor_frames[edge]),
                    stop_frame=int(parsed.anchor_frames[edge + 1]),
                    human_status=state[name],
                    human_basis=state["basis"],
                    human_notes=state["notes"],
                    trajectory_constraint_created=False,
                )
            )
    summary = dict(
        schema=module.IMPORT_STAGE,
        status="ok",
        review_status_preserved=parsed.review["status"],
        review_clip_range=module.CLIP,
        source_frame_range=[89640, 91300],
        pack_seal_sha256=module.PACK_SHA,
        parser_source_sha256=module.PINS["work/three_body/import_long_review.py"],
        original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
        external_input_count=1,
        original_review_bytes_preserved=True,
        endpoints=endpoints,
        links=links,
        proposed_to_returned_anchor_changes=changes,
        original_anchor_proposal=proposal,
        biological_naming_preserved_not_applied=parsed.review["biological_naming"],
        anonymous_identity_scope=parsed.review["anonymous_identity_scope"],
        pack_outputs_verified=1096,
        original_video_stat_or_payload_reads=0,
        SLEAP_mask_candidate_checkpoint_reads=0,
        image_decodes=0,
        GPU_used=False,
        identity_solver_calls=0,
        trajectory_modified=False,
        constraints_created=False,
        labels_filled=False,
        physical_identity_certified=False,
        all_five_minute_identity_truth=False,
        human_truth_created=False,
    )
    return summary, proposal


@pytest.mark.parametrize(
    "mutation", [None, "drop_endpoint", "invent_link", "slot", "original_bytes", "biological"]
)
def test_actual_import_summary_contract_preserves_original_nine_and_six(mutation):
    parsed = parsed_fixture()
    summary, proposal = importer_fixture(parsed)
    if mutation is None:
        module.validate_import_summary(parsed, summary, proposal)
        assert summary["proposed_to_returned_anchor_changes"][0]["changed"] is True
        return
    if mutation == "drop_endpoint":
        summary["endpoints"].pop()
    elif mutation == "invent_link":
        summary["links"][0]["human_status"] = "unreviewed"
    elif mutation == "slot":
        summary["endpoints"][0]["model_slot"] = 0
    elif mutation == "biological":
        summary["biological_naming_preserved_not_applied"] = None
    else:
        summary["original_review_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="imported review schema/content"):
        module.validate_import_summary(parsed, summary, proposal)


def lineage_fixture(root):
    shared = {str(root / f"cal{i}.json"): str(i + 1) * 64 for i in range(4)}
    shared.update({str(root / f"sl{i}.npz"): "a" * 64 for i in range(3)})
    hardlinks, videos, raw = {}, {}, {}
    for i, view in enumerate(module.VIEWS):
        sha = str(i + 1) * 64
        canonical, staged = str(root / f"source_{view}.mp4"), str(root / f"chunk_{view}.mp4")
        videos[view] = dict(canonical_path=canonical, staged_path=staged, sha256=sha)
        hardlinks[view] = dict(source_path=canonical, path=staged, sha256=sha)
        raw[view] = dict(sha256=sha, source_offset_frames=89640, source_total_frames=348600)
    chunk = dict(
        index=24, start=89640, stop=91300, manifest_sha256="e" * 64, raw_hardlinks=hardlinks
    )
    index = dict(
        original_binding=dict(inputs=dict(seed=dict(SL_and_calibration_sha256=shared))),
        prefix_chunks=[chunk],
    )
    pack = dict(
        consumed_record=chunk,
        source_manifest=str(root / "manifest.json"),
        source_manifest_sha256="e" * 64,
    )
    imported = dict(
        pack_summary=pack,
        parsed=SimpleNamespace(manifest=dict(raw_storage="original_source_video", raw=raw)),
        summary=dict(source_video_bindings=videos),
    )
    inputs = {
        **shared,
        pack["source_manifest"]: "e" * 64,
        **{v["canonical_path"]: v["sha256"] for v in videos.values()},
    }
    return (
        imported,
        {a: copy.deepcopy(index) for a in module.ARMS},
        {a: dict(inputs=inputs.copy()) for a in module.ARMS},
    )


@pytest.mark.parametrize("mutation", [None, "chunk", "calibration", "video", "offset"])
def test_lineage_checks_metadata_without_opening_named_video_or_SL(isolated_work, mutation):
    imported, indices, producers = lineage_fixture(isolated_work)
    assert list(isolated_work.iterdir()) == []  # Every named asset is nonexistent.
    if mutation == "chunk":
        indices[module.ARMS[1]]["prefix_chunks"][0]["index"] = 25
    elif mutation == "calibration":
        key = next(
            iter(
                indices[module.ARMS[0]]["original_binding"]["inputs"]["seed"][
                    "SL_and_calibration_sha256"
                ]
            )
        )
        producers[module.ARMS[2]]["inputs"][key] = "f" * 64
    elif mutation == "video":
        imported["summary"]["source_video_bindings"][module.VIEWS[0]]["sha256"] = "f" * 64
    elif mutation == "offset":
        imported["parsed"].manifest["raw"][module.VIEWS[0]]["source_offset_frames"] = 0
    if mutation is None:
        assert len(module.validate_lineage(imported, indices, producers)) == 7
    else:
        with pytest.raises(ValueError):
            module.validate_lineage(imported, indices, producers)


@pytest.mark.parametrize("enabled,outside", [(False, False), (True, True)])
def test_execution_and_workspace_guards_cannot_reach_real_input(
    monkeypatch, isolated_work, tmp_path, enabled, outside
):
    def forbidden(*args, **kwargs):
        pytest.fail("INPUT ENTRY MUST NOT BE CALLED")

    monkeypatch.setattr(module, "EXECUTION_ENABLED", enabled)
    for name in ("source_code", "load_imported_review", "load_headers", "load_rig_and_lineage"):
        monkeypatch.setattr(module, name, forbidden)
    target = tmp_path / "outside" if outside else isolated_work / "disabled"
    with pytest.raises(ValueError):
        module.run(SimpleNamespace(output=target))
    assert not target.exists()
