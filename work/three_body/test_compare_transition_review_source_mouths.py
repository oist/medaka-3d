"""Pure synthetic fixtures; actual human exports and recorded inputs are never opened."""

import builtins
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import compare_transition_review_source_mouths as module
import numpy as np
import pytest
from test_compare_fast_tail_transition_review import Rig, parsed_fixture, rows_fixture


@pytest.fixture(autouse=True)
def isolated_workspace(monkeypatch, tmp_path):
    root = tmp_path / "repository"
    work = root / "work/three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "REPO", root)
    monkeypatch.setattr(module.upstream, "WORK", work)
    monkeypatch.setattr(module.classical, "WORK", work)
    monkeypatch.setattr(module.classical, "REPO", root)
    monkeypatch.setattr(module.classical, "INPUT_ROOT", root / "work/inputs")
    return root, work


def fixture():
    def mark_all_views(answer):
        for anchor in answer["anchors"]:
            for i, animal in enumerate(module.upstream.human.ANIMALS):
                anchor["animals"][animal]["points"] = {v: [20 + 30 * i, 30] for v in module.VIEWS}

    parsed = parsed_fixture(mark_all_views)
    source = np.broadcast_to(np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], np.int8), (3, 3, 3)).copy()
    cache = {}
    # Local detection row IDs differ across cameras, with no sorting or joining by row number.
    for view, order in zip(module.VIEWS, ([20, 50, 80], [80, 20, 50], [50, 80, 20]), strict=True):
        uv = np.array([[x, 30] for x in order], np.float32)
        cache[view] = dict(
            frame_idx=parsed.anchor_frames.copy(),
            offsets=np.array([0, 3, 6, 9]),
            mouth_uv=np.tile(uv, (3, 1)),
        )
    return parsed, source, cache


def test_common_lookup_uses_each_views_own_local_row_and_same_human_labels():
    parsed, source, cache = fixture()
    points, reasons = module.assigned_mouths(source, parsed.anchor_frames, cache)
    assert np.array_equal(points, parsed.human_points)
    assert np.all(reasons == "assigned_finite_mouth")
    result = module.score_stream(
        parsed, source, cache, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    assert result["scores"][module.LAYER]["matched_anchors"] == 9
    assert result["scores"][module.LAYER]["interval_counts"] == dict(
        preserved=6, changed=0, unscored=0
    )
    assert result["equivalent_review_flag_available"] is False
    assert result["original_review"] is None


def test_c34_and_3D_common_layer_match_even_when_3D_pose_changes():
    parsed, source, cache = fixture()
    rows = rows_fixture()
    rows["sources"] = np.repeat(source[:, :, :, None].astype(np.int64), 3, axis=3)
    rows["pose_xyz"] += 12345  # Not projected in the common observation layer.
    mapping = dict(permutation_by_id=np.tile([0, 1, 2], (3, 1)))
    new = module.evaluate_three_dimensional(parsed, rows, mapping, cache)
    old = module.score_stream(
        parsed, source, cache, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    assert old["scores"][module.LAYER] == new["raw_slots"]["scores"][module.LAYER]
    assert old["scores"][module.LAYER] == new["automatic_initial_slot_IDs"]["scores"][module.LAYER]


def test_unavailable_label_or_review_cannot_hide_a_finite_competitor():
    parsed, source, cache = fixture()
    source[0, 1] = source[0, 0]
    eligible, review = np.ones((3, 3), bool), np.zeros((3, 3), bool)
    eligible[0, 1] = False
    review[0, 1] = True
    result = module.score_stream(
        parsed, source, cache, eligible, review=review, eligibility_name="saved_observed_mouth"
    )
    for score in result["scores"].values():
        a = score["anchors"][0]["animals"][0]
        assert a["slot"] is None and a["reason"] == "nonunique one-to-one correspondence"


def test_source_only_label_support_and_review_are_separate_and_immutable():
    parsed, source, cache = fixture()
    eligible, review = np.ones((3, 3), bool), np.zeros((3, 3), bool)
    eligible[1, 0] = False
    review[2, 1] = True
    before = source.tobytes(), eligible.tobytes(), review.tobytes()
    result = module.score_stream(
        parsed, source, cache, eligible, review=review, eligibility_name="saved_observed_mouth"
    )
    assert [s["matched_anchors"] for s in result["scores"].values()] == [9, 8, 7]
    assert result["original_eligibility"][1][0] is False
    assert result["original_review"][2][1] is True
    assert before == (source.tobytes(), eligible.tobytes(), review.tobytes())


def test_C34_label_eligibility_is_not_called_strict_mouth_support():
    parsed, source, cache = fixture()
    eligible = np.ones((3, 3), bool)
    eligible[1, 0] = False
    result = module.score_stream(
        parsed, source, cache, eligible, eligibility_name="C34_saved_label_available"
    )
    assert result["scores"][module.LAYER]["matched_anchors"] == 9
    assert result["scores"]["C34_saved_label_available"]["matched_anchors"] == 8
    assert result["frozen_core_reason_translation"] == {
        "mouth lacks declared measured support": "C34 saved label unavailable"
    }


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_assigned_SL_mouth_is_paired_nan_and_not_reconstructed(value):
    parsed, source, cache = fixture()
    cache["top_cam"]["mouth_uv"][3, 0] = value
    points, reasons = module.assigned_mouths(source, parsed.anchor_frames, cache)
    assert np.isnan(points[1, 0, 0]).all()
    assert reasons[1, 0, 0] == "nonfinite_assigned_mouth"
    result = module.score_stream(
        parsed, source, cache, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    primary = result["scores"][module.LAYER]
    assert primary["matched_anchors"] == 8
    assert primary["interval_counts"] == dict(preserved=4, changed=0, unscored=2)
    assert primary["entry_exit_only_counts"]["same_as_entry"] == 3
    assert result["local_human_endpoint_links"]["counts"]["unscored"] == 2


def test_unassigned_source_does_not_use_negative_index_or_another_marked_view():
    parsed, source, cache = fixture()
    source[1, 0, 0] = -1
    points, reasons = module.assigned_mouths(source, parsed.anchor_frames, cache)
    assert np.isnan(points[1, 0, 0]).all() and reasons[1, 0, 0] == "unassigned_source"
    result = module.score_stream(
        parsed, source, cache, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    assert result["scores"][module.LAYER]["anchors"][1]["animals"][0]["slot"] is None


@pytest.mark.parametrize("fault", ["float_row", "negative", "outside", "frame_order"])
def test_invalid_source_reference_is_a_failure_not_abstention(fault):
    parsed, source, cache = fixture()
    if fault == "float_row":
        source = source.astype(float)
    elif fault == "negative":
        source[0, 0, 0] = -2
    elif fault == "outside":
        source[0, 0, 0] = 3
    else:
        cache["top_cam"]["frame_idx"] = cache["top_cam"]["frame_idx"][::-1]
    with pytest.raises(ValueError):
        module.assigned_mouths(source, parsed.anchor_frames, cache)


def test_unknown_entry_chain_keeps_confirmed_second_local_link():
    parsed, source, cache = fixture()

    def edit(answer):
        answer["continuity"]["A"]["entry_to_after"] = "unreviewed"
        answer["biological_naming"]["A"]["sex"] = "female"

    parsed = parsed_fixture(edit)
    result = module.score_stream(
        parsed, source, cache, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    assert result["scores"][module.LAYER]["interval_counts"]["unscored"] == 2
    assert result["local_human_endpoint_links"]["counts"] == dict(same=5, different=0, unscored=1)
    assert parsed.review["biological_naming"]["A"]["sex"] == "female"


def test_selected_initial_P210_and_whole_body_reorder_are_valid():
    parsed, source, cache = fixture()
    rows = rows_fixture()
    rows["sources"] = np.repeat(source[:, :, :, None].astype(np.int64), 3, axis=3)
    rows["identity_review"][0, 2] = True
    state = np.array([5, 0, 5], np.int64)
    p = module.upstream.prior.PERMS[state]
    mapping = dict(
        frame_idx=parsed.anchor_frames.copy(),
        path_state=state,
        permutation_by_id=p,
        conditional_identity_alternative_gap=np.ones((3, 3)),
        conditional_identity_alternative_status=np.full((3, 3), "finite_gap", dtype="U32"),
    )
    saved = dict(
        permutation_by_id=p.tolist(),
        conditional_ID_alternative_gap=np.ones((3, 3)).tolist(),
        conditional_ID_alternative_status=mapping[
            "conditional_identity_alternative_status"
        ].tolist(),
    )
    module.validate_selected_mapping(mapping, parsed.anchor_frames, saved)
    before = {k: a.tobytes() for k, a in rows.items()}
    result = module.evaluate_three_dimensional(parsed, rows, mapping, cache)
    automatic = result["automatic_initial_slot_IDs"]
    assert automatic["source_rows"][0][0] == source[0, 2].tolist()
    assert automatic["original_review"][0] == [True, False, False]
    assert automatic["scores"][module.LAYER]["returning_changes_seen"] == 2
    assert before == {k: a.tobytes() for k, a in rows.items()}


def test_select_only_three_sparse_frames_preserves_original_row_order():
    frames = parsed_fixture().anchor_frames
    index = np.array([frames[0], frames[0], frames[0] + 1, frames[1], frames[2], frames[2]])
    points = np.arange(len(index) * 6, dtype=np.float32).reshape(-1, 3, 2)
    selected = module.select_mouths(index, points, frames)
    assert selected["offsets"].tolist() == [0, 2, 3, 5]
    assert selected["original_flat_row_ranges"].tolist() == [[0, 2], [3, 4], [4, 6]]
    assert np.array_equal(selected["mouth_uv"], points[[0, 1, 3, 4, 5], 0])
    with pytest.raises(ValueError, match="native index"):
        module.select_mouths(index[::-1].astype(np.uint64), points, frames)


def descriptor(path):
    data = path.read_bytes()
    return dict(
        path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


def test_SL_capture_reads_one_archive_preserves_missing_frame_and_never_loads_scores(
    isolated_workspace, monkeypatch
):
    root, work = isolated_workspace
    frames = parsed_fixture().anchor_frames
    index = np.array([frames[0], frames[0], frames[2]])
    points = np.arange(18, dtype=np.float32).reshape(3, 3, 2)
    path = work / "synthetic_SL.npz"
    np.savez_compressed(
        path,
        experiment=module.classical.baseline.EXPERIMENT,
        view="top_cam",
        nodes=np.array(["mouth", "pectoral", "tail"]),
        n_frames=348600,
        frame_idx=index,
        points=points,
        score=np.ones(3),
    )
    ref = descriptor(path)
    opened, fields = [], []
    original_read, original_get = Path.read_bytes, np.lib.npyio.NpzFile.__getitem__

    def read_bytes(self):
        assert self.resolve().is_relative_to(root), "NO ACTUAL INPUT READ ALLOWED"
        opened.append(self.resolve())
        return original_read(self)

    def getitem(self, key):
        fields.append(key)
        return original_get(self, key)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", read_bytes)
        patch.setattr(np.lib.npyio.NpzFile, "__getitem__", getitem)
        result, capture = module.load_mouths(ref, "top_cam", frames)
    assert opened == [path.resolve()]
    assert "score" not in fields and fields.count("points") == fields.count("frame_idx") == 1
    assert result["offsets"].tolist() == [0, 2, 2, 3]
    assert capture["archive_reads"] == 1 and capture["score_decoded"] is False
    assert all(not a.flags.writeable for a in result.values())


class Evidence:
    def __init__(self, root):
        self.root = root

    def add(self, path, expected=None):
        assert Path(path).resolve().is_relative_to(self.root), "NO ACTUAL INPUT READ ALLOWED"
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected


def registered_fixture(root, monkeypatch):
    input_root = module.classical.INPUT_ROOT
    entries = []
    for name in [
        *[
            f"{module.classical.baseline.BASELINE}/C34-label__{module.classical.baseline.EXPERIMENT}__lam{x}.npz"
            for x in module.classical.LAMBDAS
        ],
        *[
            f"work/c01_s16_predictions/bottomup/{module.classical.baseline.EXPERIMENT}__{v}.npz"
            for v in module.VIEWS
        ],
    ]:
        path = input_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic header fixture only")
        rec = descriptor(path)
        entries.append(
            dict(relative=name, local=rec["path"], bytes=rec["size_bytes"], sha256=rec["sha256"])
        )
    catalog = input_root / "staging_manifest.json"
    catalog.write_text(json.dumps(dict(files=entries)))
    monkeypatch.setattr(module.classical, "CATALOG_SHA", descriptor(catalog)["sha256"])
    shared = {r["local"]: r["sha256"] for r in entries if "c01_s16_predictions" in r["relative"]}
    return shared


@pytest.mark.parametrize("fault", [None, "wrong_top", "missing_top"])
def test_registered_top_cache_must_match_new_producers_too(isolated_workspace, monkeypatch, fault):
    root, _ = isolated_workspace
    shared = registered_fixture(root, monkeypatch)
    top = next(p for p in shared if "top_cam" in p)
    if fault == "wrong_top":
        shared[top] = "f" * 64
    elif fault == "missing_top":
        del shared[top]
    if fault is None:
        c34, sl = module.registered_references(shared, Evidence(root))
        assert len(c34) == 7 and set(sl) == set(module.VIEWS)
    else:
        with pytest.raises(ValueError, match="top_cam"):
            module.registered_references(shared, Evidence(root))


def test_upstream_flag_only_activation_does_not_pin_a_mutable_false_executable(monkeypatch):
    reviewed = b'EXECUTION_ENABLED = False\nTEXT = "EXECUTION_ENABLED = True"\n'
    monkeypatch.setattr(module, "UPSTREAM_FALSE_SHA", hashlib.sha256(reviewed).hexdigest())
    enabled = reviewed.replace(b"= False", b"= True", 1)
    assert (
        module.enabled_source_sha(reviewed)
        == module.enabled_source_sha(enabled)
        == hashlib.sha256(enabled).hexdigest()
    )
    with pytest.raises(ValueError, match="beyond"):
        module.enabled_source_sha(enabled + b"OTHER = 1\n")


@pytest.mark.parametrize("enabled,outside", [(False, False), (True, True)])
def test_run_guards_never_reach_actual_upstream_or_data(
    monkeypatch, isolated_workspace, tmp_path, enabled, outside
):
    _, work = isolated_workspace
    monkeypatch.setattr(module, "EXECUTION_ENABLED", enabled)

    def forbidden(*args, **kwargs):
        pytest.fail("ACTUAL INPUT ENTRY MUST NEVER BE CALLED")

    for name in ("source_code", "load_upstream", "registered_references", "load_mouths"):
        monkeypatch.setattr(module, name, forbidden)
    target = tmp_path / "outside" if outside else work / "disabled"
    with pytest.raises(ValueError):
        module.run(SimpleNamespace(output=target))
    assert not target.exists()


def test_run_synthetic_end_to_end(monkeypatch, isolated_workspace):
    """Real selected-six-archive loader -> real matcher -> all 13 saved streams.

    Only the already-qualified full5/source trust boundary is synthetic. All
    registrations, captures, selected codecs, row lookup, reorder, scoring,
    output writers and final guards run normally. No real payload is reachable.
    """
    root, work = isolated_workspace
    monkeypatch.setattr(module.upstream, "REPO", root)
    monkeypatch.setitem(module.proof.Evidence.add.__globals__, "REPO", root)
    monkeypatch.setattr(module.classical.baseline, "BASELINE", "synthetic_c34")
    # Retain native anchor coordinates; shrink only this synthetic cache extent.
    monkeypatch.setattr(module.classical, "FRAME_COUNT", 90381)
    assert all(
        Path(path).resolve().is_relative_to(root)
        for path in (
            module.REPO,
            module.WORK,
            module.upstream.REPO,
            module.upstream.WORK,
            module.classical.REPO,
            module.classical.WORK,
            module.classical.INPUT_ROOT,
        )
    )

    def edit(answer):
        for anchor in answer["anchors"]:
            for i, animal in enumerate(module.upstream.human.ANIMALS):
                anchor["animals"][animal]["points"] = {v: [20 + 30 * i, 30] for v in module.VIEWS}
        answer["continuity"]["A"]["entry_to_after"] = "unreviewed"

    parsed = parsed_fixture(edit)
    _, source, cache = fixture()
    # B's middle top mouth is unavailable; both adjacent B links stay unscored.
    source[1, 1, 0] = -1
    source_file = work / "synthetic_upstream_source.py"
    false_source = b"# SYNTHETIC TRUST BOUNDARY ONLY\nEXECUTION_ENABLED = False\n"
    source_file.write_bytes(false_source.replace(b"= False", b"= True"))
    source_digest = descriptor(source_file)["sha256"]
    code = {str(source_file.resolve()): source_digest}
    monkeypatch.setattr(module.upstream, "__file__", str(source_file))
    monkeypatch.setattr(module, "UPSTREAM_FALSE_SHA", hashlib.sha256(false_source).hexdigest())
    monkeypatch.setattr(module, "source_code", lambda: code.copy())

    input_root = module.classical.INPUT_ROOT
    entries, shared = [], {}

    def register(relative, arrays):
        path = input_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        row = descriptor(path)
        entries.append(
            dict(
                relative=relative, local=row["path"], bytes=row["size_bytes"], sha256=row["sha256"]
            )
        )
        return row

    for view in module.VIEWS:
        points = np.zeros((9, 3, 2), np.float32)
        points[:, 0] = cache[view]["mouth_uv"]
        row = register(
            f"work/c01_s16_predictions/bottomup/{module.classical.baseline.EXPERIMENT}__{view}.npz",
            dict(
                experiment=module.classical.baseline.EXPERIMENT,
                view=view,
                nodes=np.array(["mouth", "pectoral", "tail"]),
                n_frames=module.classical.FRAME_COUNT,
                frame_idx=np.repeat(parsed.anchor_frames, 3),
                points=points,
                score=np.ones(9),
            ),
        )
        shared[row["path"]] = row["sha256"]
    full_source = np.full((module.classical.FRAME_COUNT, 3, 3), -1, np.int8)
    full_labels = np.full((module.classical.FRAME_COUNT, 3), -1, np.int32)
    full_source[parsed.anchor_frames] = source
    full_labels[parsed.anchor_frames] = np.arange(3)
    full_labels[parsed.anchor_frames[2], 2] = -1  # Source exists, C34 label is unavailable.
    for lam in module.classical.LAMBDAS:
        register(
            f"synthetic_c34/C34-label__{module.classical.baseline.EXPERIMENT}__lam{lam}.npz",
            dict(
                source=full_source,
                slot_track=full_labels,
                **{
                    key: np.empty(0)
                    for key in (
                        "xyz",
                        "residual_mm",
                        "margin_mm",
                        "n_views_used",
                        *[f"track__{v}" for v in module.VIEWS],
                    )
                },
            ),
        )
    catalog = input_root / "staging_manifest.json"
    catalog.write_bytes(module.proof.json_bytes(dict(files=entries)))
    monkeypatch.setattr(module.classical, "CATALOG_SHA", descriptor(catalog)["sha256"])

    parent = work / "synthetic_upstream"
    out = module.proof.Output(parent)
    rows = rows_fixture()
    rows["sources"] = np.repeat(source[:, :, :, None].astype(np.int64), 3, axis=3)
    rows["observed"][1, 0, 0] = False  # Finite SL competitor remains in the primary.
    rows["identity_review"][2, 2] = True
    state = np.array([5, 0, 5], np.int64)  # Selected first row P210, not full5 initial P012.
    mapping = dict(
        frame_idx=parsed.anchor_frames.copy(),
        path_state=state,
        permutation_by_id=module.upstream.prior.PERMS[state],
        conditional_identity_alternative_gap=np.ones((3, 3)),
        conditional_identity_alternative_status=np.full((3, 3), "finite_gap", dtype="U32"),
    )
    mapping["conditional_identity_alternative_gap"][0, 0] = np.inf
    mapping["conditional_identity_alternative_status"][0, 0] = "not_retained_unknown"
    results = {}
    for arm in module.upstream.ARMS:
        # Use the real upstream result writer's pure evaluator and its actual five-field schemas.
        results[arm] = module.upstream.evaluate_arm(parsed, rows, Rig(), mapping)
        out.write(f"anchors/{arm}.npz", rows)
        out.write(f"identity_anchors/{arm}.npz", mapping)
    summary = dict(
        schema=module.upstream.STAGE,
        status="ok",
        native_interval=[66400, 116200],
        terminal_horizon=116200,
        review_clip_range=module.upstream.CLIP,
        anchor_frames=parsed.anchor_frames.tolist(),
        requested_endpoints_per_arm=9,
        requested_links_per_arm=6,
        candidate_fields_decoded=5,
        candidate_member_count=21,
        mapping_fields_decoded=5,
        mapping_member_count=14,
        terminal_mapping_arrays_decoded=False,
        trajectory_modified=False,
        reviews_cleared=False,
        human_labels_created=False,
        biological_identity_assigned=False,
        all_five_minute_identity_truth=False,
        dense_swap_zero_certified=False,
        results=results,
        original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
        review_status_preserved=parsed.review["status"],
        biological_naming_preserved_not_applied=parsed.review.get("biological_naming"),
        source_sha256=code,
        lineage=dict(shared_SL_calibration_sha256=shared, synthetic=True),
        identity_bindings={
            arm: dict(synthetic_full5_trust_only=True) for arm in module.upstream.ARMS
        },
    )
    out.write("original_review.json", parsed.review_bytes)
    out.write("source_manifest.json", parsed.manifest_bytes)
    out.write("summary.json", summary)
    parent_seal = parent / "comparison.provenance.json"
    parent_seal.write_bytes(
        module.proof.json_bytes(
            dict(
                stage=module.upstream.STAGE,
                status="ok",
                inputs=code,
                outputs=out.records,
                synthetic_fixture_only=True,
            )
        )
    )
    args = SimpleNamespace(
        human_comparison_root=parent,
        human_comparison_sha=descriptor(parent_seal)["sha256"],
        output=work / "synthetic_result",
    )
    input_paths = [
        source_file,
        catalog,
        parent_seal,
        *[Path(x["local"]) for x in entries],
        *[Path(x["path"]) for x in out.records.values()],
    ]
    before = {str(p): descriptor(p) for p in input_paths}

    def forbidden(*args, **kwargs):
        pytest.fail("FULL5/HUMAN/CALIBRATION/SL DISCOVERY OR REAL PIPELINE ENTRY FORBIDDEN")

    # Deny obsolete/full-payload loaders too, rather than relying only on synthetic paths.
    for name in (
        "load_headers",
        "load_imported_review",
        "load_rig_and_lineage",
        "load_anchor_rows",
        "load_mapping",
    ):
        monkeypatch.setattr(module.upstream, name, forbidden)
    monkeypatch.setattr(module.classical, "load_headers", forbidden)
    monkeypatch.setattr(module.classical, "source_code", forbidden)
    opened = []

    def guarded_open(original):
        def wrapped(file, *args, **kwargs):
            path = Path(file).resolve()
            assert path.is_relative_to(root), "NO FILE ACCESS OUTSIDE SYNTHETIC ROOT"
            opened.append(path)
            return original(file, *args, **kwargs)

        return wrapped

    # Apply guards before enabling this in-memory test invocation. Disk source stays False.
    with monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", guarded_open(builtins.open))
        guarded.setattr(io, "open", guarded_open(io.open))
        guarded.setattr(module, "EXECUTION_ENABLED", True)
        module.run(args)
    assert module.EXECUTION_ENABLED is False
    assert opened and all(p.is_relative_to(root) for p in opened)
    assert before == {str(p): descriptor(p) for p in input_paths}
    result = json.loads((args.output / "summary.json").read_bytes())
    seal = json.loads((args.output / "comparison.provenance.json").read_bytes())
    assert seal["status"] == "ok" and set(seal["inputs"]) == set(before)
    assert len(seal["outputs"]) == 14
    for row in seal["outputs"].values():
        assert descriptor(Path(row["path"])) == row
    assert result["streams"] == len(result["results"]) == 13
    assert result["unchanged_3D_primary_reference"] == out.records["summary.json"]
    assert (args.output / "original_review.json").read_bytes() == parsed.review_bytes
    assert result["full_3D_candidate_or_mapping_reads"] == 0
    assert result["selected_3D_anchor_archive_reads"] == 6
    assert result["SL_archive_reads"] == 3 and result["C34_archive_reads"] == 7
    for key, stream in result["results"].items():
        common = stream["scores"][module.LAYER]
        assert common["requested_anchors"] == 9 and common["requested_animal_intervals"] == 6
        assert common["matched_anchors"] == 8
        assert common["interval_counts"]["unscored"] == 4
        assert stream["local_human_endpoint_links"]["counts"]["unscored"] == 3
        assert common["entry_exit_only_counts"]["same_as_entry"] == 2
        if key.startswith("C34_"):
            assert stream["scores"]["C34_saved_label_available"]["matched_anchors"] == 7
            assert stream["original_eligibility"][2][2] is False
        else:
            assert stream["scores"]["saved_observed_mouth"]["matched_anchors"] == 7
            assert (
                stream["scores"]["saved_observed_mouth_and_original_review"]["matched_anchors"] == 6
            )
    for arm in module.upstream.ARMS:
        auto = result["results"][arm + "/automatic_initial_slot_IDs"]
        assert auto["source_rows"][0][0] == source[0, 2].tolist()
        assert auto["original_review"][2] == [True, False, False]
        inherited = result["inherited_conditional_ID_ambiguity"][arm]
        assert inherited["permutation_by_id"][0] == [2, 1, 0]
        assert inherited["conditional_ID_alternative_gap"][0][0] == "+Infinity"
        assert inherited["conditional_ID_alternative_status"][0][0] == "not_retained_unknown"
