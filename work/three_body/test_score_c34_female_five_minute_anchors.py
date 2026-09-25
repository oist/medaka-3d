"""Small synthetic C34 fixtures. WORK/REPO are isolated; actual datasets never load."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import score_c34_female_five_minute_anchors as c


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    work = root / "work/three_body"
    work.mkdir(parents=True)
    monkeypatch.setattr(c, "WORK", work)
    monkeypatch.setattr(c, "REPO", root)
    monkeypatch.setattr(c, "INPUT_ROOT", root / "work/inputs")
    monkeypatch.setattr(c, "PRIOR_ROOT", work / "prior_fixture")


def rows(frames):
    frames = np.array(frames, dtype=np.int64)
    return dict(
        frame_idx=frames,
        source=np.broadcast_to(
            np.arange(3, dtype=np.int8)[None, :, None], (len(frames), 3, 3)
        ).copy(),
        slot_track=np.tile(np.arange(3, dtype=np.int32), (len(frames), 1)),
    )


def anchors(frames, identities=None):
    identities = [0] * len(frames) if identities is None else identities
    return [
        c.generative.MouthAnchor(
            int(f),
            {} if ident is None else {"side1_cam": ident},
            {} if ident is None else {"side1_cam": (20.0, 30.0)},
            {} if ident is None else {"side1_cam": 0.0},
        )
        for f, ident in zip(frames, identities, strict=True)
    ]


def frozen(scored, mode="c34_label_available"):
    return scored["results"][mode]["frozen_scorer_output"]


def test_label_availability_and_source_only_layers_keep_all_endpoints_and_reasons():
    data = rows([10, 20, 30])
    data["slot_track"][1, 0] = -1
    before = {k: c.proof.signature(v) for k, v in data.items()}
    result = c.score_rows(data, anchors(data["frame_idx"]))
    assert frozen(result, "source_association_only")["available_anchors"] == 3
    primary = frozen(result)
    assert primary["requested_anchors"] == 3 and primary["requested_intervals"] == 2
    assert primary["available_anchors"] == 2
    assert primary["interval_counts"] == dict(preserved=0, swap=0, unscored=2)
    assert primary["anchors"][1]["reason"] == "mouth not supported by two views"
    explanation = result["results"]["c34_label_available"]["C34_anchor_interpretation"][1]
    assert explanation["C34_reason"] == "C34 label unavailable"
    assert explanation["frozen_scorer_reason"] == primary["anchors"][1]["reason"]
    assert not result["strict_mouth_support_computed"]
    assert not result["equivalent_review_flag_available"]
    assert before == {k: c.proof.signature(v) for k, v in data.items()}


@pytest.mark.parametrize("conflict", ["duplicate_unavailable_slot", "view_disagreement"])
def test_unavailable_sources_are_not_hidden_before_conflict_check(conflict):
    data = rows([10])
    target = anchors([10])
    if conflict == "duplicate_unavailable_slot":
        data["source"][0, 1, 1] = 0
        data["slot_track"][0, 1] = -1
    else:
        target = [c.generative.MouthAnchor(10, {"side1_cam": 0, "side2_cam": 1}, {}, {})]
    result = c.score_rows(data, target)
    for mode in result["results"]:
        row = frozen(result, mode)["anchors"][0]
        assert row["identity"] is None and row["reason"] == "conflicting source identities"


def test_gap_change_bracket_does_not_change_frozen_adjacent_denominators():
    data = rows([10, 20, 30, 40])
    result = c.score_rows(data, anchors(data["frame_idx"], [1, None, None, 0]))
    expected = [
        dict(
            previous_available_frame=10,
            next_available_frame=40,
            previous_identity=1,
            next_identity=0,
            skipped_requested_frames=[20, 30],
        )
    ]
    for layer in result["results"].values():
        assert layer["frozen_scorer_output"]["interval_counts"] == dict(
            preserved=0, swap=0, unscored=3
        )
        assert layer["label_change_brackets_over_unscored_anchors"] == expected


@pytest.mark.parametrize("identities", [[None, 1, 1], [1, None, 1], [0, 1, 1], [1, None, None]])
def test_no_spurious_change_bracket_when_gap_missing_or_label_unchanged(identities):
    result = c.score_rows(rows([1, 2, 3]), anchors([1, 2, 3], identities))
    assert (
        result["results"]["c34_label_available"]["label_change_brackets_over_unscored_anchors"]
        == []
    )


def test_global_slot_renaming_preserves_endpoint_counts():
    frames = [10, 20, 30]
    data = rows(frames)
    target = anchors(frames, [0, 1, 1])
    original = c.score_rows(data, target)
    renamed = dict(
        frame_idx=data["frame_idx"].copy(),
        source=data["source"][:, [1, 2, 0]].copy(),
        slot_track=np.tile(np.arange(3, dtype=np.int32), (3, 1)),
    )
    actual = c.score_rows(renamed, target)
    for mode in original["results"]:
        assert c.compact(frozen(actual, mode)) == c.compact(frozen(original, mode))


@pytest.mark.parametrize(
    "fault",
    ["gated_label", "source_dtype", "source_shape", "negative_source", "frame_order", "extra_xyz"],
)
def test_wrong_c34_semantics_or_schema_rejected(fault):
    data = rows([10, 20])
    if fault == "gated_label":
        data["slot_track"][0, 1] = 19
    elif fault == "source_dtype":
        data["source"] = data["source"].astype(np.int32)
    elif fault == "source_shape":
        data["source"] = data["source"][:, :, :2]
    elif fault == "negative_source":
        data["source"][0, 0, 0] = -2
    elif fault == "frame_order":
        data["frame_idx"] = data["frame_idx"][::-1]
    else:
        data["xyz"] = np.zeros((2, 3, 3))
    with pytest.raises(ValueError):
        c.score_rows(data, anchors([10, 20]))


def fixed_anchor_documents():
    fixed = dict(
        frames=list(c.FRAMES),
        CSV_sha256=c.prior.GT_SHA,
        source_interval=[c.START, c.STOP],
        labels_created=False,
    )
    matches = [
        dict(
            frame=f,
            detection_rows={"side1_cam": 0},
            pixels={"side1_cam": [1.0, 2.0]},
            match_errors_px={"side1_cam": 0.0},
        )
        for f in c.FRAMES
    ]
    observations = dict(matches=matches, diagnostics=[dict(frame=f) for f in c.FRAMES])
    return fixed, observations


@pytest.mark.parametrize(
    "fault", [None, "drop", "reorder", "wrong_CSV", "over_5px", "foreign_view", "negative_row"]
)
def test_saved57_matching_is_reused_with_fixed_denominator_and_5px_contract(fault):
    fixed, observation = fixed_anchor_documents()
    if fault == "drop":
        observation["matches"].pop()
    elif fault == "reorder":
        observation["matches"].reverse()
    elif fault == "wrong_CSV":
        fixed["CSV_sha256"] = "0" * 64
    elif fault == "over_5px":
        observation["matches"][0]["match_errors_px"]["side1_cam"] = 5.1
    elif fault == "foreign_view":
        observation["matches"][0]["pixels"]["top_cam"] = [1, 2]
    elif fault == "negative_row":
        observation["matches"][0]["detection_rows"]["side1_cam"] = -1
    if fault is None:
        assert len(c.saved_anchors(fixed, observation)) == 57
    else:
        with pytest.raises(ValueError):
            c.saved_anchors(fixed, observation)


def synthetic_archive(tmp_path, frame_count=6):
    path = tmp_path / "synthetic_c34.npz"
    arrays = {
        "source": rows(range(frame_count))["source"],
        "slot_track": rows(range(frame_count))["slot_track"],
    }
    arrays.update(
        {
            k: np.array([123])
            for k in (
                "xyz",
                "residual_mm",
                "margin_mm",
                "n_views_used",
                *(f"track__{v}" for v in c.generative.VIEWS),
            )
        }
    )
    np.savez_compressed(path, **arrays)
    payload = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))


def test_native_indexing_two_fields_only_single_capture_and_readonly(monkeypatch):
    monkeypatch.setattr(c, "FRAME_COUNT", 6)
    reference = synthetic_archive(c.WORK)
    calls = []
    original = np.lib.npyio.NpzFile.__getitem__

    def logged(self, key):
        calls.append(key)
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", logged)
    selected, record = c.load_anchor_rows(reference, np.array([1, 4], np.int64))
    assert calls == ["source", "slot_track"]
    assert record["archive_reads"] == 1 and record["decoded_fields"] == ["source", "slot_track"]
    assert selected["frame_idx"].tolist() == [1, 4]
    assert all(not value.flags.writeable for value in selected.values())


@pytest.mark.parametrize("fault", ["sha", "size", "native_count", "negative_frame"])
def test_corrupt_archive_binding_or_native_extent_rejected(monkeypatch, fault):
    monkeypatch.setattr(c, "FRAME_COUNT", 6)
    reference = synthetic_archive(c.WORK)
    frames = np.array([1, 4], np.int64)
    if fault == "sha":
        reference["sha256"] = "0" * 64
    elif fault == "size":
        reference["size_bytes"] += 1
    elif fault == "native_count":
        monkeypatch.setattr(c, "FRAME_COUNT", 7)
    else:
        frames[0] = -1
    with pytest.raises(ValueError):
        c.load_anchor_rows(reference, frames)


def test_catalog_normalized_duplicates_and_escape_rejected():
    with pytest.raises(ValueError):
        c.catalog_entries(dict(files=[dict(relative="a/b.npz"), dict(relative="a\\b.npz")]))
    with pytest.raises(ValueError):
        c.catalog_entries(dict(files=[dict(relative="../escape.npz")]))


def synthetic_run_context():
    frames = np.array(c.FRAMES, np.int64)
    observations = anchors(frames)
    reference = frozen(c.score_rows(rows(frames), observations))
    refs = {}
    for value in c.LAMBDAS:
        path = c.WORK / ("synthetic_lambda_" + value + ".dat")
        path.write_bytes(value.encode())
        refs[value] = dict(
            path=str(path),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
        )
    return dict(
        anchors=observations,
        c34_references=refs,
        binding=dict(reference_3D_summary={"synthetic": True}),
        reference_results={
            method: {mode: copy.deepcopy(reference) for mode in ("raw", "with_original_review")}
            for method in c.REFERENCE_METHODS
        },
    )


def test_execute_reports_all7_and_reference_layers_without_any_actual_input(monkeypatch):
    context = synthetic_run_context()
    calls = []
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    monkeypatch.setattr(c, "load_headers", lambda evidence: context)

    def synthetic_loader(reference, frames):
        value = next(v for v, row in context["c34_references"].items() if row == reference)
        calls.append(value)
        selected = rows(frames)
        index = c.LAMBDAS.index(value)
        selected["slot_track"][:index, 0] = -1
        return selected, dict(**reference, file_stat=c.file_stat(Path(reference["path"])))

    monkeypatch.setattr(c, "load_anchor_rows", synthetic_loader)
    output = c.WORK / "synthetic_output"
    c.execute(SimpleNamespace(execute=True, output=output))
    summary = json.loads((output / "summary.json").read_text())
    assert calls == list(c.LAMBDAS)
    assert [r["lambda_value"] for r in summary["results_table"]] == list(c.LAMBDAS)
    for index, result in enumerate(summary["results_table"]):
        assert result["source_association_only"]["available_anchors"] == 57
        assert result["c34_label_available"]["available_anchors"] == 57 - index
        assert result["c34_label_available"]["requested_intervals"] == 56
    assert not summary["lambda_selected"] and not summary["whole_three_physical_ID_accuracy"]
    assert not summary["dense_swap_zero_certified"] and summary["SLEAP_payload_reads"] == 0
    assert set(summary["existing_3D_reference_counts"]) == set(c.REFERENCE_METHODS)


def test_disabled_or_outside_run_cannot_reach_any_real_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "load_headers", lambda *_: pytest.fail("actual input loader forbidden"))
    monkeypatch.setattr(c, "EXECUTION_READY", False)
    with pytest.raises(ValueError, match="not enabled"):
        c.execute(SimpleNamespace(execute=True, output=c.WORK / "output"))
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    with pytest.raises(ValueError, match="fresh workspace"):
        c.execute(SimpleNamespace(execute=True, output=tmp_path / "outside_WORK"))
