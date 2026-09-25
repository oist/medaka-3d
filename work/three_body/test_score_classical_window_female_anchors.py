"""Synthetic metadata and arrays only; all input/output roots are isolated."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import score_classical_window_female_anchors as c


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    work = root / "work/three_body"
    work.mkdir(parents=True)
    for name, value in dict(
        REPO=root,
        WORK=work,
        PRODUCER=work / "producer.py",
        OLD_ROOT=work / "old",
        ANCHOR_ROOT=work / "anchors",
        EXECUTION_READY=False,
    ).items():
        monkeypatch.setattr(c, name, value)
    # Frozen utility's path guard is redirected only inside this isolated test process.
    monkeypatch.setattr(c.proof, "REPO", root)
    for name, value in dict(
        REPO=root,
        WORK=work,
        INPUT_ROOT=root / "unused_old_inputs",
        PRIOR_ROOT=root / "unused_old_prior",
    ).items():
        monkeypatch.setattr(c.old, name, value)

    def forbidden_old_io(*args, **kwargs):
        pytest.fail("old payload/header loader must not run in this synthetic test")

    for name in ("load_anchor_rows", "load_headers", "source_code"):
        monkeypatch.setattr(c.old, name, forbidden_old_io)


def arrays():
    count = c.PROCESS[1] - c.PROCESS[0]
    return dict(
        frame_idx=np.arange(*c.PROCESS, dtype=np.int64),
        source=np.broadcast_to(np.arange(3, dtype=np.int8)[None, :, None], (count, 3, 3)).copy(),
        slot_track=np.tile(np.arange(3, dtype=np.int32), (count, 1)),
    )


def full_arrays():
    out = arrays()
    count = len(out["frame_idx"])
    out.update(
        xyz=np.zeros((count, 3, 3), np.float32),
        residual_mm=np.zeros((count, 3), np.float32),
        slot_rays=np.full((count, 3), 3, np.int8),
        margin_mm=np.zeros(count, np.float32),
        n_views_used=np.full(count, 3, np.int8),
    )
    for view in c.VIEWS:
        out["track__" + view] = np.empty(0, np.int32)
        out["instance_frame_idx__" + view] = np.empty(0, np.int32)
    return out


def requested():
    return np.asarray(c.old.FRAMES, np.int64)


def anchors():
    return [
        c.old.generative.MouthAnchor(
            int(f),
            {"side1_cam": 0, "side2_cam": 0},
            {"side1_cam": [1.0, 2.0], "side2_cam": [3.0, 4.0]},
            {"side1_cam": 0.0, "side2_cam": 0.0},
        )
        for f in requested()
    ]


def test_native_mapping_uses_process_start_not_science_start_or_native_row_number():
    data = arrays()
    idx = requested() - c.PROCESS[0]
    data["source"][idx, 0] = 6
    selected, indices = c.select_rows(data, requested())
    assert np.array_equal(indices, idx)
    assert indices[0] == c.old.FRAMES[0] - 49800
    assert requested()[0] > len(data["frame_idx"])  # direct native indexing would fail
    assert (selected["source"][:, 0] == 6).all()
    assert selected["frame_idx"].tolist() == list(c.old.FRAMES)
    assert all(not value.flags.writeable for value in selected.values())


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "offset",
        "reordered",
        "dtype",
        "bad_source",
        "source_dtype",
        "track_ID",
    ],
)
def test_invalid_frame_or_source_arrays_rejected(change):
    data = arrays()
    if change == "missing":
        data = {k: v[:-1] for k, v in data.items()}
    elif change == "duplicate":
        data["frame_idx"][10] = data["frame_idx"][9]
    elif change == "offset":
        data["frame_idx"] += 16600
    elif change == "reordered":
        data["frame_idx"] = data["frame_idx"][::-1]
    elif change == "dtype":
        data["frame_idx"] = data["frame_idx"].astype(np.int32)
    elif change == "bad_source":
        data["source"][0, 0, 0] = -2
    elif change == "source_dtype":
        data["source"] = data["source"].astype(np.int64)
    else:
        data["slot_track"][0, 0] = 42
    with pytest.raises(ValueError):
        c.select_rows(data, requested())


def test_missing_source_eligibility_competition_and_gap_keep57_56():
    data = arrays()
    at = requested() - c.PROCESS[0]
    # first label1, unavailable middle anchor, then label0: frozen56 stays U/U.
    data["source"][at[0]] = data["source"][at[0], [1, 0, 2]]
    data["slot_track"][at[1]] = -1
    # Both source competitors remain visible even with every label unavailable.
    data["source"][at[4], 1, 1] = 0
    data["slot_track"][at[4]] = -1
    data["source"][at[6]] = -1
    rows, _ = c.select_rows(data, requested())
    result = c.old.score_rows(rows, anchors())
    primary = result["results"]["c34_label_available"]
    raw = primary["frozen_scorer_output"]
    assert (raw["requested_anchors"], raw["requested_intervals"]) == (57, 56)
    assert raw["anchors"][4]["reason"] == "conflicting source identities"
    assert raw["anchors"][6]["identity"] is None
    bracket = primary["label_change_brackets_over_unscored_anchors"][0]
    assert bracket["previous_identity"] == 1 and bracket["next_identity"] == 0
    assert bracket["skipped_requested_frames"] == [int(requested()[1])]
    assert sum(raw["interval_counts"].values()) == 56
    assert result["source_filtered_before_conflict_check"] is False


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".npz") and isinstance(value, dict):
        np.savez_compressed(path, **value)
    else:
        path.write_bytes(value if isinstance(value, bytes) else c.proof.json_bytes(value))
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size, sha256=c.proof.sha(path))


def compact_table():
    basic = dict(
        requested_anchors=57,
        available_anchors=57,
        requested_intervals=56,
        interval_counts=dict(preserved=56, swap=0, unscored=0),
        label_change_brackets_over_unscored_anchors=[],
    )
    return [
        dict(
            lambda_value=value,
            source_association_only=copy.deepcopy(basic),
            c34_label_available=copy.deepcopy(basic),
        )
        for value in c.LAMBDAS
    ]


@pytest.fixture
def completed_fixture(monkeypatch):
    c.PRODUCER.write_text("# synthetic producer source only\n", encoding="utf-8")
    monkeypatch.setattr(c, "PRODUCER_SHA", c.proof.sha(c.PRODUCER))
    monkeypatch.setattr(c, "source_code", lambda seal, ev: ev.add(c.PRODUCER, c.PRODUCER_SHA))
    fixed = dict(
        frames=list(c.old.FRAMES),
        CSV_sha256=c.old.prior.GT_SHA,
        source_interval=list(c.SCIENCE),
        labels_created=False,
    )
    observations = dict(
        matches=[
            dict(
                frame=a.frame,
                detection_rows=a.detection_rows,
                pixels=a.pixels,
                match_errors_px=a.match_errors_px,
            )
            for a in anchors()
        ],
        diagnostics=[dict(frame=int(f)) for f in requested()],
    )
    anchor_outputs = {
        name: write(c.ANCHOR_ROOT, name, value)
        for name, value in (
            ("fixed_female_anchor_set.json", fixed),
            ("anchor_observations.json", observations),
        )
    }
    anchor_seal = dict(stage=c.old.prior.STAGE, status="ok", outputs=anchor_outputs)
    row = write(c.ANCHOR_ROOT, "evaluation.provenance.json", anchor_seal)
    monkeypatch.setattr(c, "ANCHOR_SEAL_SHA", row["sha256"])
    records = []
    for name in (
        "catalog.json",
        "seed.json",
        *(v + ".npz" for v in c.VIEWS),
        "manifest.json",
        "cam0.json",
        "cam1.json",
        "cam2.json",
    ):
        # Only metadata references: deliberately not decodable SL/rig payloads.
        records.append(write(c.REPO / "registered", name, b"synthetic reference, do not decode"))
    shared = {
        r["path"]: r["sha256"]
        for r in records
        if r["path"].endswith(("side1_cam.npz", "side2_cam.npz"))
    }
    binding = dict(
        prior_evaluation_seal_sha256=c.ANCHOR_SEAL_SHA,
        fixed_anchor_output=anchor_outputs["fixed_female_anchor_set.json"],
        matched_anchor_output=anchor_outputs["anchor_observations.json"],
        annotation_CSV_sha256=c.old.prior.GT_SHA,
        matched_SL_references=shared,
    )
    old_summary = dict(
        schema=c.old.STAGE,
        status="ok",
        native_interval=list(c.SCIENCE),
        requested_anchors=57,
        requested_adjacent_intervals=56,
        all7_lambdas_scored=True,
        lambda_selected=False,
        evidence_binding=binding,
        results_table=compact_table(),
        existing_3D_reference_counts={key: {} for key in c.old.REFERENCE_METHODS},
    )
    old_outputs = {"summary.json": write(c.OLD_ROOT, "summary.json", old_summary)}
    row = write(
        c.OLD_ROOT,
        "evaluation.provenance.json",
        dict(stage=c.old.STAGE, status="ok", inputs={}, outputs=old_outputs),
    )
    monkeypatch.setattr(c, "OLD_SEAL_SHA", row["sha256"])
    root = c.WORK / "bounded_producer"
    sources = {str(c.PRODUCER.resolve()): c.PRODUCER_SHA}
    plan = dict(
        schema=c.PRODUCER_STAGE,
        process_interval=list(c.PROCESS),
        science_interval=list(c.SCIENCE),
        process_frames=c.PROCESS[1] - c.PROCESS[0],
        lambdas=[float(v) for v in c.LAMBDAS],
        initial_seed_shared=False,
        initialization="unconstrained C25 first-frame costs",
        future_horizon_exclusive=c.PROCESS[1],
        time_chunking=False,
        outside_observations_used=False,
        original_C25_objective_preserved=True,
        transition_cache_identity_bug_corrected=True,
        source_binding=dict(sources=sources),
        inputs=dict(
            records=records,
            seed_reference=dict(frame=c.PROCESS[0]),
            selected={
                v: dict(
                    native_interval=list(c.PROCESS),
                    full_native_index_read=True,
                    outside_points_scores_deserialized=False,
                )
                for v in c.VIEWS
            },
        ),
    )
    outputs = {"plan.json": write(root, "plan.json", plan)}
    outputs["stage_a.json"] = write(root, "stage_a.json", b"not inspected fixture")
    outputs["stage_a.npz"] = write(root, "stage_a.npz", b"not inspected fixture")
    results = {}
    for i, value in enumerate(c.LAMBDAS):
        data = full_arrays()
        data["slot_track"][requested()[:i] - c.PROCESS[0]] = -1
        results["lam" + value] = dict(
            objective=1.0, array_signatures={k: c.signature(v) for k, v in data.items()}
        )
        name = f"lam{value}/C34-label.npz"
        outputs[name] = write(root, name, data)
        for extra in ("c28.npz", "c33.npz", "c30.npz", "diagnostics.json"):
            name = f"lam{value}/{extra}"
            outputs[name] = write(root, name, b"not inspected fixture")
    summary = dict(
        stage=c.PRODUCER_STAGE,
        status="ok",
        process_interval=list(c.PROCESS),
        science_interval=list(c.SCIENCE),
        all7_lambdas=True,
        stage_a_calls=3,
        c25_whole_window_calls=7,
        manual_reinitializations=0,
        shared_seed=False,
        final_future_horizon=c.PROCESS[1],
        results=results,
    )
    outputs["summary.json"] = write(root, "summary.json", summary)
    seal = dict(
        stage=c.PRODUCER_STAGE,
        status="ok",
        outputs=outputs,
        inputs={**sources, **{r["path"]: r["sha256"] for r in records}},
    )
    row = write(root, "replay.provenance.json", seal)
    return SimpleNamespace(
        root=root,
        seal=seal,
        plan=plan,
        summary=summary,
        replay_sha=row["sha256"],
        references=dict(binding=binding),
    )


def test_complete_fixture_executes_all7_with_fixed_denominators_and_no_old_payloads(
    completed_fixture, monkeypatch
):
    fixture = completed_fixture
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    calls = []
    real_load = c.load_rows

    def capture(*args):
        calls.append(args[0]["path"])
        return real_load(*args)

    monkeypatch.setattr(c, "load_rows", capture)
    output = c.WORK / "evaluation"
    c.execute(
        SimpleNamespace(
            output=output, execute=True, producer_root=fixture.root, replay_sha=fixture.replay_sha
        )
    )
    summary = json.loads((output / "summary.json").read_text())
    assert len(calls) == 7 and all("bounded_producer" in path for path in calls)
    assert [r["lambda_value"] for r in summary["results_table"]] == list(c.LAMBDAS)
    for i, row in enumerate(summary["results_table"]):
        assert row["source_association_only"]["available_anchors"] == 57
        assert row["c34_label_available"]["available_anchors"] == 57 - i
        assert sum(row["c34_label_available"]["interval_counts"].values()) == 56
    assert summary["shared_seed"] is False and summary["physical_all3_ID_truth_validated"] is False
    seal = json.loads((output / "evaluation.provenance.json").read_text())
    assert seal["status"] == "ok"


@pytest.mark.parametrize(
    "change",
    [
        "partial",
        "wrong_stage",
        "reset",
        "shared_seed",
        "wrong_stop",
        "missing_lambda",
        "wrong_SL",
        "missing_input_edge",
        "wrong_producer",
        "bad_dtype_name",
    ],
)
def test_header_contract_rejects_unqualified_replay(completed_fixture, change):
    f = completed_fixture
    plan, summary, seal = copy.deepcopy((f.plan, f.summary, f.seal))
    if change == "partial":
        seal["status"] = "running"
    elif change == "wrong_stage":
        summary["stage"] = c.old.STAGE
    elif change == "reset":
        summary["manual_reinitializations"] = 1
    elif change == "shared_seed":
        plan["initial_seed_shared"] = True
    elif change == "wrong_stop":
        summary["final_future_horizon"] = 116201
    elif change == "missing_lambda":
        summary["results"].pop("lam8")
    elif change == "wrong_SL":
        path = next(iter(f.references["binding"]["matched_SL_references"]))
        seal["inputs"][path] = "f" * 64
    elif change == "missing_input_edge":
        seal["inputs"].pop(f.plan["inputs"]["records"][0]["path"])
    elif change == "wrong_producer":
        seal["inputs"][str(c.PRODUCER.resolve())] = "f" * 64
    else:
        summary["results"]["lam0"]["array_signatures"]["source"]["dtype"] = "|i1"
    with pytest.raises(ValueError):
        c.validate_producer_headers(plan, summary, seal, f.references)


def test_archive_reads_only3_fields_and_checks_producer_dtype_not_oldproof(
    completed_fixture, monkeypatch
):
    f = completed_fixture
    seen = []
    cls = np.lib.npyio.NpzFile
    original = cls.__getitem__

    def inspect(self, key):
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(cls, "__getitem__", inspect)
    ref = f.seal["outputs"]["lam0/C34-label.npz"]
    rows, capture = c.load_rows(ref, f.summary["results"]["lam0"]["array_signatures"], requested())
    assert seen == list(c.DECODE_FIELDS)
    assert capture["independently_checked_signatures"]["source"]["dtype"] == "int8"
    assert c.old.proof.signature(rows["source"])["dtype"] == "|i1"


def test_disabled_cannot_load_inputs(monkeypatch):
    monkeypatch.setattr(c, "load_context", lambda *a: pytest.fail("input read reached"))
    with pytest.raises(ValueError, match="disabled"):
        c.execute(SimpleNamespace(output=c.WORK / "out", execute=True))
