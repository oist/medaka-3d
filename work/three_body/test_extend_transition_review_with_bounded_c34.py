"""Synthetic-only full adapter tests; actual recording entrypoints are forbidden."""

import copy
import hashlib
import json
from types import SimpleNamespace

import extend_transition_review_with_bounded_c34 as c
import numpy as np
import pytest
from test_compare_transition_review_source_mouths import fixture as review_fixture
from test_score_classical_window_female_anchors import full_arrays


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    work = repo / "work/three_body"
    work.mkdir(parents=True)
    for module in (c, c.common, c.window, c.common.upstream, c.window.old):
        monkeypatch.setattr(module, "REPO", repo)
        monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(c.proof, "REPO", repo)
    monkeypatch.setattr(c, "EXECUTION_READY", False)
    monkeypatch.setattr(c.window, "PRODUCER", work / "producer.py")
    for module, names in (
        (c.window, ("OLD_ROOT", "ANCHOR_ROOT")),
        (c.window.old, ("INPUT_ROOT", "PRIOR_ROOT")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, repo / ("forbidden_" + name))

    def forbidden(*args, **kwargs):
        pytest.fail("real/old payload or workflow loader is forbidden")

    for module, names in (
        (c.window, ("load_context", "load_reference_headers", "source_code", "execute")),
        (c.common, ("load_upstream", "registered_references", "load_mouths", "source_code", "run")),
        (c.window.old, ("load_headers", "load_anchor_rows", "source_code")),
        (c.common.upstream, ("load_headers", "load_rig_and_lineage", "load_imported_review")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if name.endswith(".npz") and isinstance(value, dict):
        np.savez_compressed(path, **value)
    else:
        path.write_bytes(value if isinstance(value, bytes) else c.proof.json_bytes(value))
    return dict(path=str(path.resolve()), size_bytes=path.stat().st_size, sha256=c.proof.sha(path))


def refresh(fixture, which, member=None):
    root, seal = getattr(fixture, which + "_root"), getattr(fixture, which + "_seal")
    if member is not None:
        seal["outputs"][member] = write(
            root, member, getattr(fixture, which + "_" + member.split(".")[0])
        )
    name = "comparison.provenance.json" if which == "common" else "replay.provenance.json"
    row = write(root, name, seal)
    setattr(fixture, which + "_sha", row["sha256"])


@pytest.fixture
def completed(monkeypatch):
    c.window.PRODUCER.write_text("# isolated synthetic producer\n", encoding="utf-8")
    monkeypatch.setattr(c.window, "PRODUCER_SHA", c.proof.sha(c.window.PRODUCER))
    monkeypatch.setattr(
        c, "source_code", lambda seal, summary, ev: ev.add(c.window.PRODUCER, c.window.PRODUCER_SHA)
    )
    parsed, source, caches = review_fixture()
    for cache in caches.values():
        cache["original_flat_row_ranges"] = np.array([[100, 103], [400, 403], [900, 903]], np.int64)
    records = [
        write(c.REPO / "registered", name, b"synthetic reference - must never decode")
        for name in (
            "catalog.json",
            "seed.json",
            *(v + ".npz" for v in c.VIEWS),
            "manifest.json",
            "cam0.json",
            "cam1.json",
            "cam2.json",
        )
    ]
    sl = {v: records[2 + i] for i, v in enumerate(c.VIEWS)}
    captures = {
        "SL/" + v: dict(
            **sl[v],
            archive_reads=1,
            numeric_fields_decoded=["frame_idx", "points"],
            score_decoded=False,
            selected_signatures={k: c.proof.signature(a) for k, a in caches[v].items()},
        )
        for v in c.VIEWS
    }
    result = c.common.score_stream(
        parsed, source, caches, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    results = {key: copy.deepcopy(result) for key in c.ORIGINAL_KEYS}
    # Retain method-specific fields verbatim rather than synthesizing an equivalent score.
    results[c.ORIGINAL_KEYS[-1]]["inherited_extra_marker"] = {
        "gap": None,
        "status": "not_retained_unknown",
    }
    common_summary = dict(
        schema=c.common.STAGE,
        status="ok",
        layer=c.common.LAYER,
        streams=13,
        requested_endpoints_per_stream=9,
        requested_links_per_stream=6,
        anchor_frames=parsed.anchor_frames.tolist(),
        results=results,
        original_review_sha256=hashlib.sha256(parsed.review_bytes).hexdigest(),
        all7_lambdas_scored=True,
        lambda_selected=False,
        all_three_physical_ID_truth=False,
        dense_swap_zero_certified=False,
        SL3_references=sl,
        inherited_lineage={
            "shared_SL_calibration_sha256": {r["path"]: r["sha256"] for r in records[2:]}
        },
        unchanged_3D_primary_reference={
            "path": str(c.WORK / "unread_3D/summary.json"),
            "sha256": "a" * 64,
        },
        inherited_identity_bindings={"synthetic_only": True},
        inherited_conditional_ID_ambiguity={"unknown": None},
        source_sha256={},
    )
    common_root, producer_root = c.WORK / "common", c.WORK / "bounded"
    outputs = {
        name: write(common_root, name, value)
        for name, value in (
            ("summary.json", common_summary),
            ("original_review.json", parsed.review_bytes),
            ("source_manifest.json", parsed.manifest_bytes),
            ("captured_new_inputs.json", captures),
        )
    }
    for v in c.VIEWS:
        name = f"SL_anchors/{v}.npz"
        outputs[name] = write(common_root, name, caches[v])
    for lam in c.window.LAMBDAS:
        name = f"C34_anchors/lambda_{lam}.npz"
        outputs[name] = write(common_root, name, b"unread old C34 placeholder")
    common_seal = dict(
        stage=c.common.STAGE,
        status="ok",
        inputs={r["path"]: r["sha256"] for r in sl.values()},
        outputs=outputs,
    )
    sources = {str(c.window.PRODUCER.resolve()): c.window.PRODUCER_SHA}
    producer_plan = dict(
        schema=c.window.PRODUCER_STAGE,
        process_interval=list(c.window.PROCESS),
        science_interval=list(c.window.SCIENCE),
        process_frames=66400,
        lambdas=[float(v) for v in c.window.LAMBDAS],
        initial_seed_shared=False,
        initialization="unconstrained C25 first-frame costs",
        future_horizon_exclusive=116200,
        time_chunking=False,
        outside_observations_used=False,
        original_C25_objective_preserved=True,
        transition_cache_identity_bug_corrected=True,
        source_binding={"sources": sources},
        inputs=dict(
            records=records,
            seed_reference={"frame": 49800},
            selected={
                v: dict(
                    native_interval=list(c.window.PROCESS),
                    full_native_index_read=True,
                    outside_points_scores_deserialized=False,
                )
                for v in c.VIEWS
            },
        ),
    )
    outputs = {"plan.json": write(producer_root, "plan.json", producer_plan)}
    for name in ("stage_a.json", "stage_a.npz"):
        outputs[name] = write(producer_root, name, b"unread stage A placeholder")
    producer_results = {}
    for i, lam in enumerate(c.window.LAMBDAS):
        data = full_arrays()
        data["source"][parsed.anchor_frames - 49800] = source
        data["slot_track"][parsed.anchor_frames[: min(i, 3)] - 49800] = -1
        producer_results["lam" + lam] = {
            "array_signatures": {k: c.window.signature(a) for k, a in data.items()}
        }
        name = f"lam{lam}/C34-label.npz"
        outputs[name] = write(producer_root, name, data)
        for extra in ("c28.npz", "c33.npz", "c30.npz", "diagnostics.json"):
            name = f"lam{lam}/{extra}"
            outputs[name] = write(producer_root, name, b"unread other-stage placeholder")
    producer_summary = dict(
        stage=c.window.PRODUCER_STAGE,
        status="ok",
        process_interval=list(c.window.PROCESS),
        science_interval=list(c.window.SCIENCE),
        all7_lambdas=True,
        stage_a_calls=3,
        c25_whole_window_calls=7,
        manual_reinitializations=0,
        shared_seed=False,
        final_future_horizon=116200,
        results=producer_results,
    )
    outputs["summary.json"] = write(producer_root, "summary.json", producer_summary)
    producer_seal = dict(
        stage=c.window.PRODUCER_STAGE,
        status="ok",
        inputs={**sources, **{r["path"]: r["sha256"] for r in records}},
        outputs=outputs,
    )
    f = SimpleNamespace(
        common_root=common_root,
        common_seal=common_seal,
        common_summary=common_summary,
        common_captured_new_inputs=captures,
        producer_root=producer_root,
        producer_seal=producer_seal,
        producer_plan=producer_plan,
        producer_summary=producer_summary,
        parsed=parsed,
        source=source,
        caches=caches,
    )
    refresh(f, "common")
    refresh(f, "producer")
    return f


def load(f):
    return c.load_context(
        f.common_root, f.common_sha, f.producer_root, f.producer_sha, c.proof.Evidence()
    )


def test_full_synthetic_path_preserves13_adds7_native_9_by6(completed, monkeypatch):
    f = completed
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    calls, real = [], c.window.load_rows

    def capture(ref, descriptors, frames):
        calls.append(ref["path"])
        return real(ref, descriptors, frames)

    monkeypatch.setattr(c.window, "load_rows", capture)
    out = c.WORK / "extension"
    c.execute(
        SimpleNamespace(
            output=out,
            execute=True,
            common_root=f.common_root,
            common_sha=f.common_sha,
            producer_root=f.producer_root,
            replay_sha=f.producer_sha,
        )
    )
    result = json.loads((out / "summary.json").read_text())
    assert len(calls) == 7 and all(str(f.producer_root) in p for p in calls)
    assert result["streams"] == 20 and len(result["results"]) == 20
    assert {k: result["results"][k] for k in c.ORIGINAL_KEYS} == f.common_summary["results"]
    assert json.loads((out / "original_common_summary.json").read_text()) == f.common_summary
    assert (out / "original_review.json").read_bytes() == f.parsed.review_bytes
    assert (out / "source_manifest.json").read_bytes() == f.parsed.manifest_bytes
    for i, lam in enumerate(c.window.LAMBDAS):
        r = result["results"]["bounded_C34_lambda_" + lam]
        assert r["source_rows"] == f.source.tolist()
        assert r["scores"][c.common.LAYER]["matched_anchors"] == 9
        assert r["scores"]["C34_saved_label_available"]["matched_anchors"] == 9 - 3 * min(i, 3)
        for score in r["scores"].values():
            assert score["requested_anchors"] == 9 and score["requested_animal_intervals"] == 6
            assert sum(score["interval_counts"].values()) == 6
    captures = json.loads((out / "captured_bounded_inputs.json").read_text())
    assert captures["0"]["selected_array_rows"] == (f.parsed.anchor_frames - 49800).tolist()
    assert captures["0"]["decoded_fields"] == ["frame_idx", "source", "slot_track"]
    assert (
        result["unchanged_3D_primary_reference"]
        == f.common_summary["unchanged_3D_primary_reference"]
    )
    assert not result["shared_seed"] and not result["all_three_physical_ID_truth"]
    seal = json.loads((out / "comparison.provenance.json").read_text())
    assert seal["status"] == "ok" and len(seal["outputs"]) == 12


@pytest.mark.parametrize(
    "change",
    [
        "common_failed",
        "replay_failed",
        "missing_lambda",
        "wrong_SL",
        "wrong_horizon",
        "manual_reset",
    ],
)
def test_all_success_gates_precede_any_npz_decode(completed, monkeypatch, change):
    f = completed
    if change == "common_failed":
        f.common_seal["status"] = "running"
        refresh(f, "common")
    elif change == "replay_failed":
        f.producer_seal["status"] = "running"
        refresh(f, "producer")
    elif change == "missing_lambda":
        del f.producer_seal["outputs"]["lam8/C34-label.npz"]
        refresh(f, "producer")
    elif change == "wrong_SL":
        ref = f.common_summary["SL3_references"][c.VIEWS[0]]
        f.producer_seal["inputs"][ref["path"]] = "b" * 64
        refresh(f, "producer")
    else:
        f.producer_summary[
            "final_future_horizon" if change == "wrong_horizon" else "manual_reinitializations"
        ] = 1
        refresh(f, "producer", "summary.json")
    monkeypatch.setattr(
        np, "load", lambda *a, **kw: pytest.fail("NPZ decoded before all success gates")
    )
    with pytest.raises(ValueError):
        load(f)


@pytest.mark.parametrize(
    "change",
    [
        "SL_capture",
        "source_assignment",
        "pixel_row",
        "review_bytes",
        "denominator",
        "cache_signature",
    ],
)
def test_common_cache_review_and_assignment_binding(completed, change):
    f = completed
    if change == "SL_capture":
        f.common_captured_new_inputs["SL/" + c.VIEWS[0]]["sha256"] = "b" * 64
        refresh(f, "common", "captured_new_inputs.json")
    elif change == "cache_signature":
        f.caches[c.VIEWS[0]]["mouth_uv"][0, 0] += 1
        name = f"SL_anchors/{c.VIEWS[0]}.npz"
        f.common_seal["outputs"][name] = write(f.common_root, name, f.caches[c.VIEWS[0]])
        refresh(f, "common")
    else:
        r = f.common_summary["results"][c.ORIGINAL_KEYS[0]]
        if change == "source_assignment":
            r["source_rows"][0][0][0] = 1
        elif change == "pixel_row":
            r["assigned_SL_mouth_pixels"][0][0][0][0] += 1
        elif change == "review_bytes":
            f.common_summary["original_review_sha256"] = "b" * 64
        else:
            r["scores"][c.common.LAYER]["requested_anchors"] = 8
        refresh(f, "common", "summary.json")
    with pytest.raises(ValueError):
        load(f)


@pytest.mark.parametrize("flag", ["EXECUTION_ENABLED", "EXECUTION_READY"])
@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_only_real_assignment_can_change(flag, newline):
    data = (
        newline.join([(flag + " = False").encode(), ('EXAMPLE = "' + flag + ' = False"').encode()])
        + newline
    )
    expected = hashlib.sha256(data).hexdigest()
    actual = data.replace(b"= False", b"= True", 1)
    assert c.flag_digest(actual, flag, expected) == hashlib.sha256(actual).hexdigest()
    assert (
        c.flag_digest(actual, flag, expected, require_enabled=True)
        == hashlib.sha256(actual).hexdigest()
    )
    assert c.flag_digest(data, flag, expected) == expected
    with pytest.raises(ValueError, match="requires enabled assignment"):
        c.flag_digest(data, flag, expected, require_enabled=True)
    for bad in (actual.replace(b"EXAMPLE", b"OTHER"), data + data, b"# " + data):
        with pytest.raises(ValueError):
            c.flag_digest(bad, flag, expected)


@pytest.mark.parametrize("change", ["out_of_frame", "duplicate", "missing"])
def test_assigned_rows_are_not_clamped_or_prefiltered(change):
    parsed, source, caches = review_fixture()
    available = np.ones((3, 3), bool)
    if change == "out_of_frame":
        source[0, 0, 0] = 3
        with pytest.raises(ValueError, match="outside native SL frame"):
            c.common.score_stream(
                parsed, source, caches, available, eligibility_name="C34_saved_label_available"
            )
        return
    if change == "duplicate":
        source[0, 1] = source[0, 0]
        available[0, 1] = False
    else:
        source[0, 0] = -1
    result = c.common.score_stream(
        parsed, source, caches, available, eligibility_name="C34_saved_label_available"
    )
    for score in result["scores"].values():
        assert score["anchors"][0]["animals"][0]["slot"] is None
        assert sum(score["interval_counts"].values()) == 6


def test_unknown_and_missing_middle_do_not_certify_the_interior():
    parsed, source, caches = review_fixture()
    document = copy.deepcopy(parsed.review)
    document["continuity"]["A"]["entry_to_after"] = "ambiguous"
    parsed = c.common.upstream.human.parse_review(
        c.proof.json_bytes(document), parsed.manifest_bytes, clip_range=c.common.upstream.CLIP
    )
    # Keep B's endpoints, remove only the middle source assignment in all views.
    source[1, 1] = -1
    result = c.common.score_stream(
        parsed, source, caches, np.ones((3, 3), bool), eligibility_name="C34_saved_label_available"
    )
    score = result["scores"][c.common.LAYER]
    assert score["requested_anchors"] == 9 and score["requested_animal_intervals"] == 6
    assert result["local_human_endpoint_links"]["counts"]["unscored"] == 3
    assert score["entry_exit_only_counts"]["same_as_entry"] == 2
    assert not result["interior_identity_certified"]


def test_disabled_and_external_output_guards_precede_input_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(c, "load_context", lambda *a, **kw: pytest.fail("input opened"))
    args = SimpleNamespace(output=tmp_path / "outside", execute=True)
    with pytest.raises(ValueError, match="disabled"):
        c.execute(args)
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    with pytest.raises(ValueError, match="fresh workspace"):
        c.execute(args)


def test_failure_preserved_without_retry_or_partial_success(completed, monkeypatch):
    f = completed
    monkeypatch.setattr(c, "EXECUTION_READY", True)
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("synthetic bound payload failure")

    monkeypatch.setattr(c.window, "load_rows", fail)
    output = c.WORK / "failure"
    with pytest.raises(ValueError, match="synthetic bound payload"):
        c.execute(
            SimpleNamespace(
                output=output,
                execute=True,
                common_root=f.common_root,
                common_sha=f.common_sha,
                producer_root=f.producer_root,
                replay_sha=f.producer_sha,
            )
        )
    assert len(calls) == 1 and not (output / "summary.json").exists()
    assert json.loads((output / "comparison.provenance.json").read_text())["status"] == "failed"
