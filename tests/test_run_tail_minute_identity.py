"""New minute adapter boundaries; frozen DP arithmetic is not retested broadly."""

import argparse
import copy
import dataclasses
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import run_tail_minute_identity as mod  # noqa: E402


def candidate(first=10, count=5):
    arrays = {}
    for key, (tail, kind) in mod.LAYOUT.items():
        dtype = {"i": np.int64, "f": np.float64, "b": bool, "U": "U9"}[kind]
        arrays[key] = np.zeros((count, *tail), dtype=dtype)
    arrays["frame_idx"] = np.arange(first, first + count, dtype=np.int64)
    arrays["observed"][:] = True
    arrays["body_supported"][:] = True
    arrays["identity_review"][:, 2] = True
    for slot in range(3):
        arrays["pose_xyz"][:, slot, :, 0] = slot * 10
        arrays["sources"][:, slot] = slot
    arrays["xyz"] = arrays["pose_xyz"][:, :, 1].copy()
    arrays["body_xyz"] = arrays["xyz"].copy()
    arrays["accepted_xyz"] = np.where(arrays["identity_review"][..., None], np.nan, arrays["xyz"])
    return arrays


def packed(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def test_support_preflight_only_two_arrays_and_native_prefix(monkeypatch):
    values = candidate()
    values["observed"][3, 0, 1] = False
    payload = packed(values)
    observed_keys = []
    original = np.lib.npyio.NpzFile.__getitem__

    def get(self, key):
        observed_keys.append(key)
        assert key in ("frame_idx", "observed"), "preflight decompressed a numerical geometry field"
        return original(self, key)

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", get)
    got, headers = mod.read_candidate(payload, support_only=True, first=10, start=12, stop=15)
    assert observed_keys == ["frame_idx", "observed"]
    assert len(headers) == 21
    np.testing.assert_array_equal(got["frame_idx"], [12, 13, 14])
    assert not got["observed"][1, 0, 1]
    np.testing.assert_array_equal(got["observed"], values["observed"][2:])


@pytest.mark.parametrize("fault", ["native", "missing_key", "shape", "object", "initial_missing"])
def test_candidate_boundary_rejections(fault):
    values = candidate()
    if fault == "native":
        values["frame_idx"] += 1
    elif fault == "missing_key":
        del values["body_xyz"]
    elif fault == "shape":
        values["sources"] = values["sources"][:, :, :, 0]
    elif fault == "object":
        values["proposal_kind"] = np.array([object()] * 5)
    else:
        values["observed"][2, 1, 1] = False
    with pytest.raises(ValueError):
        mod.read_candidate(packed(values), support_only=True, first=10, start=12, stop=15)


def test_solve_tiny_permutation_graph_links_review_and_all_arrays_immutable():
    values = candidate(count=4)
    # Raw geometric rows 0/1 exchange while the physical points remain fixed.
    for key in ("pose_xyz", "xyz", "body_xyz", "accepted_xyz", "sources"):
        values[key][2:, [0, 1]] = values[key][2:, [1, 0]]
    before = {k: mod.signature(v) for k, v in values.items()}
    count = mod.counter.reachable_counts(values["frame_idx"], values["observed"][:, :, 1])
    result, identity, links, changes, report = mod.solve_one(values, dataclasses.asdict(count), [])
    np.testing.assert_array_equal(result.permutation_by_id, [[0, 1, 2]] * 2 + [[1, 0, 2]] * 2)
    assert report["path_cost"] == 0
    assert report["P012_link_cost"] > 0
    assert report["changes"]["direct_involved"]["identity_frames"] == 4
    assert not changes["latent_only"].any()
    assert identity["raw_identity_review"][:, 2].all()
    assert np.isnan(identity["conservative_accepted_xyz"][:, 2]).all()
    assert not links["terminal_unsettled_gap"].any()
    assert before == {k: mod.signature(v) for k, v in values.items()}


def test_graph_count_mismatch_refused():
    values = candidate(count=2)
    with pytest.raises(ValueError, match="graph differs"):
        mod.solve_one(values, dict(state_counts=np.array([1, 7]), edge_counts=np.array([6])), [])


def test_changed_intervals_preserve_latent_and_direct_categories():
    frames = np.arange(90, 94, dtype=np.int64)
    before = np.tile(np.arange(3), (4, 1))
    after = before.copy()
    after[1:, :2] = [1, 0]
    direct = np.ones((4, 3), bool)
    direct[1:3, :2] = False
    arrays, report = mod.mapping_comparison(frames, before, after, direct)
    assert arrays["latent_only"][1:3, :2].all()
    assert arrays["direct_involved"][3, :2].all()
    assert report["latent_only"]["intervals"] == [
        dict(identity=i, start=91, stop=93, native_frames=2) for i in range(2)
    ]


def test_preflight_rejects_capacity_or_changed_source_binding():
    binding = dict(native_interval=[66400, 76360], source_pins={"new.py": "a"})
    plan = dict(schema=mod.STAGES["preflight"], binding=binding)
    summary = dict(
        schema=mod.STAGES["preflight"],
        status="ok",
        all_within_fixed_caps=True,
        methods={m: dict(completed=True) for m in mod.METHODS},
    )
    mod.validate_preflight(plan, summary, binding)
    bad = copy.deepcopy(summary)
    bad["all_within_fixed_caps"] = False
    with pytest.raises(ValueError, match="bounded preflight"):
        mod.validate_preflight(plan, bad, binding)
    with pytest.raises(ValueError, match="exact same source"):
        mod.validate_preflight(plan, summary, {**binding, "source_pins": {"new.py": "b"}})


def test_female_endpoint_row_mapping_retains_review():
    values = candidate(count=2)
    anchors = [
        dict(frame=t, detection_rows={"top_cam": 1}, pixels={}, match_errors_px={})
        for t in (10, 11)
    ]
    raw = mod.female_scores(
        values["frame_idx"],
        values["sources"],
        values["observed"],
        values["identity_review"],
        anchors,
    )
    assert raw["raw"]["interval_counts"] == dict(preserved=1, swap=0, unscored=0)
    values["identity_review"][1, 1] = True
    held = mod.female_scores(
        values["frame_idx"],
        values["sources"],
        values["observed"],
        values["identity_review"],
        anchors,
    )
    assert held["raw"] == raw["raw"]
    assert held["with_original_review"]["interval_counts"]["unscored"] == 1


def test_legacy_manifest_named_path_and_input_shapes(tmp_path):
    payload = b"success"
    path = tmp_path / "candidate.npz"
    path.write_bytes(payload)
    seal = dict(
        inputs={str(path.resolve()): dict(sha256=mod.sha(payload))},
        outputs={
            "arbitrary_producer_role": dict(
                path=str(path), size=len(payload), sha256=mod.sha(payload)
            )
        },
    )
    evidence = mod.Evidence()
    mod.input_member(seal, path, mod.sha(payload))
    assert mod.named_read(seal, tmp_path, "candidate.npz", evidence) == payload
    seal["outputs"]["duplicate"] = dict(seal["outputs"]["arbitrary_producer_role"])
    assert mod.named_read(seal, tmp_path, "candidate.npz", evidence) == payload
    seal["outputs"]["duplicate"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="conflicting named"):
        mod.named_read(seal, tmp_path, "candidate.npz", evidence)


@pytest.mark.requires_artifacts(
    mod.COMPARISON / "comparison.provenance.json",
    mod.FEMALE / "evaluation.provenance.json",
    *(run["root"] / run["seal"] for run in mod.RUNS.values()),
)
def test_actual_small_json_contract_has_no_npz_or_solver(monkeypatch):
    original = Path.read_bytes

    def read(path):
        assert path.suffix in (".py", ".json"), "metadata probe may not read payload"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("actual NPZ prohibited"))
    monkeypatch.setattr(
        mod.memory, "solve_identity_memory_dp", lambda *a, **k: pytest.fail("DP prohibited")
    )
    evidence = mod.Evidence()
    binding, summary, seals, anchors = mod.load_inputs(evidence)
    assert len(seals) == 2 and len(anchors) == 11
    assert binding["candidate_references"]["tail"]["sha256"] == mod.RUNS["tail"]["candidate"]
    assert summary["all21_arrays_exact_backtrace"] is True
    assert not any(Path(p).suffix == ".npz" for p in evidence.inputs)


def test_preflight_mode_never_calls_solver_and_saves_cap_stop(tmp_path, monkeypatch):
    values = candidate(first=mod.FIRST, count=mod.STOP - mod.FIRST)
    # This fixture's call is replaced with an exact explicit stopped counter,
    # so it never enumerates a full real minute or loads existing arrays.
    stop = mod.counter.ReachableCount(
        False,
        values["frame_idx"][mod.START - mod.FIRST :],
        np.array([1], np.int64),
        np.empty(0, np.int64),
        1,
        0,
        dict(reason="maximum_total_states", attempted_frame=mod.START + 1),
        None,
    )
    monkeypatch.setattr(mod, "WORK", tmp_path)
    binding = dict(source_pins={})
    monkeypatch.setattr(
        mod, "load_inputs", lambda evidence: (binding, {}, dict(original={}, tail={}), [])
    )
    monkeypatch.setattr(mod, "named_read", lambda *args: packed(values))
    monkeypatch.setattr(mod.counter, "reachable_counts", lambda *args: stop)
    monkeypatch.setattr(
        mod.memory, "solve_identity_memory_dp", lambda *a, **k: pytest.fail("preflight solved DP")
    )
    out = tmp_path / "preflight"
    mod.run(argparse.Namespace(output=out, mode="preflight"))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "ok" and not summary["all_within_fixed_caps"]
    assert not summary["full_DP_executed"]
    assert json.loads((out / "preflight.provenance.json").read_text())["status"] == "ok"


def test_dataclass_archive_preserves_all_fields_and_nonfinite_scalars(tmp_path, monkeypatch):
    values = candidate(count=1)
    counted = mod.counter.reachable_counts(values["frame_idx"], values["observed"][:, :, 1])
    result, *_ = mod.solve_one(values, dataclasses.asdict(counted), [])
    monkeypatch.setattr(mod, "WORK", tmp_path)
    out = mod.Output(tmp_path / "output")
    mod.save_result(out, "original", result)
    scalars = json.loads((out.root / "original/dp_scalars.json").read_text())
    with np.load(out.root / "original/dp_arrays.npz", allow_pickle=False) as saved:
        assert set(saved.files) | set(scalars) == {f.name for f in dataclasses.fields(result)}
        for key in saved.files:
            assert mod.signature(saved[key]) == mod.signature(getattr(result, key))
    assert scalars["global_alternative_gap"] == {"nonfinite_float64": "+inf"}
