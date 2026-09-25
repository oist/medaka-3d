"""Synthetic saved-file boundaries; no real trajectory, image or model arrays."""

import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import qualify_temporal_prefix_deltas64 as m


def npz_bytes(arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def query_row(queries=2, first=333519):
    arrays = dict(
        xy=np.zeros((queries, 2), np.float32),
        mask_index=np.zeros(queries, np.int64),
        corner_index=np.arange(queries, dtype=np.int64),
        frame_idx=np.asarray(first, np.int64),
        available=np.asarray(True),
    )
    raw = npz_bytes(arrays)
    row = dict(
        view="top_cam",
        frame_idx=first,
        query_count=queries,
        sha256=m.sha_bytes(raw),
        array_signatures=m.codec.signatures(arrays),
        rows=[],
    )
    return arrays, raw, row


def saved_view():
    first, q = 333519, 2
    _, raw, row = query_row(q, first)
    files = {"top_cam/queries.npz": raw}
    prefixes = []
    ledger = m.core.EmissionLedger(64, q)
    for i, (lo, hi) in enumerate(m.WINDOWS):
        arrays = {}
        for key in m.codec.FIELDS:
            shape, dtype = m.codec.layout(key, hi, q)
            arrays[key] = np.zeros(shape, dtype)
            if i:
                arrays[key][: hi - 8] = prefixes[-1][key]
            arrays[key][lo:hi] = bool(i % 2) if key == "reported" else i / 8
        if i == 3:
            # Retain an old row NaN payload exactly, far outside the active window.
            arrays["product"][1, 0] = np.array([0x7FC00039], np.uint32).view(np.float32)[0]
            arrays["xy"][2, 0, 0] = -0.0
        meta = dict(
            call_index=i,
            input_relative=[lo, hi],
            input_native=[first + lo, first + hi],
            maximum_seen_native=first + hi - 1,
            output_prefix=hi,
            query_count=q,
            forward_called=True,
            status="ok",
            forward_seconds=0.123,
            extra={"full": [False, None, "preserved"]},
        )
        stem = "top_cam/online/call_" + str(i)
        files[stem + ".json"] = json.dumps(meta).encode()
        files[stem + ".npz"] = npz_bytes(arrays)
        prefixes.append(arrays)
        ledger.add(arrays["xy"], arrays["reported"], input_start=lo, input_stop=hi)
    emissions = ledger.arrays()
    emissions["not_rederived_extra"] = np.asarray(17, np.int64)
    files["top_cam/online_emissions.npz"] = npz_bytes(emissions)
    return dict(name="synthetic", first=first), row, files, prefixes


def test_disk_replay_uses_previous_reconstruction_and_exact_ledger(tmp_path):
    spec, row, files, originals = saved_view()
    output = m.Output(tmp_path / "new")
    requested = []

    def read(role):
        requested.append(role)
        return files[role]

    result = m.replay_view(spec, "top_cam", row, read, output)
    assert len(requested) == 16 and len(set(requested)) == 16
    assert len(result["calls"]) == 7 and result["ledger_fields_exact"]
    old_change = result["calls"][3]["outside_changed_rows"]
    assert old_change["product"] == dict(relative=[1], native=[333520])
    assert old_change["xy"] == dict(relative=[2], native=[333521])
    previous = None
    for i in range(7):
        root = output.root / "synthetic/top_cam"
        header = m.parse_json((root / f"call_{i}.json").read_bytes())
        payload = m.arrays_from_bytes((root / f"call_{i}.npz").read_bytes())
        previous = m.codec.decode_delta(previous, header, payload)
        m.exact_arrays(previous, originals[i], m.codec.FIELDS)
    output.guard()
    replay = m.parse_json((output.root / "synthetic/top_cam/replay.json").read_bytes())
    assert replay["saved_emission_keys_not_rederived"] == ["not_rederived_extra"]


@pytest.mark.parametrize("kind", ["origin", "order", "queries", "ledger", "query_bytes"])
def test_replay_refuses_cross_call_or_ledger_binding_drift(tmp_path, kind):
    spec, row, files, _ = saved_view()
    if kind in ("origin", "order", "queries"):
        role = "top_cam/online/call_2.json"
        meta = m.parse_json(files[role])
        if kind == "origin":
            meta["input_native"] = [x + 100 for x in meta["input_native"]]
            meta["maximum_seen_native"] += 100
        elif kind == "order":
            meta["call_index"] = 1
        else:
            meta["query_count"] = 1
        files[role] = json.dumps(meta).encode()
    elif kind == "ledger":
        role = "top_cam/online_emissions.npz"
        data = m.arrays_from_bytes(files[role])
        data["first_horizon"][0] += 1
        files[role] = npz_bytes(data)
    else:
        data = m.arrays_from_bytes(files["top_cam/queries.npz"])
        data["mask_index"][0] = 99
        files["top_cam/queries.npz"] = npz_bytes(data)
    with pytest.raises(ValueError):
        m.replay_view(spec, "top_cam", row, files.__getitem__, m.Output(tmp_path / "bad"))


@pytest.mark.parametrize("kind", ["path", "size", "sha", "escape"])
def test_named_output_requires_exact_role_size_path_and_sha(tmp_path, kind):
    path = tmp_path / "one.json"
    path.write_bytes(b"{}")
    role = "one.json"
    entry = dict(path=str(path), size_bytes=2, sha256=m.file_sha(path))
    if kind == "path":
        entry["path"] = str(tmp_path / "different.json")
    elif kind == "size":
        entry["size_bytes"] = 3
    elif kind == "sha":
        entry["sha256"] = "0" * 64
    else:
        role = "../one.json"
    with pytest.raises(ValueError):
        m.named_bytes(tmp_path, dict(outputs={role: entry}), role, m.Evidence())


def test_selected_output_reader_does_not_open_unrelated_outputs(tmp_path):
    path = tmp_path / "wanted.json"
    path.write_bytes(b"{}")
    entry = dict(path=str(path), size_bytes=2, sha256=m.file_sha(path))
    evidence = m.Evidence()
    seal = dict(outputs={"wanted.json": entry, "unused_image.png": dict(path="not present")})
    assert m.named_bytes(tmp_path, seal, "wanted.json", evidence) == b"{}"
    assert set(evidence.inputs) == {str(path.resolve())}
    path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="input changed"):
        evidence.guard()


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_nonfinite_or_duplicate_metadata_rejected(raw):
    with pytest.raises(ValueError):
        m.parse_json(raw)


def test_query_logical_signature_prevents_relabel_even_with_updated_npz_hash():
    arrays, _, row = query_row()
    arrays["mask_index"][0] = 4
    row["sha256"] = m.sha_bytes(npz_bytes(arrays))
    with pytest.raises(ValueError, match="logical bytes"):
        m.validate_queries(arrays, row, 333519, 2)


def test_output_guard_requires_exact_membership_and_bytes(tmp_path):
    output = m.Output(tmp_path / "new")
    path = output.json("value.json", dict(v=1))
    output.guard()
    (output.root / "unexpected").write_bytes(b"x")
    with pytest.raises(ValueError, match="membership"):
        output.guard()
    (output.root / "unexpected").unlink()
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed"):
        output.guard()


def test_failure_seal_retains_partials_and_does_not_claim_success(tmp_path, monkeypatch):
    def stop(spec, evidence):
        raise ValueError("synthetic source binding failure")

    monkeypatch.setattr(m, "load_dataset", stop)
    monkeypatch.setattr(m, "PINS", {})
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="synthetic source"):
        m.run(destination)
    seal = m.parse_json((destination / "qualification.provenance.json").read_bytes())
    assert seal["status"] == "failed"
    assert "failure.json" in seal["outputs"] and "plan.json" in seal["outputs"]
    assert "summary.json" not in seal["outputs"]
    assert set(seal["outputs"]) == {
        p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
    } - {"qualification.provenance.json"}


def test_real_small_json_scope_contract_only():
    # Reads selected original source/JSON metadata, never an NPZ/image or model.
    for spec in m.DATASETS:
        evidence = m.Evidence()
        root, seal, plan = m.load_dataset(spec, evidence)
        for i, view in enumerate(m.VIEWS):
            row = plan["all_query_rows"][i]
            assert row["sha256"] == seal["outputs"][view + "/queries.npz"]["sha256"]
            for index in range(7):
                role = view + f"/online/call_{index}.json"
                meta = m.parse_json(m.named_bytes(root, seal, role, evidence))
                m.validate_call(meta, spec["first"], index, spec["queries"][i])
        assert all(not path.endswith(".npz") for path in evidence.inputs)


def test_scope_refuses_query_group_reordering():
    spec = m.DATASETS[0]
    root = m.WORK / spec["name"]
    plan = m.parse_json((root / "plan.json").read_bytes())
    summary = m.parse_json((root / "summary.json").read_bytes())
    plan = copy.deepcopy(plan)
    plan["all_query_rows"].reverse()
    with pytest.raises(ValueError, match="ordered query"):
        m.validate_scope(spec, plan, summary)


def test_successful_orchestration_seals_all_42_synthetic_calls(tmp_path, monkeypatch):
    spec, row, files, _ = saved_view()
    specs = [dict(spec, name=name) for name in ("fixture_a", "fixture_b")]
    rows = [dict(row, view=view) for view in m.VIEWS]
    monkeypatch.setattr(m, "DATASETS", specs)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(
        m, "load_dataset", lambda spec, evidence: (None, {}, dict(all_query_rows=rows))
    )

    def named(root, seal, role, evidence):
        return files["top_cam/" + role.split("/", 1)[1]]

    monkeypatch.setattr(m, "named_bytes", named)
    destination = tmp_path / "success"
    m.run(destination)
    seal = m.parse_json((destination / "qualification.provenance.json").read_bytes())
    summary = m.parse_json((destination / "summary.json").read_bytes())
    assert seal["status"] == "ok" and seal["final_guards_passed"] is True
    assert summary["calls"] == 42 and summary["logical_prefix_fields"] == 252
    assert summary["model_forwards"] == 0 and summary["GPU_used"] is False
    assert len(summary["view_results"]) == 6
    assert set(seal["outputs"]) == {
        p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file()
    } - {"qualification.provenance.json"}
    for item in seal["outputs"].values():
        assert m.file_sha(item["path"]) == item["sha256"]
        assert Path(item["path"]).stat().st_size == item["size_bytes"]
