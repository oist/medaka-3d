"""New metadata/rig/background edges only; all checkpoints here are tiny fakes."""

import copy
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import receipted_final_inputs as a  # noqa: E402


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, expected=None):
        path = Path(path).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or digest == expected
        assert self.paths.get(str(path), digest) == digest
        self.paths[str(path)] = digest
        return digest


def written(path, value, evidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(a.boundary.codec._canonical(value))
    digest = evidence.add(path)
    return dict(path=str(path), sha256=digest, size=path.stat().st_size)


def endpoint_fixture(tmp_path):
    e = Evidence()
    old = dict(code_sha256={f"old{i}": str(i) for i in range(36)})
    new = dict(code_sha256={f"new{i}": str(i) for i in range(40)})
    rows, start = [], a.FIRST
    for i, stop in enumerate(a.boundary.endpoint_stops()):
        run = tmp_path / f"run{i}"
        binding = old if i <= 10 else new
        metadata = dict(
            processed_frames=stop - a.FIRST,
            previous_frame=stop - 1,
            initialize_count=1,
            restore_count=i,
        )
        meta = written(run / "checkpoint/metadata.json", metadata, e)
        # This is deliberately not an NPZ: this adapter must never deserialize it.
        payload = written(run / "checkpoint/arrays.npz", {"fixture": i}, e)
        manifest = dict(
            schema=a.boundary.codec.SCHEMA,
            status="ok",
            bindings=binding,
            bindings_sha256=a.boundary.codec._sha(a.boundary.codec._canonical(binding)),
            metadata=dict(file="metadata.json", bytes=meta["size"], sha256=meta["sha256"]),
            payload=dict(file="arrays.npz", bytes=payload["size"], sha256=payload["sha256"]),
        )
        man = written(run / "checkpoint/manifest.json", manifest, e)
        record = dict(
            stage="three_body_long_continuation" if i <= 10 else a.chain_api.m.STAGE,
            status="ok",
            params=dict(through_frame=stop, bindings=binding),
            outputs={"checkpoint_manifest": man, meta["path"]: meta, payload["path"]: payload},
        )
        seal = written(run / "continuation.provenance.json", record, e)
        rows.append(
            dict(run=str(run), interval=[start, stop], restores=i, seal_sha256=seal["sha256"])
        )
        start = stop
    chain = dict(
        chain_version="old_prefix_then_receipted_child_v1",
        final_audit_complete=False,
        ancestors=rows[:4],
        phases=rows[4:],
        prefix=dict(binding=old, prefix_counts=dict(checkpoint_restores=10)),
        binding=new,
        final_run=Path(rows[-1]["run"]),
    )
    return chain, e


def test_all33_own_binding_and_no_payload_decode(tmp_path):
    chain, e = endpoint_fixture(tmp_path)
    refs = a.make_endpoint_refs(chain, e)
    assert [r["stop"] for r in refs] == list(a.boundary.endpoint_stops())
    assert [r["restore_count"] for r in refs] == list(range(33))
    assert [len(r["bindings"]["code_sha256"]) for r in refs] == [36] * 11 + [40] * 22


@pytest.mark.parametrize(
    "kind", ["shuffle", "duplicate", "binding", "missing_proof", "role", "final"]
)
def test_endpoint_edges_reject(tmp_path, kind):
    chain, e = endpoint_fixture(tmp_path)
    if kind == "shuffle":
        chain["phases"][1:3] = reversed(chain["phases"][1:3])
    elif kind == "duplicate":
        chain["phases"][1]["run"] = chain["phases"][0]["run"]
    elif kind == "binding":
        chain["prefix"]["binding"]["code_sha256"]["old0"] = "changed"
    elif kind == "missing_proof":
        del e.paths[str(Path(chain["ancestors"][0]["run"]) / "checkpoint/arrays.npz")]
    elif kind == "final":
        chain["final_run"] = Path(chain["phases"][-2]["run"])
    else:
        path = Path(chain["ancestors"][0]["run"]) / "continuation.provenance.json"
        d = json.loads(path.read_bytes())
        d["outputs"]["wrong_role"] = d["outputs"].pop("checkpoint_manifest")
        del e.paths[str(path)]
        written(path, d, e)
    with pytest.raises((ValueError, AssertionError)):
        a.make_endpoint_refs(chain, e)


@dataclasses.dataclass
class Camera:
    view: str


def geometry_fixture(tmp_path, monkeypatch):
    e = Evidence()
    models = []
    shared = {}
    for view in a.VIEWS:
        p = tmp_path / f"{view}.json"
        shared[str(p)] = written(p, {}, e)["sha256"]
        models.append(dict(path=p.name))
    p = tmp_path / "calibration_manifest.json"
    shared[str(p)] = written(p, dict(model_files=models), e)["sha256"]
    nodes = {}
    for view in a.VIEWS:
        p = tmp_path / f"{a.chain_api.native.EXPERIMENT}__{view}.npz"
        shared[str(p)] = written(p, {}, e)["sha256"]
        nodes[view] = p
    rig = SimpleNamespace(cameras=[Camera(v) for v in a.VIEWS], views=a.VIEWS, world_units="mm")
    context = dict(
        cameras=a.COMMON.normalize([dataclasses.asdict(c) for c in rig.cameras]),
        views=list(a.VIEWS),
        world_units="mm",
    )
    monkeypatch.setattr(a, "load_rig", lambda p: rig)
    return shared, context, nodes, e, rig


def test_registered_rig_no_second_SL_loader(tmp_path, monkeypatch):
    shared, c, n, e, rig = geometry_fixture(tmp_path, monkeypatch)
    result, proof = a.bind_rig(shared, c, n, e)
    assert result is rig and proof["SL_reloaded"] is False


@pytest.mark.parametrize("kind", ["escaped_model", "missing_view", "context", "units"])
def test_bad_geometry_binding(tmp_path, monkeypatch, kind):
    shared, c, n, e, rig = geometry_fixture(tmp_path, monkeypatch)
    if kind == "missing_view":
        n.pop(a.VIEWS[0])
    elif kind == "context":
        c["cameras"][0]["view"] = "different"
    elif kind == "units":
        rig.world_units = "cm"
    else:
        p = tmp_path / "calibration_manifest.json"
        del e.paths[str(p)]
        shared[str(p)] = written(p, dict(model_files=[dict(path="../escaped.json")] * 3), e)[
            "sha256"
        ]
    with pytest.raises((ValueError, KeyError)):
        a.bind_rig(shared, c, n, e)


def background_fixture(tmp_path, shape=(1200, 1200, 3)):
    e = Evidence()
    records = {}
    expected = {}
    for i, v in enumerate(a.VIEWS):
        img = np.full(shape, i * 30, np.uint8)
        ok, raw = cv2.imencode(".png", img)
        assert ok
        p = tmp_path / (v + ".png")
        p.write_bytes(raw.tobytes())
        expected[v] = e.add(p)
        records[v] = dict(path=p.name, sha256=expected[v])
    chunks = [
        dict(root=tmp_path, manifest=dict(backgrounds=copy.deepcopy(records))) for _ in range(180)
    ]
    return chunks, expected, e


def test_actual_encoded_background_decode_proof(tmp_path):
    chunks, expected, e = background_fixture(tmp_path)
    arrays, records, proof = a.decode_backgrounds(chunks, expected, e)
    assert all(proof[v]["decoded_from_encoded_verified_here"] for v in a.VIEWS)
    inherited = a.boundary.background_binding(arrays, records)
    assert all(inherited[v]["decoded_from_encoded_verified_here"] is False for v in a.VIEWS)
    assert np.all(arrays["side1_cam"] == 30)


@pytest.mark.parametrize("kind", ["changed_chunk", "shape", "changed_encoded"])
def test_bad_background_input(tmp_path, kind):
    chunks, expected, e = background_fixture(
        tmp_path, shape=(2, 2, 3) if kind == "shape" else (1200, 1200, 3)
    )
    if kind == "changed_chunk":
        chunks[-1]["manifest"]["backgrounds"]["top_cam"]["sha256"] = "0" * 64
    if kind == "changed_encoded":
        (tmp_path / "top_cam.png").write_bytes(b"changed")
    with pytest.raises((ValueError, AssertionError)):
        a.decode_backgrounds(chunks, expected, e)


@pytest.mark.requires_artifacts(
    ROOT
    / "work/three_body/long_remaining_receipted_supervisor_v1/phases"
    / "f126160-136120/tracking/continuation.provenance.json"
)
def test_existing_completed_metadata_roles_only():
    """Read sealed small JSON and payload stat only; never hash/open long NPZ."""
    root = ROOT / (
        "work/three_body/long_remaining_receipted_supervisor_v1/phases/f126160-136120/tracking"
    )
    record = json.loads((root / "continuation.provenance.json").read_bytes())
    manifest = json.loads((root / "checkpoint/manifest.json").read_bytes())
    a.output_record(record, root / "checkpoint/manifest.json", role="checkpoint_manifest")
    assert "checkpoint_metadata" not in record["outputs"]
    for key in ("metadata", "payload"):
        spec = manifest[key]
        assert (
            a.output_record(record, root / "checkpoint" / spec["file"], expected=spec["sha256"])
            == spec["sha256"]
        )
    assert len(record["params"]["bindings"]["code_sha256"]) == 36
    assert all(Path(p).is_absolute() for p in record["params"]["bindings"]["code_sha256"])


def header_fixture(tmp_path, change=None):
    e = Evidence()
    summary = dict(
        schema=a.chain_api.SUPERVISOR_STAGE,
        status="complete",
        through=a.STOP,
        full_sequence_complete=True,
        prefix_backtrace_provisional=False,
        logical_initialization_count=1,
        checkpoint_restore_count=32,
        external_pose_reset_count=0,
        active_beam_count=6,
        tracking_processes=33,
        consumed_chunk_count=180,
        stop_request=None,
        incomplete_phase=None,
        independent_ID_truth=False,
        final_audit_complete=False,
        detector_started=False,
        old_process_stopped=False,
        next_phase_retried=False,
        phases=[{}] * 28,
    )
    if change == "incomplete":
        summary["full_sequence_complete"] = False
    if change == "bool_count":
        summary["logical_initialization_count"] = True
    handoff = dict(old_completed_phases=0, initial_parent="synthetic")
    policy = dict(
        schema=a.chain_api.SUPERVISOR_STAGE,
        initial_parent="synthetic",
        initial_stop=76360,
        initial_restores=4,
        target_interval=[a.FIRST, a.STOP],
        phase_intervals=[list(p) for p in a.chain_api.expected_new_phases()],
        new_external_pose_seed=False,
        failed_phase_retry=False,
        scheduling_changes_scientific_protocol=False,
    )
    outputs = {
        role: written(tmp_path / name, data, e)
        for role, name, data in (
            ("summary", "summary.json", summary),
            ("plan", "supervisor_plan.json", policy),
            ("handoff", "handoff.json", handoff),
        )
    }
    header = dict(
        stage=a.chain_api.SUPERVISOR_STAGE,
        status="ok",
        finished_at="2026-09-10T00:00:00+00:00",
        outputs=outputs,
    )
    if change == "no_finish":
        header.pop("finished_at")
    if change == "wrong_role":
        header["outputs"]["wrong"] = header["outputs"].pop("summary")
    sealed = written(tmp_path / "supervisor.provenance.json", header, e)
    return sealed["sha256"], e


def test_final_header_uses_only_small_named_json(tmp_path):
    digest, e = header_fixture(tmp_path)
    assert a.completed_header(tmp_path, digest, e)["status"] == "ok"
    assert len(e.paths) == 4


@pytest.mark.parametrize("change", ["incomplete", "bool_count", "no_finish", "wrong_role"])
def test_incomplete_or_untyped_final_header(tmp_path, change):
    digest, e = header_fixture(tmp_path, change)
    with pytest.raises(ValueError):
        a.completed_header(tmp_path, digest, e)
