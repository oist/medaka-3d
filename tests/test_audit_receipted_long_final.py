"""Finite caller orchestration with explicit synthetic helpers, never real payloads."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import audit_receipted_long_final as c  # noqa: E402


def fixture(tmp_path, monkeypatch, *, fault=None):
    events = []
    args = SimpleNamespace(
        output=str(tmp_path / "audit"),
        supervisor=str(tmp_path),
        expected_supervisor_sha256="a" * 64,
        owner_exit_evidence="owner.json",
        expected_owner_exit_sha256="b" * 64,
        d2_invocation_seal="d2.json",
        expected_d2_invocation_sha256="c" * 64,
    )
    candidate = {f"candidate{i}": np.zeros(1) for i in range(21)}
    checkpoint = {f"state{i}": np.zeros(1) for i in range(39)}
    counts = dict(direct_node_counts_fish_node=[[0] * 3 for _ in range(3)])
    chain = dict(
        chain_version="old_prefix_then_receipted_child_v1",
        final_audit_complete=False,
        tracking_processes=33,
        checkpoint_restores=32,
        ancestors=[],
        phases=[],
        final_run=tmp_path,
        final_record=dict(outputs=dict(candidate=dict(sha256="d" * 64))),
        binding=dict(opencv_threads=32),
    )
    metadata = dict(
        processed_frames=c.COUNT,
        previous_frame=c.STOP - 1,
        initialize_count=1,
        restore_count=32,
        context={},
    )
    completed = dict(
        chain=chain,
        candidate=candidate,
        checkpoint=checkpoint,
        checkpoint_metadata=metadata,
        counts=counts,
        exact_backtrace={},
        final_audit_complete=False,
    )
    context = SimpleNamespace(nodes=object())
    geometry = dict(
        rig=object(),
        backgrounds=object(),
        encoded_background_records=object(),
        protocol=dict(flow_config=object()),
        proof=dict(fixture=True),
    )
    refs = [dict(stop=s, restore_count=i) for i, s in enumerate(c.boundary.endpoint_stops())]

    class Evidence:
        def __init__(self):
            self.paths = {}

        def final_guard(self):
            events.append("input_guard")
            if fault == "input_guard":
                raise ValueError("fixture input changed")

    def owner(*values, **kw):
        events.append("owner")
        assert values[0:4] == ("owner.json", "b" * 64, tmp_path, {"header": True})
        assert kw["audit_started_at"] and isinstance(kw["evidence"], Evidence)
        if fault == "owner":
            raise ValueError("fixture owner rejected")
        return dict(verified=True, fixture=True)

    def header(*values):
        events.append("header")
        if fault == "header":
            raise ValueError("fixture unfinished")
        return {"header": True}

    def load(root, evidence):
        assert events[:2] == ["header", "owner"]
        events.append("load")
        return completed

    def native_inputs(given, evidence):
        assert given is chain
        events.append("native_inputs")
        return context

    def bind_geometry(given, n, evidence):
        assert given is completed and n is context
        events.append("geometry")
        return geometry

    def endpoint(given, evidence):
        assert given is chain
        events.append("endpoints")
        return refs

    def assembly(*values):
        assert values[2] is chain and values[3] == ["consumed"]
        events.append("D2_metadata")
        return dict(summary={"metadata_verified": True})

    def packed(ctx, *, on_chunk):
        events.append("D2_packed")
        for vi, v in enumerate(c.VIEWS):
            for i in range(180):
                if fault == "missing_D2" and vi == 2 and i == 179:
                    continue
                on_chunk(
                    dict(
                        view=v,
                        index=i,
                        start=c.FIRST + i * c.BLOCK,
                        stop=c.FIRST + (i + 1) * c.BLOCK,
                        frames=c.BLOCK,
                        exact_fields=13,
                    )
                )
        return dict(
            chunk_view_comparisons=540,
            logical_field_comparisons=7020,
            packed_arrays_verified=True,
            final_audit_complete=False,
        )

    support_result = dict(
        native_frames=c.COUNT,
        node_denominator=c.COUNT * 9,
        pectoral_denominator=c.COUNT * 3,
        maximum_view_node_pairs=c.COUNT * 27,
        block_frames=c.BLOCK,
        blocks=180,
        frame_range=[c.FIRST, c.STOP],
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        temporal_subsampling=False,
        source_rows_checked=0,
        observed_nodes_checked=0,
        maximum_supported_ray_distance_mm=None,
        maximum_supported_pixel_error=None,
    )

    def support(cand, nodes, rig, *, block_frames, on_block):
        assert cand is candidate and nodes is context.nodes and rig is geometry["rig"]
        assert block_frames == c.BLOCK
        events.append("support")
        data = dict(
            valid_source=np.zeros((c.BLOCK, 3, 3, 3), bool),
            ray_distance_mm=np.full((c.BLOCK, 3, 3, 3), np.nan),
            native_pixel_error=np.full((c.BLOCK, 3, 3, 3), np.nan),
        )
        for i in range(180):
            lo, hi = i * c.BLOCK, (i + 1) * c.BLOCK
            data["frame_idx"] = np.arange(c.FIRST + lo, c.FIRST + hi, dtype=np.int64)
            on_block(
                dict(
                    start_index=lo,
                    stop_index=hi,
                    frame_range=[c.FIRST + lo, c.FIRST + hi],
                    source_rows_checked=0,
                    observed_nodes_checked=0,
                ),
                data,
            )
        return copy.deepcopy(support_result)

    class Writer:
        def __init__(self, root, binding):
            self.root = root
            self.root.mkdir()
            self.binding = binding
            self.closed = False
            events.append("spool_new")

        def abort(self, error):
            if self.closed:
                return
            self.closed = True
            events.append("spool_abort")
            (self.root / "failure.json").write_text(json.dumps({"error": str(error)}))

        def finish(self):
            events.append("spool_finish")
            self.closed = True
            p = self.root / "fixture.json"
            p.write_text("{}")
            m = dict(
                callback_records=1629,
                saved_views=1077,
                outputs={p.name: dict(sha256=c.sha(p))},
                final_audit_complete=False,
            )
            (self.root / "spool.manifest.json").write_bytes(c.canonical(m))
            return m

    def samples(given, *, on_sample):
        assert given is context
        events.append("samples")
        if fault == "sample_save":
            (on_sample.root / "partial.npz").write_bytes(b"partial fixture")
            raise OSError("fixture sample save failed")
        return [{"fixture": True}] * 1629

    class Reader:
        def __init__(self, *args, **kw):
            events.append("reader")

        def require_final_scope(self):
            events.append("reader_scope")

        def verify_all(self):
            events.append("reader_guard")

    def flow(reader, state, meta, rig, **kw):
        assert state is checkpoint and meta is metadata and rig is geometry["rig"]
        assert kw["backgrounds"] is geometry["backgrounds"]
        assert kw["restore_starts"] == [r["stop"] for r in refs[:-1]]
        events.append("flow")
        for i in range(179):
            n = c.FIRST + (i + 1) * c.BLOCK
            kw["on_flow"](dict(frame_idx=n, previous_frame=n - 1), {"fixture": np.zeros(1)})
        return dict(
            restored_boundaries=32, same_process_reader_boundaries=147, final_audit_complete=False
        )

    def previous(reader, given, ctx, **kw):
        assert given is refs and ctx is metadata["context"]
        events.append("previous")
        return dict(count=33, checkpoints=[{}] * 33, final_audit_complete=False)

    def fake_npz(out, name, arrays):
        # This test checks orchestration; archive roundtrip is tested separately below.
        return out.bytes(name, b"not a real payload; synthetic helper fixture")

    def residuals(blocks, root):
        assert len(blocks) == 180 and root == Path(args.output)
        events.append("residuals")
        return dict(
            native_frames=c.COUNT,
            valid_source_count=0,
            metrics={
                k: dict(finite_supported_count=0, nonfinite_supported_count=0, max=None)
                for k in ("ray_distance_mm", "native_pixel_error")
            },
        )

    monkeypatch.setattr(c, "owner_verifier", lambda: owner)
    monkeypatch.setattr(c, "bind_sources", lambda evidence, out: {})
    monkeypatch.setattr(c, "source_guard", lambda src: events.append("source_guard"))
    monkeypatch.setattr(c.Outputs, "npz", fake_npz)
    monkeypatch.setattr(
        c,
        "adapters",
        SimpleNamespace(
            completed_header=header,
            bind_geometry_and_backgrounds=bind_geometry,
            make_endpoint_refs=endpoint,
            named_json=lambda *x: ["consumed"],
        ),
    )
    monkeypatch.setattr(
        c,
        "chain_api",
        SimpleNamespace(load_completed_final=load, load_native_sample_inputs=native_inputs),
    )
    monkeypatch.setattr(
        c, "d2", SimpleNamespace(load_completed_assembly=assembly, verify_packed_assembly=packed)
    )
    monkeypatch.setattr(
        c, "support_api", SimpleNamespace(Digests=Evidence, verify_support_partitioned=support)
    )
    monkeypatch.setattr(
        c,
        "native",
        SimpleNamespace(fixed_sample_frames=lambda: np.arange(543), read_native_samples=samples),
    )
    monkeypatch.setattr(
        c,
        "spool",
        SimpleNamespace(signatures=c.spool.signatures, NativeSpool=Writer, SpoolReader=Reader),
    )
    monkeypatch.setattr(
        c,
        "boundary",
        SimpleNamespace(verify_flow_boundaries=flow, verify_previous_observations=previous),
    )
    monkeypatch.setattr(
        c,
        "report",
        SimpleNamespace(
            candidate_metrics=lambda given: dict(direct_counts=counts), residual_metrics=residuals
        ),
    )
    return args, events, completed


def test_full_orchestration_once_same_objects_no_helper_promotion(tmp_path, monkeypatch):
    args, events, completed = fixture(tmp_path, monkeypatch)
    seal = c.run(args)
    assert seal["status"] == "ok" and seal["final_audit_complete"] is True
    assert (
        completed["final_audit_complete"] is False
        and completed["chain"]["final_audit_complete"] is False
    )
    for event in [
        "owner",
        "load",
        "native_inputs",
        "geometry",
        "endpoints",
        "D2_metadata",
        "D2_packed",
        "support",
        "samples",
        "spool_finish",
        "reader",
        "flow",
        "previous",
        "residuals",
        "reader_guard",
        "input_guard",
    ]:
        assert events.count(event) == 1, event
    assert (
        events.index("owner")
        < events.index("load")
        < events.index("samples")
        < events.index("input_guard")
    )
    root = Path(args.output)
    for row in seal["outputs"].values():
        assert c.sha(row["path"]) == row["sha256"]
    assert set(seal["outputs"]) == {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "audit.provenance.json"
    }


@pytest.mark.parametrize("fault", ["header", "owner", "missing_D2", "sample_save", "input_guard"])
def test_failure_never_promotes_and_retains_partial(tmp_path, monkeypatch, fault):
    args, events, _ = fixture(tmp_path, monkeypatch, fault=fault)
    with pytest.raises((ValueError, OSError)):
        c.run(args)
    p = Path(args.output)
    seal = json.loads((p / "audit.provenance.json").read_text())
    assert seal["status"] == "failed" and seal["final_audit_complete"] is False
    if fault in ("header", "owner"):
        assert "load" not in events
    if fault == "missing_D2":
        assert "support" not in events
    if fault == "sample_save":
        assert "spool_abort" in events and "flow" not in events
        assert (p / "native_spool/partial.npz").exists()
        assert "native_spool/partial.npz" in seal["partial_file_paths"]


def test_missing_owner_module_before_header(tmp_path, monkeypatch):
    args, events, _ = fixture(tmp_path, monkeypatch)

    def absent():
        raise ModuleNotFoundError("no integrated verifier")

    monkeypatch.setattr(c, "owner_verifier", absent)
    with pytest.raises(ModuleNotFoundError):
        c.run(args)
    assert events == []
    assert (
        json.loads((Path(args.output) / "audit.provenance.json").read_text())["status"] == "failed"
    )


def test_output_roundtrip_mutation_and_nested_seal_not_exempt(tmp_path):
    o = c.Outputs(tmp_path / "outputs")
    a = {"x": np.asarray([np.nan, 0.0, 1.0])}
    saved = o.npz("block.npz", a)
    assert saved["arrays"]["x"]["shape"] == [3]
    o.guard()
    with pytest.raises(ValueError, match="duplicate"):
        o.add("block.npz")
    (o.root / "unexpected").mkdir()
    (o.root / "unexpected/audit.provenance.json").write_text("{}")
    with pytest.raises(ValueError, match="membership"):
        o.guard()


def test_output_changed_after_registration(tmp_path):
    o = c.Outputs(tmp_path / "outputs")
    o.bytes("x", b"old")
    (o.root / "x").write_bytes(b"new")
    with pytest.raises(ValueError, match="changed"):
        o.guard()


@pytest.mark.parametrize("value", [True, None, "A" * 64, "0" * 63, "g" * 64])
def test_explicit_sha_contract(value):
    with pytest.raises(ValueError):
        c.valid_sha(value)


def test_owner_verifier_changed_source_rejected_before_import(monkeypatch):
    monkeypatch.setitem(c.PINS, c.WORK / "receipted_final_owner_exit.py", "0" * 64)
    with pytest.raises(ValueError, match="reviewed final owner verifier unavailable"):
        c.owner_verifier()


def test_source_inventory_change_rejected(tmp_path, monkeypatch):
    p = tmp_path / "source.py"
    p.write_text("original")
    monkeypatch.setattr(c, "source_paths", lambda: {p})
    sources = {str(p): c.sha(p)}
    c.source_guard(sources)
    p.write_text("changed")
    with pytest.raises(ValueError, match="executed source changed"):
        c.source_guard(sources)
    monkeypatch.setattr(c, "source_paths", lambda: {p, tmp_path / "late.py"})
    with pytest.raises(ValueError, match="late local import"):
        c.source_guard(sources)
