"""Retry wiring only: real anonymous codec plus bounded 200-frame CPU fixtures."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_mask_transport200_v2 as m


class MemoryOutput:
    def __init__(self):
        self.values = {}

    def write(self, name, value):
        assert name not in self.values
        self.values[name] = copy.deepcopy(value)


class NoFileEvidence:
    def add(self, *args):
        pass


@pytest.mark.parametrize("bad", [None, "state_bits", "binding", "metadata"])
def test_real_codec_anonymous_storage_is_decoded_before_all39_gate(tmp_path, monkeypatch, bad):
    candidate = {"frame_idx": np.arange(m.FIRST, m.STOP, dtype=np.int64)}
    candidate.update({f"candidate_{i}": np.array([np.nan, -0.0]) for i in range(20)})
    core = {"history_offsets": np.arange(201, dtype=np.int64) * 2}
    core.update({"history_" + k: np.arange(400) for k in m.RECORD_FIELDS})
    core.update({f"state_{i}": np.array([0.0, -0.0]) for i in range(19)})
    binding, metadata = {"synthetic": True}, {"lifecycle": "complete"}
    manifest = m.checkpoint.write_checkpoint(
        tmp_path / "checkpoint", core, metadata, bindings=binding
    )
    with np.load(tmp_path / "checkpoint/arrays.npz", allow_pickle=False) as stored:
        assert len(stored.files) == 39
        assert all(k.startswith("array_") for k in stored.files)
        assert "history_pose_xyz" not in stored.files
    monkeypatch.setattr(m, "REFERENCE", tmp_path)
    monkeypatch.setattr(m, "REFERENCE_ARRAYS_SHA", manifest["payload"]["sha256"])
    calls = []

    def candidate_only(root, members, relative):
        assert relative == "candidate.npz", "old anonymous NPZ must use frozen codec"
        calls.append(relative)
        return candidate

    monkeypatch.setattr(m, "read_arrays", candidate_only)
    ref = dict(members={}, binding=copy.deepcopy(binding), metadata=copy.deepcopy(metadata))
    current = copy.deepcopy(core)
    if bad == "state_bits":
        current["state_18"][1] = 0.0
    elif bad == "binding":
        ref["binding"]["synthetic"] = 1
    elif bad == "metadata":
        ref["metadata"]["lifecycle"] = "different"
    out = MemoryOutput()
    if bad:
        with pytest.raises(ValueError):
            m.original_gate(candidate, current, ref, NoFileEvidence(), finished=True, output=out)
        if bad == "state_bits":
            assert out.values["original_control.json"]["state"]["byte_exact"] is False
    else:
        result = m.original_gate(
            candidate, current, ref, NoFileEvidence(), finished=True, output=out
        )
        assert result["passed"] is True
        assert [result[k]["array_count"] for k in ("candidate", "history", "state")] == [21, 20, 39]
    assert calls == ["candidate.npz"]


@pytest.mark.parametrize("fault", [None, "extra_context", "mutation", "restore"])
def test_cpu_semantic_scratch_preserves_saved_metadata_and_never_restores(monkeypatch, fault):
    original = dict(
        implementation="Original",
        tracker_config={"device": "cuda"},
        recovery_config={},
        persistence_config={},
    )
    context = {
        **copy.deepcopy(original),
        "implementation": "MaskTransportTracker",
        "mask_transport": {
            "coefficient": 0.0,
            "input_binding": {
                "arm": "original",
                "schema": "mask_transport200_three_arm_tracking_v1",
            },
        },
    }
    metadata = dict(
        context=context,
        initialize_count=1,
        restore_count=0,
        processed_frames=200,
        previous_frame=m.STOP - 1,
    )
    untouched = copy.deepcopy(metadata)
    core = {"signed": np.array([-0.0, np.nan])}
    side, sidemeta, events = {}, {"saved": "original GPU context"}, []
    expected_cpu = copy.deepcopy(context)
    expected_cpu["tracker_config"]["device"] = "cpu"

    class Validator:
        initialize_count = restore_count = 0
        _states = None

        def __init__(self, *args, **kwargs):
            assert kwargs["config"].device == "cpu"
            assert kwargs["input_binding"] == context["mask_transport"]["input_binding"]
            with pytest.raises(AssertionError, match="cannot request votes"):
                kwargs["vote_provider"](m.FIRST)

        def _context(self):
            value = copy.deepcopy(expected_cpu)
            if fault == "extra_context":
                value["implementation"] = "Changed"
            return value

        def _decode_state(self, arrays, scratch):
            events.append("decode")
            assert scratch is not metadata and scratch["context"] == expected_cpu
            assert metadata == untouched
            if fault == "mutation":
                arrays["signed"][0] = 0.0
            if fault == "restore":
                self.restore_count = 1

        def _validated_sidecar(self, arrays, saved, side_arrays, side_metadata):
            events.append("sidecar")
            assert saved is metadata and saved == untouched
            assert side_arrays is side and side_metadata is sidemeta
            return {"validated": True}

    monkeypatch.setattr(m.tracking, "MaskTransportTracker", Validator)
    points = SimpleNamespace(group_counts=(3, 4, 3), query_sha256=("a" * 64,) * 3)
    common = dict(original_context=original, rig=object(), initial=np.zeros((3, 3, 3)))
    if fault:
        with pytest.raises(ValueError):
            m.validate_original_sidecar(core, metadata, side, sidemeta, common, points)
    else:
        assert m.validate_original_sidecar(core, metadata, side, sidemeta, common, points) == {
            "validated": True
        }
        assert events == ["decode", "sidecar"]
    assert metadata == untouched


@pytest.fixture
def reuse200(monkeypatch):
    """Exercise every reuse row without re-testing the frozen state/geometry math."""
    offsets = np.arange(201, dtype=np.int64) * 2
    core = {"history_offsets": offsets}
    core.update({"history_" + k: np.arange(400, dtype=np.int64) for k in m.RECORD_FIELDS})
    original = {"implementation": "Original", "tracker_config": {"device": "cuda"}}
    metadata = dict(
        context={**original, "implementation": "New", "mask_transport": {}},
        initialize_count=1,
        restore_count=0,
    )
    oldmeta = {**metadata, "context": original}
    initial = dict(
        detections={v: np.zeros((2, 3, 2), np.float32) for v in m.VIEWS},
        masks={v: np.zeros((2, 2, 2), bool) for v in m.VIEWS},
    )
    samples, fits, frames = {}, [], {}
    for t, n in enumerate(range(m.FIRST, m.STOP)):
        sample = dict(
            frame_idx=np.asarray(n, np.int64),
            fit_pixels=np.zeros((3, 1024, 2), np.int64),
            fit_target=np.zeros((3, 1024), np.float32),
            fit_weights=np.ones((3, 1024), np.float32),
            fit_foreground=np.zeros((3, 1024), bool),
            unused_pixels=np.zeros((3, 256, 2), np.int64),
            unused_fit_overlap=np.zeros((3, 256), bool),
        )
        samples[n] = sample
        fit = {k: value.copy() for k, value in sample.items() if k.startswith("fit_")}
        fit["parent"] = np.array([0, 0], np.int64)
        fits.append(fit)
        frames[n] = dict(
            fit=copy.deepcopy(fit),
            anchor_ids=np.array([0, 1]),
            records=[
                {k: core["history_" + k][i].copy() for k in m.RECORD_FIELDS}
                for i in range(t * 2, t * 2 + 2)
            ],
            release_native=m.horizon(n),
            advance_seconds=0.25,
        )
    anchors = [
        SimpleNamespace(links=tuple(np.zeros((g, 3), bool) for g in (3, 4, 3))) for _ in range(2)
    ]
    facts = [
        dict(references={"decision": "conditional"}, geometry={"x": np.array([-0.0])})
        for _ in range(2)
    ]
    content = dict(
        fits=fits,
        anchor_history=[np.array([0, 1])] * 200,
        initial_observation=initial,
        anchors=[dict(links=a.links) for a in anchors],
        anchor_facts=facts,
    )
    summary = dict(
        arm="original",
        frames=200,
        coefficient=0.0,
        initialize_count=1,
        restore_count=0,
        initial_anchors=2,
        retained_rows=400,
        proposal_counts=[2] * 200,
        advance_and_synchronize_seconds=50.0,
    )
    rows = {
        (m.FIRST, v): dict(
            mask_available=True,
            observation_array_signatures={
                k: m.tracking.signature(initial[k][v]) for k in ("masks", "detections")
            },
        )
        for v in m.VIEWS
    }
    common = dict(original_context=original, reader=SimpleNamespace(rows=rows), rig=object())
    seen, copied = [], []
    originals = [
        f"original/frames/native_{n}.{ext}"
        for n in range(m.FIRST, m.STOP)
        for ext in ("npz", "json")
    ] + [f"original/other_{i}" for i in range(8)]
    monkeypatch.setattr(m, "failed_original_members", lambda ev: ({}, originals))
    monkeypatch.setattr(m, "validate_original_sidecar", lambda *args: content)
    monkeypatch.setattr(
        m,
        "original_gate",
        lambda *args, **kw: {
            "passed": True,
            **{k: {"byte_exact": True} for k in ("candidate", "history", "state")},
        },
    )
    monkeypatch.setattr(m.tracking, "initial_anchors", lambda *args: (anchors, facts))
    monkeypatch.setattr(m.torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(m, "copy_original_file", lambda output, members, name: copied.append(name))

    def arrays(root, members, relative):
        if root == m.SAMPLES:
            n = int(Path(relative).stem.split("_")[-1])
            seen.append(n)
            return samples[n]
        return core if relative.endswith("/arrays.npz") else {}

    def small(root, members, relative):
        return (
            metadata
            if relative.endswith("/metadata.json")
            else (summary if relative.endswith("/summary.json") else {})
        )

    def tree(members, name):
        if name == "original/initial_anchors":
            return dict(anchors=content["anchors"], facts=facts)
        return frames[int(name.rsplit("_", 1)[-1])]

    monkeypatch.setattr(m, "read_arrays", arrays)
    monkeypatch.setattr(m.base, "bound_json", small)
    monkeypatch.setattr(m, "read_tree", tree)
    args = (
        common,
        {},
        SimpleNamespace(group_counts=(3, 4, 3)),
        {"metadata": oldmeta},
        NoFileEvidence(),
        MemoryOutput(),
    )
    return SimpleNamespace(
        args=args,
        seen=seen,
        copied=copied,
        frames=frames,
        fits=fits,
        rows=rows,
        initial=initial,
        summary=summary,
        originals=originals,
    )


@pytest.mark.parametrize("fault", [None, "last_fit", "last_record", "last_release", "initial_SL"])
def test_all200_rows_before_any_adoption_and_actual_b0_preserved(reuse200, fault):
    f = reuse200
    if fault == "last_fit":
        f.fits[-1]["fit_target"][0, -1] = -0.0
    elif fault == "last_record":
        f.frames[m.STOP - 1]["records"][-1][m.RECORD_FIELDS[-1]] += 1
    elif fault == "last_release":
        f.frames[m.STOP - 1]["release_native"] -= 1
    elif fault == "initial_SL":
        f.initial["detections"][m.VIEWS[-1]][-1, -1, -1] = -0.0
    if fault:
        with pytest.raises(ValueError):
            m.qualify_original_reuse(*f.args)
        assert f.copied == []
    else:
        control, records0, anchors, summary = m.qualify_original_reuse(*f.args)
        assert control["passed"] and len(records0) == len(anchors) == 2
        assert f.copied == f.originals and len(f.copied) == 408
        proof = f.args[-1].values["original_reuse_qualification.json"]
        assert proof["actual_fit4_byte_exact_frames"] == 200
        assert proof["recomputed_initial_link_cells"] == 60
        assert proof["GPU_used"] is False and proof["validation_restore_calls"] == 0
        assert summary["computed_in_this_process"] is False
    assert f.seen == list(range(m.FIRST, m.STOP))


def test_tree_and_container_copy_preserve_nan_and_signed_zero(tmp_path, monkeypatch):
    old, new = tmp_path / "failed", tmp_path / "retry"
    old.mkdir()
    original = m.base.Output(old / "payload")
    value = {"nested": [np.array([-0.0, np.nan]), np.asarray("kind", dtype="U9")]}
    m.save_tree(original, "original/example", value)
    members = {row["path"]: row for row in original.records.values()}
    monkeypatch.setattr(m, "FAILED", original.root)
    restored = m.read_tree(members, "original/example")
    m.same_tree(restored, value, "roundtrip")
    changed = copy.deepcopy(restored)
    changed["nested"][0][0] = 0.0
    with pytest.raises(ValueError, match="bytes differ"):
        m.same_tree(restored, changed, "signed zero")
    output = m.base.Output(new)
    for name in original.records:
        m.copy_original_file(output, members, name)
        assert (output.root / name).read_bytes() == (original.root / name).read_bytes()
    with pytest.raises(ValueError, match="exclusive"):
        m.copy_original_file(output, members, "original/example.npz")
    output.final_guard()


def test_reuse_rejection_is_persisted_before_runtime_or_any_new_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "validate_execution", lambda args: None)
    cfg = {
        "tracker_config": dict(device="cuda", flow_proposals=True, max_hypotheses=18, opt_steps=14)
    }
    monkeypatch.setattr(m, "load_inputs", lambda *args: ({"original_context": cfg}, {}, None, {}))
    monkeypatch.setattr(m.prepared, "snapshot_sources", lambda *args: None)
    monkeypatch.setattr(m.cv2, "setNumThreads", lambda n: None)
    monkeypatch.setattr(m.torch, "set_num_threads", lambda n: None)

    def reject(*args):
        raise ValueError("synthetic reuse rejection")

    def forbidden(*args, **kwargs):
        raise AssertionError("runtime or GPU started before CPU proof")

    monkeypatch.setattr(m, "qualify_original_reuse", reject)
    monkeypatch.setattr(m, "runtime", forbidden)
    monkeypatch.setattr(m, "gpu_memory", forbidden)
    dest = tmp_path / "new"
    with pytest.raises(ValueError, match="synthetic reuse"):
        m.run(SimpleNamespace(output=dest, samples_seal_sha256=m.SAMPLES_SHA))
    failure = m.base.read_json(dest / "failure.json")
    assert failure["active_arm"] == "original_reuse_validation"
    assert failure["completed_arms"] == []
    assert m.base.read_json(dest / "tracking.provenance.json")["status"] == "failed"


def test_only_gate_run_changed_and_two_active_arm_loop_is_explicit():
    def definitions(path):
        return {
            n.name: n
            for n in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    old = definitions(Path(m.__file__).with_name("run_mask_transport200.py"))
    new = definitions(Path(m.__file__))
    changed = {
        k
        for k in old.keys() & new.keys()
        if ast.dump(old[k], include_attributes=False) != ast.dump(new[k], include_attributes=False)
    }
    assert changed == {"original_gate", "run"}
    assert len(old.keys() & new.keys()) - len(changed) == 22
    loops = [
        node
        for node in ast.walk(new["run"])
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "arm"
    ]
    assert len(loops) == 1 and ast.unparse(loops[0].iter) == "ARMS[1:]"
