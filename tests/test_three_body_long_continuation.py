"""CPU I/O integration for long-prefix execution; the tracker/reader are tiny fakes.

The real checkpoint codec, prefix-proof validator, output validators and provenance
writer are retained. These tests do not claim numerical fitting or image-reader QA;
those are covered separately by the frozen core and native reader tests.
"""

import copy
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def isolated_module():
    names = {path.stem for path in LAB.glob("*.py")}
    prior = {key: value for key, value in sys.modules.items() if key in names}
    old_path = sys.path.copy()
    try:
        for key in names:
            sys.modules.pop(key, None)
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location(
            "long_continuation_integration_tests", LAB / "run_long_continuation.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key in names:
            sys.modules.pop(key, None)
        sys.modules.update(prior)
        sys.path[:] = old_path


M = isolated_module()
FIRST = 49800


def tiny_result(frames):
    count = len(frames)
    pose = np.tile(np.arange(27, dtype=np.float64).reshape(1, 3, 3, 3), (count, 1, 1, 1))
    # A future suffix may change even the first provisional output coordinate.
    pose += count / 100
    observed = np.ones((count, 3, 3), bool)
    review = np.zeros((count, 3), bool)
    sources = np.tile(np.arange(3, dtype=np.int32)[None, :, None, None], (count, 1, 3, 3))
    return dict(
        frame_idx=np.asarray(frames, np.int64),
        pose_xyz=pose,
        sources=sources,
        observed=observed,
        identity_review=review,
        mask_supported=np.ones((count, 3), bool),
        body_supported=np.ones((count, 3), bool),
        missing=np.zeros((count, 3), np.int64),
        costs=np.zeros(count),
        path_costs=np.zeros(count),
        loss_parts=np.zeros((count, 5)),
        score_margin=np.full(count, np.inf),
        proposal_kind=np.full(count, "predict"),
        hypotheses=np.ones(count, np.int64),
        proposal_flow_nodes=np.ones((count, 3, 3), bool),
        proposal_flow_parent=np.r_[-1, np.zeros(max(count - 1, 0), np.int64)],
        node_missing=np.zeros((count, 3, 3), np.int64),
        direct_pectoral_age=np.tile(np.arange(1, count + 1)[:, None], (1, 3)),
        xyz=pose[:, :, 1].copy(),
        accepted_xyz=pose[:, :, 1].copy(),
        body_xyz=pose[:, :, 1].copy(),
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    registration = seed / "registration.json"
    registration.write_text('{"synthetic_first_frame":49800}', encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()
    payload = tmp_path / "input.bin"
    payload.write_bytes(b"synthetic sealed input")
    roots = []
    records = []
    for index in range(3):
        root = stage / f"chunk{index}"
        root.mkdir()
        first = FIRST + 2 * index
        path = root / "manifest.json"
        path.write_text(json.dumps({"frames": list(range(first, first + 2))}), encoding="utf-8")
        roots.append(root)
        records.append(
            dict(
                index=index,
                start=first,
                stop=first + 2,
                manifest_sha256=M.base.sha256(path),
                attempt_path=str(root),
            )
        )
    protocol_path = tmp_path / "protocol.json"
    input_protocol = {"experiment": "toy", "chunk_frames": 2}
    input_protocol_path = tmp_path / "long_input_protocol.json"
    input_protocol_path.write_text(json.dumps(input_protocol), encoding="utf-8")
    versions = dict(
        numpy=M.np.__version__,
        scipy=M.scipy.__version__,
        opencv=M.cv2.__version__,
        torch=str(M.torch.__version__),
        cuda=M.torch.version.cuda,
        cuda_device_name="CPU integration fixture",
    )
    protocol = dict(
        schema="three_body_long_tracking_protocol_v1",
        source_frame_range=[FIRST, FIRST + 6],
        seed_registration_sha256=M.base.sha256(registration),
        frozen_tracker_sources_sha256={},
        input_protocol_sha256=M.base.sha256(input_protocol_path),
        torch_cpu_threads=1,
        package_versions=versions,
        tracker_config=dataclasses.asdict(
            M.TrackerConfig(device="cpu", mode="nodes", flow_proposals=True)
        ),
        node_recovery_config=dataclasses.asdict(M.NodeRecoveryConfig()),
        persistence_config=dataclasses.asdict(M.PersistentNodeConfig()),
        flow_config=dataclasses.asdict(M.FlowConfig()),
        limitations=["Synthetic CPU integration only"],
    )
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    events = {"instances": [], "readers": [], "seeks": [], "closed": [], "advances": []}
    options = {"mutation": None, "change_proof": False, "change_fixed": False}
    original_threads = M.torch.get_num_threads()
    monkeypatch.setattr(M, "PROTOCOL_PATH", protocol_path)
    monkeypatch.setattr(M, "LAB", tmp_path)
    monkeypatch.setattr(M, "_sources", lambda: [Path(M.__file__).resolve(), protocol_path])
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(M.torch.cuda, "get_device_name", lambda index: "CPU integration fixture")
    monkeypatch.setattr(M.torch.cuda, "synchronize", lambda: None)

    def load(seed_root, stage_root, input_root, *, through_frame):
        assert seed_root == seed and stage_root == stage
        if through_frame not in (FIRST + 2, FIRST + 4, FIRST + 6):
            raise ValueError("requested endpoint is not a registered chunk boundary")
        count = (through_frame - FIRST) // 2
        proof = copy.deepcopy(records[:count])
        if options["change_proof"]:
            proof[0]["manifest_sha256"] = "f" * 64
        hashes = {
            str(payload): M.base.sha256(payload),
            str(registration): M.base.sha256(registration),
        }
        hashes.update(
            {
                str(root / "manifest.json"): M.base.sha256(root / "manifest.json")
                for root in roots[:count]
            }
        )
        return SimpleNamespace(
            initial_pose_xyz=np.arange(27, dtype=np.float64).reshape(3, 3, 3),
            rig=None,
            protocol=input_protocol.copy(),
            full_frame_range=(FIRST, FIRST + 6),
            frame_idx=np.arange(FIRST, through_frame, dtype=np.int64),
            chunk_roots=tuple(roots[:count]),
            input_root=input_root,
            fixed_bindings={
                "seed": M.base.sha256(registration),
                "changed": options["change_fixed"],
            },
            verified_hashes=hashes,
            chunk_records=proof,
        )

    class Tracker:
        def __init__(self, rig, initial, config, **kwargs):
            self.frames, self.previous_image = [], None
            self.initialize_count, self.restore_count = 0, 0
            self.finished = False
            events["instances"].append(self)

        @property
        def processed_frames(self):
            return len(self.frames)

        @property
        def previous_frame(self):
            return self.frames[-1] if self.frames else None

        @property
        def node_recovery_diagnostics(self):
            return {"calls": len(self.frames)}

        def begin(self):
            assert self.initialize_count == 0
            self.initialize_count = 1

        def restore_state(self, arrays, metadata):
            assert self.initialize_count == 0
            self.frames = arrays["saved_frames"].tolist()
            self.previous_image = arrays["previous_image"].copy()
            self.restore_count = metadata["restore_count"] + 1
            self.initialize_count = 1

        def advance(self, frame):
            assert not self.finished and self.initialize_count == 1
            if self.previous_frame is not None:
                assert frame.frame_idx == self.previous_frame + 1
                assert int(self.previous_image[0]) == self.previous_frame
            events["advances"].append(frame.frame_idx)
            self.frames.append(frame.frame_idx)
            self.previous_image = frame.image.copy()
            if options["mutation"] == "input_change":
                payload.write_bytes(b"changed during processing")

        def export_state(self):
            assert not self.finished, "snapshot must precede provisional finish"
            return dict(
                saved_frames=np.array(self.frames, np.int64),
                previous_image=self.previous_image.copy(),
                history_offsets=np.arange(len(self.frames) + 1, dtype=np.int64),
                state_pose=np.zeros((1, 3, 3, 3)),
            ), dict(
                processed_frames=len(self.frames),
                previous_frame=self.previous_frame,
                restore_count=self.restore_count,
                initialize_count=self.initialize_count,
            )

        def finish(self):
            self.finished = True
            return tiny_result(self.frames)

    def reader(root, input_root):
        events["readers"].append(root)
        return SimpleNamespace(
            root=root, frames=json.loads((root / "manifest.json").read_text())["frames"]
        )

    def iterate(reader, start):
        events["seeks"].append(start)
        try:
            frames = [frame for frame in reader.frames if frame >= start]
            if options["mutation"] == "skip":
                frames = frames[1:]
            elif options["mutation"] == "extra":
                frames += [frames[-1] + 1]
            elif options["mutation"] == "duplicate":
                frames.insert(1, frames[0])
            for native in frames:
                yield SimpleNamespace(frame_idx=native, image=np.array([native], np.int64))
        finally:
            events["closed"].append(reader.root)

    monkeypatch.setattr(M, "load_long_inputs", load)
    monkeypatch.setattr(M, "AcceleratedContinuousPersistentNodeTracker", Tracker)
    monkeypatch.setattr(M, "SourceSequenceReader", reader)
    monkeypatch.setattr(M, "iter_from", iterate)
    monkeypatch.setattr(
        M, "CachedNodes", lambda *args: SimpleNamespace(at=lambda view, native: np.zeros((3, 3, 2)))
    )

    def run(through, name, resume=None):
        out = tmp_path / name
        argv = [
            "--seed-root",
            str(seed),
            "--input-stage-root",
            str(stage),
            "--input-root",
            str(tmp_path),
            "--through-frame",
            str(through),
            "--output",
            str(out),
        ]
        if resume is not None:
            argv += ["--resume", str(resume)]
        return out, M.main(argv)

    yield SimpleNamespace(
        run=run,
        root=tmp_path,
        events=events,
        options=options,
        payload=payload,
        roots=roots,
        protocol_path=protocol_path,
    )
    M.torch.set_num_threads(original_threads)


def sealed(root):
    record = json.loads((root / "continuation.provenance.json").read_text())
    assert record["status"] == "ok"
    for info in record["outputs"].values():
        assert M.base.sha256(info["path"]) == info["sha256"]
    return record


def test_two_chunk_boundary_preserves_one_state_and_previous_flow_image(setup):
    out, summary = setup.run(FIRST + 4, "single")
    assert setup.events["advances"] == list(range(FIRST, FIRST + 4))
    assert setup.events["seeks"] == [FIRST, FIRST + 2]
    assert len(setup.events["instances"]) == 1
    assert setup.events["closed"] == setup.roots[:2]
    assert summary["logical_initialization_count"] == 1
    assert summary["external_pose_reset_count"] == summary["checkpoint_restore_count"] == 0
    assert summary["prefix_backtrace_provisional"] is True
    assert summary["full_sequence_complete"] is False
    record = sealed(out)
    assert record["params"]["all_input_source_snapshot_output_sha_unchanged"] is True
    with np.load(out / "candidate.npz", allow_pickle=False) as arrays:
        assert len(arrays.files) == 21
        assert arrays["frame_idx"].tolist() == list(range(FIRST, FIRST + 4))


def test_resume_uses_only_new_chunks_and_keeps_prefix_provisional(setup):
    first_out, first = setup.run(FIRST + 2, "prefix")
    second_out, second = setup.run(FIRST + 4, "resumed", resume=first_out)
    assert setup.events["advances"] == list(range(FIRST, FIRST + 4))
    assert setup.events["seeks"] == [FIRST, FIRST + 2]
    assert second["phase_frames"] == [FIRST + 2, FIRST + 4]
    assert second["checkpoint_restore_count"] == 1
    assert second["logical_initialization_count"] == 1
    assert first["prefix_backtrace_provisional"] and second["prefix_backtrace_provisional"]
    with (
        np.load(first_out / "candidate.npz", allow_pickle=False) as left,
        np.load(second_out / "candidate.npz", allow_pickle=False) as right,
    ):
        assert not np.array_equal(left["pose_xyz"], right["pose_xyz"][:2])
    record = sealed(second_out)
    assert str(first_out / "checkpoint/manifest.json") in record["inputs"]
    final_out, final = setup.run(FIRST + 6, "complete", resume=second_out)
    assert final["full_sequence_complete"] is True
    assert final["prefix_backtrace_provisional"] is False
    assert final["checkpoint_restore_count"] == 2
    assert setup.events["advances"] == list(range(FIRST, FIRST + 6))
    sealed(final_out)


@pytest.mark.parametrize("change", ["change_proof", "change_fixed"])
def test_resume_rejects_changed_consumed_proof_or_fixed_binding_before_advance(setup, change):
    out, _ = setup.run(FIRST + 2, "prefix")
    setup.options[change] = True
    with pytest.raises(ValueError, match=r"changed|different"):
        setup.run(FIRST + 4, "invalid_resume", resume=out)
    assert setup.events["advances"] == [FIRST, FIRST + 1]
    assert not (setup.root / "invalid_resume").exists()


@pytest.mark.parametrize("mutation", ["skip", "extra", "duplicate"])
def test_bad_native_stream_never_publishes_success_or_checkpoint(setup, mutation):
    setup.options["mutation"] = mutation
    with pytest.raises(ValueError, match=r"native|boundary"):
        setup.run(FIRST + 4, "failed_stream")
    out = setup.root / "failed_stream"
    record = json.loads((out / "continuation.provenance.json").read_text())
    assert record["status"] == "failed"
    assert not (out / "checkpoint/manifest.json").exists()
    assert setup.events["closed"] == setup.roots[:1]


def test_input_mutation_during_tracking_invalidates_phase_seal(setup):
    setup.options["mutation"] = "input_change"
    with pytest.raises(RuntimeError, match="evidence changed"):
        setup.run(FIRST + 2, "changed_input")
    out = setup.root / "changed_input"
    record = json.loads((out / "continuation.provenance.json").read_text())
    assert record["status"] == "failed"
    # A codec checkpoint alone is insufficient: parent loader requires phase success.
    setup.options["mutation"] = None
    with pytest.raises(ValueError, match="successfully sealed"):
        setup.run(FIRST + 4, "invalid_parent", resume=out)


def test_nonregistered_endpoint_and_no_progress_resume_are_rejected(setup):
    with pytest.raises(ValueError, match="boundary"):
        setup.run(FIRST + 1, "not_endpoint")
    first, _ = setup.run(FIRST + 2, "prefix")
    with pytest.raises(ValueError, match="advance"):
        setup.run(FIRST + 2, "no_progress", resume=first)
    assert setup.events["advances"] == [FIRST, FIRST + 1]
