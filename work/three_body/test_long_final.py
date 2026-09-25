"""Bounded CPU support and completed-chain contracts; no full terminal data used."""

import copy
import hashlib
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_long_final as module


class Camera:
    def __init__(self, axes):
        self.axes = tuple(axes)
        self.depth = next(i for i in range(3) if i not in axes)

    def project(self, xyz):
        return np.asarray(xyz)[..., self.axes]

    def rays(self, uv):
        uv = np.asarray(uv)
        origin = np.zeros((*uv.shape[:-1], 3))
        origin[..., self.axes] = uv
        direction = np.zeros_like(origin)
        direction[..., self.depth] = 1
        return origin, direction


class Rig:
    views = module.SUPPORT.base.VIEWS

    def __init__(self, axes=((0, 1), (0, 2), (1, 2))):
        self.cameras = dict(zip(self.views, (Camera(a) for a in axes), strict=True))

    def camera(self, view):
        return self.cameras[view]


class Nodes:
    def __init__(self, frames, pose, rig):
        self.values = {
            (view, int(frame)): rig.camera(view).project(pose[t]).copy()
            for t, frame in enumerate(frames)
            for view in rig.views
        }

    def at(self, view, frame):
        return self.values[view, int(frame)]


def supported_data(count=7, *, rig=None):
    rig = rig or Rig()
    frames = np.arange(100, 100 + count, dtype=np.int64)
    pose = np.broadcast_to(
        np.arange(27, dtype=np.float64).reshape(3, 3, 3) + 40, (count, 3, 3, 3)
    ).copy()
    sources = np.broadcast_to(np.arange(3)[None, :, None, None], pose.shape).copy()
    arrays = dict(
        frame_idx=frames, pose_xyz=pose, sources=sources, observed=np.ones((count, 3, 3), bool)
    )
    return arrays, Nodes(frames, pose, rig), rig


def test_partitioned_all_frames_matches_frozen_scan_and_preserves_inputs():
    arrays, nodes, rig = supported_data()
    arrays["sources"][2:4] = -1
    arrays["observed"][2:4] = False
    arrays["pose_xyz"][2:4] = np.nan
    # One-view support still has a finite latent location but is not observed.
    arrays["sources"][5, 0, 1:, 0] = -1
    arrays["observed"][5, 0, 0] = False
    old_summary, old_arrays = module.SUPPORT.verify_all_support(arrays, nodes, rig)
    originals = {key: value.tobytes() for key, value in arrays.items()}
    blocks = []
    total = module.verify_support_partitioned(
        arrays, nodes, rig, block_frames=2, on_block=lambda info, data: blocks.append((info, data))
    )
    assert total["blocks"] == 4 and total["zero_support_blocks"] == 1
    assert total["node_denominator"] == 63 and total["pectoral_denominator"] == 21
    assert total["maximum_view_node_pairs"] == 189
    assert total["source_rows_checked"] == old_summary["source_rows_checked"] == 133
    assert total["observed_nodes_checked"] == old_summary["observed_nodes_checked"] == 44
    for key, value in old_arrays.items():
        np.testing.assert_equal(np.concatenate([row[1][key] for row in blocks]), value)
    assert [(row[0]["start_index"], row[0]["stop_index"]) for row in blocks] == [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 7),
    ]
    blocks[0][1]["frame_idx"][:] = -1
    assert all(arrays[key].tobytes() == value for key, value in originals.items())


def test_all_missing_preserves_denominators_without_optical_reduction(monkeypatch):
    arrays, nodes, rig = supported_data(3)
    arrays["sources"][:] = -1
    arrays["observed"][:] = False
    arrays["pose_xyz"][:] = np.nan
    monkeypatch.setattr(
        module.SUPPORT,
        "verify_all_support",
        lambda *_: pytest.fail("empty source set must not call max reducer"),
    )
    result = module.verify_support_partitioned(arrays, nodes, rig, block_frames=2)
    assert result["observed_nodes_checked"] == result["source_rows_checked"] == 0
    assert result["node_denominator"] == 27 and result["frame_range"] == [100, 103]
    assert result["maximum_supported_ray_distance_mm"] is None
    assert result["maximum_supported_pixel_error"] is None
    assert result["zero_support_blocks"] == 2


def test_float32_cached_ray_is_retained_before_residual():
    rig = Rig(((0, 1),) * 3)
    arrays, nodes, _ = supported_data(1, rig=rig)
    arrays["pose_xyz"][..., 0] = 16777218.0
    for value in nodes.values.values():
        value[..., 0] = 16777215.6
    result = module.verify_support_partitioned(arrays, nodes, rig)
    assert result["maximum_supported_ray_distance_mm"] == 2.0
    assert result["maximum_supported_pixel_error"] == pytest.approx(2.4)
    assert 16777218.0 - 16777215.6 > 2.0  # Pure float64 origins would wrongly reject it.


@pytest.mark.parametrize(
    "fault",
    [
        "out_of_pool",
        "nonexclusive",
        "nonfinite_uv",
        "ray_gate",
        "observed",
        "below_missing",
        "float_rows",
        "partial_nan",
        "nan_declared",
        "inf",
        "gap",
        "frame_overflow",
        "row_overflow",
    ],
)
def test_invalid_support_declarations_raise(fault):
    arrays, nodes, rig = supported_data(2)
    if fault == "out_of_pool":
        arrays["sources"][0, 0, 0, 0] = 3
    elif fault == "nonexclusive":
        arrays["sources"][0, 1, 0, 0] = 0
    elif fault == "nonfinite_uv":
        nodes.values[rig.views[0], 100][0, 0] = np.nan
    elif fault == "ray_gate":
        arrays["pose_xyz"][0, 0, 0] += 4
    elif fault == "observed":
        arrays["observed"][0, 0, 0] = False
    elif fault == "below_missing":
        arrays["sources"][0, 0, 0, 0] = -2
    elif fault == "float_rows":
        arrays["sources"] = arrays["sources"].astype(float)
    elif fault == "partial_nan":
        arrays["pose_xyz"][0, 0, 0, 0] = np.nan
    elif fault == "nan_declared":
        arrays["pose_xyz"][0, 0, 0] = np.nan
    elif fault == "inf":
        arrays["pose_xyz"][0, 0, 0] = np.inf
    elif fault == "gap":
        arrays["frame_idx"][1] += 1
    elif fault == "frame_overflow":
        arrays["frame_idx"] = np.array([2**63, 2**63 + 1], dtype=np.uint64)
    else:
        arrays["sources"] = arrays["sources"].astype(np.uint64)
        arrays["sources"][0, 0, 0, 0] = 2**63
    with pytest.raises(ValueError):
        module.verify_support_partitioned(arrays, nodes, rig)


@pytest.mark.parametrize("block", [0, -1, True, 2.0])
def test_block_size_cannot_skip_or_implicitly_round_frames(block):
    arrays, nodes, rig = supported_data(1)
    with pytest.raises(ValueError, match="positive integer"):
        module.verify_support_partitioned(arrays, nodes, rig, block_frames=block)


def test_native_plan_contains_28_six_chunk_phases_and_last_two():
    groups = module.expected_phases()
    assert len(groups) == 29 and groups[0] == (66400, 76360)
    assert groups[-1] == (345280, 348600)
    assert sum(stop - start for start, stop in groups) == 282200
    assert all(a[1] == b[0] for a, b in pairwise(groups))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def entry(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Evidence:
    def __init__(self):
        self.paths = {}

    def add(self, path, digest=None):
        path = Path(path).resolve(strict=True)
        actual = entry(path)["sha256"]
        if digest is not None and digest != actual:
            raise ValueError("evidence SHA mismatch")
        if str(path) in self.paths and self.paths[str(path)] != actual:
            raise ValueError("evidence changed")
        self.paths[str(path)] = actual


def seal(root, name, stage, outputs, *, inputs=(), params=None):
    return write_json(
        root / name,
        dict(
            status="ok",
            stage=stage,
            started_at="2026-09-09T00:01:00+00:00",
            finished_at="2026-09-09T00:02:00+00:00",
            params=dict(all_input_source_snapshot_output_sha_unchanged=True, **(params or {})),
            inputs={str(p): entry(p) for p in inputs},
            outputs={key: entry(path) for key, path in outputs.items()},
        ),
    )


def audit_data(run, sha, start, stop, restores):
    return dict(
        schema="long_continuation_generic_phase_audit_v1",
        status="ok",
        run=str(run),
        frame_range=[module.FIRST, stop],
        new_interval=[start, stop],
        candidate_sha256=sha,
        all_39_checkpoint_arrays_verified=True,
        logical_initialization_count=1,
        checkpoint_restore_count=restores,
        external_pose_reset_count=0,
        full_sequence_complete=stop == module.FINAL_STOP,
        prefix_backtrace_provisional=stop != module.FINAL_STOP,
        independently_redecoded_raw_images=0,
        independent_geometric_reprojection=False,
        independent_ID_truth=False,
        all_21_candidate_arrays_exact_backtrace={
            str(i): dict(dtype="<f8", shape=[1], sha256="0" * 64) for i in range(21)
        },
        history=dict(
            active_beam_count=6,
            parent_all_history_fields_exact=True,
            all_suffix_parent_recurrences_exact=True,
        ),
    )


@pytest.fixture
def completed_chain(tmp_path, monkeypatch):
    """All 29 phase seals are real tiny files; no synthetic tracking arrays/truth."""
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "PINS", {})
    root = tmp_path / "supervisor"
    parent = tmp_path / "work/three_body/long_continuation_through66400_v1"
    binding = {"immutable": "context"}
    candidate = write_json(parent / "candidate.npz", {"deliberately": "not an NPZ"})
    parent_seal = seal(
        parent,
        "continuation.provenance.json",
        "three_body_long_continuation",
        dict(candidate=candidate),
        params=dict(bindings=binding),
    )
    minute = tmp_path / "minute"
    minute_seal = write_json(minute / "audit.provenance.json", {"old": "pinned audit"})
    monkeypatch.setattr(module.phase, "MINUTE_AUDIT", minute)
    monkeypatch.setattr(module.phase, "MINUTE_AUDIT_SHA", entry(minute_seal)["sha256"])
    bootstrap = root / "bootstrap"
    audit = audit_data(parent, entry(candidate)["sha256"], 59760, 66400, 3)
    audit["bootstrap_lineage"] = dict(audit_seal_sha256=entry(minute_seal)["sha256"])
    previous_audit = write_json(bootstrap / "summary.json", audit)
    previous_seal = seal(
        bootstrap,
        "audit.provenance.json",
        "long_supervised_bootstrap_audit",
        dict(summary=previous_audit),
        inputs=[minute_seal],
    )
    output_paths = [previous_audit, previous_seal]
    rows = []
    for index, (start, stop) in enumerate(module.expected_phases()):
        operation = root / "phases" / f"f{start:06d}-{stop:06d}"
        registration, run = operation / "registration", operation / "tracking"
        plan = write_json(
            registration / "plan.json",
            dict(
                parent_phase=str(parent),
                expected_output=str(run),
                next_native_interval=[start, stop],
                expected_restores=index + 4,
                frozen_input_and_code_binding=binding,
                parent_seal_sha256=entry(parent_seal)["sha256"],
            ),
        )
        config = write_json(
            registration / "execution.json",
            dict(
                independent_parent_audit=dict(
                    root=str(previous_seal.parent),
                    seal_sha256=entry(previous_seal)["sha256"],
                    summary_sha256=entry(previous_audit)["sha256"],
                    parent_candidate_sha256=entry(candidate)["sha256"],
                )
            ),
        )
        registered = seal(
            registration,
            "registration.provenance.json",
            "register_long_supervised_phase",
            dict(plan=plan, execution=config),
            inputs=[previous_seal, previous_audit, parent_seal],
        )
        candidate = write_json(run / "candidate.npz", {"unopened": index})
        child_seal = seal(
            run,
            "continuation.provenance.json",
            "three_body_long_continuation",
            dict(candidate=candidate),
            params=dict(bindings=binding, resume_phase=str(parent), through_frame=stop),
        )
        child = json.loads(child_seal.read_text())
        child["started_at"] = "2026-09-09T00:02:00+00:00"
        write_json(child_seal, child)
        audit_file = write_json(
            operation / "audit.json",
            audit_data(run, entry(candidate)["sha256"], start, stop, index + 4),
        )
        execution = write_json(
            operation / "child_execution.json", dict(returncode=0, child_alive_after_return=False)
        )
        operation_seal = seal(
            operation,
            "operation.provenance.json",
            "long_supervised_phase_operation",
            dict(audit=audit_file, child_execution=execution),
            inputs=[child_seal],
            params=dict(tracking_child_exit_code=0),
        )
        output_paths.extend([registered, plan, config, operation_seal, audit_file, execution])
        rows.append(dict(parent=str(parent), run=str(run), through=stop))
        parent, parent_seal = run, child_seal
        previous_seal, previous_audit = operation_seal, audit_file
    summary = write_json(
        root / "summary.json",
        dict(
            status="ok",
            through=module.FINAL_STOP,
            full_sequence_complete=True,
            logical_initialization_count=1,
            external_pose_reset_count=0,
            independent_ID_truth=False,
            phases=rows,
            final_run=str(parent),
        ),
    )
    plan = write_json(
        root / "supervisor_plan.json",
        dict(
            schema="long_remaining_operational_supervisor_v1",
            initial_stop=66400,
            initial_parent=str(tmp_path / "work/three_body/long_continuation_through66400_v1"),
            phase_intervals=[list(p) for p in module.expected_phases()],
            target_interval=[module.FIRST, module.FINAL_STOP],
            detector_started=False,
            new_external_pose_seed=False,
            fixed_bindings=binding,
            source_hashes={},
        ),
    )
    seal(
        root,
        "supervisor.provenance.json",
        "long_remaining_supervisor",
        dict(summary=summary, plan=plan, **{str(p): p for p in output_paths}),
    )
    return root


def test_completed_chain_binds_all_phases_without_deserializing_candidate(
    completed_chain, monkeypatch
):
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("chain must not open arrays"))
    evidence = Evidence()
    result = module.load_supervisor_chain(completed_chain, evidence)
    assert len(result["phases"]) == 29 and result["phases"][-1]["restores"] == 32
    assert result["phases"][-1]["interval"] == [345280, 348600]
    assert len(evidence.paths) > 230


@pytest.mark.parametrize("status", ["running", "failed", "missing"])
def test_unfinished_supervisor_refused_before_arrays(tmp_path, monkeypatch, status):
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("unfinished array opened"))
    if status != "missing":
        write_json(tmp_path / "supervisor.provenance.json", dict(status=status))
    with pytest.raises((ValueError, FileNotFoundError)):
        module.load_completed_final(tmp_path, Evidence())


def reseal_references(root, changed):
    """Repair hashes after deliberate contract change to test logic beyond checksums."""
    pending = [changed]
    while pending:
        path = pending.pop(0)
        for seal_path in root.rglob("*.provenance.json"):
            data = json.loads(seal_path.read_text())
            touched = False
            for category in ("inputs", "outputs"):
                for item in data.get(category, {}).values():
                    if Path(item["path"]) == path and item["sha256"] != entry(path)["sha256"]:
                        item["sha256"] = entry(path)["sha256"]
                        touched = True
            if touched:
                write_json(seal_path, data)
                pending.append(seal_path)


@pytest.mark.parametrize(
    "fault",
    [
        "gap",
        "restore",
        "failed_child",
        "exit",
        "lineage",
        "binding",
        "late_registration",
        "missing_backtrace",
    ],
)
def test_logical_chain_corruption_rejected_even_with_updated_output_hashes(
    completed_chain, monkeypatch, fault
):
    operation = completed_chain / "phases/f076360-086320"
    if fault == "gap":
        path = completed_chain / "summary.json"
    elif fault == "restore":
        path = operation / "registration/plan.json"
    elif fault in ("failed_child", "binding"):
        path = operation / "tracking/continuation.provenance.json"
    elif fault == "exit":
        path = operation / "child_execution.json"
    elif fault == "lineage":
        path = operation / "registration/execution.json"
    elif fault == "late_registration":
        path = operation / "registration/registration.provenance.json"
    else:
        path = operation / "audit.json"
    value = json.loads(path.read_text())
    if fault == "gap":
        value["phases"][1]["through"] += 1
    elif fault == "restore":
        value["expected_restores"] += 1
    elif fault == "failed_child":
        value["status"] = "failed"
    elif fault == "exit":
        value["returncode"] = 1
    elif fault == "lineage":
        value["independent_parent_audit"]["parent_candidate_sha256"] = "0" * 64
    elif fault == "binding":
        value["params"]["bindings"] = {"foreign": "candidate"}
    elif fault == "late_registration":
        value["finished_at"] = "2026-09-10T00:00:00+00:00"
    else:
        value["all_21_candidate_arrays_exact_backtrace"]["0"] = {}
    write_json(path, value)
    reseal_references(completed_chain, path)
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("bad chain opened arrays"))
    with pytest.raises(ValueError):
        module.load_completed_final(completed_chain, Evidence())


def test_changed_sealed_candidate_bytes_rejected_without_array_decode(completed_chain, monkeypatch):
    path = completed_chain / "phases/f076360-086320/tracking/candidate.npz"
    path.write_bytes(b"changed")
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("bad SHA opened arrays"))
    with pytest.raises(ValueError, match="SHA"):
        module.load_completed_final(completed_chain, Evidence())


def test_output_cannot_escape_its_stage_root(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "REPO", tmp_path)
    outside = write_json(tmp_path / "foreign.json", {})
    root = tmp_path / "stage"
    seal(root, "audit.provenance.json", "test", dict(payload=outside))
    with pytest.raises(ValueError, match="foreign stage"):
        module.read_record(root, "audit.provenance.json", "test", Evidence())


def test_terminal_loader_keeps_reprojection_and_raw_audit_explicitly_pending(
    completed_chain, monkeypatch
):
    chain = module.load_supervisor_chain(completed_chain, Evidence())
    arrays, _, _ = supported_data(2)
    arrays["identity_review"] = np.zeros((2, 3), bool)
    arrays["xyz"] = arrays["pose_xyz"][:, :, 1].copy()
    arrays["accepted_xyz"] = arrays["xyz"].copy()
    for i in range(14):
        arrays[f"extra_{i}"] = np.zeros(2)
    assert len(arrays) == 21
    chain["final_audit"]["counts"] = module.SUPPORT.direct_counts(arrays)
    chain["final_audit"]["all_21_candidate_arrays_exact_backtrace"] = {
        key: module.COMMON.digest_array(value) for key, value in arrays.items()
    }
    monkeypatch.setattr(module, "load_supervisor_chain", lambda *_: chain)
    monkeypatch.setattr(module.phase, "validate_summary", lambda *_a, **_k: None)
    monkeypatch.setattr(module, "FIRST", 100)
    monkeypatch.setattr(module, "FINAL_STOP", 102)
    state = {str(i): np.zeros(1) for i in range(38)}
    state["state_pose"] = np.zeros((6, 3, 3, 3))
    metadata = dict(processed_frames=2, previous_frame=101, initialize_count=1, restore_count=32)
    record = chain["final_record"]
    record["outputs"]["summary"] = dict(path=str(chain["final_run"] / "summary.json"))
    record["outputs"]["checkpoint_manifest"] = dict(
        path=str(chain["final_run"] / "checkpoint/manifest.json")
    )
    write_json(chain["final_run"] / "summary.json", {})
    monkeypatch.setattr(
        module.phase.old.prior, "read_checkpoint", lambda *_a, **_k: (state, metadata, {})
    )
    monkeypatch.setattr(module.phase.old, "read_npz", lambda *_: copy.deepcopy(arrays))
    monkeypatch.setattr(module.COMMON, "traced_candidate", lambda *_: copy.deepcopy(arrays))
    result = module.load_completed_final(completed_chain, Evidence())
    assert result["final_audit_complete"] is False and len(result["remaining_checks"]) == 4
    assert result["counts"]["direct_pectoral_fish_frames"] == 6
