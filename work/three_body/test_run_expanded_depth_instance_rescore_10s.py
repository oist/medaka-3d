"""CPU-only caller contracts: disabled execution, actual source closure and evidence edges."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import run_expanded_depth_instance_rescore_10s as run
import test_instance_rescore_tracker as toy


def test_execution_disabled_before_outputs_or_any_scientific_input(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("disabled preparation may not open data, initialize GPU or write a stage")

    monkeypatch.setattr(run.base, "load_parent", forbidden)
    monkeypatch.setattr(run.proof, "Output", forbidden)
    monkeypatch.setattr(run.torch.cuda, "set_per_process_memory_fraction", forbidden)
    with pytest.raises(ValueError, match="not enabled"):
        run.main(
            SimpleNamespace(
                execute=True,
                tracker_sha=run.TRACKER_SHA,
                output=tmp_path / "never",
                receipt_root=tmp_path,
            )
        )
    assert not (tmp_path / "never").exists()
    assert run.EXECUTION_READY is False


def test_actual_import_closure_contains_all_eight_pins_and_preserves_original_contract(monkeypatch):
    code = run.source_closure()
    assert str(Path(run.__file__).resolve()) in code
    assert len(run.PINS) == 8
    assert not run.torch.cuda.is_initialized()
    broken = dict(code)
    broken.pop(str((run.WORK / "expanded_depth_instance_rescore_tracker.py").resolve()))
    monkeypatch.setattr(run.prior, "local_code", lambda: broken)
    with pytest.raises(ValueError, match="source missing"):
        run.source_closure()


def summaries():
    free_records, bounded_records = [], []
    for frame in (77709, 77723):
        free_trials, bounded_trials = [], []
        for index in range(4):
            gates = dict(
                finite=True,
                forward=True,
                ray_rms=True,
                box=False,
                anatomy=True,
                source_exclusive=True,
                mouth_direct=True,
                chest_direct=True,
                no_other_direct_support_loss=True,
            )
            row = dict(
                index=index,
                views=["top_cam", "side1_cam"],
                rows=[1, 2],
                mouth_xyz=[70.0, 150.0, -9.0],
                gates=gates,
            )
            free_trials.append(row)
            bounded_trials.append(
                dict(
                    index=index,
                    views=row["views"],
                    rows=row["rows"],
                    free_point_xyz=row["mouth_xyz"],
                    free_gates=copy.deepcopy(gates),
                    gates={**gates, "box": True, "ray_rms": False},
                    ray_rms_mm=1.7,
                    passes_existing_spatial_gates=False,
                )
            )
        free_records.append(dict(frame_idx=frame, trials=free_trials))
        bounded_records.append(dict(frame_idx=frame, trials=bounded_trials))
    return dict(
        free=dict(records=free_records),
        bounded=dict(records=bounded_records),
        sparse=dict(
            native_frames=1660,
            views_compared=4980,
            all_saved_dense_fields_BYTEEXACT=True,
            all_normalized_and_adjusted_costs_BYTEEXACT=True,
            saved_workload_equivalence_only=True,
            continuous_rollout_equivalence=False,
        ),
        dense_control=dict(
            process_interval=[76360, 78020],
            retained_interval=[49800, 78020],
            parent_audit_sha256=run.AUDIT_SHA,
            parent_checkpoint_sha256=run.base.audit.CHECKPOINT_SHA,
            initialize_count=1,
            restore_count=5,
            all39_roundtrip=True,
        ),
    )


@pytest.mark.parametrize(
    "damage",
    [
        "missing_subset",
        "source_rows",
        "missing_gate",
        "nondepth_bound",
        "no_rms_failure",
        "partial_sparse",
        "claimed_rollout",
        "different_parent",
    ],
)
def test_required_rationale_rejects_wrong_or_incomplete_evidence(damage):
    data = summaries()
    run.validate_rationale(data)
    if damage == "missing_subset":
        data["free"]["records"][1]["trials"].pop()
    elif damage == "source_rows":
        data["bounded"]["records"][0]["trials"][0]["rows"] = [2, 1]
    elif damage == "missing_gate":
        data["free"]["records"][0]["trials"][0]["gates"].pop("anatomy")
    elif damage == "nondepth_bound":
        data["free"]["records"][0]["trials"][0]["mouth_xyz"][0] = -1
    elif damage == "no_rms_failure":
        data["bounded"]["records"][0]["trials"][0]["ray_rms_mm"] = 1.4
    elif damage == "partial_sparse":
        data["sparse"]["native_frames"] = 3
    elif damage == "claimed_rollout":
        data["sparse"]["continuous_rollout_equivalence"] = True
    else:
        data["dense_control"]["parent_audit_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        run.validate_rationale(data)


@pytest.fixture
def saved_rationale(tmp_path, monkeypatch):
    roots = {role: tmp_path / path.name for role, path in run.ROOTS.items()}
    seal_specs = {}
    data = summaries()
    for role in ("free", "bounded", "dense_control", "sparse"):
        root = roots[role]
        output = run.proof.Output(root)
        output.write("summary.json", dict(schema=root.name, status="ok", **data[role]))
        if role == "dense_control":
            output.write(
                "plan.json",
                dict(original_binding={"synthetic": True}, parent_audit_sha256=run.AUDIT_SHA),
            )
        inputs = {}
        parent = {"bounded": "free", "sparse": "dense_control"}.get(role)
        if parent:
            filename, sha = seal_specs[parent]
            inputs[str((roots[parent] / filename).resolve())] = sha
        filename = run.SEALS[role][0]
        seal_path = root / filename
        seal_path.write_bytes(
            run.proof.json_bytes(
                dict(stage=root.name, status="ok", inputs=inputs, outputs=output.records)
            )
        )
        seal_specs[role] = (filename, run.prior.checks.sha(seal_path))
    monkeypatch.setattr(run, "ROOTS", roots)
    monkeypatch.setattr(run, "SEALS", seal_specs)
    return roots, seal_specs


def test_bound_rationale_reads_only_sealed_small_members_and_links_inputs(saved_rationale):
    roots, specs = saved_rationale
    evidence = run.original.phase.Digests()
    refs, plan = run.load_rationale(evidence)
    assert set(refs) == set(roots)
    assert all(refs[k]["sha256"] == specs[k][1] for k in refs)
    assert plan["original_binding"] == {"synthetic": True}
    assert len(evidence.paths) == 9  # Four seals, four summaries, control plan.
    assert all(Path(path).suffix == ".json" for path in evidence.paths)
    evidence.final_guard()


def test_bound_member_rejects_wrong_path_and_changed_summary(saved_rationale):
    roots, specs = saved_rationale
    root = roots["free"]
    seal = run.proof.read_json(root / specs["free"][0])
    altered = copy.deepcopy(seal)
    altered["outputs"]["summary.json"]["path"] = str(roots["bounded"] / "summary.json")
    with pytest.raises(ValueError, match="descriptor"):
        run.bound_member(root, altered, "summary.json", run.original.phase.Digests())
    path = root / "summary.json"
    path.write_bytes(path.read_bytes().replace(b'"ok"', b'"no"'))  # Same size, wrong SHA.
    with pytest.raises(ValueError):
        run.load_rationale(run.original.phase.Digests())


def test_missing_cross_stage_seal_edge_is_rejected(saved_rationale):
    roots, specs = saved_rationale
    path = roots["bounded"] / specs["bounded"][0]
    seal = run.proof.read_json(path)
    seal["inputs"] = {}
    path.write_bytes(run.proof.json_bytes(seal))
    specs["bounded"] = (path.name, run.prior.checks.sha(path))
    with pytest.raises(ValueError, match="seal edge"):
        run.load_rationale(run.original.phase.Digests())


def test_scientific_scope_and_every_frame_policy_survive_frozen_codec(tmp_path):
    tracker = run.variant.ExpandedDepthInstanceRescoreContinuousTracker(*toy.inputs.__wrapped__())
    policies = run.policy_binding(tracker)
    assert policies["scientific_scope"]["post_fit_candidate_selection_only"] is False
    assert policies["scientific_scope"]["proposal_and_fitting_domains_changed"] is True
    assert policies["new_scoring_policy"]["post_fit_only"] is True  # Only the mask term.
    tracker.last_instance_diagnostics = dict(
        frame_idx=run.START,
        candidate_index=np.arange(2),
        gate_preparation_seconds=0.0,
        candidate_scoring_seconds=0.0,
        views={"top_cam": dict(eligible=True, cost=np.array([np.inf, 1.0]))},
        policy=policies["new_scoring_policy"],
        distance_backend=policies["distance_backend"],
    )
    output = run.proof.Output(tmp_path / "diagnostics")
    row = run.diagnostic.save_diagnostic(output, tracker, run.START)
    saved = json.loads((output.root / f"diagnostics/f{run.START}.json").read_text())
    with np.load(saved["archive"]["path"], allow_pickle=False) as archive:
        restored = run.diagnostic.restore_diagnostic(
            saved["tree"], {key: archive[key] for key in archive.files}
        )
    assert restored["expanded_depth_policy"] == policies["expanded_depth_policy"]
    assert restored["distance_backend"] == policies["distance_backend"]
    assert restored["policy"] == policies["new_scoring_policy"]
    assert np.isposinf(restored["views"]["top_cam"]["cost"][0])
    assert row["frame_idx"] == 76360 and row["candidate_count"] == 2
    assert tracker.initialize_count == 0 and not run.torch.cuda.is_initialized()
