"""Synthetic source/binding tests only; real candidate/solver/GPU access is forbidden."""

import copy
from pathlib import Path

import numpy as np
import pytest
import run_expanded_depth_five_minute_identity_beam as run
from test_compare_fast_tail_two_minutes import candidate, refresh_derived


@pytest.fixture(autouse=True)
def forbid_solver_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("actual solver/GPU prohibited")

    monkeypatch.setattr(run.frozen.beam, "solve_identity_node_memory_beam", forbidden)
    torch = run.frozen.sys.modules.get("torch")
    if torch is not None:
        assert not torch.cuda.is_initialized()
        monkeypatch.setattr(torch.cuda, "is_available", forbidden)


def test_disabled_before_input_or_output(tmp_path, monkeypatch):
    assert run.EXECUTION_READY is False
    monkeypatch.setattr(run.proof, "Output", lambda *a: pytest.fail("created output"))
    with pytest.raises(ValueError, match="not enabled"):
        run.run("0" * 64, tmp_path / "never")
    assert not (tmp_path / "never").exists()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "WORK", tmp_path)
    monkeypatch.setattr(run, "COMPARISON_ROOT", tmp_path / run.comparison.STAGE)
    runner = tmp_path / "continue_expanded_depth_five_minutes.py"
    tracker = tmp_path / run.TRACKER_NAME
    control_spec = copy.deepcopy(run.comparison.saved.SPECS[run.comparison.ARMS[0]])
    control_runner = tmp_path / control_spec["runner"]
    for path in (runner, tracker, control_runner):
        path.write_text("# synthetic source " + path.name)
    monkeypatch.setattr(run.comparison, "VARIANT_RUNNER", runner)
    monkeypatch.setattr(run, "RUNNER_SHA", run.proof.sha(runner))
    monkeypatch.setattr(run, "TRACKER_SHA", run.proof.sha(tracker))
    control_spec["source_sha"] = run.proof.sha(control_runner)
    monkeypatch.setitem(run.comparison.saved.SPECS, run.comparison.ARMS[0], control_spec)
    raw, _ = candidate(8, 13)
    raw["identity_review"][:, 1] = True
    refresh_derived(raw)
    references, producers = {}, {}
    inputs = {str(Path(run.comparison.__file__).resolve()): run.COMPARISON_SHA}
    for arm in run.comparison.ARMS:
        stage = run.comparison.VARIANT_STAGE if arm == run.ARM else control_spec["stage"]
        root = tmp_path / stage
        root.mkdir()
        source = runner if arm == run.ARM else control_runner
        source_inputs = {str(source.resolve()): run.proof.sha(source)}
        if arm == run.ARM:
            source_inputs[str(tracker.resolve())] = run.TRACKER_SHA
        final = str((root / "candidate.npz").resolve())
        milestone = str((root / "milestones/through_116200/candidate.npz").resolve())
        # Original archives are intentionally absent; comparison output signatures bind them.
        producer = dict(
            stage=stage,
            status="ok",
            inputs=source_inputs,
            outputs={
                "candidate.npz": dict(path=final, sha256="a" * 64, size_bytes=101),
                "milestones/through_116200/candidate.npz": dict(
                    path=milestone, sha256="b" * 64, size_bytes=202
                ),
            },
        )
        producer_path = root / "tracking.provenance.json"
        producer_path.write_bytes(run.proof.json_bytes(producer))
        producer_sha = run.proof.sha(producer_path)
        references[arm] = dict(
            path=final,
            sha256="a" * 64,
            milestone_path=milestone,
            milestone_sha256="b" * 64,
            terminal_horizon=116200,
            arrays=run.signature(raw),
            tracking_seal_sha256=producer_sha,
        )
        inputs.update(
            {
                str(producer_path.resolve()): producer_sha,
                final: "a" * 64,
                milestone: "b" * 64,
                **source_inputs,
            }
        )
        producers[arm] = (producer_path, producer)
    scope = dict(
        schema=run.comparison.STAGE,
        native_interval=[66400, 116200],
        native_frames=49800,
        intervals={k: list(v) for k, v in run.comparison.INTERVALS.items()},
        terminal_horizon_by_arm=dict.fromkeys(run.comparison.ARMS, 116200),
        lower_z_mm_by_arm={run.comparison.ARMS[0]: -6.0, run.ARM: -12.0},
        depth_and_mask_change_native=76360,
        sparse_backend_start_by_arm={run.comparison.ARMS[0]: 78020, run.ARM: 76360},
        strict_support_ray_mm=2.0,
        strict_support_pixel_gate=12.0,
        physical_ID_truth=False,
        physical_depth_truth=False,
        automatic_adoption=False,
        candidate_references=references,
    )
    out = run.proof.Output(run.COMPARISON_ROOT)
    out.write(
        "plan.json",
        dict(**scope, arms=list(run.comparison.ARMS), variant_runner_sha256=run.RUNNER_SHA),
    )
    out.write(
        "summary.json",
        dict(
            **scope,
            status="ok",
            candidate_modified=False,
            review_modified=False,
            all_three_physical_ID_truth=False,
            human_truth_created=False,
            adopted=False,
        ),
    )
    out.write(f"candidates/{run.ARM}.npz", raw)
    seal = dict(stage=run.comparison.STAGE, status="ok", inputs=inputs, outputs=out.records)
    path = run.COMPARISON_ROOT / "comparison.provenance.json"
    path.write_bytes(run.proof.json_bytes(seal))
    return dict(raw=raw, seal=seal, path=path, producers=producers)


def reseal(saved):
    saved["path"].write_bytes(run.proof.json_bytes(saved["seal"]))
    return run.proof.sha(saved["path"])


def update_doc(saved, name, mutate):
    path = run.COMPARISON_ROOT / name
    value = run.proof.read_json(path)
    mutate(value)
    path.write_bytes(run.proof.json_bytes(value))
    saved["seal"]["outputs"][name].update(
        sha256=run.proof.sha(path), size_bytes=path.stat().st_size
    )


def test_comparison_copy_one_decode_all21_readonly_no_original_archive(saved, monkeypatch):
    evidence = run.proof.Evidence()
    seal, binding = run.comparison_headers(reseal(saved), evidence)
    monkeypatch.setattr(run, "FIRST", 8)
    monkeypatch.setattr(run, "STOP", 13)
    original_load, opened = np.load, []

    def counted(path, **kwargs):
        opened.append(Path(path))
        return original_load(path, **kwargs)

    monkeypatch.setattr(run.np, "load", counted)
    raw, signature, ref = run.load_candidate(seal, binding, evidence)
    assert opened == [Path(ref["path"])]
    assert signature == run.signature(saved["raw"]) == binding["raw_candidate_reference"]["arrays"]
    assert ref["sha256"] not in ("a" * 64, "b" * 64)
    assert len(raw) == 21 and all(not value.flags.writeable for value in raw.values())
    assert raw["identity_review"][:, 1].all()
    assert binding["original_candidate_payloads_rehashed_here"] is False
    assert binding["original_final_milestone_equality_verified_by_successful_comparison"] is True
    for role in ("path", "milestone_path"):
        assert not Path(binding["raw_candidate_reference"][role]).exists()
    evidence.final_guard()


@pytest.mark.parametrize(
    "damage",
    [
        "failed_comparison",
        "comparison_source",
        "runner_plan",
        "variant_tracker_edge",
        "control_runner_edge",
        "milestone_edge",
        "final_edge",
        "horizon",
        "depth_start",
        "physical_truth",
        "review_modified",
        "signature_mismatch",
        "escape_path",
    ],
)
def test_bad_headers_fail_before_candidate(saved, monkeypatch, damage):
    if damage == "failed_comparison":
        saved["seal"]["status"] = "failed"
    elif damage == "comparison_source":
        saved["seal"]["inputs"][str(Path(run.comparison.__file__).resolve())] = "0" * 64
    elif damage.endswith("edge"):
        refs = run.proof.read_json(run.COMPARISON_ROOT / "plan.json")["candidate_references"]
        paths = dict(
            variant_tracker_edge=str((run.WORK / run.TRACKER_NAME).resolve()),
            control_runner_edge=str(
                (run.WORK / run.comparison.saved.SPECS[run.comparison.ARMS[0]]["runner"]).resolve()
            ),
            milestone_edge=refs[run.ARM]["milestone_path"],
            final_edge=refs[run.ARM]["path"],
        )
        saved["seal"]["inputs"].pop(paths[damage])
    elif damage == "runner_plan":
        update_doc(saved, "plan.json", lambda d: d.update(variant_runner_sha256="0" * 64))
    elif damage == "horizon":
        update_doc(
            saved, "summary.json", lambda d: d["terminal_horizon_by_arm"].update({run.ARM: 86320})
        )
    elif damage == "depth_start":
        update_doc(saved, "summary.json", lambda d: d.update(depth_and_mask_change_native=78020))
    elif damage == "physical_truth":
        update_doc(saved, "summary.json", lambda d: d.update(physical_ID_truth=True))
    elif damage == "review_modified":
        update_doc(saved, "summary.json", lambda d: d.update(review_modified=True))
    elif damage == "signature_mismatch":
        update_doc(
            saved,
            "summary.json",
            lambda d: d["candidate_references"][run.ARM]["arrays"]["sources"].update(
                sha256="0" * 64
            ),
        )
    else:
        for name in ("plan.json", "summary.json"):
            update_doc(
                saved,
                name,
                lambda d: d["candidate_references"][run.ARM].update(
                    path=str(run.WORK / "escape.npz")
                ),
            )
    monkeypatch.setattr(run.np, "load", lambda *a, **k: pytest.fail("candidate opened"))
    with pytest.raises(ValueError):
        run.comparison_headers(reseal(saved), run.proof.Evidence())


@pytest.mark.parametrize("arm", run.comparison.ARMS)
@pytest.mark.parametrize("damage", ["failed", "runner", "candidate_sha"])
def test_both_producer_success_sources_and_archives_bound(saved, arm, damage, monkeypatch):
    path, producer = saved["producers"][arm]
    if damage == "failed":
        producer["status"] = "failed"
    elif damage == "runner":
        producer["inputs"].clear()
    else:
        producer["outputs"]["candidate.npz"]["sha256"] = "c" * 64
    path.write_bytes(run.proof.json_bytes(producer))
    digest = run.proof.sha(path)
    saved["seal"]["inputs"][str(path.resolve())] = digest
    for name in ("plan.json", "summary.json"):
        update_doc(
            saved,
            name,
            lambda d: d["candidate_references"][arm].update(tracking_seal_sha256=digest),
        )
    monkeypatch.setattr(run.np, "load", lambda *a, **k: pytest.fail("candidate opened"))
    with pytest.raises(ValueError):
        run.comparison_headers(reseal(saved), run.proof.Evidence())


def test_all21_reference_including_review_is_rechecked_after_decode(saved, monkeypatch):
    seal, binding = run.comparison_headers(reseal(saved), run.proof.Evidence())
    binding["raw_candidate_reference"]["arrays"]["identity_review"]["sha256"] = "0" * 64
    monkeypatch.setattr(run, "FIRST", 8)
    monkeypatch.setattr(run, "STOP", 13)
    with pytest.raises(ValueError, match="all21"):
        run.load_candidate(seal, binding, run.proof.Evidence())


def test_whole49800_once_fixed_helpers_and_capacity(monkeypatch):
    for name in (
        "qualification",
        "scientific_geometry",
        "solve_once",
        "checked_ledger",
        "resource_report",
        "write_result",
        "explicit_json",
        "failure_record",
    ):
        assert getattr(run, name) is getattr(run.frozen, name)
    part = dict(
        frame_idx=np.arange(66400, 116200, dtype=np.int64),
        pose_xyz=np.broadcast_to(0.0, (49800, 3, 3, 3)),
        observed=np.broadcast_to(True, (49800, 3, 3)),
    )
    calls, marker = [], object()

    def stub(frames, unary, xyz, direct, **kwargs):
        calls.append(1)
        assert len(frames) == 49800 and frames[0] == 66400 and frames[-1] == 116199
        assert unary.shape == (49800, 6) and unary.dtype == np.float64 and not unary.any()
        assert xyz.shape == (49800, 3, 2, 3) and direct.all()
        assert kwargs == dict(beam_width=256, node_indices=(1, 0), **run.CONFIG)
        assert kwargs["max_total_states"] == 1000000 and kwargs["max_total_edges"] == 6000000
        assert kwargs["max_layer_states"] == 100000 and kwargs["initial_state"] == 0
        return marker

    monkeypatch.setattr(run.frozen.beam, "solve_identity_node_memory_beam", stub)
    assert run.solve_once(part) is marker and calls == [1]
    assert run.explicit_json(dict(gap=np.inf, missing=np.nan)) == dict(
        gap="+Infinity", missing="NaN"
    )


def test_missing_first_direct_cannot_shift_initialization():
    n = run.STOP - run.FIRST
    observed = np.ones((n, 3, 3), bool)
    observed[run.START - run.FIRST, 2, 0] = False
    with pytest.raises(ValueError, match="no shifted initialization"):
        run.scientific_geometry(
            dict(
                frame_idx=np.arange(run.FIRST, run.STOP),
                pose_xyz=np.broadcast_to(0.0, (n, 3, 3, 3)),
                observed=observed,
            )
        )


def test_capacity_failure_one_call_retains_raw_and_full_unresolved(saved, monkeypatch):
    raw = saved["raw"]
    for value in raw.values():
        value.setflags(write=False)
    signatures = run.signature(raw)
    monkeypatch.setattr(run, "EXECUTION_READY", True)
    monkeypatch.setattr(run, "source_code", lambda: {})
    monkeypatch.setattr(run, "qualification", lambda e: {})
    monkeypatch.setattr(run, "comparison_headers", lambda s, e: ({}, {}))
    monkeypatch.setattr(run, "load_candidate", lambda *a: (raw, signatures, {}))
    monkeypatch.setattr(run, "scientific_geometry", lambda raw: {})
    calls = []

    def fail(part):
        calls.append(1)
        raise run.frozen.beam.IdentityBeamCapacityError(20000, "total_states", 1000001, 1000000)

    monkeypatch.setattr(run, "solve_once", fail)
    out = run.WORK / "failed_identity"
    with pytest.raises(run.frozen.beam.IdentityBeamCapacityError):
        run.run("0" * 64, out)
    failure = run.proof.read_json(out / "failure.json")
    assert calls == [1] and failure["solve_calls"] == 1 and failure["native_frame"] == 86400
    assert failure["unresolved_interval"] == [66400, 116200]
    assert failure["raw21_unchanged_at_failure"] and run.signature(raw) == signatures
    for key in (
        "automatic_retry",
        "cap_increase",
        "intermediate_ID_reset",
        "physical_ID_truth",
        "partial_identity_solution_published",
        "complete_identity_result",
        "exact_DP_rerun",
    ):
        assert failure[key] is False
    assert not (out / "identity_mapping_and_ambiguity.npz").exists()
    assert run.proof.read_json(out / "identity.provenance.json")["status"] == "failed"
