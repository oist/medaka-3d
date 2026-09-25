"""Synthetic caller binding/failure tests; no recording or actual ID solver is opened."""

import copy
from pathlib import Path

import numpy as np
import pytest
import run_fast_tail_five_minute_identity_beam as run
from test_compare_fast_tail_two_minutes import candidate, refresh_derived


@pytest.fixture(autouse=True)
def forbid_solver_gpu(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("actual solver/GPU prohibited in caller tests")

    monkeypatch.setattr(run.frozen.beam, "solve_identity_node_memory_beam", forbidden)
    torch = run.frozen.sys.modules.get("torch")
    if torch is not None:
        assert not torch.cuda.is_initialized()
        monkeypatch.setattr(torch.cuda, "is_available", forbidden)


def test_readiness_gate_precedes_all_input_and_output(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "EXECUTION_READY", False)
    monkeypatch.setattr(run.proof, "Output", lambda *a: pytest.fail("output created"))
    with pytest.raises(ValueError, match="not enabled"):
        run.run(run.COMPARISON_SEAL_SHA, tmp_path / "never")
    assert not (tmp_path / "never").exists()


@pytest.fixture
def saved(tmp_path, monkeypatch):
    cmp_root = tmp_path / "comparison"
    track_root = tmp_path / "tracking"
    monkeypatch.setattr(run, "COMPARISON_ROOT", cmp_root)
    monkeypatch.setattr(run, "TRACKING_ROOT", track_root)
    raw, _ = candidate(8, 13)
    raw["identity_review"][:, 1] = True
    refresh_derived(raw)
    track = run.proof.Output(track_root)
    final_ref = track.write("candidate.npz", raw)
    milestone_ref = track.write("milestones/through_116200/candidate.npz", raw)
    assert milestone_ref["sha256"] == final_ref["sha256"]
    milestone = dict(
        schema=run.comparison.runner.STAGE + "_milestone",
        status="ok",
        horizon_stop=116200,
        scientific_interval=[66400, 116200],
        native_frames=49800,
        initialize_count=1,
        restore_count=5,
        parent_audit_sha256=run.comparison.AUDIT_SHA,
        external_pose_resets=0,
        finish_calls_so_far=0,
        all39_roundtrip=True,
        prior_history_prefix_unchanged=True,
        physical_ID_truth=False,
        five_minute_candidate=True,
        whole_process_completion_pending=True,
        source_sha256={
            str(Path(run.comparison.runner.__file__).resolve()): run.comparison.RUNNER_SHA
        },
        candidate=milestone_ref,
    )
    track.write("milestones/through_116200/summary.json", milestone)
    track.write(
        "summary.json",
        dict(
            schema=run.comparison.runner.STAGE,
            status="ok",
            scientific_interval=[66400, 116200],
            process_interval=[76360, 116200],
            native_frames=49800,
            process_frames=39840,
            initialize_count=1,
            restore_count=5,
            external_pose_resets=0,
            new_begin_calls=0,
            finish_calls=1,
            physical_ID_truth=False,
            checkpoints=[dict(previous_frame=n - 1) for n in range(78020, 116201, 1660)],
        ),
    )
    tracking = dict(
        stage=run.comparison.runner.STAGE,
        status="ok",
        inputs={str(Path(run.comparison.runner.__file__).resolve()): run.comparison.RUNNER_SHA},
        outputs=track.records,
    )
    tp = track_root / "tracking.provenance.json"
    tp.write_bytes(run.proof.json_bytes(tracking))
    tracking_sha = run.proof.sha(tp)
    milestone_sha = track.records["milestones/through_116200/summary.json"]["sha256"]
    for key, value in dict(
        TRACKING_SEAL_SHA=tracking_sha,
        MILESTONE_SHA=milestone_sha,
        CANDIDATE_SHA=final_ref["sha256"],
    ).items():
        monkeypatch.setattr(run, key, value)
    reference = dict(
        path=milestone_ref["path"], sha256=milestone_ref["sha256"], arrays=run.signature(raw)
    )
    windows = {k: list(v) for k, v in run.comparison.intervals(116200).items()}
    common = dict(
        schema=run.comparison.STAGE,
        horizon_stop=116200,
        intervals=windows,
        candidate_references=dict(original={}, fast_tail=reference),
    )
    output = run.proof.Output(cmp_root)
    output.write(
        "summary.json",
        dict(
            **common,
            status="ok",
            native_interval=[66400, 116200],
            native_frames=49800,
            whole_five_minute_process_complete=True,
            final_tracking_seal_verified=True,
            same_parent_and_horizon=True,
            same_registered_input_binding=True,
            logical_initialization_count=1,
            external_pose_resets=0,
            candidate_all21_and_source_arrays_unchanged=True,
            strict_support_recomputed_arms=2,
            strict_support_native_frames=99600,
            physical_ID_truth=False,
            all_three_ID_truth=False,
            swap_zero_verified=False,
            review_modified=False,
            identity_solver_included=False,
        ),
    )
    output.write(
        "plan.json",
        dict(
            **common,
            final_tracking_sha256=tracking_sha,
            milestone_sha256=milestone_sha,
            original_source_seal_sha256=run.comparison.ORIGINALS[116200]["seal"],
            support_gates=dict(ray_mm=2.0, native_pixel=12.0, block_frames=1660),
        ),
    )
    inputs = {
        str(Path(run.comparison.__file__).resolve()): run.COMPARISON_CODE_SHA,
        str(tp.resolve()): tracking_sha,
        str((track_root / "milestones/through_116200/summary.json").resolve()): milestone_sha,
        final_ref["path"]: final_ref["sha256"],
        milestone_ref["path"]: milestone_ref["sha256"],
    }
    seal = dict(stage=run.comparison.STAGE, status="ok", inputs=inputs, outputs=output.records)
    cp = cmp_root / "comparison.provenance.json"
    cp.write_bytes(run.proof.json_bytes(seal))
    monkeypatch.setattr(run, "COMPARISON_SEAL_SHA", run.proof.sha(cp))
    return dict(raw=raw, seal=seal, path=cp, track=track, tracking=tracking, tracking_path=tp)


def reseal_comparison(saved, monkeypatch):
    saved["path"].write_bytes(run.proof.json_bytes(saved["seal"]))
    monkeypatch.setattr(run, "COMPARISON_SEAL_SHA", run.proof.sha(saved["path"]))


def rewrite_json(root, records, name, mutate):
    path = root / name
    doc = run.proof.read_json(path)
    mutate(doc)
    path.write_bytes(run.proof.json_bytes(doc))
    records[name].update(sha256=run.proof.sha(path), size_bytes=path.stat().st_size)


def test_direct_milestone_reference_no_fake_copy_and_all21_readonly(saved, monkeypatch):
    evidence = run.proof.Evidence()
    binding = run.comparison_headers(run.COMPARISON_SEAL_SHA, evidence)
    assert binding["comparison_candidate_copy_available"] is False
    assert not (run.COMPARISON_ROOT / "candidates").exists()
    # Archive fixture is deliberately five synthetic frames; no real payload is decoded.
    monkeypatch.setattr(run, "FIRST", 8)
    monkeypatch.setattr(run, "STOP", 13)
    load, opened = np.load, []

    def counted(path, **kwargs):
        opened.append(str(path))
        return load(path, **kwargs)

    monkeypatch.setattr(run.np, "load", counted)
    raw, signatures = run.load_candidate(binding, evidence)
    assert opened == [binding["candidate_archive_references"]["milestone"]["path"]]
    assert signatures == run.signature(saved["raw"])
    assert len(raw) == 21 and all(not v.flags.writeable for v in raw.values())
    assert raw["identity_review"][:, 1].all()
    evidence.final_guard()


@pytest.mark.parametrize(
    "damage",
    ["failed", "source", "final_edge", "milestone_edge", "horizon", "reset", "reference_path"],
)
def test_bad_headers_rejected_before_archive_decode(saved, monkeypatch, damage):
    seal = saved["seal"]
    if damage == "failed":
        seal["status"] = "failed"
    elif damage == "source":
        seal["inputs"][str(Path(run.comparison.__file__).resolve())] = "0" * 64
    elif damage in {"final_edge", "milestone_edge"}:
        name = (
            "candidate.npz" if damage == "final_edge" else "milestones/through_116200/candidate.npz"
        )
        seal["inputs"].pop(str((run.TRACKING_ROOT / name).resolve()))
    elif damage in {"horizon", "reset"}:
        key, value = ("horizon_stop", 86320) if damage == "horizon" else ("external_pose_resets", 1)
        rewrite_json(
            run.COMPARISON_ROOT, seal["outputs"], "summary.json", lambda d: d.update({key: value})
        )
    else:
        for name in ["plan.json", "summary.json"]:
            rewrite_json(
                run.COMPARISON_ROOT,
                seal["outputs"],
                name,
                lambda d: d["candidate_references"]["fast_tail"].update(
                    path=str(run.COMPARISON_ROOT.parent / "escape.npz")
                ),
            )
    reseal_comparison(saved, monkeypatch)
    monkeypatch.setattr(run.np, "load", lambda *a, **k: pytest.fail("premature archive decode"))
    with pytest.raises(ValueError):
        run.comparison_headers(run.COMPARISON_SEAL_SHA, run.proof.Evidence())


def test_failed_final_cannot_hide_behind_successful_comparison(saved, monkeypatch):
    saved["tracking"]["status"] = "failed"
    saved["tracking_path"].write_bytes(run.proof.json_bytes(saved["tracking"]))
    digest = run.proof.sha(saved["tracking_path"])
    monkeypatch.setattr(run, "TRACKING_SEAL_SHA", digest)
    saved["seal"]["inputs"][str(saved["tracking_path"].resolve())] = digest
    rewrite_json(
        run.COMPARISON_ROOT,
        saved["seal"]["outputs"],
        "plan.json",
        lambda d: d.update(final_tracking_sha256=digest),
    )
    reseal_comparison(saved, monkeypatch)
    with pytest.raises(ValueError, match="successful final tracking"):
        run.comparison_headers(run.COMPARISON_SEAL_SHA, run.proof.Evidence())


@pytest.mark.parametrize("damage", ["review_signature", "final_payload", "milestone_payload"])
def test_candidate_guards_cover_all21_and_both_archives(saved, monkeypatch, damage):
    evidence = run.proof.Evidence()
    binding = run.comparison_headers(run.COMPARISON_SEAL_SHA, evidence)
    monkeypatch.setattr(run, "FIRST", 8)
    monkeypatch.setattr(run, "STOP", 13)
    if damage == "review_signature":
        binding["comparison_candidate_reference"]["arrays"]["identity_review"]["sha256"] = "0" * 64
    else:
        role = damage.removesuffix("_payload")
        path = Path(binding["candidate_archive_references"][role]["path"])
        data = bytearray(path.read_bytes())
        data[-1] ^= 1  # Same size: the SHA guard, not the size guard, must reject it.
        path.write_bytes(data)
        monkeypatch.setattr(run.np, "load", lambda *a, **k: pytest.fail("tampered archive decoded"))
    with pytest.raises(ValueError):
        run.load_candidate(binding, evidence)


def test_pure_frozen_helpers_are_reused_and_whole49800_call_has_fixed_caps(monkeypatch):
    for name in [
        "qualification",
        "scientific_geometry",
        "solve_once",
        "checked_ledger",
        "resource_report",
        "write_result",
    ]:
        assert getattr(run, name) is getattr(run.frozen, name)
    frames = np.arange(run.START, run.STOP, dtype=np.int64)
    poses = np.broadcast_to(np.zeros((3, 3, 3)), (len(frames), 3, 3, 3))
    observed = np.broadcast_to(True, (len(frames), 3, 3))
    calls, marker = [], object()

    def stub(f, unary, xyz, direct, **kwargs):
        calls.append(1)
        assert len(f) == 49800 and f[0] == 66400 and f[-1] == 116199
        assert unary.shape == (49800, 6) and not unary.any()
        assert xyz.shape == (49800, 3, 2, 3) and direct.all()
        assert kwargs == dict(beam_width=256, node_indices=(1, 0), **run.CONFIG)
        assert kwargs["max_layer_states"] == 100000 and kwargs["max_total_states"] == 1000000
        assert kwargs["max_total_edges"] == 6000000 and kwargs["initial_state"] == 0
        return marker

    monkeypatch.setattr(run.frozen.beam, "solve_identity_node_memory_beam", stub)
    assert run.solve_once(dict(frame_idx=frames, pose_xyz=poses, observed=observed)) is marker
    assert calls == [1]


def test_first_missing_direct_cannot_move_start():
    n = run.STOP - run.FIRST
    observed = np.ones((n, 3, 3), bool)
    observed[run.START - run.FIRST, 2, 0] = False
    raw = dict(
        frame_idx=np.arange(run.FIRST, run.STOP),
        pose_xyz=np.broadcast_to(0.0, (n, 3, 3, 3)),
        observed=observed,
    )
    with pytest.raises(ValueError, match="no shifted initialization"):
        run.scientific_geometry(raw)


def test_capacity_failure_one_call_full_unresolved_no_partial_output(saved, monkeypatch, tmp_path):
    raw = copy.deepcopy(saved["raw"])
    for v in raw.values():
        v.setflags(write=False)
    before = run.signature(raw)
    monkeypatch.setattr(run, "WORK", tmp_path)
    monkeypatch.setattr(run, "EXECUTION_READY", True)
    monkeypatch.setattr(run, "source_code", lambda: {})
    monkeypatch.setattr(run, "qualification", lambda evidence: dict(synthetic_fixture=True))
    monkeypatch.setattr(
        run, "comparison_headers", lambda sha, evidence: dict(synthetic_fixture=True)
    )
    monkeypatch.setattr(run, "load_candidate", lambda binding, evidence: (raw, before))
    monkeypatch.setattr(run, "scientific_geometry", lambda raw: dict(synthetic_fixture=True))
    calls = []

    def capacity(part):
        calls.append(1)
        raise run.frozen.beam.IdentityBeamCapacityError(20000, "total_states", 1000001, 1000000)

    monkeypatch.setattr(run, "solve_once", capacity)
    out = tmp_path / "capacity_result"
    with pytest.raises(run.frozen.beam.IdentityBeamCapacityError):
        run.run(run.COMPARISON_SEAL_SHA, out)
    failure = run.proof.read_json(out / "failure.json")
    assert calls == [1] and failure["solve_calls"] == 1
    assert failure["native_frame"] == 86400 and failure["attempted"] == 1000001
    assert failure["unresolved_interval"] == [66400, 116200]
    assert failure["raw21_unchanged_at_failure"] is True and run.signature(raw) == before
    assert failure["complete_identity_result"] is False
    for key in [
        "partial_identity_solution_published",
        "automatic_retry",
        "cap_increase",
        "intermediate_ID_reset",
        "physical_ID_truth",
        "exact_DP_rerun",
    ]:
        assert failure[key] is False
    assert not (out / "identity_mapping_and_ambiguity.npz").exists()
    assert not (out / "summary.json").exists()
    assert run.proof.read_json(out / "identity.provenance.json")["status"] == "failed"
