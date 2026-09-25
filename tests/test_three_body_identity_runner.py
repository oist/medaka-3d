"""CPU integration checks for full-frame scoring and sealed fixed-geometry ID exports."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_module():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location("_identity_runner_audit",
                                                     LAB / "run_identity_dp.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


RUN = load_module()


def geometry(n=168, change_at=166):
    centers = np.array([[10, 10, 20], [30, 30, 20], [50, 50, 20]], float)
    pose = np.broadcast_to(centers[:, None] + [[2, 0, 0], [0, 0, 0], [-4, 1, 0]],
                           (n, 3, 3, 3)).copy()
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    pose[change_at:] = pose[change_at:, [1, 2, 0]]
    source[change_at:] = source[change_at:, [1, 2, 0]]
    observed, review = np.ones((n, 3, 3), bool), np.zeros((n, 3), bool)
    review[:, 1] = True
    xyz = pose[:, :, 1].copy()
    return dict(frame_idx=np.arange(76620, 76620+n), pose_xyz=pose, sources=source,
                observed=observed, identity_review=review, xyz=xyz,
                accepted_xyz=np.where(~review[..., None], xyz, np.nan))


class Reader:
    def __init__(self, frames, yielded=None):
        self.frames = np.asarray(frames).copy()
        self.yielded = list(self.frames if yielded is None else yielded)
        self.closed, self.started, self.read = False, False, []

    def __iter__(self):
        self.started = True
        try:
            for frame in self.yielded:
                self.read.append(frame)
                yield SimpleNamespace(frame_idx=frame)
        finally:
            self.closed = True


def fake_emission(*_args, **_kwargs):
    costs = np.ones(6)
    costs[np.flatnonzero((RUN.PERMUTATIONS == [2, 0, 1]).all(1))[0]] = 0
    return SimpleNamespace(
        costs=costs, per_view_costs=np.column_stack((costs, np.zeros((6, 2)))),
        enabled_views=np.array([True, False, False]), sampled_pixels=np.array([7, 0, 0]),
        valid_pixels=np.array([5, 0, 0]), available_core_pixels=np.array([20, 0, 0]),
        visible_mass=np.array([[1, 2, 3], [0, 0, 0], [0, 0, 0]], float),
        rendering_performed=True,
    )


def enrollment():
    return {"templates": np.zeros((3, 3, 6)), "enabled_views": np.array([True, False, False])}


def test_collect_reads_every_native_frame_but_never_scores_initial_166(monkeypatch):
    original, colors = geometry(), enrollment()
    before = {key: value.tobytes() for key, value in original.items()}
    reader, calls, progress = Reader(original["frame_idx"]), [], []

    def score(rig, frame, raw_pose, templates, enabled, **kwargs):
        calls.append(frame.frame_idx)
        t = frame.frame_idx - original["frame_idx"][0]
        assert t >= 166 and kwargs["device"] == "cpu"
        np.testing.assert_array_equal(raw_pose, original["pose_xyz"][t])
        np.testing.assert_array_equal(templates, colors["templates"])
        np.testing.assert_array_equal(enabled, colors["enabled_views"])
        return fake_emission()

    monkeypatch.setattr(RUN, "score_identity_permutations", score)
    scores = RUN.collect_scores(reader, object(), original, colors, RUN.registered_config(),
                                device="cpu", progress=progress.append)
    assert calls == original["frame_idx"][166:].tolist()
    assert reader.read == original["frame_idx"].tolist() and reader.closed
    assert [event["processed"] for event in progress] == [100, 168]
    for key, values in scores.items():
        if key != "frame_idx":
            assert not values[:166].any(), key
    np.testing.assert_array_equal(scores["unary"][166:],
                                  np.broadcast_to(fake_emission().costs, (2, 6)))
    assert {key: value.tobytes() for key, value in original.items()} == before


@pytest.mark.parametrize("failure", ["early_eof", "extra_frame", "wrong_frame", "emission_error"])
def test_collect_closes_on_incomplete_extra_misaligned_or_failed_observation(monkeypatch, failure):
    original = geometry()
    frames = original["frame_idx"].tolist()
    expected = "complete geometry interval"
    if failure == "early_eof":
        frames.pop()
    elif failure == "extra_frame":
        frames.append(frames[-1]+1)
        expected = "extra or incorrectly aligned"
    elif failure == "wrong_frame":
        frames[167] += 1
        expected = "extra or incorrectly aligned"
    else:
        expected = "synthetic emission failure"
    reader = Reader(original["frame_idx"], frames)

    def score(*args, **kwargs):
        if failure == "emission_error":
            raise ValueError(expected)
        return fake_emission()

    monkeypatch.setattr(RUN, "score_identity_permutations", score)
    with pytest.raises(ValueError, match=expected):
        RUN.collect_scores(reader, None, original, enrollment(), RUN.registered_config(),
                           device="cpu")
    assert reader.started and reader.closed


@pytest.mark.parametrize("which", ["both_gap", "reader_only", "unsigned_wrap"])
def test_collect_rejects_non_native_sequences_before_starting_reader(which):
    original = geometry(2)
    if which == "both_gap":
        original["frame_idx"][1] += 1
        reader = Reader(original["frame_idx"])
    elif which == "unsigned_wrap":
        original["frame_idx"] = np.array([np.iinfo(np.uint64).max, 0], np.uint64)
        reader = Reader(original["frame_idx"])
    else:
        reader = Reader(original["frame_idx"] + 1)
    with pytest.raises(ValueError):
        RUN.collect_scores(reader, None, original, enrollment(), RUN.registered_config(),
                           device="cpu")
    assert not reader.started


def test_optimize_and_export_correct_the_synthetic_raw_cycle_without_changing_geometry():
    original = geometry(5, change_at=2)
    before = {key: value.copy() for key, value in original.items()}
    scores = {"frame_idx": original["frame_idx"].copy(), "unary": np.zeros((5, 6))}
    result, output = RUN.optimize_scores(original, scores, RUN.registered_config())
    np.testing.assert_array_equal(result.permutation_by_id[:2], [[0, 1, 2]]*2)
    np.testing.assert_array_equal(result.permutation_by_id[2:], [[2, 0, 1]]*3)
    np.testing.assert_array_equal(output["proposed_pose_xyz"],
                                  np.broadcast_to(original["pose_xyz"][0], (5, 3, 3, 3)))
    np.testing.assert_array_equal(output["prior_review_for_proposed_identity"][2:],
                                  [[False, False, True]]*3)
    assert np.isnan(output["conservative_accepted_xyz"][2:, 2]).all()
    assert scores["transitions"].shape == (4, 6, 6)
    for key, value in original.items():
        np.testing.assert_array_equal(value, before[key])


def test_optimize_refuses_costs_for_different_native_frames_before_mutating_scores():
    original = geometry(3)
    scores = {"frame_idx": original["frame_idx"]+1, "unary": np.zeros((3, 6))}
    with pytest.raises(ValueError, match="score frames must match"):
        RUN.optimize_scores(original, scores, RUN.registered_config())
    assert set(scores) == {"frame_idx", "unary"}


@pytest.fixture
def hermetic_main(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    repo.mkdir()
    paths = {}
    for name in ("geometry_run", "enrollment_run", "enrollment_audit", "sequence", "input_root"):
        paths[name] = repo / name
        paths[name].mkdir()
    source = repo / "lab" / "source.py"
    source.parent.mkdir()
    source.write_text("# synthetic source snapshot\n", encoding="utf-8")
    protocol = source.with_name("PROTOCOL.md")
    protocol.write_text("Synthetic protocol only", encoding="utf-8")
    evidence = repo / "sealed_input.bin"
    evidence.write_bytes(b"synthetic independently verified input")
    original, reader = geometry(), Reader(geometry()["frame_idx"])
    original_before = {key: value.copy() for key, value in original.items()}
    actual_sha = RUN.base.sha256
    loaded = []
    inputs = SimpleNamespace(
        geometry=original, enrollment=enrollment(), frame_idx=original["frame_idx"].copy(),
        geometry_sequence_root=paths["sequence"], input_root=paths["input_root"],
        verified_hashes={str(evidence): actual_sha(evidence)},
        prefix_audit={"synthetic_fixture": True, "post_inference_assets_opened": False},
    )

    def loader(*args):
        expected = tuple(paths[key] for key in
                         ("geometry_run", "enrollment_run", "enrollment_audit"))
        assert args == expected
        loaded.append(args)
        return inputs

    def sequence(sequence_root, input_root):
        assert sequence_root == paths["sequence"] and input_root == paths["input_root"]
        return reader

    base = SimpleNamespace(REPO=repo, LAB=source.parent, CALIBRATION_DIR="calibration",
                           sha256=actual_sha, SequenceReader=sequence, load_rig=lambda _p: object())
    monkeypatch.setattr(RUN, "base", base)
    monkeypatch.setattr(RUN, "load_identity_inputs", loader)
    monkeypatch.setattr(RUN, "code_paths", lambda: [source, protocol])
    monkeypatch.setattr(RUN, "package_versions", lambda: {"synthetic_cpu": True})
    monkeypatch.setattr(RUN.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(RUN.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(RUN, "score_identity_permutations", fake_emission)
    output = repo / "new_output"
    argv = [argument for key in ("geometry_run", "enrollment_run", "enrollment_audit")
            for argument in ("--" + key.replace("_", "-"), str(paths[key]))]
    argv += ["--output", str(output)]
    previous_threads = RUN.torch.get_num_threads()
    try:
        yield SimpleNamespace(repo=repo, paths=paths, source=source, protocol=protocol,
                              evidence=evidence, original=original, before=original_before,
                              reader=reader, loaded=loaded, output=output, argv=argv,
                              sha=actual_sha)
    finally:
        RUN.torch.set_num_threads(previous_threads)


def test_main_cpu_fixture_seals_outputs_snapshots_and_no_evaluation_inputs(hermetic_main):
    test = hermetic_main
    summary = RUN.main(test.argv)
    assert summary["technical_completion"] and summary["returned_frames"] == 168
    assert summary["scored_frames"] == 2 and summary["sampled_pixels"] == [14, 0, 0]
    assert not summary["evaluation_data_opened_during_inference"]
    assert summary["geometry_fits_in_this_stage"] == summary["external_reset_count"] == 0
    assert len(test.loaded) == 1 and test.reader.closed
    record = json.loads((test.output / "experiment.provenance.json").read_text())
    assert record["status"] == "ok" and record["stage"] == RUN.STAGE
    assert set(record["inputs"]) == {str(test.evidence)}
    assert not record["params"]["evaluation_data_opened_during_inference"]
    for item in record["outputs"].values():
        assert test.sha(item["path"]) == item["sha256"]
    manifest = json.loads((test.output / "source_manifest.json").read_text())
    for relative, item in manifest.items():
        assert test.sha(test.output / item["snapshot"]) == test.sha(test.repo / relative)
        assert item["sha256"] == test.sha(test.repo / relative)
    with np.load(test.output / "identity.npz") as arrays:
        np.testing.assert_array_equal(arrays["raw_identity_review"],
                                      test.original["identity_review"])
        np.testing.assert_array_equal(arrays["identity_to_raw_slot"][166:], [[2, 0, 1]]*2)
    for key, value in test.original.items():
        np.testing.assert_array_equal(value, test.before[key])


@pytest.mark.parametrize("failure", [
    "loader", "early_eof", "input_mutation", "source_mutation", "snapshot_mutation",
])
def test_main_failure_is_sealed_without_success_claim(hermetic_main, monkeypatch, failure):
    test = hermetic_main
    if failure == "loader":
        def bad_loader(*args):
            raise ValueError("synthetic seal rejected")
        monkeypatch.setattr(RUN, "load_identity_inputs", bad_loader)
    elif failure == "early_eof":
        test.reader.yielded.pop()
    else:
        def mutate(*args, **kwargs):
            path = test.evidence if failure == "input_mutation" else test.source
            if failure == "snapshot_mutation":
                path = test.output / "code_snapshot" / test.source.relative_to(test.repo)
            path.write_text("changed during inference", encoding="utf-8")
            return fake_emission()
        monkeypatch.setattr(RUN, "score_identity_permutations", mutate)
    with pytest.raises((ValueError, RuntimeError)):
        RUN.main(test.argv)
    record = json.loads((test.output / "experiment.provenance.json").read_text())
    assert record["status"] == "failed" and record["error"]
    assert not (test.output / "summary.json").exists()
    if failure != "loader":
        assert test.reader.closed


def test_main_never_overwrites_existing_output_or_starts_loader(hermetic_main):
    test = hermetic_main
    test.output.mkdir()
    marker = test.output / "preserved.bin"
    marker.write_bytes(b"existing")
    with pytest.raises(ValueError, match="new output directory"):
        RUN.main(test.argv)
    assert marker.read_bytes() == b"existing" and not test.loaded


def test_main_outside_workspace_is_rejected_before_loader(hermetic_main):
    test = hermetic_main
    argv = [*test.argv[:-1], str(test.repo.parent / "outside")]
    with pytest.raises(ValueError, match="inside the workspace"):
        RUN.main(argv)
    assert not test.loaded and not (test.repo.parent / "outside").exists()
