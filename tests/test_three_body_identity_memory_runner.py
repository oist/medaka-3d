"""The new memory search is connected to full-frame evidence and failure provenance."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"


def load_runner():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {name: sys.modules.get(name) for name in names}, sys.path.copy()
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location(
            "_memory_runner", LAB / "run_identity_memory.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


RUN = load_runner()


def geometry(n=168):
    centers = np.array([[10., 10, 20], [30, 30, 20], [50, 50, 20]])
    pose = np.broadcast_to(centers[:, None] + [[2, 0, 0], [0, 0, 0], [-4, 1, 0]],
                           (n, 3, 3, 3)).copy()
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    pose[166:], source[166:] = pose[166:, [1, 2, 0]], source[166:, [1, 2, 0]]
    observed, review = np.ones((n, 3, 3), bool), np.zeros((n, 3), bool)
    review[:, 1] = True
    xyz = pose[:, :, 1].copy()
    return dict(frame_idx=np.arange(76620, 76620+n), pose_xyz=pose, sources=source,
                observed=observed, identity_review=review, xyz=xyz,
                accepted_xyz=np.where(~review[..., None], xyz, np.nan))


def fake_emission(*_args, **_kwargs):
    return SimpleNamespace(
        costs=np.full(6, .25), per_view_costs=np.tile([.25, 0, 0], (6, 1)),
        enabled_views=np.array([True, False, False]), sampled_pixels=np.array([7, 0, 0]),
        valid_pixels=np.array([7, 0, 0]), available_core_pixels=np.array([9, 0, 0]),
        visible_mass=np.zeros((3, 3)), rendering_performed=True,
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    repo = tmp_path / "workspace"
    repo.mkdir()
    paths = {}
    for key in ("geometry_run", "enrollment_run", "enrollment_audit", "sequence", "input_root"):
        paths[key] = repo / key
        paths[key].mkdir()
    source = repo / "lab" / "source.py"
    source.parent.mkdir()
    source.write_text("# test source\n")
    evidence = repo / "evidence.bin"
    evidence.write_bytes(b"input")
    original = geometry()
    enrollment = {"templates": np.zeros((3, 3, 6)), "enabled_views": np.ones(3, bool)}
    reader = SimpleNamespace(frames=original["frame_idx"], closed=False)

    class Reader:
        frames = reader.frames

        def __iter__(self):
            try:
                for frame in self.frames:
                    yield SimpleNamespace(frame_idx=int(frame))
            finally:
                reader.closed = True

    context = SimpleNamespace(
        geometry=original, enrollment=enrollment, frame_idx=original["frame_idx"],
        geometry_sequence_root=paths["sequence"], input_root=paths["input_root"],
        verified_hashes={str(evidence): RUN.base.sha256(evidence)},
        prefix_audit={"test_only": True},
    )
    base = SimpleNamespace(REPO=repo, LAB=source.parent, CALIBRATION_DIR="calibration",
                           sha256=RUN.base.sha256, SequenceReader=lambda *_: Reader(),
                           load_rig=lambda *_: object())
    loaded = []

    def load(*args):
        loaded.append(args)
        return context

    monkeypatch.setattr(RUN, "base", base)
    monkeypatch.setattr(RUN.common, "base", base)
    monkeypatch.setattr(RUN.common, "load_identity_inputs", load)
    monkeypatch.setattr(RUN.common, "package_versions", lambda: {"cpu_fixture": True})
    monkeypatch.setattr(RUN.common, "score_identity_permutations", fake_emission)
    monkeypatch.setattr(RUN, "code_paths", lambda: [source])
    monkeypatch.setattr(RUN.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(RUN.torch.cuda, "synchronize", lambda: None)
    output = repo / "output"
    argv = [arg for key in ("geometry_run", "enrollment_run", "enrollment_audit")
            for arg in ("--" + key.replace("_", "-"), str(paths[key]))]
    argv += ["--output", str(output)]
    threads = RUN.torch.get_num_threads()
    try:
        yield SimpleNamespace(original=original, reader=reader, evidence=evidence, source=source,
                              context=context, repo=repo, output=output, argv=argv, loaded=loaded)
    finally:
        RUN.torch.set_num_threads(threads)


def test_full_cpu_fixture_saves_memory_graph_links_and_preserved_geometry(setup):
    before = {key: value.copy() for key, value in setup.original.items()}
    summary = RUN.main(setup.argv)
    assert summary["returned_frames"] == 168 and summary["scored_frames"] == 2
    assert summary["sampled_pixels"] == [14, 0, 0]
    assert summary["reobservation_links"] == 167*3
    assert summary["terminal_unobserved_age_frames"] == [0, 0, 0]
    assert summary["external_reset_count"] == summary["geometry_fits_in_this_stage"] == 0
    record = json.loads((setup.output / "experiment.provenance.json").read_text())
    assert record["status"] == "ok" and record["stage"] == RUN.STAGE
    assert set(record["inputs"]) == {str(setup.evidence)}
    for item in record["outputs"].values():
        assert RUN.base.sha256(item["path"]) == item["sha256"]
    with np.load(setup.output / "identity.npz") as identity:
        np.testing.assert_array_equal(identity["identity_to_raw_slot"][166:], [[2, 0, 1]]*2)
        assert np.isnan(identity["conservative_accepted_xyz"][166:, 2]).all()
    with np.load(setup.output / "dp.npz") as dp:
        np.testing.assert_array_equal(dp["state_permutation"][dp["path_node"]], dp["path_state"])
    for key in before:
        np.testing.assert_array_equal(setup.original[key], before[key])
    assert setup.reader.closed and len(setup.loaded) == 1


@pytest.mark.parametrize("failure", ["capacity", "input", "snapshot", "objective"])
def test_new_stage_failure_seals_without_claiming_success(setup, monkeypatch, failure):
    if failure == "capacity":
        def limited(*_args, **_kwargs):
            raise RuntimeError("capacity exceeded; no exact solution")
        monkeypatch.setattr(RUN, "solve_identity_memory_dp", limited)
    elif failure == "objective":
        original = RUN.chosen_memory_links

        def incorrect(*args, **kwargs):
            links = original(*args, **kwargs)
            links["brownian_motion_cost_by_id"][1, 0] += 7
            return links
        monkeypatch.setattr(RUN, "chosen_memory_links", incorrect)
    else:
        def mutate(*args, **kwargs):
            path = setup.evidence if failure == "input" else (
                setup.output / "code_snapshot" / setup.source.relative_to(setup.repo)
            )
            path.write_bytes(b"changed")
            return fake_emission()
        monkeypatch.setattr(RUN.common, "score_identity_permutations", mutate)
    with pytest.raises((RuntimeError, ValueError)):
        RUN.main(setup.argv)
    record = json.loads((setup.output / "experiment.provenance.json").read_text())
    assert record["status"] == "failed"
    assert not (setup.output / "summary.json").exists()
    assert setup.reader.closed


def test_existing_output_is_preserved_before_loading(setup):
    setup.output.mkdir()
    marker = setup.output / "preserved"
    marker.write_bytes(b"existing")
    with pytest.raises(ValueError, match="new output"):
        RUN.main(setup.argv)
    assert not setup.loaded and marker.read_bytes() == b"existing"


def test_misaligned_image_scores_are_rejected():
    raw = geometry(3)
    scores = {"frame_idx": raw["frame_idx"]+1, "unary": np.zeros((3, 6))}
    with pytest.raises(ValueError, match="frames differ"):
        RUN.optimize_memory(raw, scores, RUN.registered_config())
