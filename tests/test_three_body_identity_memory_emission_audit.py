"""Memory image audit binds its own producer and preserves the frozen raw replay math."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"
prior_spec = importlib.util.spec_from_file_location(
    "_memory_audit_old_fixture",
    Path(__file__).with_name("test_three_body_identity_emission_audit.py"),
)
prior = importlib.util.module_from_spec(prior_spec)
prior_spec.loader.exec_module(prior)


def load_module():
    names = {path.stem for path in LAB.glob("*.py")}
    saved, old_path = {n: sys.modules.get(n) for n in names}, sys.path.copy()
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location(
            "_memory_image_audit_test", LAB / "audit_identity_memory_emissions.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_module()


def values(n=168):
    geometry, colors, old_scores, _ = prior.fixture_values(n)
    saved = {key: value for key, value in old_scores.items() if key in m.IMAGE_FIELDS}
    return geometry, colors, saved, copy.deepcopy(m.evidence.FIXED_CONFIG)


def test_relocated_loop_exactly_matches_legacy_arrays_hashes_and_prefix_behavior(monkeypatch):
    geometry, colors, scores, config = values()
    old_geometry, old_colors, old_scores, old_config = prior.fixture_values()
    calls = []

    def score(rig, frame, *args, **kwargs):
        calls.append(frame.frame_idx)
        return prior.emission(rig, frame, *args, **kwargs)

    monkeypatch.setattr(m, "score_identity_permutations", score)
    monkeypatch.setattr(prior.m, "score_identity_permutations", prior.emission)
    reader = prior.Reader(geometry["frame_idx"])
    actual = m.replay_emissions(reader, None, geometry, colors, scores, config)
    expected = prior.m.replay_emissions(
        prior.Reader(old_geometry["frame_idx"]), None,
        old_geometry, old_colors, old_scores, old_config,
    )
    for key in m.IMAGE_FIELDS:
        assert actual[0][key].dtype == expected[0][key].dtype
        assert actual[0][key].tobytes() == expected[0][key].tobytes()
    assert actual[1:] == expected[1:]
    assert reader.closed and calls == geometry["frame_idx"][166:].tolist()
    assert not any(key in actual[0] for key in prior.m.MOTION_FIELDS)


@pytest.mark.parametrize("change", ["extra_motion", "wrong_memory", "short_prefix", "int32"])
def test_memory_schema_refuses_fake_motion_fields_and_wrong_registration(change):
    geometry, colors, scores, config = values()
    if change == "extra_motion":
        scores["transitions"] = np.empty(0)
    elif change == "wrong_memory":
        config["memory"]["max_layer_states"] += 1
    elif change == "int32":
        scores["valid_pixels"] = scores["valid_pixels"].astype(np.int32)
    else:
        geometry["frame_idx"] = geometry["frame_idx"][:165]
    with pytest.raises(ValueError):
        m.validate_image_arrays(scores, geometry["frame_idx"], colors, config)


@pytest.mark.parametrize("change", ["early_eof", "extra", "wrong_pixel", "reordered"])
def test_new_loop_fails_on_raw_alignment_or_pixel_changes_and_closes(change, monkeypatch):
    geometry, colors, scores, config = values()
    frames = [prior.frame(int(frame)) for frame in geometry["frame_idx"]]
    if change == "early_eof":
        frames.pop()
    elif change == "extra":
        frames.append(prior.frame(int(geometry["frame_idx"][-1]+1)))
    elif change == "wrong_pixel":
        frames[-1].images_bgr[m.base.VIEWS[0]][0, 0, 0] += 1
    else:
        frames[3], frames[4] = frames[4], frames[3]
    reader = prior.Reader(geometry["frame_idx"], frames)
    monkeypatch.setattr(m, "score_identity_permutations", prior.emission)
    with pytest.raises(ValueError):
        m.replay_emissions(reader, None, geometry, colors, scores, config)
    assert reader.closed


@pytest.fixture
def main_inputs(tmp_path, monkeypatch):
    origin_data = (LAB / "audit_identity_emissions.py").read_bytes()
    run, output = tmp_path / "run", tmp_path / "output"
    run.mkdir()
    source = tmp_path / "audit_fixture.py"
    source.write_text("fixed fixture code", encoding="utf-8")
    origin = tmp_path / m.LOOP_ORIGIN
    origin.parent.mkdir(parents=True)
    origin.write_bytes(origin_data)
    geometry, colors, saved, config = values(200)
    np.savez(run / "scores.npz", **saved)
    (run / "experiment.provenance.json").write_text(json.dumps({
        "params": {"package_versions": m.common.runtime_versions()},
    }), encoding="utf-8")
    context = SimpleNamespace(frame_idx=geometry["frame_idx"], geometry=geometry,
                              enrollment=colors, geometry_sequence_root=tmp_path,
                              input_root=tmp_path)
    hashes = {str(run / "scores.npz"): m.base.sha256(run / "scores.npz")}
    monkeypatch.setattr(m.base, "REPO", tmp_path)
    monkeypatch.setattr(m, "code_paths", lambda: [origin, source])
    monkeypatch.setattr(
        m.evidence, "verify_completed", lambda run: ({"config": config}, context, hashes)
    )
    monkeypatch.setattr(m.common, "SequenceReader", lambda *_args: prior.Reader(context.frame_idx))
    monkeypatch.setattr(m.base, "load_rig", lambda *_args: object())
    monkeypatch.setattr(m, "score_identity_permutations", prior.emission)
    return run, output, source


@pytest.mark.parametrize("tamper", [None, "source", "snapshot", "scores"])
def test_main_seals_only_verified_replay_and_restores_cpu_threads(main_inputs, monkeypatch, tamper):
    run, output, source = main_inputs
    original = m.replay_emissions

    def replay(*args, **kwargs):
        result = original(*args, **kwargs)
        path = {"source": source, "snapshot": output / "code_snapshot/audit_fixture.py",
                "scores": run / "scores.npz"}.get(tamper)
        if path:
            path.write_bytes(b"changed after replay")
        return result

    def no_gpu(*args, **kwargs):
        raise AssertionError("raw CPU image audit must not initialize or synchronize CUDA")

    monkeypatch.setattr(m, "replay_emissions", replay)
    monkeypatch.setattr(m.torch.cuda, "is_available", no_gpu)
    monkeypatch.setattr(m.torch.cuda, "synchronize", no_gpu)
    threads = m.torch.get_num_threads()
    if tamper:
        with pytest.raises(RuntimeError, match="changed"):
            m.main(["--run", str(run), "--output", str(output)])
    else:
        report = m.main(["--run", str(run), "--output", str(output)])
        assert report["decoded_images"] == 600 and report["rendered_frames"] == 34
        assert not report["geometry_fit_or_identity_optimization_performed"]
        assert not report["ground_truth_or_reference_arrays_opened"]
        arrays = m.common.evidence.npz(output / "recomputed_image_scores.npz")
        assert set(arrays) == set(m.IMAGE_FIELDS)
    seal = m.common.evidence.read_json(output / "audit.provenance.json")
    assert seal["status"] == ("failed" if tamper else "ok")
    assert m.torch.get_num_threads() == threads
    with pytest.raises(ValueError, match="new workspace"):
        m.main(["--run", str(run), "--output", str(output)])


def test_unsealed_memory_producer_is_not_treated_as_old_stage(main_inputs, monkeypatch):
    run, output, _ = main_inputs

    def reject(_run):
        raise ValueError("memory producer is not sealed")

    monkeypatch.setattr(m.evidence, "verify_completed", reject)
    with pytest.raises(ValueError, match="not sealed"):
        m.main(["--run", str(run), "--output", str(output)])
    assert not output.exists()
