"""Raw replay catches native alignment, six-state score and finite/dtype corruption."""

import copy
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
    saved, old_path = {n: sys.modules.get(n) for n in names}, sys.path.copy()
    try:
        for name in names:
            sys.modules.pop(name, None)
        sys.path.insert(0, str(LAB))
        spec = importlib.util.spec_from_file_location(
            "_identity_emission_audit_tests", LAB / "audit_identity_emissions.py"
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


def frame(number):
    return SimpleNamespace(
        frame_idx=number,
        images_bgr={v: np.full((2, 3, 3), 128, np.uint8) for v in m.base.VIEWS},
        masks={v: np.ones((1, 2, 3), bool) for v in m.base.VIEWS},
    )


class Reader:
    def __init__(self, frames, yielded=None):
        self.frames = frames
        self.yielded = [frame(int(f)) for f in frames] if yielded is None else yielded
        self.closed = False
        self.read = []

    def __iter__(self):
        try:
            for value in self.yielded:
                self.read.append(value.frame_idx)
                yield value
        finally:
            self.closed = True


def emission(_rig, current, *_args, **kwargs):
    assert kwargs["device"] == "cpu"
    pixel = current.images_bgr[m.base.VIEWS[0]][0, 0, 0] / 255
    costs = np.arange(6, dtype=np.float64) / 10 + pixel
    return SimpleNamespace(
        costs=costs,
        per_view_costs=np.column_stack((costs, np.zeros((6, 2)))),
        enabled_views=np.array([True, False, False]),
        sampled_pixels=np.array([5, 0, 0], np.int64),
        valid_pixels=np.array([4, 0, 0], np.int64),
        available_core_pixels=np.array([6, 0, 0], np.int64),
        visible_mass=np.array([[0.1, 0.2, 0.3], [0, 0, 0], [0, 0, 0]]),
        rendering_performed=True,
    )


def fixture_values(n=168):
    frames = np.arange(76620, 76620 + n, dtype=np.int64)
    geometry = dict(frame_idx=frames, pose_xyz=np.zeros((n, 3, 3, 3)))
    colors = dict(templates=np.zeros((3, 3, 6)), enabled_views=np.array([True, False, False]))
    config = copy.deepcopy(m.evidence.FIXED_CONFIG)
    saved = {k: np.zeros((n, *tail), dtype=dtype) for k, (tail, dtype) in m.IMAGE_FIELDS.items()}
    saved["frame_idx"][:] = frames
    saved.update({k: np.zeros((n - 1, 6, 6)) for k in m.MOTION_FIELDS})
    for index in range(166, n):
        result = emission(None, frame(int(frames[index])), device="cpu")
        saved["unary"][index] = result.costs
        for key in m.IMAGE_FIELDS.keys() - {"frame_idx", "unary"}:
            saved[key][index] = getattr(result, key)
    return geometry, colors, saved, config


def test_replay_all_six_states_and_native_frames_with_no_prefix_render_or_cuda(monkeypatch):
    geometry, colors, saved, config = fixture_values()
    reader, calls = Reader(geometry["frame_idx"]), []

    def checked(rig, current, pose, templates, enabled, **kwargs):
        calls.append(current.frame_idx)
        assert current.frame_idx >= geometry["frame_idx"][166]
        np.testing.assert_array_equal(pose, geometry["pose_xyz"][0])
        np.testing.assert_array_equal(templates, colors["templates"])
        return emission(rig, current, pose, templates, enabled, **kwargs)

    def no_cuda(*_args, **_kwargs):
        raise AssertionError("CPU audit must not access CUDA")

    monkeypatch.setattr(m, "score_identity_permutations", checked)
    monkeypatch.setattr(m.torch.cuda, "is_available", no_cuda)
    monkeypatch.setattr(m.torch.cuda, "synchronize", no_cuda)
    actual, comparisons, pixels, rendered = m.replay_emissions(
        reader, object(), geometry, colors, saved, config
    )
    assert calls == geometry["frame_idx"][166:].tolist()
    assert reader.read == geometry["frame_idx"].tolist() and reader.closed
    assert rendered == 2 and len(pixels) == 168
    assert len(np.unique(actual["unary"][-1])) == 6
    for key in m.IMAGE_FIELDS:
        np.testing.assert_array_equal(actual[key], saved[key])
        assert comparisons[key].get("within_tolerance", comparisons[key].get("exact"))


@pytest.mark.parametrize(
    "bad",
    [
        "prefix_cost",
        "prefix_mass",
        "prefix_render",
        "state5_cost",
        "view_cost",
        "mass",
        "flag",
        "count",
        "dtype",
        "nan",
        "inf",
    ],
)
def test_saved_gpu_score_corruption_is_rejected(monkeypatch, bad):
    geometry, colors, saved, config = fixture_values()
    if bad == "prefix_cost":
        saved["unary"][165, 0] = 0.1
    elif bad == "prefix_mass":
        saved["visible_mass"][165, 0, 0] = 0.1
    elif bad == "prefix_render":
        saved["rendering_performed"][165] = True
    elif bad == "state5_cost":
        saved["unary"][166, 5] += 1e-6
    elif bad == "view_cost":
        saved["per_view_costs"][166, 4, 0] += 1e-6
    elif bad == "mass":
        saved["visible_mass"][166, 0, 1] += 1e-6
    elif bad == "flag":
        saved["enabled_views"][166, 0] = False
    elif bad == "count":
        saved["valid_pixels"][166, 0] -= 1
    elif bad == "dtype":
        saved["sampled_pixels"] = saved["sampled_pixels"].astype(np.int32)
    else:
        saved["unary"][166, 0] = np.nan if bad == "nan" else np.inf
    monkeypatch.setattr(m, "score_identity_permutations", emission)
    with pytest.raises(ValueError):
        m.replay_emissions(Reader(geometry["frame_idx"]), None, geometry, colors, saved, config)


@pytest.mark.parametrize(
    "bad", ["early_eof", "extra", "reorder", "different_reader", "changed_pixel"]
)
def test_raw_frame_errors_fail_and_close_decoder(monkeypatch, bad):
    geometry, colors, saved, config = fixture_values()
    values = [frame(int(f)) for f in geometry["frame_idx"]]
    if bad == "early_eof":
        values.pop()
    elif bad == "extra":
        values.append(frame(int(geometry["frame_idx"][-1] + 1)))
    elif bad == "reorder":
        values[5], values[6] = values[6], values[5]
    elif bad == "changed_pixel":
        values[-1].images_bgr[m.base.VIEWS[0]][0, 0, 0] += 1
    reader = Reader(geometry["frame_idx"], values)
    if bad == "different_reader":
        reader.frames = reader.frames + 1
    monkeypatch.setattr(m, "score_identity_permutations", emission)
    with pytest.raises(ValueError):
        m.replay_emissions(reader, None, geometry, colors, saved, config)
    assert reader.closed or bad == "different_reader"


@pytest.mark.parametrize("delta,allowed", [(5e-13, True), (2e-12, False)])
def test_double_precision_comparison_uses_fixed_tolerance(delta, allowed):
    actual, expected = np.array([delta]), np.array([0.0])
    if allowed:
        assert m.compare_image_array(actual, expected, "unary")["within_tolerance"]
    else:
        with pytest.raises(ValueError, match="tolerance"):
            m.compare_image_array(actual, expected, "unary")


def test_failed_upstream_seal_creates_no_audit_output(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = tmp_path / "out"
    monkeypatch.setattr(m.base, "REPO", tmp_path)
    monkeypatch.setattr(m, "code_paths", lambda: [])

    def reject(_run):
        raise ValueError("producer is not complete")

    monkeypatch.setattr(m.evidence, "verify_completed", reject)
    with pytest.raises(ValueError, match="not complete"):
        m.main(["--run", str(run), "--output", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("tamper", [None, "source", "snapshot", "scores"])
def test_main_seals_cpu_result_and_restores_threads(tmp_path, monkeypatch, tamper):
    run, source, output = tmp_path / "run", tmp_path / "audit.py", tmp_path / "out"
    run.mkdir()
    source.write_text("fixture audit source", encoding="utf-8")
    geometry, colors, saved, config = fixture_values(200)
    np.savez(run / "scores.npz", **saved)
    (run / "experiment.provenance.json").write_text(
        json.dumps(dict(params=dict(package_versions=m.runtime_versions()))), encoding="utf-8"
    )
    hashes = {str(run / "scores.npz"): m.base.sha256(run / "scores.npz")}
    context = SimpleNamespace(
        frame_idx=geometry["frame_idx"],
        geometry=geometry,
        enrollment=colors,
        geometry_sequence_root=tmp_path,
        input_root=tmp_path,
    )
    monkeypatch.setattr(m.base, "REPO", tmp_path)
    monkeypatch.setattr(m.base, "load_rig", lambda _path: object())
    monkeypatch.setattr(m, "code_paths", lambda: [source])
    monkeypatch.setattr(
        m.evidence, "verify_completed", lambda _run: (dict(config=config), context, hashes)
    )
    monkeypatch.setattr(m, "SequenceReader", lambda *_args: Reader(geometry["frame_idx"]))
    mutated = False

    def checked_emission(*args, **kwargs):
        nonlocal mutated
        if tamper and not mutated:
            target = {
                "source": source,
                "snapshot": output / "code_snapshot/audit.py",
                "scores": run / "scores.npz",
            }[tamper]
            with target.open("ab") as stream:
                stream.write(b"changed during replay")
            mutated = True
        return emission(*args, **kwargs)

    monkeypatch.setattr(m, "score_identity_permutations", checked_emission)
    before_threads = m.torch.get_num_threads()
    argv = ["--run", str(run), "--output", str(output)]
    if tamper:
        with pytest.raises(RuntimeError, match="changed during replay"):
            m.main(argv)
        assert m.torch.get_num_threads() == before_threads
        record = json.loads((output / "audit.provenance.json").read_text())
        assert record["status"] == "failed"
        return
    result = m.main(argv)
    assert result["native_frames"] == 200 and result["rendered_frames"] == 34
    assert m.torch.get_num_threads() == before_threads
    record = json.loads((output / "audit.provenance.json").read_text())
    assert record["status"] == "ok"
    assert record["params"]["device"] == "cpu"
    for item in record["outputs"].values():
        assert m.base.sha256(item["path"]) == item["sha256"]
