"""New two-minute wiring and shared CPU pixel-score boundaries; no real inputs."""

import copy
from pathlib import Path
from types import SimpleNamespace

import evaluate_fast_tail_two_minute_pixels as m
import numpy as np
import pytest

from medaka3d.calib import Rig


@pytest.fixture(autouse=True)
def no_gpu_or_raw(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("GPU or raw image read is outside this test")

    for name in ("is_available", "init", "mem_get_info"):
        monkeypatch.setattr(m.old.torch.cuda, name, prohibited)
    monkeypatch.setattr(m.producer.original.cv2, "VideoCapture", prohibited)
    monkeypatch.setattr(m.producer.original.cv2, "imread", prohibited)
    before = m.old.torch.get_num_threads()
    m.old.torch.set_num_threads(1)
    yield
    m.old.torch.set_num_threads(before)


def samples():
    return dict(
        fit_pixels=np.tile(np.column_stack((np.arange(1024), np.zeros(1024))), (3, 1, 1)),
        fit_target=np.ones((3, 1024), np.float32),
        fit_weights=np.full((3, 1024), 1 / 1024, np.float32),
        fit_foreground=np.ones((3, 1024), bool),
        pixels=np.tile(np.column_stack((np.arange(256), np.ones(256))), (3, 1, 1)),
        target=np.tile(np.r_[np.ones(96), np.zeros(160)].astype(np.float32), (3, 1)),
        weights=np.tile(
            np.r_[np.full(96, 1 / 96), np.full(96, 0.75 / 96), np.full(64, 0.25 / 64)].astype(
                np.float32
            ),
            (3, 1),
        ),
        reused_fitted_coordinate=np.zeros((3, 256), bool),
    )


class Camera:
    def __init__(self, view, invalid=False):
        self.view, self.invalid = view, invalid

    def rays(self, xy):
        origin = np.zeros((len(xy), 3), np.float64)
        origin[:, :2] = xy * 0.1
        direction = np.tile(np.array([0.0, 0.0, 1.0]), (len(xy), 1))
        if self.invalid:
            direction[17] = np.nan
        return origin, direction


def poses():
    seed = np.array(
        [
            [[2.0, 0.0, 10.0], [6.0, 0.0, 10.0], [12.0, 1.0, 10.0]],
            [[11.0, 2.0, 15.0], [15.0, 2.0, 15.0], [20.0, 3.0, 15.0]],
            [[-5.0, 4.0, 20.0], [-1.0, 4.0, 20.0], [5.0, 5.0, 20.0]],
        ]
    )
    return np.stack((seed, seed + np.array([0.4, 0.7, -0.1])))


def test_old_cpu_f64_renderer_and_weighted_union_bytes_unchanged():
    rig = Rig(tuple(Camera(v, i == 1) for i, v in enumerate(m.old.VIEWS)), Path("synthetic"))
    points, value = poses(), samples()
    before = m.comparison.signature(value), m.proof.signature(points)
    actual = m.score_frame(rig, points, value)
    renderer = m.old.RefractiveFishRenderer(
        rig,
        {v: value["pixels"][i] for i, v in enumerate(m.old.VIEWS)},
        m.old.RendererConfig(width_ratio=0.06),
        device="cpu",
        dtype=m.old.torch.float64,
    )
    with m.old.torch.no_grad():
        alpha = renderer.render(points).union_alpha.numpy()
    valid = renderer.ray_valid.numpy()
    expected = ((alpha - value["target"]) ** 2 * value["weights"] * valid).sum(-1)
    for key, a in (("union_alpha", alpha), ("ray_valid", valid), ("per_view_error", expected)):
        m.proof.same(actual[key], a, key)
    assert np.isfinite(expected).all() and not valid[1, 17]
    assert np.any(expected[0] != expected[1])
    assert before == (m.comparison.signature(value), m.proof.signature(points))


@pytest.mark.parametrize("corruption", ("overlap", "width", "nonfinite_pose"))
def test_scoring_rejects_changes_before_renderer(monkeypatch, corruption):
    value, points = samples(), poses()
    if corruption == "overlap":
        value["pixels"][2, 11] = value["fit_pixels"][2, 100]
    elif corruption == "width":
        value["weights"] = value["weights"].astype(np.float64)
    else:
        points[1, 1, 1, 1] = np.nan
    monkeypatch.setattr(
        m.old, "RefractiveFishRenderer", lambda *a, **k: pytest.fail("renderer entered")
    )
    with pytest.raises(ValueError):
        m.score_frame(None, points, value)


def membership(native, value):
    return [
        dict(
            frame=native,
            view=v,
            source_mask_available=True,
            source_foreground_pixels=900,
            available_unused_pool_sizes=[800, 700, 800000],
            overlap_count=0,
            sampler_seed=native * 7 + i + m.old.SEED_OFFSET,
            known_weight_sum=float(value["weights"][i].sum()),
        )
        for i, v in enumerate(m.old.VIEWS)
    ]


def test_unavailable_and_empty_are_separate_neutral_views():
    value, native = samples(), m.FRAMES[0]
    value["weights"][:2] = 0
    rows = membership(native, value)
    for row in rows[:2]:
        row.update(source_foreground_pixels=0, available_unused_pool_sizes=[0, 0, 10000])
    rows[0]["source_mask_available"] = False
    result = m.view_counts(value, rows, native)
    assert result[0]["neutral_zero_weight"] and not result[0]["available_empty_foreground"]
    assert result[1]["neutral_zero_weight"] and result[1]["available_empty_foreground"]
    assert not result[2]["neutral_zero_weight"] and result[2]["weighted_draws"] == 256


@pytest.mark.parametrize("change", ("weight", "seed", "order", "pool"))
def test_membership_attachment_rejects_other_frame_or_samples(change):
    value, native = samples(), m.FRAMES[50]
    rows = membership(native, value)
    if change == "weight":
        rows[0]["known_weight_sum"] += 0.01
    elif change == "seed":
        rows[2]["sampler_seed"] += 1
    elif change == "order":
        rows.reverse()
    else:
        rows[0]["available_unused_pool_sizes"][0] = 0
    with pytest.raises(ValueError):
        m.view_counts(value, rows, native)


def score_rows():
    value = samples()
    return [
        dict(
            frame_idx=n,
            view_counts=m.view_counts(value, membership(n, value), n),
            invalid_ray_draws_by_view=[0, 0, 1],
        )
        for n in m.FRAMES
    ]


def test_full_and_half_statistics_keep_all_views_and_exact_ties():
    errors = np.ones((120, 2, 3), np.float64)
    errors[:60, 1] = 0.5
    errors[60:, 1] = 2
    errors[0, 1] = 1
    rows = score_rows()
    full = m.summarize(errors, rows, range(120))
    first = m.summarize(errors, rows, range(60))
    second = m.summarize(errors, rows, range(60, 120))
    assert [
        full[k] for k in ("frames_lower_error", "frames_higher_error", "frames_equal_error")
    ] == [59, 60, 1]
    assert full["frame_view_slots"] == 360 and full["pixel_draws_per_arm"] == 92160
    assert first["frames_lower_error"] == 59 and second["frames_higher_error"] == 60
    assert (
        full["fast_tail_mean_error"]
        == (first["fast_tail_mean_error"] + second["fast_tail_mean_error"]) / 2
    )
    assert full["invalid_ray_draws_per_arm"] == 120


def test_neutral_zero_scores_do_not_create_percentage_or_disappear():
    errors = np.zeros((120, 2, 3), np.float64)
    result = m.summarize(errors, score_rows(), range(120))
    assert result["frames_equal_error"] == 120 and result["original_mean_error"] == 0
    assert result["relative_error_reduction_percent"] is None
    assert result["relative_reduction_unavailable_reason"] == "zero_original_mean"
    assert result["neutral_views_retained_in_means"]


def test_custom_size_bytes_seal_reads_member_and_rejects_resize(tmp_path, monkeypatch):
    root = tmp_path / "sample"
    out = m.proof.Output(root)
    out.write("summary.json", {"ok": True})
    seal = dict(stage=m.prep.STAGE, status="ok", inputs={}, outputs=out.records)
    (root / "preparation.provenance.json").write_bytes(m.proof.json_bytes(seal))
    evidence = SimpleNamespace(add=lambda path, expected=None: m.proof.sha(path))
    bundle = m.custom_seal(root, m.proof.sha(root / "preparation.provenance.json"), evidence)
    assert m.custom_member(bundle, "summary.json", evidence) == root / "summary.json"
    (root / "summary.json").write_text("different length")
    with pytest.raises(ValueError, match="resized"):
        m.custom_member(bundle, "summary.json", evidence)


def test_source_pin_pending_blocks_before_input_or_cuda(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "PREPARATION_CODE_SHA", None)
    monkeypatch.setattr(m, "load_context", lambda *a: pytest.fail("actual inputs read"))
    args = SimpleNamespace(
        execute=True,
        preparation_sha="a" * 64,
        comparison_sha="b" * 64,
        tracking_sha="c" * 64,
        output=tmp_path / "never",
    )
    with pytest.raises(ValueError, match="pin pending"):
        m.run(args)
    assert not args.output.exists()


def mask_contract(monkeypatch, tmp_path):
    sources = {}
    for first in range(m.START, m.STOP, m.prep.CHUNK):
        for view in m.old.VIEWS:
            sources[(first, view)] = (tmp_path / f"{first}_{view}.npz", "a" * 64)
    original = dict(inputs={str(p): dict(path=str(p), sha256=h) for p, h in sources.values()})
    tail = copy.deepcopy(original)
    summary = dict(source_mask_sha256={str(p): h for p, h in sources.values()})
    previous_summary = dict(
        source_mask_sha256={str(p): h for (first, _), (p, h) in sources.items() if first < m.HALF}
    )
    previous = dict(
        seal=dict(
            inputs={
                str(p): dict(path=str(p), sha256=h)
                for (first, _), (p, h) in sources.items()
                if first < m.HALF
            }
        )
    )
    prepared = dict(
        seal=dict(inputs={str(p): h for (first, _), (p, h) in sources.items() if first >= m.HALF})
    )

    def lookup(seal, first, view):
        p, _ = sources[first, view]
        return p, seal["inputs"][str(p)]["sha256"]

    monkeypatch.setattr(m.prep, "mask_input", lookup)
    return summary, previous_summary, prepared, previous, (dict(seal=original), dict(seal=tail))


def test_all36_mask_edges_reuse_first18_transitively_no_payload_read(monkeypatch, tmp_path):
    contract = mask_contract(monkeypatch, tmp_path)
    assert len(m.bind_mask_sources(*contract)) == 36
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("where", ("arm", "prepared", "previous", "extra"))
def test_changed_mask_or_foreign_extra_source_rejected(monkeypatch, tmp_path, where):
    summary, prior_summary, prepared, prior, arms = mask_contract(monkeypatch, tmp_path)
    key = next(iter(summary["source_mask_sha256"]))
    if where == "arm":
        arms[1]["seal"]["inputs"][key]["sha256"] = "b" * 64
    elif where == "prepared":
        summary["source_mask_sha256"][key] = "b" * 64
    elif where == "previous":
        prior_summary["source_mask_sha256"][key] = "b" * 64
    else:
        summary["source_mask_sha256"]["unused.npz"] = "b" * 64
    with pytest.raises(ValueError):
        m.bind_mask_sources(summary, prior_summary, prepared, prior, arms)
