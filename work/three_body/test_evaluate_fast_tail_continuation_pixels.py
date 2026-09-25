"""Synthetic source/lineage/scope checks; no real tracking or images are evaluated."""

import copy
from types import SimpleNamespace

import evaluate_fast_tail_continuation_pixels as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def prohibit_gpu_and_raw(monkeypatch):
    def prohibited(*args, **kwargs):
        raise AssertionError("GPU, model inference or raw video is outside this test")

    for name in ("is_available", "init", "mem_get_info"):
        monkeypatch.setattr(m.old.torch.cuda, name, prohibited)
    monkeypatch.setattr(m.producer.original.cv2, "VideoCapture", prohibited)
    monkeypatch.setattr(m.producer.original.cv2, "imread", prohibited)


def score_rows(frames):
    return [
        dict(
            frame_idx=native,
            view_counts=[
                dict(
                    source_available=True,
                    available_empty_foreground=False,
                    neutral_zero_weight=False,
                    no_unused_foreground=False,
                    weighted_draws=256,
                )
                for _ in range(3)
            ],
            invalid_ray_draws_by_view=[0, 1, 0],
        )
        for native in frames
    ]


def test_two_minute_uses_frozen_formula_exactly():
    frames = m.frame_schedule(86320)
    errors = np.random.default_rng(8).random((120, 2, 3))
    rows = score_rows(frames)
    for indices in (range(120), range(60), range(60, 120)):
        assert m.summarize(errors, rows, indices, frames) == m.frozen.summarize(
            errors, rows, indices
        )


def test_dynamic_300_formula_on120_subset_exactly_matches_frozen():
    frames = m.frame_schedule(116200)
    errors = np.random.default_rng(42).random((300, 2, 3))
    rows = score_rows(frames)
    errors[17, 1] = errors[17, 0]
    rows[13]["view_counts"][0].update(
        neutral_zero_weight=True, no_unused_foreground=True, weighted_draws=0
    )
    for indices in (range(120), range(60), range(60, 120)):
        actual = m.summarize(errors, rows, indices, frames)
        expected = m.frozen.summarize(errors[:120], rows[:120], indices)
        assert actual == expected
        assert m.proof.json_bytes(actual) == m.proof.json_bytes(expected)


def test_five_minutes_all_view_denominator_includes_unavailable_and_neutral():
    frames = m.frame_schedule(116200)
    rows = score_rows(frames)
    errors = np.ones((300, 2, 3), np.float64)
    errors[:, :, 0] = 0
    for row in rows:
        row["view_counts"][0].update(
            source_available=False,
            neutral_zero_weight=True,
            no_unused_foreground=True,
            weighted_draws=0,
        )
    rows[1]["view_counts"][0].update(source_available=True, available_empty_foreground=True)
    result = m.summarize(errors, rows, range(300), frames)
    assert result["original_mean_error"] == 2 / 3
    assert result["frame_view_slots"] == 900 and result["pixel_draws_per_arm"] == 230400
    assert result["unavailable_source_views"] == 299
    assert result["available_empty_foreground_views"] == 1
    assert result["neutral_zero_weight_views"] == 300
    assert result["weighted_draws_per_arm"] == 300 * 2 * 256
    assert result["frames_equal_error"] == 300
    for i in range(5):
        minute = m.summarize(errors, rows, range(i * 60, (i + 1) * 60), frames)
        assert minute["frame_view_slots"] == 180 and minute["frames"] == 60


def test_zero_original_score_keeps_missing_percentage_explicit():
    frames = m.frame_schedule(116200)
    result = m.summarize(np.zeros((300, 2, 3)), score_rows(frames), range(300), frames)
    assert result["relative_error_reduction_percent"] is None
    assert result["relative_reduction_unavailable_reason"] == "zero_original_mean"


@pytest.mark.parametrize(
    "change",
    ("dtype", "NaN", "missing_view", "wrong_frame", "negative", "duplicate_index", "empty_index"),
)
def test_summary_rejects_changed_denominator_or_scope(change):
    frames = m.frame_schedule(116200)
    errors = np.ones((300, 2, 3))
    rows = score_rows(frames)
    indices = range(300)
    if change == "dtype":
        errors = errors.astype(np.float32)
    elif change == "NaN":
        errors[0, 0, 0] = np.nan
    elif change == "missing_view":
        errors = errors[:, :, :2]
    elif change == "wrong_frame":
        rows[4]["frame_idx"] += 1
    elif change == "negative":
        errors[0, 0, 0] = -1
    elif change == "duplicate_index":
        indices = [0, 0]
    else:
        indices = []
    with pytest.raises(ValueError):
        m.summarize(errors, rows, indices, frames)


def comparison_metadata(stop):
    refs = {arm: dict(path=f"/{arm}", sha256="a" * 64, arrays={}) for arm in m.ARMS}
    windows = {k: list(v) for k, v in m.comparison.intervals(stop).items()}
    plan = dict(
        schema=m.comparison.STAGE,
        horizon_stop=stop,
        candidate_references=refs,
        original_source_seal_sha256=m.comparison.ORIGINALS[stop]["seal"],
        milestone_sha256="b" * 64,
        final_tracking_sha256="c" * 64,
        intervals=windows,
    )
    summary = dict(
        schema=m.comparison.STAGE,
        status="ok",
        native_interval=[m.START, stop],
        native_frames=stop - m.START,
        horizon_stop=stop,
        same_parent_and_horizon=True,
        candidate_all21_and_source_arrays_unchanged=True,
        strict_support_recomputed_arms=2,
        strict_support_native_frames=2 * (stop - m.START),
        external_pose_resets=0,
        final_tracking_seal_verified=True,
        whole_five_minute_process_complete=True,
        milestone_is_immutable_completed_checkpoint=True,
        all_three_ID_truth=False,
        image_evaluation_included=False,
        candidate_references=refs,
        intervals=windows,
    )
    return summary, plan


@pytest.mark.parametrize("stop", (86320, 116200))
def test_completed_comparison_metadata_accepted(stop):
    summary, plan = comparison_metadata(stop)
    assert len(m.comparison_contract(summary, plan, stop)) in (120, 300)


@pytest.mark.parametrize(
    "change", ("interim", "missing_final", "other_candidate", "wrong_scope", "bool_count")
)
def test_comparison_requires_final_consumed_mask_proof_and_same_candidates(change):
    summary, plan = comparison_metadata(116200)
    if change == "interim":
        summary["final_tracking_seal_verified"] = False
    elif change == "missing_final":
        plan["final_tracking_sha256"] = None
    elif change == "other_candidate":
        plan["candidate_references"] = copy.deepcopy(plan["candidate_references"])
        plan["candidate_references"]["fast_tail"]["sha256"] = "f" * 64
    elif change == "wrong_scope":
        plan["intervals"] = dict(full=[m.START, 86320])
    else:
        summary["strict_support_recomputed_arms"] = True
    with pytest.raises(ValueError):
        m.comparison_contract(summary, plan, 116200)


def mask_context(monkeypatch, tmp_path, stop):
    originals = {}
    masks = {}
    for first in range(m.START, stop, m.prep.CHUNK):
        for view in m.old.VIEWS:
            path = (tmp_path / f"{first}_{view}.npz").resolve()
            masks[first, view] = (path, "a" * 64)
            originals[str(path)] = {"path": str(path), "sha256": "a" * 64}
    final = {str(path): digest for path, digest in masks.values()}
    summaries = {}
    bundles = {}
    for count, stop_frame in ((60, m.HALF), (120, 86320), (300, 116200)):
        if stop_frame > stop:
            continue
        prefix = {
            str(path): digest for (first, _), (path, digest) in masks.items() if first < stop_frame
        }
        summaries[count] = dict(source_mask_sha256=prefix)
        lo = {60: m.START, 120: m.HALF, 300: 86320}[count]
        consumed = {
            str(path): digest
            for (first, _), (path, digest) in masks.items()
            if lo <= first < stop_frame
        }
        bundles[count] = dict(seal=dict(inputs=consumed))
    first_minute = {
        str(path): dict(path=str(path), sha256=digest)
        for (first, _), (path, digest) in masks.items()
        if first < m.HALF
    }

    def lookup(seal, first, view):
        path, digest = masks[first, view]
        assert seal["inputs"][str(path)]["sha256"] == digest
        return path, digest

    monkeypatch.setattr(m.prep, "mask_input", lookup)
    return (
        stop,
        summaries,
        bundles,
        dict(seal=dict(inputs=originals)),
        dict(seal=dict(inputs=final)),
        dict(seal=dict(inputs=first_minute)),
    )


@pytest.mark.parametrize("stop,count", ((86320, 36), (116200, 90)))
def test_all_mask_lineages_without_payload_reads(monkeypatch, tmp_path, stop, count):
    args = mask_context(monkeypatch, tmp_path, stop)
    assert len(m.bind_mask_sources(*args)) == count
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "where", ("final", "first_minute", "prep60", "prep120", "prep300", "extra")
)
def test_corrupt_actual_consumption_or_transitive_prep_edge_rejected(monkeypatch, tmp_path, where):
    args = mask_context(monkeypatch, tmp_path, 116200)
    _stop, summaries, bundles, _original, final, first_minute = args
    if where == "final":
        final["seal"]["inputs"].pop(next(iter(final["seal"]["inputs"])))
    elif where == "first_minute":
        first_minute["seal"]["inputs"][next(iter(first_minute["seal"]["inputs"]))]["sha256"] = (
            "b" * 64
        )
    elif where == "extra":
        summaries[300]["source_mask_sha256"]["foreign.npz"] = "b" * 64
    else:
        count = int(where[4:])
        key = next(iter(bundles[count]["seal"]["inputs"]))
        bundles[count]["seal"]["inputs"][key] = "b" * 64
    with pytest.raises(ValueError):
        m.bind_mask_sources(*args)


def test_foreign_record_input_path_rejected(tmp_path):
    path = (tmp_path / "a").resolve()
    seal = dict(inputs={str(path): dict(path=str(tmp_path / "b"), sha256="a" * 64)})
    with pytest.raises(ValueError, match="alias path"):
        m.input_edge(seal, path, "a" * 64)


def test_frozen_cpu_scoring_is_reused_without_modification():
    assert m.score_frame is m.frozen.score_frame
    assert m.view_counts is m.frozen.view_counts


@pytest.mark.parametrize("change", ("float32_pose", "float64_weights", "fit_overlap"))
def test_frozen_scoring_rejects_corrupt_samples_before_render(monkeypatch, change):
    samples = dict(
        fit_pixels=np.tile(
            np.column_stack((np.arange(1024, dtype=np.float64), np.zeros(1024))), (3, 1, 1)
        ),
        fit_target=np.ones((3, 1024), np.float32),
        fit_weights=np.ones((3, 1024), np.float32),
        fit_foreground=np.ones((3, 1024), bool),
        pixels=np.tile(
            np.column_stack((np.arange(256, dtype=np.float64), np.ones(256))), (3, 1, 1)
        ),
        target=np.ones((3, 256), np.float32),
        weights=np.ones((3, 256), np.float32),
        reused_fitted_coordinate=np.zeros((3, 256), bool),
    )
    poses = np.ones((2, 3, 3, 3), np.float64)
    if change == "float32_pose":
        poses = poses.astype(np.float32)
    elif change == "float64_weights":
        samples["weights"] = samples["weights"].astype(np.float64)
    else:
        samples["pixels"][1, 0] = samples["fit_pixels"][1, 0]
    monkeypatch.setattr(
        m.old, "RefractiveFishRenderer", lambda *a, **k: pytest.fail("renderer entered")
    )
    with pytest.raises(ValueError):
        m.score_frame(None, poses, samples)


def test_pending_source_pin_prevents_any_real_input_or_output(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "COMPARISON_CODE_SHA", None)
    monkeypatch.setattr(m, "load_context", lambda *a: pytest.fail("real inputs entered"))
    args = SimpleNamespace(
        execute=True,
        horizon_stop=116200,
        comparison_sha="a" * 64,
        preparation_sha="b" * 64,
        output=tmp_path / "never",
    )
    with pytest.raises(ValueError, match="pin pending"):
        m.run(args)
    assert not args.output.exists()
