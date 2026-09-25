"""Synthetic CPU callbacks only: no real model/crop/geometry files."""

import copy

import diagnose_dino_minute_images as m
import numpy as np
import pytest


def prepared(nonempty=True):
    image = np.zeros((8, 8, 3), np.uint8)
    mask = np.zeros((8, 8), bool)
    if nonempty:
        mask[3, 3] = True
    arrays, meta = m.crops.prepare_crop(image, mask)
    row = dict(meta, crop_id="synthetic", frame_idx=66483, view="top_cam", mask_row=0)
    return row, arrays


def test_nonempty_eligible0_receives_both_forwards_in_fixed_order_and_all256():
    row, arrays = prepared()
    before = m.store.signatures(arrays)
    called = []
    saved = []

    def forward(chw):
        called.append(chw.copy())
        return np.full((256, 384), len(called), np.float32), 0.01

    result = m.infer_prepared(
        [row],
        lambda _: arrays,
        forward,
        lambda r, c, t, g: saved.append((c, t.copy(), copy.deepcopy(g))),
    )
    assert len(called) == 2 and [r["condition"] for r in result] == ["raw", "masked"]
    assert all(r["forward_called"] and r["eligible_token_count"] == 0 for r in result)
    assert before == m.store.signatures(arrays)
    for i, condition in enumerate(("raw", "masked")):
        assert m.store.same(called[i], arrays[condition + "_normalized_chw"])
        assert saved[i][0] == condition and saved[i][1].shape == (256, 384)


def test_empty_row_retained_nan_and_no_forward():
    row, arrays = prepared(False)
    saved = []
    result = m.infer_prepared(
        [row],
        lambda _: arrays,
        lambda _: pytest.fail("empty forwarded"),
        lambda r, c, t, g: saved.append(t.copy()),
    )
    assert len(result) == 2 and not any(r["forward_called"] for r in result)
    assert all(np.isnan(a).all() for a in saved)


@pytest.mark.parametrize("fault", ["dtype", "shape", "input_mutation", "missing_array"])
def test_input_or_model_shape_mutation_rejected(fault):
    row, arrays = prepared()
    if fault == "missing_array":
        del arrays["token_index"]

    def forward(chw):
        if fault == "input_mutation":
            chw[0, 0, 0] += 1
        return np.zeros(
            (255 if fault == "shape" else 256, 384), np.float64 if fault == "dtype" else np.float32
        ), 0.01

    with pytest.raises(ValueError):
        m.infer_prepared([row], lambda _: arrays, forward, lambda *a: None)


def synthetic_plan():
    counts = {(n, v): 3 for n in m.crops.FRAMES for v in m.crops.VIEWS}
    increments = [
        list(range(1, 19)),
        list(range(1, 36, 2)),
        list(range(1, 8)) + list(range(9, 29, 2)),
    ]
    for view, indices in zip(m.crops.VIEWS, increments, strict=True):
        for i in indices:
            counts[m.crops.FRAMES[i], view] = 4
    plan = m.crops.enumerate_plan(counts)
    assert len(plan["crops"]) == 593 and len(plan["pairs"]) == 1934
    native = [
        dict(frame_idx=n, view=v, mask_count=counts[n, v])
        for n in m.crops.FRAMES
        for v in m.crops.VIEWS
    ]
    summary = dict(
        status="ok",
        mask_crops=593,
        mask_pairs=1934,
        model_forward_calls=0,
        candidate_arrays_read=False,
        raw_video_delivered_images=180,
        geometry_join_performed=False,
    )
    return plan, copy.deepcopy(plan["crops"]), native, summary


def test_full593_1934_synthetic_registered_cardinality():
    values = synthetic_plan()
    before = copy.deepcopy(values)
    m.validate_preparation(*values)
    assert values == before


@pytest.mark.parametrize("fault", ["row_reorder", "skip_native", "pair_bridge", "summary_model"])
def test_exact_native_rows_pair_adjacency_and_independent_prep_required(fault):
    plan, rows, native, summary = synthetic_plan()
    if fault == "row_reorder":
        rows[:2] = rows[1::-1]
    elif fault == "skip_native":
        native.pop()
    elif fault == "pair_bridge":
        plan["pairs"][0]["later_frame"] += 166
    else:
        summary["model_forward_calls"] = 1
    with pytest.raises(ValueError):
        m.validate_preparation(plan, rows, native, summary)


def test_all_conditions_and_unscorable_pair_kept():
    token = np.array([[1, 0], [1, 0]], np.float32)
    geometry = dict(
        token_eligible=np.zeros(2, bool),
        token_center_crop_xy=np.zeros((2, 2)),
        token_center_native_xy=np.zeros((2, 2)),
    )

    class Source:
        def read(self, key):
            return token, geometry

    pair = dict(
        pair_index=0,
        request_index=0,
        earlier_frame=1,
        later_frame=2,
        view="top_cam",
        earlier_crop_id="a",
        later_crop_id="b",
    )
    saved = []
    rows = m.calculate_pairs(dict(pairs=[pair]), Source(), lambda *a: saved.append(a))
    assert [r["condition"] for r in rows] == ["raw", "masked"]
    assert all(not r["comparable"] and r["status"] == "no_eligible_tokens" for r in rows)
    assert len(saved) == 2 and all(np.isnan(r[3]["distances"]).all() for r in saved)


def test_successful_full_compatibility_required():
    summary = dict(
        schema="dino_shared_store_compatibility_v1",
        status="ok",
        passed=True,
        all_pairs_byte_exact=True,
        all_scalar_metadata_exact=True,
        original_token_bytes_exact=True,
        full_distances_and_all_NN_ties_preserved=True,
        shared_crops=71,
        shared_conditions=142,
        pair_conditions=1400,
        logical_arrays_restored=44800,
        model_calls=0,
        raw_images_opened=0,
    )
    registration = dict(
        bindings=dict(
            codec_source_sha256=m.PINS[m.Path(m.store.__file__).resolve()],
            approved_plan_sha256=m.crops.PLAN_SHA,
        )
    )
    m.validate_compatibility(summary, registration)
    summary["all_pairs_byte_exact"] = False
    with pytest.raises(ValueError):
        m.validate_compatibility(summary, registration)


def test_run_refuses_existing_output_before_input(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m.old, "CPUModel", lambda *a: pytest.fail("model initialized"))
    with pytest.raises(ValueError):
        m.run(tmp_path, "a" * 64)
