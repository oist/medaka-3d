"""Independent synthetic CPU checks; no model, GPU or experimental artifact."""

import copy
import json

import diagnose_dino_fixed_masks as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic tests must not construct a model")

    monkeypatch.setattr(m, "CPUModel", forbidden)


def fixed_plan():
    # Cardinalities alone; no actual image, mask, token or geometric evidence.
    values = ((4, 3, 5), (4, 4, 4), (3, 4, 4), (5, 5, 4), (3, 4, 4), (3, 3, 5))
    counts = {
        (native, view): values[i][j]
        for i, native in enumerate(m.prep.FRAMES)
        for j, view in enumerate(m.prep.VIEWS)
    }
    plan = m.prep.make_plan(counts)
    return plan, copy.deepcopy(plan["crops"])


def crop_fixture(kinds=("ready", "empty", "thin")):
    rows, arrays = [], {}
    for index, kind in enumerate(kinds):
        image = np.full((32, 32, 3), [40 + index, 70, 100], np.uint8)
        mask = np.zeros((32, 32), bool)
        if kind == "ready":
            mask[3:29, 3:29] = True
        elif kind == "thin":
            mask[12, 8:24] = True
        crop = m.partial.prepare_mask_crop(image, mask)
        crop_id = f"synthetic_{index}"
        row = dict(
            crop_id=crop_id,
            frame_idx=100 + index,
            view="top_cam",
            mask_row=index,
            crop_status=crop["crop_status"],
            token_status=crop["token_status"],
        )
        arrays[crop_id] = {
            key: value for key, value in crop.items() if isinstance(value, np.ndarray)
        }
        rows.append(row)
    first = {
        condition: arrays[rows[0]["crop_id"]][condition + "_normalized_chw"].copy()
        for condition in m.prep.CONDITIONS
    }
    # Every token/feature carries information, including the last nonselected row.
    golden = np.arange(256 * 384, dtype=np.float32).reshape(1, 256, 384) / 1024
    return rows, arrays, first, golden


def infer(rows, arrays, first, golden, forward, save=lambda *args: None):
    return m.infer_crops(
        rows,
        lambda row: arrays[row["crop_id"]],
        forward,
        save,
        first_inputs=first,
        smoke_tokens=golden,
    )


def test_inference_retains_all_rows_raw_then_masked_and_calls_tokenless_nonempty_crops():
    rows, arrays, first, golden = crop_fixture()
    before = {
        name: {key: m.prep.signature(value) for key, value in crop.items()}
        for name, crop in arrays.items()
    }
    events, saved = [], []
    buffer = golden[0].copy()

    def read(row):
        events.append(("read", row["crop_id"]))
        return arrays[row["crop_id"]]

    def forward(chw):
        events.append(("forward", m.prep.signature(chw)))
        return buffer, 0.125

    def save(record, tokens, geometry):
        events.append(("save", record["crop_id"], record["condition"]))
        record["synthetic_saved_path"] = record["crop_id"] + "/" + record["condition"]
        saved.append((copy.deepcopy(record), tokens.copy(), copy.deepcopy(geometry)))

    tokens, geometry, records = m.infer_crops(
        rows, read, forward, save, first_inputs=first, smoke_tokens=golden
    )
    assert len(records) == len(tokens) == len(saved) == 6
    assert list(tokens) == [
        (row["crop_id"], condition) for row in rows for condition in m.prep.CONDITIONS
    ]
    assert [record["forward_called"] for record in records] == [
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert [record["eligible_token_count"] for record in records][4:] == [0, 0]
    assert [e[0] for e in events] == [
        "read",
        "forward",
        "save",
        "forward",
        "save",
        "read",
        "save",
        "save",
        "read",
        "forward",
        "save",
        "forward",
        "save",
    ]
    assert len([e for e in events if e[0] == "forward"]) == 4
    for record, value, stored_geometry in saved:
        assert record["synthetic_saved_path"].endswith(record["condition"])
        assert value.shape == (256, 384) and value.dtype == np.float32
        assert set(stored_geometry) == set(m.GEOMETRY_KEYS)
        assert record["token_signature"] == m.prep.signature(value)
        if record["crop_status"] == "empty_mask":
            assert np.isnan(value).all()
            assert record["forward_seconds"] is None and record["finite_token_count"] == 0
        else:
            assert m.byte_exact(value, golden[0])
            assert record["finite_token_count"] == 256
    assert before == {
        name: {key: m.prep.signature(value) for key, value in crop.items()}
        for name, crop in arrays.items()
    }
    buffer[:] = -20
    assert m.byte_exact(tokens[rows[0]["crop_id"], "raw"], golden[0])
    geometry[rows[0]["crop_id"]]["token_index"][0] = -1
    assert arrays[rows[0]["crop_id"]]["token_index"][0] == 0


@pytest.mark.parametrize("condition", ["raw", "masked"])
def test_first_crop_input_byte_guard_includes_signed_zero(condition):
    rows, arrays, first, golden = crop_fixture(("ready",))
    values = arrays[rows[0]["crop_id"]][condition + "_normalized_chw"]
    values[0, 0, 0] = 0.0
    first[condition] = values.copy()
    first[condition][0, 0, 0] = -0.0
    np.testing.assert_array_equal(first[condition], values)
    with pytest.raises(ValueError, match="first crop/prepared source"):
        infer(rows, arrays, first, golden, lambda _: (golden[0].copy(), 0.1))


def test_first_smoke_guard_checks_last_token_and_feature_before_saving():
    rows, arrays, first, golden = crop_fixture(("ready",))
    changed = golden[0].copy()
    changed[-1, -1] = np.nextafter(changed[-1, -1], np.float32(np.inf))
    saved = []
    with pytest.raises(ValueError, match="tokens differ bytewise"):
        infer(
            rows, arrays, first, golden, lambda _: (changed, 0.1), lambda *args: saved.append(args)
        )
    assert not saved


@pytest.mark.parametrize("fault", ["dtype", "token_count", "feature_count", "batch_axis", "list"])
def test_inference_rejects_changed_model_output_contract(fault):
    rows, arrays, first, golden = crop_fixture(("ready",))
    result = {
        "dtype": golden[0].astype(np.float64),
        "token_count": golden[0, :255],
        "feature_count": golden[0, :, :383],
        "batch_axis": golden,
        "list": [],
    }[fault]
    with pytest.raises(ValueError, match="all256 original float32"):
        infer(rows, arrays, first, golden, lambda _: (result, 0.1))


def test_forward_input_mutation_fails_before_result_is_saved():
    rows, arrays, first, golden = crop_fixture(("ready",))
    saved = []

    def forward(values):
        values[0, 0, 0] += 1
        return golden[0].copy(), 0.1

    with pytest.raises(ValueError, match="mutated prepared crop"):
        infer(rows, arrays, first, golden, forward, lambda *args: saved.append(args))
    assert not saved


def test_nonfinite_model_token_rows_remain_recorded_instead_of_disappearing():
    rows, arrays, first, golden = crop_fixture(("ready",))
    count = 0

    def forward(_):
        nonlocal count
        result = golden[0].copy()
        if count:
            result[23, 8] = np.nan
            result[100, 7] = np.inf
        count += 1
        return result, 0.1

    tokens, _, records = infer(rows, arrays, first, golden, forward)
    assert records[1]["finite_token_count"] == 254
    assert tokens[rows[0]["crop_id"], "masked"].shape == (256, 384)
    assert np.isnan(tokens[rows[0]["crop_id"], "masked"][23, 8])
    assert np.isinf(tokens[rows[0]["crop_id"], "masked"][100, 7])


def test_empty_crop_cannot_substitute_a_finite_zero_image():
    rows, arrays, first, golden = crop_fixture(("ready", "empty"))
    arrays[rows[1]["crop_id"]]["raw_normalized_chw"][:] = 0
    with pytest.raises(ValueError, match="empty mask input must stay NaN"):
        infer(rows, arrays, first, golden, lambda _: (golden[0].copy(), 0.1))


def restore_packed(metadata, arrays):
    if isinstance(metadata, dict):
        if "array" in metadata:
            value = arrays[metadata["array"]]
            assert {key: metadata[key] for key in ("dtype", "shape", "sha256")} == m.prep.signature(
                value
            )
            return value.copy()
        if "nonfinite" in metadata:
            return {"NaN": np.nan, "+inf": np.inf, "-inf": -np.inf}[metadata["nonfinite"]]
        return {key: restore_packed(value, arrays) for key, value in metadata.items()}
    if isinstance(metadata, list):
        return [restore_packed(value, arrays) for value in metadata]
    return metadata


def assert_lossless(original, restored):
    if isinstance(original, np.ndarray):
        assert m.byte_exact(original, restored)
    elif isinstance(original, dict):
        assert set(original) == set(restored)
        for key in original:
            assert_lossless(original[key], restored[key])
    elif isinstance(original, (tuple, list)):
        assert len(original) == len(restored)
        for left, right in zip(original, restored, strict=True):
            assert_lossless(left, right)
    elif isinstance(original, (float, np.floating)) and np.isnan(original):
        assert np.isnan(restored)
    else:
        assert original == restored


def small_geometry(count, eligible=None):
    centers = np.column_stack((np.arange(count), np.arange(count) + 0.5))
    return dict(
        token_eligible=np.ones(count, bool) if eligible is None else np.asarray(eligible, bool),
        token_center_crop_xy=centers,
        token_center_native_xy=centers * 2 + 3,
    )


@pytest.mark.parametrize("kind", ["ties", "partial", "impossible"])
def test_nested_correspondence_packing_preserves_every_array_and_missing_reason(kind):
    left = np.array([[1, 0], [1, 0], [0, 1]], np.float32)
    right = np.array([[1, 0], [0, 1], [0, 1]], np.float32)
    left_geometry, right_geometry = small_geometry(3), small_geometry(3)
    if kind == "partial":
        left[1] = 0
    elif kind == "impossible":
        right_geometry["token_eligible"][:] = False
    result = m.partial.token_correspondence(left, right, left_geometry, right_geometry)
    arrays, metadata = m.pack_correspondence(result)
    restored = restore_packed(json.loads(json.dumps(metadata, allow_nan=False)), arrays)
    assert_lossless(result, restored)
    assert len(arrays) > 25
    assert arrays["distances"].shape == (3, 3)
    if kind == "ties":
        np.testing.assert_array_equal(
            arrays["correspondences__pairs"], [[0, 0], [1, 0], [2, 1], [2, 2]]
        )
        assert arrays["correspondences__mutual"].all()
    elif kind == "partial":
        assert metadata["status"] == "partial_invalid_tokens"
        assert np.isnan(arrays["distances"][1]).all()
        assert metadata["left"]["matched_fraction"] == pytest.approx(2 / 3)
    else:
        assert metadata["status"] == "no_eligible_tokens"
        assert metadata["left"]["matched_fraction"] == {"nonfinite": "NaN"}
        assert metadata["finite_distance_distribution"]["count"] == 0
    arrays["distances"][:] = -99
    assert not np.any(result["distances"] == -99)


def test_packing_scalar_nonfinite_and_ieee_array_values_are_distinct_and_lossless():
    value = dict(
        a=np.array([0.0, -0.0, np.nan, np.inf, -np.inf], np.float32),
        nested=[
            dict(n=np.float64(np.nan), p=np.float64(np.inf), m=-np.inf),
            (np.int64(5), np.bool_(True)),
        ],
    )
    arrays, metadata = m.pack_correspondence(value)
    assert metadata["nested"][0] == {
        "n": {"nonfinite": "NaN"},
        "p": {"nonfinite": "+inf"},
        "m": {"nonfinite": "-inf"},
    }
    assert_lossless(
        value, restore_packed(json.loads(json.dumps(metadata, allow_nan=False)), arrays)
    )
    assert np.signbit(arrays["a"][1])


@pytest.mark.parametrize(
    "value",
    [
        {"a__b": np.ones(1), "a": {"b": np.zeros(1)}},
        {"a": np.array([object()], object)},
        {"a": {1, 2}},
    ],
)
def test_packing_rejects_ambiguous_array_names_and_nonportable_values(value):
    with pytest.raises(ValueError):
        m.pack_correspondence(value)


def test_correspondence_rows_keeps_all700_pairs_and_both_conditions_in_plan_order(monkeypatch):
    plan, rows = fixed_plan()
    tokens = {
        (row["crop_id"], condition): np.array([[1.0, 0.0]], np.float32)
        for row in rows
        for condition in m.prep.CONDITIONS
    }
    geometry = {row["crop_id"]: small_geometry(1) for row in rows}
    geometry[rows[0]["crop_id"]]["token_eligible"][:] = False
    signatures = {key: m.prep.signature(value) for key, value in tokens.items()}
    saved = []

    def save(record, arrays):
        assert arrays["distances"].shape == (1, 1)
        record["synthetic_saved"] = True
        saved.append((record["pair_index"], record["condition"], record["status"]))

    result = m.correspondence_rows(plan["pairs"], tokens, geometry, save)
    expected = [
        (pair["pair_index"], condition) for pair in plan["pairs"] for condition in m.prep.CONDITIONS
    ]
    assert len(result) == len(saved) == 1400
    assert [(row["pair_index"], row["condition"]) for row in result] == expected
    assert [(index, condition) for index, condition, _ in saved] == expected
    assert all(row["synthetic_saved"] for row in result)
    expected_missing = 2 * sum(
        pair["earlier_crop_id"] == rows[0]["crop_id"] or pair["later_crop_id"] == rows[0]["crop_id"]
        for pair in plan["pairs"]
    )
    assert sum(row["status"] == "no_eligible_tokens" for row in result) == expected_missing > 0
    assert signatures == {key: m.prep.signature(value) for key, value in tokens.items()}


def test_correspondence_input_mutation_is_detected_before_saving(monkeypatch):
    tokens = {("left", c): np.ones((1, 2), np.float32) for c in m.prep.CONDITIONS}
    tokens.update({("right", c): np.ones((1, 2), np.float32) for c in m.prep.CONDITIONS})
    geometry = {key: small_geometry(1) for key in ("left", "right")}
    pair = dict(earlier_crop_id="left", later_crop_id="right", pair_index=0)
    saved = []

    def changed(left, *_):
        left[0, 0] = 123
        return {}

    monkeypatch.setattr(m.partial, "token_correspondence", changed)
    with pytest.raises(ValueError, match="mutated model outputs"):
        m.correspondence_rows([pair], tokens, geometry, lambda *args: saved.append(args))
    assert not saved


def test_complete_crop_plan_validation_accepts_fixed_denominators_without_model():
    plan, rows = fixed_plan()
    m.validate_crop_plan(plan, rows)
    assert len(rows) == 71 and len(plan["pairs"]) == 700


@pytest.mark.parametrize(
    "fault",
    [
        "frames",
        "views",
        "crop_missing",
        "row_order",
        "mask_row",
        "crop_id",
        "pairs",
        "requests",
        "conditions",
        "budget",
        "first",
        "view_counts",
    ],
)
def test_plan_validator_rejects_reselection_and_pair_or_condition_scope_drift(fault):
    plan, rows = fixed_plan()
    if fault in ("frames", "views"):
        plan[fault].reverse()
    elif fault == "crop_missing":
        rows.pop()
    elif fault == "row_order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "mask_row":
        rows[0]["mask_row"] = True
    elif fault == "crop_id":
        rows[0]["crop_id"] = rows[1]["crop_id"]
    elif fault in ("pairs", "requests"):
        plan[fault].pop()
    elif fault == "conditions":
        plan["conditions"] = ["raw"]
    elif fault == "budget":
        plan["maximum_forward_calls"] = 140
    elif fault == "first":
        plan["first_crop"]["mask_row"] = 1
    else:
        plan["mask_pair_counts_by_view"][m.prep.VIEWS[0]] += 1
    with pytest.raises(ValueError):
        m.validate_crop_plan(plan, rows)


@pytest.fixture
def smoke_summary(monkeypatch):
    runtime = {"synthetic_version": "cpu-fixture"}
    monkeypatch.setattr(m.assets, "runtime", lambda: runtime.copy())
    return dict(
        schema=m.smoke.STAGE,
        status="ok",
        model=m.assets.MODEL,
        commit=m.assets.COMMIT,
        crop_count=1,
        forward_calls=2,
        repeat_same_crop_byte_exact=True,
        patch_shape=[1, 256, 384],
        final_layernorm_not_global_pool=True,
        device="cpu",
        dtype="float32",
        torch_cpu_threads=2,
        XFORMERS_DISABLED=True,
        torch_load_weights_only=True,
        strict_state_load=True,
        GPU_used=False,
        training=False,
        runtime=runtime.copy(),
        XFORMERS_module_flags={
            key: dict(enabled=False, available=False)
            for key in ("attention", "block", "swiglu_ffn")
        },
        preprocessing=dict(resize_in_this_runner=False, normalization_in_this_runner=False),
    )


def test_smoke_validator_accepts_same_cpu2_layernorm_path(smoke_summary):
    m.validate_smoke(smoke_summary)


@pytest.mark.parametrize(
    "fault",
    [
        "device",
        "dtype",
        "threads",
        "bool_count",
        "repeat",
        "patch",
        "weights",
        "strict",
        "GPU",
        "training",
        "runtime",
        "backend",
        "resize",
        "normalize",
    ],
)
def test_smoke_validator_rejects_model_runtime_and_preprocessing_drift(smoke_summary, fault):
    updates = {
        "device": ("device", "cuda"),
        "dtype": ("dtype", "float64"),
        "threads": ("torch_cpu_threads", 4),
        "bool_count": ("crop_count", True),
        "repeat": ("repeat_same_crop_byte_exact", False),
        "patch": ("patch_shape", [1, 257, 384]),
        "weights": ("torch_load_weights_only", False),
        "strict": ("strict_state_load", False),
        "GPU": ("GPU_used", True),
        "training": ("training", True),
    }
    if fault in updates:
        key, value = updates[fault]
        smoke_summary[key] = value
    elif fault == "runtime":
        smoke_summary["runtime"]["synthetic_version"] = "different"
    elif fault == "backend":
        smoke_summary["XFORMERS_module_flags"]["attention"]["available"] = True
    else:
        key = "resize_in_this_runner" if fault == "resize" else "normalization_in_this_runner"
        smoke_summary["preprocessing"][key] = True
    with pytest.raises(ValueError):
        m.validate_smoke(smoke_summary)
