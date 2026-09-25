"""Synthetic saved-data arithmetic only; no model or actual artifacts."""

import copy
from itertools import pairwise
from typing import ClassVar

import numpy as np
import pytest
import summarize_dino_minute_images as m


def fixture(legacy=False):
    frames = [1, 2, 3]
    views = list(m.prep.VIEWS)
    counts = {(n, v): 0 for n in frames for v in views}
    if legacy:
        counts.update({(1, views[0]): 5, (2, views[0]): 20, (3, views[0]): 30, (1, views[1]): 16})
    else:
        counts.update({(1, views[0]): 1, (2, views[0]): 1, (3, views[0]): 1, (1, views[1]): None})
    crops = [
        dict(frame_idx=n, view=v, mask_row=i, crop_id=f"f{n}_{v}_m{i:03d}")
        for n in frames
        for v in views
        for i in range(counts[n, v] or 0)
    ]
    requests, pairs = [], []
    for a, b in pairwise(frames):
        for v in views:
            x, y = counts[a, v], counts[b, v]
            req = dict(
                request_index=len(requests),
                earlier_frame=a,
                later_frame=b,
                view=v,
                earlier_mask_count=x,
                later_mask_count=y,
                status="view_unavailable"
                if x is None or y is None
                else "no_mask_pairs"
                if not x or not y
                else "available",
                mask_pair_count=(x or 0) * (y or 0),
            )
            requests.append(req)
            for i in range(x or 0):
                for j in range(y or 0):
                    pairs.append(
                        dict(
                            pair_index=len(pairs),
                            request_index=req["request_index"],
                            earlier_frame=a,
                            later_frame=b,
                            view=v,
                            earlier_mask_row=i,
                            later_mask_row=j,
                            earlier_crop_id=f"f{a}_{v}_m{i:03d}",
                            later_crop_id=f"f{b}_{v}_m{j:03d}",
                        )
                    )
    plan = dict(
        frames=frames,
        views=views,
        conditions=["raw", "masked"],
        crops=crops,
        requests=requests,
        pairs=pairs,
    )
    inference = [
        dict(
            **c,
            condition=k,
            eligible_token_count=8 if legacy or c["frame_idx"] != 3 else 0,
            finite_token_count=256,
            crop_status="ready",
        )
        for c in crops
        for k in plan["conditions"]
    ]
    lookup = {(r["crop_id"], r["condition"]): r for r in inference}
    rows, bitmaps = [], {}
    for p in pairs:
        for c in plan["conditions"]:
            lc, rc = (
                lookup[p[k], c]["eligible_token_count"]
                for k in ("earlier_crop_id", "later_crop_id")
            )
            comparable = bool(lc and rc)
            value = np.zeros((256, 256), bool)
            if comparable:
                value[0, 0] = True
                value[1, 1 if c == "raw" else 2] = True

            def side(count, available=comparable):
                return dict(
                    eligible_count=count,
                    valid_count=count,
                    invalid_eligible_count=0,
                    matched_fraction=2 / count if available else {"nonfinite": "NaN"},
                    tie_fraction=0.0 if available else {"nonfinite": "NaN"},
                )

            row = dict(
                **p,
                condition=c,
                comparable=comparable,
                status="comparable" if comparable else "no_eligible_tokens",
                shared_pair_id=f"p{p['pair_index']:04d}_{c}",
                metadata=dict(
                    left=side(lc),
                    right=side(rc),
                    mutual_distance_distribution=dict(
                        count=int(value.sum()),
                        mean=(0.2 if c == "raw" else 0.1) if comparable else {"nonfinite": "NaN"},
                    ),
                    mutual_nn=dict(array="mutual_nn", **m.store.signature(value)),
                ),
            )
            rows.append(row)
            bitmaps[row["shared_pair_id"]] = value
    return plan, inference, rows, bitmaps


def test_missing_requests_zeroeligible_nan_and_empty_equality_remain():
    plan, inference, rows, values = fixture()
    before = copy.deepcopy((plan, inference, rows))
    paired, summary = m.analyze(plan, inference, rows, lambda r: values[r["shared_pair_id"]])
    assert len(paired) == 2 and summary["pair_condition_records"] == 4
    assert summary["requested_pair_views"] == 6
    assert summary["request_status_counts"] == {
        "available": 2,
        "no_mask_pairs": 3,
        "view_unavailable": 1,
    }
    assert summary["zero_eligible_token_crops"] == 1
    assert plan["requests"][1]["earlier_mask_count"] is None
    assert plan["requests"][2]["earlier_mask_count"] == 0
    assert paired[0]["mutual_edge_jaccard"] == pytest.approx(1 / 3)
    assert paired[0]["masked_minus_raw_mean_mutual_distance"] == pytest.approx(-0.1)
    assert paired[1]["exact_mutual_edge_set_equal"] is True
    assert np.isnan(paired[1]["mutual_edge_jaccard"])
    assert np.isnan(paired[1]["raw"]["coverage_mean"])
    zero = summary["zero_eligible_pairs"]
    assert zero["pairs"] == zero["both_empty_mutual_edge_sets"] == 1
    assert zero["nonempty_exact_mutual_edge_set_equal"] == 0
    assert zero["raw"]["coverage_mean"]["finite_count"] == 0
    assert zero["raw"]["coverage_mean"]["median"] is None
    assert summary["all_views_descriptive_pool"]["raw"]["coverage_mean"]["count"] == 2
    assert summary["all_views_descriptive_pool"]["raw"]["coverage_mean"]["nonfinite_count"] == 1
    assert before == (plan, inference, rows)
    arrays = m.numeric_arrays(paired)
    assert arrays["raw__coverage_mean"].shape == (2,) and np.isnan(arrays["raw__coverage_mean"][1])
    assert arrays["raw__mutual_edge_count"].tolist() == [2.0, 0.0]


def assert_common_equal(old, new):
    if isinstance(old, dict):
        for k, v in old.items():
            assert k in new
            assert_common_equal(v, new[k])
    elif isinstance(old, list):
        assert len(old) == len(new)
        for a, b in zip(old, new, strict=True):
            assert_common_equal(a, b)
    elif isinstance(old, float) and np.isnan(old):
        assert np.isnan(new)
    else:
        assert old == new


def test_original_fixed700_arithmetic_is_identical_on_synthetic_legacy_scope():
    plan, inference, rows, values = fixture(legacy=True)
    assert len(plan["crops"]) == 71 and len(plan["pairs"]) == 700

    def read(r):
        return values[r["shared_pair_id"]]

    a, sa = m.prior.analyze(inference, rows, read)
    b, sb = m.analyze(plan, inference, rows, read)
    assert_common_equal(a, b)
    for key in (
        "by_view",
        "all_views_descriptive_pool",
        "by_exact_minimum_eligible_tokens",
        "crop_eligible_token_counts",
    ):
        assert_common_equal(sa[key], sb[key])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_pair",
        "reorder",
        "wrong_crop",
        "same_shared_id",
        "count_none_to_zero",
        "request_bridge",
        "eligible_change",
        "bitmap_count",
    ],
)
def test_binding_and_denominator_drift_rejected(fault):
    plan, inference, rows, values = fixture()
    if fault == "missing_pair":
        rows.pop()
    elif fault == "reorder":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "wrong_crop":
        inference[0]["crop_id"] = "other"
    elif fault == "same_shared_id":
        rows[1]["shared_pair_id"] = rows[0]["shared_pair_id"]
    elif fault == "count_none_to_zero":
        plan["requests"][1]["earlier_mask_count"] = 0
    elif fault == "request_bridge":
        plan["requests"][0]["later_frame"] = 3
    elif fault == "eligible_change":
        rows[0]["metadata"]["left"]["eligible_count"] = 7
    else:
        values[rows[0]["shared_pair_id"]][0, 0] = False
    with pytest.raises(ValueError):
        m.analyze(plan, inference, rows, lambda r: values[r["shared_pair_id"]])


def test_full_ties_are_retained_and_jaccard_is_bitmap_not_best_one():
    plan, inference, rows, values = fixture()
    for r in rows[:2]:
        a = values[r["shared_pair_id"]]
        a[:] = False
        a[0, :2] = True
        r["metadata"]["left"]["tie_fraction"] = 0.5
    paired, _ = m.analyze(plan, inference, rows, lambda r: values[r["shared_pair_id"]])
    assert paired[0]["mutual_edge_jaccard"] == 1.0
    assert paired[0]["raw"]["forward_tie_fraction"] == 0.5
    assert paired[0]["mutual_edge_intersection"] == 2


@pytest.mark.parametrize("fault", ["identity", "metadata", "signature"])
def test_shared_reader_binding_refused(fault):
    _, _, rows, values = fixture()
    row = rows[0]

    class Reader:
        index: ClassVar[dict] = {
            "pairs": {
                row["shared_pair_id"]: dict(
                    left_id=row["earlier_crop_id"] + "__raw",
                    right_id=row["later_crop_id"] + "__raw",
                )
            }
        }

        def read_pair(self, key):
            meta = copy.deepcopy(row["metadata"])
            if fault == "metadata":
                meta["left"]["eligible_count"] = 5
            return {"mutual_nn": values[key].copy()}, meta

    if fault == "identity":
        Reader.index["pairs"][row["shared_pair_id"]]["left_id"] = "other"
    elif fault == "signature":
        row["metadata"]["mutual_nn"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.read_mutual(Reader(), row)


def test_shared_reader_returns_owned_complete_bitmap():
    _, _, rows, values = fixture()
    row = rows[0]

    class Reader:
        index: ClassVar[dict] = {
            "pairs": {
                row["shared_pair_id"]: dict(
                    left_id=row["earlier_crop_id"] + "__raw",
                    right_id=row["later_crop_id"] + "__raw",
                )
            }
        }

        def read_pair(self, key):
            return {"mutual_nn": values[key].copy()}, copy.deepcopy(row["metadata"])

    got = m.read_mutual(Reader(), row)
    assert m.store.signature(got) == m.store.signature(values[row["shared_pair_id"]])
    assert not np.shares_memory(got, values[row["shared_pair_id"]])


def test_existing_output_refused_before_input_read(tmp_path, monkeypatch):
    monkeypatch.setattr(m.prep, "WORK", tmp_path)
    monkeypatch.setattr(m.old, "verified", lambda *a: pytest.fail("input read not permitted"))
    with pytest.raises(ValueError):
        m.run(tmp_path)
