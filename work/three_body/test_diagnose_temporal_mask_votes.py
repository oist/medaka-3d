"""Focused arithmetic, missingness and source-bound saved-query fixtures only."""

from pathlib import Path
from types import SimpleNamespace

import diagnose_temporal_mask_votes as m
import numpy as np
import pytest


def measure(xy, membership, selected=None, available=True):
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    initial = np.nan_to_num(xy, copy=True) / 2
    selected = np.ones(len(xy), bool) if selected is None else np.asarray(selected, bool)
    geometry = m.base.core.mask_memberships(xy, None, (1200, 1200))
    return m.subset_metrics(
        xy,
        initial,
        np.asarray(membership, bool).reshape(len(xy), -1),
        selected,
        available=available,
        sampled=geometry["sampled"],
        inside=geometry["in_image"],
    )


def test_overlap_votes_ties_and_same_initial_pairs_have_independent_oracle():
    row, a = measure([[0, 0], [2, 0], [0, 2]], [[1, 0], [1, 1], [0, 1]])
    assert a["votes"].tolist() == [2, 2]
    assert a["single_votes"].tolist() == [1, 1]
    assert a["positive_argmax_ties"].tolist() == [True, True]
    assert row["single_membership_count"] == 2 and row["multi_membership_count"] == 1
    assert row["maximum_vote_fraction"] == 2 / 3 and row["single_maximum_share"] == 0.5
    assert row["rms_radius_px"] == pytest.approx(4 / 3)
    assert row["pair_distance_median_px"] == 2
    assert row["pair_distance_change_median_px"] == 1
    assert a["pair_query_indices"].tolist() == [[0, 1], [0, 2], [1, 2]]


@pytest.mark.parametrize(
    "available,reason,unassigned",
    [(False, "masks_unavailable", None), (True, "no_current_masks", 2)],
)
def test_none_and_available_empty_are_distinct(available, reason, unassigned):
    row, a = measure([[0, 0], [2, 0]], np.zeros((2, 0), bool), available=available)
    assert row["vote_status"] == reason and row["unassigned_count"] == unassigned
    assert a["votes"].shape == (0,)
    assert row["current_mask_count"] == (0 if available else None)


def test_zero_votes_keep_all_raw_zero_ties_but_no_positive_winner():
    row, a = measure([[0, 0], [2, 0]], [[0, 0, 0], [0, 0, 0]])
    assert row["vote_status"] == "zero_votes"
    assert a["argmax_ties"].all() and not a["positive_argmax_ties"].any()
    assert row["maximum_vote_fraction"] == 0 and row["single_maximum_share"] is None
    assert not row["concentration_comparable"]


def test_one_point_is_not_dispersion_or_multi_point_concentration():
    row, a = measure([[4, 6]], [[1, 1]])
    assert row["vote_status"] == "one_selected_point"
    assert row["rms_radius_px"] is None and row["pair_count"] == 0
    assert row["dispersion_status"] == "one_finite_point"
    assert not row["concentration_comparable"]
    assert a["centroid_xy"].tolist() == [4, 6]


def test_nonfinite_does_not_disappear_from_reported_subset_denominator():
    row, _ = measure([[4, 6], [np.nan, np.nan]], [[1], [0]])
    assert row["selected_count"] == 2 and row["finite_count"] == 1
    assert row["maximum_vote_fraction"] == 0.5
    assert row["nonfinite_count"] == 1 and not row["concentration_comparable"]


def test_continuous_outside_and_rounded_sampled_remain_distinct():
    row, _ = measure([[-0.1, 1], [2, 2]], [[1], [1]])
    assert row["sampled_count"] == 2 and row["continuous_inside_count"] == 1
    assert row["sampled_outside_continuous_count"] == 1


def test_empty_selected_has_all_reasons_and_no_input_mutation():
    xy = np.array([[0, 0], [2, 0]], np.float32)
    mm = np.ones((2, 2), bool)
    selected = np.zeros(2, bool)
    originals = [a.tobytes() for a in (xy, mm, selected)]
    row, a = m.subset_metrics(
        xy, xy, mm, selected, available=True, sampled=np.ones(2, bool), inside=np.ones(2, bool)
    )
    assert row["selected_count"] == 0 and row["vote_status"] == "no_selected_points"
    assert row["maximum_vote_fraction"] is None and np.isnan(a["centroid_xy"]).all()
    assert originals == [a.tobytes() for a in (xy, mm, selected)]


def test_subset_partition_uses_reports_only_and_all_four_parts():
    lk = np.array([1, 1, 0, 0], bool)
    ct = np.array([1, 0, 1, 0], bool)
    out = m.paired_subsets(lk, ct)
    assert {k: np.flatnonzero(v).tolist() for k, v in out.items()} == dict(
        common=[0], lk_only=[1], ct_only=[2], neither=[3]
    )
    assert np.stack(list(out.values())).sum(0).tolist() == [1, 1, 1, 1]
    out["common"][:] = False
    assert lk.tolist() == [True, True, False, False]


def membership_fixture():
    xy = np.array([[1, 1], [2, 2], [3, 3]], np.float32)
    origin = np.array([0, 0, 1], np.int64)
    saved = {
        k: np.asarray(v).copy()
        for k, v in m.base.core.mask_memberships(xy, None, (1200, 1200)).items()
    }
    saved["available"] = np.asarray(True)
    saved["membership"] = np.array([[1, 0], [1, 1], [0, 1]], bool)
    saved["native_frame"] = np.asarray(50, np.int64)
    reported = np.array([[1, 0, 1]], bool)
    metrics = dict(
        xy=xy[None],
        reported=reported,
        finite=saved["finite"][None],
        continuous_inside=saved["in_image"][None],
        rounded_sampled=saved["sampled"][None],
        mask_available=np.array([True]),
        mask_count=np.array([2]),
        union_membership=np.array([[1, 1, 1]], bool),
    )
    for key, selection in (
        ("all_query_group_mask_count", np.ones(3, bool)),
        ("reported_group_mask_count", reported[0]),
    ):
        counts = np.stack(
            [saved["membership"][(origin == g) & selection].sum(0) for g in range(2)]
        ).astype(np.int64)
        saved[key] = counts
        saved[key + "_argmax_ties"] = counts == counts.max(1, keepdims=True)
        saved[key + "_positive"] = counts.any(1)
    return saved, metrics, origin


def test_saved_recount_passes_then_one_vote_or_native_or_geometry_drift_rejected():
    s, a, origin = membership_fixture()
    before = {k: m.base.signature(v) for k, v in s.items()}
    assert m.verify_membership(s, a, origin, 2, 0, 50)
    assert before == {k: m.base.signature(v) for k, v in s.items()}
    for field in (
        "reported_group_mask_count",
        "native_frame",
        "pixel_xy",
        "all_query_group_mask_count_argmax_ties",
    ):
        bad = {k: v.copy() for k, v in s.items()}
        bad[field].flat[0] = (
            not bad[field].flat[0] if bad[field].dtype == bool else bad[field].flat[0] + 1
        )
        with pytest.raises(ValueError):
            m.verify_membership(bad, a, origin, 2, 0, 50)


def test_saved_unsigned_count_dtype_drift_is_not_equal_bytes():
    s, a, origin = membership_fixture()
    s["reported_group_mask_count"] = s["reported_group_mask_count"].astype(np.uint64)
    with pytest.raises(ValueError):
        m.verify_membership(s, a, origin, 2, 0, 50)


def test_all_eleven_selection_rows_keep_empty_groups():
    a = {k: dict(reported=np.zeros((64, 2), bool)) for k in m.ARMS}
    s = m.selections(a, 0)
    assert len(s) == 11 and sum(p == "own" for p, _, _, _ in s) == 3
    assert all(not mask.any() for *_, mask in s)


def query_fixture(tmp_path):
    prep, root = tmp_path / "prep", tmp_path / "run"
    prep.mkdir()
    root.mkdir()
    q = dict(
        xy=np.zeros((1, 2), np.float32),
        mask_index=np.array([0], np.int64),
        corner_index=np.array([0], np.int64),
        frame_idx=np.asarray(10, np.int64),
        available=np.asarray(True),
    )

    def save(path, arrays):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        return dict(path=str(path), sha256=m.base.sha(path), size_bytes=path.stat().st_size)

    pp = prep / "queries.npz"
    rp = root / "top_cam/queries.npz"
    pm = {str(pp): save(pp, q)}
    rm = {str(rp): save(rp, q)}
    for arm in m.ARMS:
        p = root / f"top_cam/{arm}/metrics.npz"
        rm[str(p)] = save(
            p, dict(xy=np.zeros((64, 1, 2), np.float32), reported=np.ones((64, 1), bool))
        )
    row = dict(
        view="top_cam",
        path="queries.npz",
        sha256=pm[str(pp)]["sha256"],
        query_count=1,
        rows=[dict(mask_index=0, query_start=0, query_stop=1, query_count=1)],
        array_signatures={k: m.base.signature(a) for k, a in q.items()},
    )
    context = dict(
        spec=dict(first=10),
        root=root,
        prep=prep,
        members=rm,
        prep_members=pm,
        queries=[row],
        seal=dict(inputs={str(pp): row["sha256"]}),
    )

    class Evidence:
        def __init__(self):
            self.paths = {}

        def add(self, path, expected=None):
            actual = m.base.sha(path)
            assert expected is None or actual == expected
            self.paths[str(path)] = actual

    return context, Evidence(), q, rp, save


def test_prepared_query_all_fields_exact_and_signed_zero_drift_rejected(tmp_path):
    ctx, e, q, rp, save = query_fixture(tmp_path)
    saved, _, _ = m.load_view(ctx, "top_cam", e)
    assert {k: m.base.signature(a) for k, a in saved.items()} == {
        k: m.base.signature(a) for k, a in q.items()
    }
    q["xy"][0, 0] = -0.0
    ctx["members"][str(rp)] = save(rp, q)
    with pytest.raises(ValueError, match="saved prepared query"):
        m.load_view(ctx, "top_cam", e)


def test_failed_comparison_is_rejected_before_query_payload(monkeypatch, tmp_path):
    def refuse(*args):
        raise ValueError("successful completed input required")

    monkeypatch.setattr(m.base, "verified", refuse)
    monkeypatch.setattr(m, "load_npz", lambda *_: pytest.fail("payload opened"))
    with pytest.raises(ValueError, match="successful"):
        m.load_dataset(m.DATASETS[0], SimpleNamespace())


def test_summary_retains_frame0_separately_and_none_denominators():
    row, _ = measure([[1, 1]], [[0]])
    base = dict(window="a", view="top", comparison="own", subset="reported", arm="lk", **row)
    rows = [dict(base, initial_query_frame=True), dict(base, initial_query_frame=False)]
    result = m.summarize(rows)[0]
    assert result["all64_group_times"] == 2 and result["tracking63_group_times"] == 1
    assert result["descriptive_distributions"]["rms_radius_px"]["finite_count"] == 0
    assert result["descriptive_distributions"]["rms_radius_px"]["denominator"] == 1


def test_full_native_orchestration_retains_empty_columns_and_all_eleven_rows(monkeypatch):
    saved, one, origin = membership_fixture()
    metrics = {arm: {k: np.repeat(v, 64, axis=0) for k, v in one.items()} for arm in m.ARMS}
    query = dict(xy=one["xy"][0].copy(), mask_index=origin)
    qrow = dict(rows=[dict(query_count=2, status="ready"), dict(query_count=1, status="ready")])
    contexts = {}
    for t in range(64):
        s = {k: v.copy() for k, v in saved.items()}
        s["native_frame"] = np.asarray(50 + t, np.int64)
        if t in (20, 21):
            s["available"] = np.asarray(t == 21)
            s["membership"] = np.zeros((3, 0), bool)
            for k in ("all_query_group_mask_count", "reported_group_mask_count"):
                s[k] = np.zeros((2, 0), np.int64)
                s[k + "_argmax_ties"] = np.zeros((2, 0), bool)
                s[k + "_positive"] = np.zeros(2, bool)
            for arm in m.ARMS:
                metrics[arm]["mask_available"][t] = t == 21
                metrics[arm]["mask_count"][t] = 0 if t == 21 else -1
                metrics[arm]["union_membership"][t] = False
        contexts[50 + t] = s
    monkeypatch.setattr(m, "load_view", lambda *_: (query, qrow, metrics))
    monkeypatch.setattr(m, "bound_path", lambda root, mem, name, evidence: Path(name))
    monkeypatch.setattr(m, "load_npz", lambda p: contexts[int(p.stem.split("_")[-1])])
    blocks = {}
    output = SimpleNamespace(write=lambda name, value: blocks.__setitem__(name, value))
    rows, votes = [], []
    context = dict(spec=dict(name="synthetic", first=50), root=Path("unused"), members={})
    result = m.analyze_view(context, "top_cam", None, output, rows, votes)
    assert result["membership_archives_verified"] == 192
    assert len(blocks) == 64 and len(rows) == 64 * 2 * 11
    assert sum(r["comparison"] == "own" for r in rows) == 64 * 2 * 3
    assert blocks["synthetic/top_cam/native_70.npz"]["votes"].shape == (22, 0)
    assert blocks["synthetic/top_cam/native_71.npz"]["votes"].shape == (22, 0)
    assert {r["vote_status"] for r in rows if r["native_frame"] == 70} == {"masks_unavailable"}
    assert {r["vote_status"] for r in rows if r["native_frame"] == 71} == {"no_current_masks"}
    assert all(b["selected"].shape == (22, 3) for b in blocks.values())


def test_empty_mask_vote_csv_still_has_a_header():
    assert (
        m.csv_bytes([], fields=["row_id", "current_mask_row"]).decode("utf-8-sig")
        == "row_id,current_mask_row\r\n"
    )
