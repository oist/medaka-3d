"""Pure synthetic tests: all paths are temporary and actual input loaders are blocked."""

from __future__ import annotations

import copy
import io
from fractions import Fraction

import numpy as np
import pytest
import render_partial_body_weight_envelope as m


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(m, "WORK", work)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m, "COMPARISON_ROOT", work / "comparison")
    monkeypatch.setattr(
        m.drawing, "load_context", lambda *a: pytest.fail("actual metadata forbidden")
    )
    monkeypatch.setattr(
        m.drawing, "photograph_and_points", lambda *a: pytest.fail("actual images forbidden")
    )
    monkeypatch.setattr(m.prior.cv2, "VideoCapture", lambda *a: pytest.fail("video forbidden"))
    real_load = np.load

    def safe_load(file, *args, **kwargs):
        assert isinstance(file, io.BytesIO), "only captured synthetic arrays may be decoded"
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", safe_load)
    return work


def rational(number):
    number = Fraction(number)
    return {
        "numerator": str(number.numerator),
        "denominator": str(number.denominator),
        "decimal_approx": float(number),
    }


def envelope_fixture(count=6):
    mode = dict(
        factor_only_minimum_indices=[5],
        weight_one_exact_minimum_indices=[1, 2, 3],
        weight_one_saved_path_minimum_indices=[1],
        envelope=dict(
            zero_weight_minimum_indices=[0, 1],
            positive_breakpoints=[dict(weight=rational(1), minimum_indices=[1, 2, 3])],
            intervals=[
                dict(lower=rational(0), upper=rational(1), minimum_indices=[1]),
                dict(lower=rational(1), upper=None, minimum_indices=[2, 3]),
            ],
            ever_optimal_indices=[0, 1, 2, 3],
        ),
    )
    return dict(
        arm=m.drawing.ARMS[0],
        frame_idx=82720,
        candidate_count=count,
        baseline_path_minimum_indices=[4],
        modes={key: copy.deepcopy(mode) for key in m.MODES},
    )


def test_complete_union_keeps_breakpoint_ties_zero_only_and_factor_minimum():
    c = envelope_fixture()
    before = copy.deepcopy(c)
    selected = m.select_candidates(c)
    assert list(selected) == [0, 1, 2, 3, 4, 5]
    assert selected[0] == ["row w=0", "row+mask w=0"]
    assert any("(tie)" in x for x in selected[3])
    assert selected[4] == ["baseline path minimum"]
    assert selected[5] == ["row factor minimum", "row+mask factor minimum"]
    assert c == before


@pytest.mark.parametrize("bad", ["missing_union", "interval", "fraction", "index"])
def test_invalid_envelope_rejected(bad):
    c = envelope_fixture()
    mode = c["modes"]["row_only"]
    if bad == "missing_union":
        mode["envelope"]["ever_optimal_indices"] = [1, 2, 3]
    elif bad == "interval":
        mode["envelope"]["intervals"][1]["lower"] = rational(2)
    elif bad == "fraction":
        mode["envelope"]["positive_breakpoints"][0]["weight"]["decimal_approx"] = 2.0
    else:
        mode["factor_only_minimum_indices"] = [6]
    with pytest.raises(ValueError):
        m.select_candidates(c)


def synthetic_case():
    e = envelope_fixture()
    count = e["candidate_count"]
    pose = np.arange(count * 27, dtype=np.float64).reshape(count, 3, 3, 3) + 5
    return dict(
        arm=e["arm"],
        frame=e["frame_idx"],
        count=count,
        arrays=dict(
            candidate_index=np.arange(count),
            candidate_pose_xyz=pose,
            candidate_parent=np.arange(count) % 2,
            candidate_plan=np.full((count, 3, 3, 3), -1, np.int32),
            base_local_cost=np.arange(count, dtype=float) + 1,
            parent_path_cost=np.arange(count, dtype=float) * 10,
        ),
        terms=dict(
            baseline=np.arange(count, dtype=float),
            row_only=np.linspace(0, 2, count),
            row_plus_mask=np.linspace(0, 3, count),
        ),
        projections={v: pose[..., :2].copy() + k for k, v in enumerate(m.VIEWS)},
        kinds=["synthetic_predict"] * count,
        selected=m.select_candidates(e),
        envelope=e,
        final_matches=[3],
        input_archive={"synthetic": True},
        selected_array_signatures={},
    )


def test_candidate_record_whole_body_costs_and_open_plan_scope():
    case = synthetic_case()
    before = case["arrays"]["candidate_pose_xyz"].copy()
    row = m.candidate_record(case, 3)
    assert row["baseline_local_cost"] == 7
    assert row["baseline_path_cost"] == 37
    assert row["parent_index"] == 1
    assert row["matches_saved_final_pose_and_sources"]
    assert row["plan_is_assigned_views_not_direct_support"]
    assert np.array_equal(row["pose_xyz"], before[3])
    assert np.array_equal(case["arrays"]["candidate_pose_xyz"], before)


def test_all_pages_use_all_selected_candidates_for_one_crop(monkeypatch):
    case = synthetic_case()
    images = {v: np.zeros((200, 200, 3), np.uint8) for v in m.VIEWS}
    points = {v: np.zeros((1, 3, 2), np.float32) for v in m.VIEWS}
    seen = []

    def crop(image, mask, sl, proposals):
        seen.append(np.stack(proposals))
        return np.array([1, 2]), np.array([180, 190])

    monkeypatch.setattr(m.drawing.visual, "display_crop", crop)
    crops = m.crop_for_case(case, images, points)
    assert len(seen) == 3 and all(a.shape[0] == 6 for a in seen)
    assert all(x == dict(lower_xy=[1, 2], upper_xy=[180, 190]) for x in crops.values())
    plots = []
    real_panel = m.drawing.draw_panel

    def panel(ax, image, sl, proposal):
        real_panel(ax, image, sl, proposal)
        plots.append(ax)

    monkeypatch.setattr(m.drawing, "draw_panel", panel)
    for page, selected in enumerate(([0, 1, 2, 3], [4, 5]), 1):
        png = m.render_page(case, selected, images, points, crops, page, 2)
        decoded = m.prior.cv2.imdecode(np.frombuffer(png, np.uint8), m.prior.cv2.IMREAD_COLOR)
        assert decoded is not None and decoded.ndim == 3
    assert len(plots) == 18
    assert all(ax.get_xlim() == (1, 180) and ax.get_ylim() == (190, 2) for ax in plots)
    body_lines = [line for ax in plots for line in ax.lines if line.get_marker() in ("o", "s", "^")]
    assert body_lines and all(line.get_markerfacecolor() == "none" for line in body_lines)


def write_synthetic_case(work):
    root = m.COMPARISON_ROOT
    out = m.proof.Output(root)
    arrays = {}

    def store(a):
        a = np.asarray(a)
        key = f"a{len(arrays):05d}"
        arrays[key] = a
        return {"__array__": key, "signature": m.proof.signature(a)}

    pose = np.arange(54, dtype=np.float64).reshape(2, 3, 3, 3) + 50
    bank = dict(
        candidate_index=store(np.arange(2)),
        candidate_pose_xyz=store(pose),
        candidate_parent=store(np.array([0, 0])),
        candidate_plan=store(np.full((2, 3, 3, 3), -1, np.int32)),
        base_local_cost=store([1.0, 2.0]),
        parent_path_cost=store([0.0, 0.0]),
        candidate_kind=["synthetic_predict", "synthetic_flow"],
    )
    terms = dict(baseline=[3.0, 0.0], row_only=[1.0, 0.0], row_plus_mask=[2.0, 0.0])
    modes = {mode: dict(sixth_term=store(values)) for mode, values in terms.items()}
    projections = {v: store(pose[..., :2] + vi) for vi, v in enumerate(m.VIEWS)}
    # This is deliberately outside the selected field schema. Reading all payloads would fail.
    arrays["unread_object_payload"] = np.array([object()], dtype=object)
    arm, frame = m.drawing.ARMS[0], 82720
    stem = f"cases/{arm}_{frame}"
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    archive_path = root / (stem + ".npz")
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(stream.getvalue())
    archive = dict(
        path=str(archive_path),
        sha256=m.proof.sha(archive_path),
        size_bytes=archive_path.stat().st_size,
    )
    out.records[stem + ".npz"] = archive
    tree = dict(
        schema=m.drawing.COMPARISON_STAGE + "_case",
        arm=arm,
        frame_idx=frame,
        baseline_bank=bank,
        projections=projections,
        modes=modes,
        parent_history_fixed=True,
        new_temporal_backtrace=False,
        final_selected_match=dict(available=False),
    )
    out.write(stem + ".json", dict(archive=archive, tree=tree))
    envelope = dict(
        arm=arm,
        frame_idx=frame,
        candidate_count=2,
        baseline_path_minimum_indices=[1],
        modes={},
        candidates=[],
    )
    for mode, bound in (("row_only", Fraction(1)), ("row_plus_mask", Fraction(1, 2))):
        envelope["modes"][mode] = dict(
            factor_only_minimum_indices=[1],
            weight_one_exact_minimum_indices=[0, 1] if mode == "row_only" else [1],
            weight_one_saved_path_minimum_indices=[0, 1] if mode == "row_only" else [1],
            envelope=dict(
                zero_weight_minimum_indices=[0],
                positive_breakpoints=[dict(weight=rational(bound), minimum_indices=[0, 1])],
                intervals=[
                    dict(lower=rational(0), upper=rational(bound), minimum_indices=[0]),
                    dict(lower=rational(bound), upper=None, minimum_indices=[1]),
                ],
                ever_optimal_indices=[0, 1],
            ),
        )
    for i in range(2):
        b, old = float(i + 1), terms["baseline"][i]
        envelope["candidates"].append(
            dict(
                candidate_index=i,
                candidate_kind=bank["candidate_kind"][i],
                candidate_parent=0,
                base_local_cost=b,
                parent_path_cost=0.0,
                original_sixth_term=old,
                original_adjusted_local_cost=b + old,
                original_adjusted_path_cost=b + old,
                row_only_term=terms["row_only"][i],
                row_plus_mask_term=terms["row_plus_mask"][i],
            )
        )
    seal = dict(outputs=out.records)
    return envelope, seal


def test_selected_schema_read_cost_binding_and_tamper_rejection(isolate):
    envelope, seal = write_synthetic_case(isolate)
    case = m.load_case(m.COMPARISON_ROOT, seal, envelope, m.proof.Evidence())
    assert list(case["selected"]) == [0, 1]
    assert len(case["selected_array_signatures"]) == 12
    assert "unread_object_payload" not in case["selected_array_signatures"]
    assert not case["arrays"]["candidate_pose_xyz"].flags.writeable
    envelope["candidates"][1]["base_local_cost"] += 0.01
    with pytest.raises(ValueError, match="cost metadata"):
        m.load_case(m.COMPARISON_ROOT, seal, envelope, m.proof.Evidence())


def test_synthetic_run_outputs_all_selected_rows_and_explicit_scope(isolate, monkeypatch):
    envelope, seal = write_synthetic_case(isolate)
    path = isolate / "fake_envelope.json"
    path.write_bytes(m.proof.json_bytes(dict(schema="fake_synthetic_only", cases=[envelope])))
    context = dict(
        envelope=dict(cases=[envelope]),
        envelope_path=str(path),
        comparison_seal=seal,
        source_code={},
    )
    monkeypatch.setattr(m, "load_context", lambda *a: context)
    monkeypatch.setattr(
        m.drawing,
        "photograph_and_points",
        lambda context, frame, view, evidence: (
            np.zeros((160, 160, 3), np.uint8),
            np.zeros((1, 3, 2), np.float32),
            dict(fake=True, frame=frame, view=view),
        ),
    )
    output = isolate / "render"
    m.run(path, m.proof.sha(path), output)
    summary = m.proof.read_json(output / "summary.json")
    provenance = m.proof.read_json(output / "render.provenance.json")
    assert provenance["status"] == "ok"
    assert summary["contact_sheets"] == 1
    assert summary["new_reranking_archives_decoded"] == 1
    assert summary["cases"][0]["selected_candidate_indices"] == [0, 1]
    assert summary["all_markers_are_open_proposals"]
    assert not summary["physical_ID_truth"] and not summary["new_trajectory_created"]
    assert len(provenance["outputs"]) == 4
    for record in provenance["outputs"].values():
        assert m.proof.sha(record["path"]) == record["sha256"]


def test_output_guard_never_reaches_real_loaders(isolate, monkeypatch):
    monkeypatch.setattr(m, "load_context", lambda *a: pytest.fail("input guard failed"))
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(isolate / "fake.json", "0" * 64, isolate.parent / "outside")
    assert not (isolate.parent / "outside").exists()
