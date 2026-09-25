"""Synthetic cost and sealed-subset integration checks; no scientific inputs."""

from __future__ import annotations

from pathlib import Path

import compare_partial_body_cost_compositions as m
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    work = root / "work"
    work.mkdir(parents=True)
    for module in (m, m.saved, m.proof):
        monkeypatch.setattr(module, "REPO", root, raising=False)
        monkeypatch.setattr(module, "WORK", work, raising=False)
    monkeypatch.setattr(m, "INPUT_ROOT", work / "synthetic_input")
    for key in ("load_headers", "load_banks", "load_observations", "load_rig", "compare_bank"):
        monkeypatch.setattr(
            m.saved, key, lambda *a, **k: pytest.fail("unused scientific loader called")
        )
    original_load = np.load

    def confined_load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(root)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", confined_load)
    return root


def case_fixture(eligible=(False, True, False), arm=None, frame=82720):
    n = 3
    bank = dict(
        frame_idx=frame,
        policy=dict(
            weight=3.0,
            bodies_per_view=3,
            fixed_camera_denominator=3,
            cap_per_view=9.0,
            camera_order=list(m.VIEWS),
        ),
        candidate_index=np.arange(n, dtype=np.int64),
        candidate_pose_xyz=np.arange(n * 27, dtype=np.float64).reshape(n, 3, 3, 3) / 10,
        candidate_parent=np.array([0, 1, 0], dtype=np.int64),
        candidate_plan=np.zeros((n, 3, 3, 3), dtype=np.int32),
        candidate_kind=["original", "proposal", "proposal"],
        base_local_cost=np.array([1.0, 2.0, 3.0]),
        parent_path_cost=np.array([1.0, 0.0, 2.0]),
        views={},
    )
    columns = []
    for vi, view in enumerate(m.VIEWS):
        raw = np.array([0.0, 3.0, 81.0]) if eligible[vi] else np.zeros(n)
        normalized = raw / 3
        capped = np.clip(normalized, 0, 9) if eligible[vi] else np.zeros(n)
        bank["views"][view] = dict(
            eligible=bool(eligible[vi]),
            reason="synthetic",
            neutral=not eligible[vi],
            raw_uncapped_scene_cost=raw,
            uncapped_mean_body_cost=normalized,
            capped_view_contribution=capped,
            nonfinite_score=~np.isfinite(raw),
            cap_applied=(normalized > 9) | ~np.isfinite(raw),
        )
        columns.append(capped)
    bank["capped_contributions_by_view"] = np.stack(columns, axis=1)
    bank["sixth_instance_term"] = 3.0 * bank["capped_contributions_by_view"].sum(axis=1) / 3
    bank["adjusted_local_cost"] = bank["base_local_cost"] + bank["sixth_instance_term"]
    bank["adjusted_path_cost"] = bank["parent_path_cost"] + bank["adjusted_local_cost"]
    modes = dict(
        baseline=dict(
            m.saved.rank_costs(bank["adjusted_local_cost"].copy(), bank["parent_path_cost"]),
            sixth_term=bank["sixth_instance_term"].copy(),
        )
    )
    for mode in m.NEW_MODES:
        raw = np.array([[0.0, 6.0, 3.0], [6.0, 3.0, 0.0], [12.0, 12.0, 12.0]])
        if mode == "row_plus_mask":
            raw += 0.75
        partial, term = m.saved.bounded_term(raw, 15.0)
        modes[mode] = dict(
            m.saved.rank_costs(bank["base_local_cost"] + term, bank["parent_path_cost"]),
            sixth_term=term,
            raw_costs_by_view=raw,
            normalized_costs_by_view=partial,
            normalizer=15.0,
            scorer_ceiling=12.0 if mode == "row_only" else 15.0,
            invalid_projection_candidates=np.zeros(n, dtype=bool),
        )
    return dict(
        schema=m.saved.STAGE + "_case",
        frame_idx=frame,
        arm=arm or m.saved.ARMS[0],
        baseline_bank=bank,
        projections={v: bank["candidate_pose_xyz"][..., :2].copy() for v in m.VIEWS},
        modes=modes,
        final_selected_match=dict(available=False, reason="synthetic only"),
        parent_history_fixed=True,
        applies_only_to_saved_fit_candidates=True,
        reoptimized_fit=False,
        new_temporal_backtrace=False,
        physical_ID_truth=False,
    )


@pytest.mark.parametrize(
    "eligible", [(False, False, False), (True, True, True), (True, False, True)]
)
@pytest.mark.parametrize("alpha", [1, 9])
def test_old_view_preservation_and_explicit_parent_arithmetic(eligible, alpha):
    case = case_fixture(eligible)
    before = m.codec.split_diagnostic(case)[0]
    result = m.compare_case(case)
    bank = case["baseline_bank"]
    assert before == m.codec.split_diagnostic(case)[0]
    assert len(result["modes"]) == 11
    np.testing.assert_array_equal(
        result["old_capped_contributions_by_view"], bank["capped_contributions_by_view"]
    )
    for mode in m.NEW_MODES:
        q = case["modes"][mode]["normalized_costs_by_view"]
        all_mode = result["modes"][f"preserve_old__{mode}__all_views__alpha{alpha}"]
        missing = result["modes"][f"preserve_old__{mode}__old_ineligible_views__alpha{alpha}"]
        wanted = np.where(np.asarray(eligible)[None, :], 0.0, q)
        np.testing.assert_array_equal(missing["selected_partial_contributions_by_view"], wanted)
        np.testing.assert_array_equal(
            missing["path_cost"],
            bank["parent_path_cost"]
            + (
                bank["base_local_cost"] + (bank["sixth_instance_term"] + alpha * wanted.sum(axis=1))
            ),
        )
        if all(eligible):
            np.testing.assert_array_equal(missing["path_cost"], bank["adjusted_path_cost"])
        if not any(eligible):
            np.testing.assert_array_equal(missing["path_cost"], all_mode["path_cost"])


def test_zero_addition_keeps_old_exact_minimum_set_and_parent_winner():
    base, parent, old = np.array([1.0, 2.0, 5.0]), np.array([1.0, 0.0, 0.0]), np.zeros(3)
    result = m.compose(base, parent, old, np.zeros((3, 3)), np.ones(3, bool), 9, "all_views")
    assert result["local_minimum_indices"].tolist() == [0]
    assert result["path_minimum_indices"].tolist() == [0, 1]
    assert result["path_order"].tolist() == [0, 1, 2]


@pytest.mark.parametrize("scope", m.SCOPES)
def test_positive_infinity_stays_invalid_even_when_view_excluded(scope):
    q = np.array([[np.inf, 0.0, 0.0], [0.0, 0.5, 0.0]])
    with np.errstate(invalid="raise"):
        result = m.compose(
            np.zeros(2), np.zeros(2), np.zeros(2), q, np.ones(3, dtype=bool), 1, scope
        )
    assert result["path_order"][0] == 1
    assert np.isposinf(result["new_partial_term"][0])
    assert not np.isnan(result["sixth_term"]).any()
    if scope == "old_ineligible_views":
        assert not result["selected_partial_contributions_by_view"].any()


@pytest.mark.parametrize("alpha", [0, 2, 9.0, True])
def test_no_extra_alpha_search(alpha):
    with pytest.raises(ValueError):
        m.compose(
            np.zeros(1),
            np.zeros(1),
            np.zeros(1),
            np.zeros((1, 3)),
            np.ones(3, bool),
            alpha,
            "all_views",
        )


def test_old_raw_nonfinite_caps_but_ineligible_remains_zero():
    case = case_fixture((True, False, False))
    bank = case["baseline_bank"]
    for view in m.VIEWS:
        d = bank["views"][view]
        d["raw_uncapped_scene_cost"][:] = np.inf
        d["uncapped_mean_body_cost"][:] = np.inf
        d["nonfinite_score"][:] = True
        d["cap_applied"][:] = True
        d["capped_view_contribution"][:] = 9.0 if d["eligible"] else 0.0
    bank["capped_contributions_by_view"] = np.array([[9.0, 0.0, 0.0]] * 3)
    bank["sixth_instance_term"][:] = 9.0
    bank["adjusted_local_cost"] = bank["base_local_cost"] + 9.0
    bank["adjusted_path_cost"] = bank["parent_path_cost"] + bank["adjusted_local_cost"]
    eligible, columns, term = m.old_components(bank)
    assert eligible.tolist() == [True, False, False]
    np.testing.assert_array_equal(columns, bank["capped_contributions_by_view"])
    np.testing.assert_array_equal(term, np.full(3, 9.0))


@pytest.mark.parametrize("field", ["sixth_instance_term", "capped_contributions_by_view"])
def test_wrong_old_ledger_rejected(field):
    case = case_fixture()
    case["baseline_bank"][field].flat[0] += 0.25
    with pytest.raises(ValueError):
        m.compare_case(case)


def test_new_all_invalid_explicit_failure():
    with pytest.raises(ValueError, match="at least one finite"):
        m.compose(
            np.zeros(2),
            np.zeros(2),
            np.zeros(2),
            np.full((2, 3), np.inf),
            np.ones(3, bool),
            1,
            "old_ineligible_views",
        )


def sealed_fixture(monkeypatch):
    out = m.proof.Output(m.INPUT_ROOT)
    summaries, render_rows = [], []
    for arm, frame in m.saved.case_keys():
        case = case_fixture(arm=arm, frame=frame)
        # An unused diagnostic array must not be decoded by this adapter.
        case["unused_diagnostic"] = np.ones((2, 2), dtype=np.float32)
        archive = m.saved.write_case(out, f"cases/{arm}_{frame}", case)
        render_rows.append(m.saved.render_selection(case))
        summaries.append(
            dict(
                arm=arm,
                frame_idx=frame,
                candidates=3,
                archive=archive,
                rankings={
                    mode: {
                        k: m.saved.small_json(d[k])
                        for k in m.RANK_FIELDS
                        if k not in ("local_cost", "path_cost", "minimum_ties_are_exact_float64")
                    }
                    for mode, d in case["modes"].items()
                },
            )
        )
    out.write(
        "plan.json",
        dict(
            schema=m.saved.STAGE,
            cases=[list(k) for k in m.saved.case_keys()],
            modes=list(m.saved.MODES),
            terminal_horizon=116200,
            normalizers=dict(row_only=15.0, row_plus_mask=15.0),
            scorer_ceilings=dict(row_only=12.0, row_plus_mask=15.0),
            fixed_parent_history=True,
            old_sixth_replaced_not_added=True,
        ),
    )
    out.write(
        "summary.json",
        dict(
            schema=m.saved.STAGE,
            status="ok",
            cases=summaries,
            saved_candidate_banks=11,
            baseline_bank_unchanged=True,
            physical_ID_truth=False,
            new_trajectory_created=False,
        ),
    )
    out.write(
        "render_payload.json",
        dict(
            schema=m.saved.STAGE + "_render_payload_v1",
            cases=render_rows,
            modes=list(m.saved.MODES),
            original_observation_evidence=[dict(synthetic=True)],
            physical_ID_truth=False,
            raw_slot_labels_only=True,
            new_trajectory_created=False,
        ),
    )
    seal = dict(stage=m.saved.STAGE, status="ok", inputs={}, outputs=out.records)
    path = m.INPUT_ROOT / "comparison.provenance.json"
    path.write_bytes(m.proof.json_bytes(seal))
    monkeypatch.setattr(m, "INPUT_SHA", m.proof.sha(path))
    monkeypatch.setattr(m, "source_code", lambda *args: {"synthetic_source_trust": "fixture"})
    return seal


def test_full_synthetic_success_has_eleven_cases_compositions_and_readonly_input(monkeypatch):
    sealed_fixture(monkeypatch)
    actual_load = m.load_case
    loads = []

    def load(*args):
        result, decoded, total = actual_load(*args)
        assert decoded < total
        assert not result["baseline_bank"]["candidate_pose_xyz"].flags.writeable
        assert not result["modes"]["row_only"]["raw_costs_by_view"].flags.writeable
        loads.append((result["arm"], result["frame_idx"]))
        return result, decoded, total

    monkeypatch.setattr(m, "load_case", load)
    output = m.WORK / "result"
    m.run(output)
    seal = m.proof.read_json(output / "comparison.provenance.json")
    summary = m.proof.read_json(output / "summary.json")
    payload = m.proof.read_json(output / "render_payload.json")
    assert seal["status"] == "ok" and len(seal["outputs"]) == 25
    assert loads == m.saved.case_keys()
    assert len(summary["cases"]) == len(payload["cases"]) == 11
    assert all(len(c["rankings"]) == 11 for c in summary["cases"])
    assert all(len(c["selections"]) == 11 for c in payload["cases"])
    assert payload["negative_OldFast_images_are_observation_context_only"] is True
    for case in payload["cases"]:
        for selection in case["selections"].values():
            assert selection["path_cost"] == selection["parent_path_cost"] + selection["local_cost"]
            assert selection["support_not_recomputed"] is True


def test_failed_input_seal_never_reads_archives(monkeypatch):
    seal = sealed_fixture(monkeypatch)
    seal["status"] = "failed"
    path = m.INPUT_ROOT / "comparison.provenance.json"
    path.write_bytes(m.proof.json_bytes(seal))
    monkeypatch.setattr(m, "INPUT_SHA", m.proof.sha(path))
    monkeypatch.setattr(m, "load_case", lambda *a: pytest.fail("archive read before success"))
    output = m.WORK / "failed"
    with pytest.raises(ValueError, match="successful input"):
        m.run(output)
    assert m.proof.read_json(output / "comparison.provenance.json")["status"] == "failed"


def test_output_guard_before_input_io(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "load_headers", lambda *a: pytest.fail("input I/O before output guard"))
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(tmp_path / "outside_work")
    with pytest.raises(ValueError, match="fresh workspace"):
        m.run(m.WORK)
