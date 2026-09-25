"""Finite Tail MP caller tests; no saved numerical payloads or old DP rerun."""

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import identity_node_memory_dp as helper  # noqa: E402
import run_tail_minute_mouth_memory_solve as mod  # noqa: E402


def geometry():
    g = {}
    for key, (tail, kind) in mod.previous.LAYOUT.items():
        dtype = {"i": np.int64, "f": np.float64, "b": bool, "U": "U9"}[kind]
        g[key] = np.zeros((3, *tail), dtype=dtype)
    g["frame_idx"] = np.arange(10, 13, dtype=np.int64)
    g["pose_xyz"][:, :, :, 0] = np.array([0, 40, 80])[None, :, None]
    g["observed"][:] = True
    g["observed"][1, [0, 2], 1] = False
    g["observed"][1, 2, 0] = False
    g["identity_review"][:, 2] = True
    g["xyz"] = np.where(g["observed"][:, :, 1, None], g["pose_xyz"][:, :, 1], np.nan)
    g["accepted_xyz"] = np.where(g["identity_review"][..., None], np.nan, g["xyz"])
    return g


def controls(g):
    before = np.tile(np.arange(3, dtype=np.int64), (3, 1))
    control = mod.previous.build_identity_output(g, before)
    optimal = np.zeros((3, 6), bool)
    optimal[:, 0] = True
    optimal[1, 5] = True
    gap = np.array([np.inf, 0.0, 1.0])
    dp = dict(
        frame_idx=g["frame_idx"],
        permutation_by_id=before,
        optimal_state_mask=optimal,
        state_alternative_gap=gap,
        min_marginal_delta=np.zeros((3, 6)),
    )
    counter = mod.preflight.counter.reachable_counts(g["frame_idx"], g["observed"][:, :, mod.NODES])
    return (
        control,
        dp,
        dict(
            frame_idx=g["frame_idx"],
            state_counts=counter.state_counts,
            edge_counts=counter.edge_counts,
        ),
    )


def test_tiny_end_to_end_counter_new_solver_links_ties_review(monkeypatch):
    g = geometry()
    control, dp, counts = controls(g)
    monkeypatch.setattr(mod, "TIE_NATIVES", (11,))
    before = {k: mod.previous.signature(v) for k, v in g.items()}
    result, identity, links, oldlinks, changes, chest, report = mod.solve(
        g, counts, control, dp, [], helper
    )
    assert result.node_indices == (1, 0)
    np.testing.assert_array_equal(result.path_state, [0, 0, 0])
    assert report["prior_ties"][0]["before_optimal_P"] == [0, 5]
    assert report["prior_ties"][0]["after_optimal_P"] == [0]
    assert report["prior_ties"][0]["after_state_alternative_gap"] > 0
    assert not changes["changed_by_ID"].any() and not chest["changed"].any()
    assert links["cost_by_event"].shape == (3, 3, 2)
    assert np.isclose(links["cost_by_event"].sum(), result.path_cost)
    assert np.isclose(
        oldlinks["cost_by_event"][:, :, 0].sum(),
        mod.previous.chosen_memory_links(g, control["identity_to_raw_slot"])[
            "brownian_motion_cost_by_id"
        ].sum(),
    )
    assert identity["raw_identity_review"][:, 2].all()
    assert np.isnan(identity["conservative_accepted_xyz"][:, 2]).all()
    assert before == {k: mod.previous.signature(v) for k, v in g.items()}


def test_MP_classification_can_be_chest_latent_but_mouth_supported():
    frames = np.arange(2, dtype=np.int64)
    before = np.tile(np.arange(3, dtype=np.int64), (2, 1))
    after = before.copy()
    after[1] = [2, 1, 0]
    direct = np.ones((2, 3, 2), bool)
    direct[1, [0, 2], 0] = False
    direct[1, 2, 1] = False
    arrays, chest, report = mod.compare_mappings(frames, before, after, direct)
    assert chest["latent_only"][1, [0, 2]].all()
    assert arrays["direct_involved_MP_cells"][1, [0, 2], 1].all()
    assert not arrays["direct_involved_MP_cells"][1, :, 0].any()
    assert report["by_node"]["1"]["latent_only_cells"] == 2
    assert report["by_node"]["0"]["direct_involved_cells"] == 2
    assert not arrays["all_MP_latent_change"].any()


@pytest.mark.parametrize("fault", ["support", "signatures", "mapping", "ties"])
def test_saved_control_rejects_mismatch(fault, monkeypatch):
    g = geometry()
    control, dp, counts = controls(g)
    monkeypatch.setattr(mod, "TIE_NATIVES", (11,))
    signs = {k: mod.previous.signature(v) for k, v in g.items()}
    support = dict(frame_idx=g["frame_idx"], direct_nodes=g["observed"][:, :, mod.NODES].copy())
    scalar = dict(
        status="optimal", initial_state=0, scale_mm=0.8, weight=0.2, approximation_used=False
    )
    mod.validate_saved_control(g, signs, control, dp, scalar, counts, support)
    if fault == "support":
        support["direct_nodes"][1, 0, 1] = False
    elif fault == "signatures":
        signs["sources"]["sha256"] = "0" * 64
    elif fault == "mapping":
        control["identity_to_raw_slot"][1] = [2, 1, 0]
    else:
        dp["optimal_state_mask"][2, 5] = True
    with pytest.raises(ValueError):
        mod.validate_saved_control(g, signs, control, dp, scalar, counts, support)


def test_independent_ledger_wrong_cost_refused(monkeypatch):
    g = geometry()
    control, dp, counts = controls(g)
    monkeypatch.setattr(mod, "TIE_NATIVES", (11,))

    def bad_links(*args, **kwargs):
        value = helper.node_memory_path_links(*args, **kwargs)
        value["cost_by_event"][1, 0, 1] += 1
        return value

    fake = SimpleNamespace(
        solve_identity_node_memory_dp=helper.solve_identity_node_memory_dp,
        node_memory_path_links=bad_links,
    )
    with pytest.raises(ValueError, match="link cost"):
        mod.solve(g, counts, control, dp, [], fake)


def test_all_new_dataclass_fields_saved_with_node_axis(tmp_path, monkeypatch):
    g = geometry()
    control, dp, counts = controls(g)
    monkeypatch.setattr(mod, "TIE_NATIVES", (11,))
    result, *_ = mod.solve(g, counts, control, dp, [], helper)
    monkeypatch.setattr(mod.previous, "WORK", tmp_path)
    out = mod.previous.Output(tmp_path / "new")
    mod.previous.save_result(out, "MP", result)
    scalar = json.loads((out.root / "MP/dp_scalars.json").read_text())
    with np.load(out.root / "MP/dp_arrays.npz", allow_pickle=False) as f:
        assert set(f.files) | set(scalar) == {r.name for r in dataclasses.fields(result)}
        assert f["state_last_event"].shape[1:] == (3, 2)
        assert f["edge_delta_frames"].shape[1:] == (3, 2)
    assert scalar["node_indices"] == [1, 0]


def test_actual_small_JSON_only_Tail_allowed_Original_failed_preserved(monkeypatch):
    read = Path.read_bytes

    def guarded(path):
        assert path.suffix in (".py", ".json"), "actual arrays forbidden"
        return read(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("no NPZ"))
    monkeypatch.setattr(
        helper, "solve_identity_node_memory_dp", lambda *a, **k: pytest.fail("no DP")
    )
    _, _, plan, _, _, anchors = mod.load_inputs(mod.previous.Evidence())
    assert plan["original_MP_solved"] is False and plan["geometry"] == "tail"
    assert plan["original_MP_capacity_failure"]["completed"] is False
    assert plan["tail_MP_capacity"]["completed"] is True and len(anchors) == 11


@pytest.mark.parametrize("fault", ["original_hidden", "tail_incomplete", "node_order"])
def test_preflight_status_not_overridden(fault):
    # Only sealed JSON is read; no array or graph input is touched.
    p = json.loads((mod.PREFLIGHT / "plan.json").read_text())
    s = json.loads((mod.PREFLIGHT / "summary.json").read_text())
    mod.validate_preflight(p, s, p)
    bad = copy.deepcopy(s)
    if fault == "original_hidden":
        bad["methods"]["original"]["completed"] = True
    elif fault == "tail_incomplete":
        bad["methods"]["tail"]["completed"] = False
    else:
        bad["node_indices"] = [0, 1]
    with pytest.raises(ValueError):
        mod.validate_preflight(p, bad, p)
