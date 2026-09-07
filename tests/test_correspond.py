"""Multi-node correspondence: more evidence, and a missing node costs only a node."""

from __future__ import annotations

import numpy as np
import pytest

from medaka3d.correspond import (
    hypothesis_costs,
    label_costs,
    node_available,
    reconstruct_from_label_states,
    reconstruct_from_states,
    reconstruct_multinode,
    state_costs,
    viterbi_path,
    viterbi_sweep,
)
from medaka3d.reconstruct import N_ANIMALS, UNRESOLVED


class FakeCamera:
    """A camera at ``centre`` whose 'pixels' are already 3D points to look at."""

    def __init__(self, centre):
        self.centre = np.asarray(centre, float)

    def rays(self, uv):
        # uv here carries the target point in its first two columns plus a fixed
        # depth, which is enough to build a well-posed ray for the test.
        target = np.concatenate([uv, np.zeros((*uv.shape[:-1], 1))], axis=-1)
        origins = np.broadcast_to(self.centre, target.shape).copy()
        directions = target - origins
        norm = np.linalg.norm(directions, axis=-1, keepdims=True)
        directions = np.divide(directions, norm, out=np.zeros_like(directions),
                               where=norm > 0)
        # The real camera turns a NaN pixel into a NaN ray, and Stage C reads
        # exactly that to mean "this view does not hold this label". A fake that
        # answered with a zero direction instead would make every absent
        # detection look present.
        absent = ~np.isfinite(uv).all(axis=-1)
        return (np.where(absent[..., None], np.nan, origins),
                np.where(absent[..., None], np.nan, directions))


class FakeRig:
    views = ("top_cam", "side1_cam", "side2_cam")

    def camera(self, view):
        return FakeCamera({"top_cam": (0.0, 0.0, 400.0),
                           "side1_cam": (400.0, 0.0, 0.0),
                           "side2_cam": (0.0, 400.0, 0.0)}[view])


def build(n_frames=6, jitter=0.0):
    rng = np.random.default_rng(1)
    base = np.array([[0.0, 0.0], [60.0, 5.0], [-40.0, 55.0]])
    uv = {}
    for view in FakeRig.views:
        uv[view] = {}
        for node in ("mouth", "pectoral"):
            offset = 0.0 if node == "pectoral" else 6.0
            points = np.broadcast_to(base + offset, (n_frames, 3, 2)).copy()
            points += rng.normal(scale=jitter, size=points.shape)
            uv[view][node] = points
    complete = {v: np.ones(n_frames, bool) for v in FakeRig.views}
    return uv, complete


def test_anchor_must_be_among_the_solved_nodes():
    uv, complete = build()
    with pytest.raises(ValueError, match="must be among"):
        reconstruct_multinode(FakeRig(), "E", uv, complete, 6,
                              anchor="tail", nodes=("mouth", "pectoral"))


def test_node_available_needs_every_named_view():
    uv, _ = build()
    assert node_available(uv, FakeRig.views, "mouth").all()
    uv["side1_cam"]["mouth"][2, 1, 0] = np.nan
    got = node_available(uv, FakeRig.views, "mouth")
    assert not got[2]
    assert got[[0, 1, 3, 4, 5]].all()


def test_a_missing_secondary_node_does_not_lose_the_frame():
    """One absent mouth must cost the mouth's evidence, not the whole frame.

    This is the failure `c13` measured on the real data: judging a camera
    complete on the mouth threw away a fifth of the recording's third view.
    """
    uv, complete = build()
    uv["side1_cam"]["mouth"][3] = np.nan
    out, tally = reconstruct_multinode(FakeRig(), "E", uv, complete, 6,
                                       anchor="pectoral", nodes=("mouth", "pectoral"))
    assert out.resolved[3], "the frame must still be solved on the pectoral alone"
    assert tally["node_frames.pectoral"] == 6
    assert tally["node_frames.mouth"] == 5


def test_two_nodes_discriminate_at_least_as_well_as_one():
    """The margin is the point of the change, so it is what is asserted."""
    uv, complete = build(n_frames=40, jitter=0.4)
    one, _ = reconstruct_multinode(FakeRig(), "E", uv, complete, 40,
                                   anchor="pectoral", nodes=("pectoral",))
    two, _ = reconstruct_multinode(FakeRig(), "E", uv, complete, 40,
                                   anchor="pectoral", nodes=("mouth", "pectoral"))
    assert np.nanmedian(two.margin_mm) > np.nanmedian(one.margin_mm)


def test_single_node_reproduces_reconstruct():
    """With one node this must be the incumbent algorithm, not a near-miss."""
    from medaka3d.reconstruct import reconstruct

    uv, complete = build(n_frames=12, jitter=0.3)
    flat = {v: uv[v]["pectoral"] for v in FakeRig.views}
    incumbent = reconstruct(FakeRig(), "E", flat, complete, 12)
    mine, _ = reconstruct_multinode(FakeRig(), "E", uv, complete, 12,
                                    anchor="pectoral", nodes=("pectoral",))
    assert np.array_equal(incumbent.source, mine.source)
    assert np.allclose(incumbent.xyz, mine.xyz, equal_nan=True)
    assert np.allclose(incumbent.residual_mm, mine.residual_mm, equal_nan=True)


def test_hypothesis_costs_needs_two_views():
    with pytest.raises(ValueError, match="at least two views"):
        hypothesis_costs(np.zeros((2, 1, 3, 3)), np.zeros((2, 1, 3, 3)))


# ---------------------------------------------------------------------------
# Stage B — the correspondence as a piecewise-constant sequence
# ---------------------------------------------------------------------------


def label_inputs(n_frames=6, jitter=0.0):
    """`build`, plus the row-of-label map an already-label-ordered array implies."""
    uv, complete = build(n_frames=n_frames, jitter=jitter)
    rows = np.broadcast_to(np.arange(3), (n_frames, 3)).copy()
    return uv, complete, {v: rows for v in FakeRig.views}


def test_zero_penalty_is_the_per_frame_winner():
    """The sequence must contain the incumbent, or it is a different algorithm.

    At lambda = 0 the objective is exactly the sum of the per-frame costs, so the
    minimising sequence is the per-frame argmin. If this ever fails, the state
    space or the label ordering is wrong — which is the failure mode that is
    silent on real data and expensive to find there.
    """
    uv, complete, _ = label_inputs(n_frames=25, jitter=0.5)
    costs = state_costs(FakeRig(), uv, complete, 25, nodes=("mouth", "pectoral"))
    path = viterbi_path(costs.cost, 0.0)
    assert np.array_equal(path, costs.cost.argmin(axis=1))


def test_a_large_penalty_collapses_to_one_state():
    uv, complete, _ = label_inputs(n_frames=30, jitter=1.5)
    costs = state_costs(FakeRig(), uv, complete, 30, nodes=("mouth", "pectoral"))
    path = viterbi_path(costs.cost, 1e9)
    assert np.unique(path).size == 1
    assert path[0] == costs.cost.sum(axis=0).argmin()


def test_changes_are_non_increasing_in_the_penalty():
    """The monotonicity the decision document's lambda argument rests on."""
    rng = np.random.default_rng(0)
    cost = rng.random((400, 36))
    sweep = viterbi_sweep(cost, np.geomspace(1e-4, 1e3, 24))
    assert np.all(np.diff(sweep["changes"]) <= 0)
    assert sweep["changes"][0] == (np.diff(cost.argmin(axis=1)) != 0).sum()
    assert sweep["changes"][-1] == 0


def test_the_sweep_agrees_with_the_backtracked_path():
    """The sweep never materialises a path; it must still describe the same one."""
    rng = np.random.default_rng(1)
    cost = rng.random((300, 36))
    held = rng.random((300, 36))
    lambdas = np.geomspace(1e-3, 3.0, 9)
    sweep = viterbi_sweep(cost, lambdas, held_out=held)
    for index, penalty in enumerate(lambdas):
        path = viterbi_path(cost, float(penalty))
        assert sweep["changes"][index] == (np.diff(path) != 0).sum()
        assert np.isclose(sweep["fit_cost"][index],
                          cost[np.arange(300), path].sum()
                          + penalty * (np.diff(path) != 0).sum())
        assert np.isclose(sweep["held_cost"][index], held[np.arange(300), path].sum())


def test_a_frame_with_no_evidence_carries_the_state():
    """Fewer than two complete views must cost nothing, not force a choice."""
    uv, complete, _ = label_inputs(n_frames=8, jitter=0.4)
    for view in ("side1_cam", "side2_cam"):
        complete[view][4] = False
    costs = state_costs(FakeRig(), uv, complete, 8, nodes=("mouth", "pectoral"))
    assert np.all(costs.cost[4] == 0.0)
    assert costs.n_views_used[4] == 1


def test_two_view_frames_collapse_to_six_distinguishable_states():
    uv, complete, _ = label_inputs(n_frames=8, jitter=0.4)
    complete["side2_cam"][3] = False
    costs = state_costs(FakeRig(), uv, complete, 8, nodes=("mouth", "pectoral"))
    assert np.unique(np.round(costs.cost[3], 9)).size == 6
    # ...and the margin must be measured over those six, not over all 36, or it
    # would read 0 mm on every two-view frame.
    assert costs.margin["total"][3] > 0


def test_reconstruction_from_states_matches_the_per_frame_solver():
    """At lambda = 0 the rebuilt geometry must equal `reconstruct_multinode`'s."""
    uv, complete, rows = label_inputs(n_frames=20, jitter=0.5)
    incumbent, _ = reconstruct_multinode(FakeRig(), "E", uv, complete, 20,
                                         anchor="pectoral", nodes=("mouth", "pectoral"))
    costs = state_costs(FakeRig(), uv, complete, 20, nodes=("mouth", "pectoral"))
    path = viterbi_path(costs.cost, 0.0)
    mine = reconstruct_from_states(FakeRig(), "E", uv, complete, rows, path, 20,
                                   anchor="pectoral", nodes=("mouth", "pectoral"))
    assert np.array_equal(incumbent.source, mine.source)
    assert np.allclose(incumbent.xyz, mine.xyz, equal_nan=True)
    # One deliberate difference: `reconstruct_multinode` reports the multi-node
    # residual as one number per frame, repeated across the three slots. Here it
    # is per slot, which is what `Reconstruction` documents. The frame means must
    # still agree exactly, or the arithmetic has diverged rather than the layout.
    assert np.allclose(mine.residual_mm.mean(axis=1), incumbent.residual_mm[:, 0],
                       equal_nan=True)


def test_source_records_detector_rows_and_not_labels():
    """`c09` inverts `source` to find the human's slot; rows and labels differ."""
    uv, complete, rows = label_inputs(n_frames=5)
    shuffled = {v: np.broadcast_to(np.array([2, 0, 1]), (5, 3)).copy()
                for v in FakeRig.views}
    costs = state_costs(FakeRig(), uv, complete, 5, nodes=("pectoral",))
    path = viterbi_path(costs.cost, 0.0)
    plain = reconstruct_from_states(FakeRig(), "E", uv, complete, rows, path, 5,
                                    anchor="pectoral", nodes=("pectoral",))
    moved = reconstruct_from_states(FakeRig(), "E", uv, complete, shuffled, path, 5,
                                    anchor="pectoral", nodes=("pectoral",))
    assert np.array_equal(moved.source, np.array([2, 0, 1])[plain.source])
    assert np.allclose(plain.xyz, moved.xyz, equal_nan=True)


def test_the_sequence_refuses_a_one_frame_flip():
    """The behaviour the stage exists for, on the smallest case that shows it.

    Frame 5 alone prefers a different correspondence. A per-frame solver takes
    it; charging for the change refuses it, and — unlike a modal window — the
    refusal costs nothing in lag when a change is genuine.
    """
    cost = np.ones((11, 36)) * 10.0
    cost[:, 0], cost[:, 1] = 0.0, 1.0
    cost[5, 0], cost[5, 1] = 1.0, 0.0
    assert viterbi_path(cost, 0.0)[5] == 1
    assert np.all(viterbi_path(cost, 3.0) == 0)
    lasting = cost.copy()
    lasting[5:, 0], lasting[5:, 1] = 1.0, 0.0
    settled = viterbi_path(lasting, 3.0)
    assert np.all(settled[:5] == 0) and np.all(settled[5:] == 1), "no lag at a real change"


# ---------------------------------------------------------------------------
# Stage C — the cardinality limiter removed
# ---------------------------------------------------------------------------


def label_uv(n_frames=6):
    """The same scene as ``build``, laid out by 2D track label rather than by row."""
    uv, _ = build(n_frames)
    return {v: {node: value.copy() for node, value in nodes.items()}
            for v, nodes in uv.items()}


def drop(uv, view, node, frame, label):
    """Remove one detection: the label that frame's view no longer holds."""
    uv[view][node][frame, label] = np.nan


def rows_all_present(n_frames):
    return {v: np.tile(np.arange(N_ANIMALS, dtype=np.int8), (n_frames, 1))
            for v in FakeRig.views}


def test_a_complete_frame_costs_exactly_the_old_cost_over_three():
    """The new cost is the old one divided by the animal count — nothing else moved.

    This is what lets kappa, chosen on the old scale, be reused on the new one.
    """
    uv, complete = build(8, jitter=0.4)
    old = state_costs(FakeRig(), uv, complete, 8, nodes=("mouth", "pectoral"))
    new = label_costs(FakeRig(), uv, 8, anchor="pectoral", nodes=("mouth", "pectoral"))
    assert np.allclose(new.cost, old.cost / N_ANIMALS, atol=1e-6)
    assert (new.terms == N_ANIMALS * 2).all()
    assert (new.cost.argmin(axis=1) == old.cost.argmin(axis=1)).all()


def test_a_missing_label_costs_one_ray_and_not_a_camera():
    """One camera losing one animal must leave the other two animals untouched."""
    uv = label_uv(4)
    drop(uv, "side1_cam", "pectoral", 2, 1)
    drop(uv, "side1_cam", "mouth", 2, 1)
    costs = label_costs(FakeRig(), uv, 4, anchor="pectoral", nodes=("mouth", "pectoral"))
    winner = costs.cost.argmin(axis=1)
    rays = costs.slot_rays[np.arange(4), winner]
    assert (rays[[0, 1, 3]] == 3).all()
    assert sorted(rays[2].tolist()) == [2, 3, 3]
    # Still fully resolvable: two cameras is a reconstruction, one is not.
    assert costs.resolvable().all()


def test_two_views_short_of_a_label_do_not_lose_the_frame():
    """Label 0 of one camera is not label 0 of another; only the state says so.

    Both side views losing *their* label 0 leaves every animal two cameras under
    some state, because the two absences can fall on different animals. The
    correspondence is what decides whether they do, which is why the frame is
    kept and scored rather than discarded on a count.
    """
    uv = label_uv(3)
    for view in ("side1_cam", "side2_cam"):
        for node in ("mouth", "pectoral"):
            drop(uv, view, node, 1, 0)
    costs = label_costs(FakeRig(), uv, 3, anchor="pectoral", nodes=("mouth", "pectoral"))
    assert costs.resolvable().all()
    assert costs.terms[1].min() < costs.terms[1].max()  # the stated bias, in one frame


def test_an_animal_seen_by_one_camera_is_not_resolved():
    uv = label_uv(3)
    for view in ("side1_cam", "side2_cam"):
        for node in ("mouth", "pectoral"):
            for label in (0, 1):
                drop(uv, view, node, 1, label)
    costs = label_costs(FakeRig(), uv, 3, anchor="pectoral", nodes=("mouth", "pectoral"))
    assert costs.resolvable().tolist() == [True, False, True]

    chosen = viterbi_path(costs.cost, 0.0)
    out, slot_rays = reconstruct_from_label_states(
        FakeRig(), "fake", uv, rows_all_present(3), chosen, 3,
        anchor="pectoral", nodes=("mouth", "pectoral"))
    assert out.resolved.tolist() == [True, False, True]
    assert np.isnan(out.xyz[1]).any()
    assert slot_rays[1].min() == 1


def test_an_absent_label_records_no_detection_row():
    """`c09` inverts ``source``; a view that saw nothing must say so, not point at row 0."""
    uv = label_uv(3)
    drop(uv, "side2_cam", "pectoral", 1, 2)
    drop(uv, "side2_cam", "mouth", 1, 2)
    rows = rows_all_present(3)
    rows["side2_cam"][1, 2] = UNRESOLVED
    costs = label_costs(FakeRig(), uv, 3, anchor="pectoral", nodes=("mouth", "pectoral"))
    out, _ = reconstruct_from_label_states(
        FakeRig(), "fake", uv, rows, viterbi_path(costs.cost, 0.0), 3,
        anchor="pectoral", nodes=("mouth", "pectoral"))
    column = FakeRig.views.index("side2_cam")
    assert (out.source[1, :, column] == UNRESOLVED).sum() == 1
    assert out.resolved.all()


def test_the_margin_ignores_states_the_frame_cannot_tell_apart():
    """A frame short of a label must not report 0 mm and read as total ambiguity."""
    uv = label_uv(3)
    for node in ("mouth", "pectoral"):
        drop(uv, "side2_cam", node, 1, 0)
    costs = label_costs(FakeRig(), uv, 3, anchor="pectoral", nodes=("mouth", "pectoral"))
    assert costs.margin["total"][1] > 0.0
