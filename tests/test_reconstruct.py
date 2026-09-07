"""Cross-view correspondence and 3D linking, on animals whose true positions are known.

The fixture rig is a pinhole (see `conftest.py`), so three synthetic fish can be
projected into three cameras exactly and the solver's answer compared against the
truth it was never given.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import project
from medaka3d.reconstruct import UNRESOLVED, hypotheses, reconstruct, solve
from medaka3d.track import METHODS, UNTRACKED, TrackConfig, _match, propagate_to_views

TRIO = np.array([[40.0, 60.0, 30.0], [120.0, 90.0, 100.0], [70.0, 150.0, 160.0]])


def views_of(rig, xyz_by_frame: np.ndarray, orders=None):
    """Project (n_frames, 3, 3) world points into every camera, optionally shuffled.

    ``orders`` is ``view -> (n_frames, 3)`` giving the order the detector reported
    the animals in. A detector emits instances in whatever order its peak finder
    produced, and that order has nothing to do with identity — reconstructing the
    permutation is the entire job.
    """
    uv, complete = {}, {}
    n_frames = xyz_by_frame.shape[0]
    for view in rig.views:
        points = project(rig.camera(view), xyz_by_frame)
        if orders is not None and view in orders:
            points = np.take_along_axis(points, orders[view][:, :, None], axis=1)
        uv[view] = points
        complete[view] = np.ones(n_frames, dtype=bool)
    return uv, complete


class TestHypotheses:
    def test_three_views_enumerate_thirty_six(self):
        h = hypotheses(3)
        assert h.shape == (36, 3, 3)
        # View 0 is the reference and is never permuted; permuting it too would
        # count every labelling six times over.
        assert (h[:, :, 0] == np.arange(3)).all()

    def test_two_views_enumerate_six(self):
        assert hypotheses(2).shape == (6, 3, 2)


class TestSolve:
    def test_recovers_a_shuffled_correspondence(self, rig):
        frames = np.tile(TRIO, (4, 1, 1))
        orders = {
            "side1_cam": np.tile([2, 0, 1], (4, 1)),
            "side2_cam": np.tile([1, 2, 0], (4, 1)),
        }
        uv, _ = views_of(rig, frames, orders)
        origins, directions = [], []
        for view in rig.views:
            o, d = rig.camera(view).rays(uv[view])
            origins.append(o)
            directions.append(d)
        assignment, xyz, residual, best, second = solve(
            np.stack(origins, axis=1), np.stack(directions, axis=1)
        )
        assert np.allclose(xyz, frames, atol=1e-6)
        assert (residual < 1e-6).all()
        assert (best < 1e-6).all()
        # The runner-up is not a near miss: it asks rays aimed at different
        # animals to meet, and they miss by centimetres.
        assert (second > 10.0).all()
        # `orders` says which animal each detector row holds; `assignment` says
        # which row each slot is in. They are inverses of one another, and
        # asserting the inverse is what makes this a check rather than a tautology.
        for column, view in enumerate(("side1_cam", "side2_cam"), start=1):
            inverse = np.argsort(orders[view], axis=1)
            assert (assignment[:, :, column] == inverse).all()


class TestReconstruct:
    def test_three_views(self, rig):
        frames = np.tile(TRIO, (5, 1, 1))
        uv, complete = views_of(rig, frames)
        out = reconstruct(rig, "E", uv, complete, 5)
        assert out.resolved.all()
        assert (out.n_views_used == 3).all()
        assert np.allclose(out.xyz, frames, atol=1e-6)
        assert (out.margin_mm > 10.0).all()

    def test_two_views_are_enough(self, rig):
        frames = np.tile(TRIO, (5, 1, 1))
        uv, complete = views_of(rig, frames)
        complete["side2_cam"][:] = False
        out = reconstruct(rig, "E", uv, complete, 5)
        assert out.resolved.all()
        assert (out.n_views_used == 2).all()
        assert np.allclose(out.xyz, frames, atol=1e-6)
        # The view that took no part is marked, not silently filled in.
        assert (out.source[:, :, rig.views.index("side2_cam")] == UNRESOLVED).all()

    def test_one_view_is_not(self, rig):
        frames = np.tile(TRIO, (5, 1, 1))
        uv, complete = views_of(rig, frames)
        complete["side1_cam"][:] = False
        complete["side2_cam"][:] = False
        out = reconstruct(rig, "E", uv, complete, 5)
        assert not out.resolved.any()
        assert np.isnan(out.xyz).all()

    def test_mixed_availability_is_grouped_not_padded(self, rig):
        frames = np.tile(TRIO, (6, 1, 1))
        uv, complete = views_of(rig, frames)
        complete["side2_cam"][[1, 3]] = False
        complete["side1_cam"][5] = False
        complete["top_cam"][5] = False
        out = reconstruct(rig, "E", uv, complete, 6)
        assert list(out.n_views_used) == [3, 2, 3, 2, 3, 0]
        assert np.allclose(out.xyz[:5], frames[:5], atol=1e-6)


class TestTrackers:
    def moving(self, n_frames=40, step=(0.5, -0.3, 0.2)):
        step = np.asarray(step, dtype=float)
        return TRIO[None, :, :] + step[None, None, :] * np.arange(n_frames)[:, None, None]

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_a_clean_sequence_yields_exactly_three_tracks(self, rig, method):
        frames = self.moving()
        rng = np.random.default_rng(0)
        orders = {
            view: np.argsort(rng.random((frames.shape[0], 3)), axis=1)
            for view in ("side1_cam", "side2_cam", "top_cam")
        }
        uv, complete = views_of(rig, frames, orders)
        out = reconstruct(rig, "E", uv, complete, frames.shape[0])
        tracks = METHODS[method](out, TrackConfig(max_step_mm=20.0, max_gap_frames=5))
        assert len(set(tracks.ravel().tolist())) == 3
        # And each track stays on one animal: its own displacement per frame is
        # the animal's, not the 100 mm hop to a neighbour.
        for track_id in set(tracks.ravel().tolist()):
            rows, cols = np.where(tracks == track_id)
            path = out.xyz[rows, cols]
            assert np.linalg.norm(np.diff(path, axis=0), axis=1).max() < 2.0

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_survives_a_gap_only_when_bridging_is_allowed(self, rig, method):
        frames = self.moving()
        uv, complete = views_of(rig, frames)
        for view in rig.views:
            complete[view][20:23] = False
        out = reconstruct(rig, "E", uv, complete, frames.shape[0])
        bridged = METHODS[method](out, TrackConfig(max_step_mm=20.0, max_gap_frames=5))
        # The three unresolved frames stay UNTRACKED — a tracker must not invent
        # a position for a frame that had no reconstruction — but the identities
        # either side of the gap are the same three.
        assert set(bridged.ravel().tolist()) == {UNTRACKED, 0, 1, 2}
        assert (bridged[20:23] == UNTRACKED).all()

    def test_gating_refuses_an_impossible_link_and_nearest_does_not(self, rig):
        # One animal teleports across the tank. `nearest` has no way to decline
        # and will hand the track to whatever is closest; `gated` opens a new one.
        frames = self.moving(n_frames=10)
        frames[5:, 0] = np.array([180.0, 180.0, 20.0])
        uv, complete = views_of(rig, frames)
        out = reconstruct(rig, "E", uv, complete, frames.shape[0])
        assert len(set(METHODS["nearest"](out, TrackConfig(max_step_mm=5.0)).ravel().tolist())) == 3
        gated = METHODS["gated"](out, TrackConfig(max_step_mm=5.0, max_gap_frames=0))
        assert len(set(gated.ravel().tolist())) > 3

    def test_a_stale_track_does_not_outbid_a_fresh_one(self):
        # Regression. The ceiling widens with how long a track has been missing,
        # so that gap bridging works. Folding it into the *cost* — dividing
        # distance by it before solving — inverts the comparison: here the
        # 10-frame-old track at 9 mm would score 0.9 against the 1-frame-old
        # track's 1.0 and steal the slot. The identity set then churns every few
        # frames while the track *count* stays low, so neither the switch rate
        # nor the track count shows it. Found in the full run, not here, which is
        # why this test exists.
        cost = np.array([[1.0, 50.0], [9.0, 50.0]])
        ceiling = np.array([1.0, 10.0])  # ages 1 and 10, gate 1 mm per frame
        assert _match(cost, ceiling) == [(0, 0)]
        # And the ceiling still rejects: slot 1 is 50 mm from both and goes to
        # neither, rather than to whichever was staler.
        assert all(slot != 1 for _, slot in _match(cost, ceiling))

    def test_the_ceiling_still_bridges_a_real_gap(self):
        # The other half: an old track *is* allowed the distance its absence
        # bought it, so bridging is not broken by the fix above.
        cost = np.array([[9.0]])
        assert _match(cost, np.array([1.0])) == []  # age 1: 9 mm is impossible
        assert _match(cost, np.array([10.0])) == [(0, 0)]  # age 10: 9 mm is fine

    def test_a_stale_tracks_unbounded_ceiling_is_what_moves_the_female(self):
        # K_1_2 frame 82,717, the reviewed swap, with the linker's real state:
        # FOUR live tracks, not three. Rows are (612) the animal that just left
        # both side cameras, (627) the female, (632) the still fish, and (636) a
        # fragment 56 frames old whose ceiling is 4 mm x 56 = 224 mm — larger
        # than the 192 mm tank, so it can claim anything in the volume.
        cost = np.array([[12.48, 39.00, 11.00],
                         [2.52, 41.65, 2.07],
                         [40.83, 0.04, 40.63],
                         [16.15, 27.57, 15.07]])
        ceiling = np.array([8.0, 8.0, 8.0, 224.0])  # ages 2, 2, 2, 56 at a 4 mm gate

        # `gated` moves her to slot 0 and `strict` does too — refusing first is
        # not enough, because the stale track is *not* refused and taking slot 2
        # with it is cheaper for the total than leaving her there.
        assert (1, 0) in _match(cost, ceiling)
        assert (1, 0) in _match(cost, ceiling, refuse_first=True)

        # Bound the ceiling and the stale track drops out of contention. Then
        # `gated` still moves her — it assigns before it refuses, so the 11.00 mm
        # link is chosen and only afterwards struck out — and `strict` keeps her.
        bounded = np.minimum(ceiling, 8.0)
        assert (1, 0) in _match(cost, bounded)
        assert _match(cost, bounded, refuse_first=True) == [(1, 2), (2, 1)]

    def test_refusing_first_changes_nothing_when_every_link_is_reachable(self):
        cost = np.array([[0.5, 30.0], [30.0, 0.4]])
        ceiling = np.full(2, 5.0)
        assert _match(cost, ceiling) == _match(cost, ceiling, refuse_first=True)

    def test_an_animal_without_a_position_is_skipped_not_matched(self, rig):
        # Stage C can resolve two animals on a frame and not the third. The
        # linker must carry the two it has rather than inventing a link, or a
        # track for a fish that is not there.
        frames = self.moving(n_frames=10)
        uv, complete = views_of(rig, frames)
        out = reconstruct(rig, "E", uv, complete, frames.shape[0])
        xyz = out.xyz.copy()
        xyz[5, 2] = np.nan  # one animal, one frame, no 3D position
        partial = dataclasses.replace(out, xyz=xyz)
        tracks = METHODS["gated"](partial, TrackConfig(max_step_mm=20.0, max_gap_frames=5))
        assert tracks[5, 2] == UNTRACKED
        assert (tracks[5, :2] >= 0).all()
        # and the animal comes back on its own track rather than a new one
        assert tracks[6, 2] == tracks[4, 2]

    def test_propagation_lands_on_the_right_instances(self, rig):
        frames = self.moving(n_frames=8)
        orders = {"side1_cam": np.tile([2, 0, 1], (8, 1))}
        uv, complete = views_of(rig, frames, orders)
        out = reconstruct(rig, "E", uv, complete, 8)
        tracks = METHODS["gated"](out, TrackConfig(max_step_mm=20.0, max_gap_frames=5))
        starts = {view: np.arange(8) * 3 for view in rig.views}
        counts = {view: 24 for view in rig.views}
        per_view = propagate_to_views(out, tracks, starts, counts)
        # side1 reported the animals in the order [2, 0, 1]; the ID that landed
        # on its row 0 must therefore be the one on top's row 2.
        assert per_view["side1_cam"][0] == per_view["top_cam"][2]
        assert (per_view["top_cam"] >= 0).all()
