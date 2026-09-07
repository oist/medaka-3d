"""Behavioural tests for the 3-node annotation seed builder (`tools/s01_pectoral_seed`).

Two properties are worth pinning down, because both would fail silently rather
than loudly:

1. **The sampler is deterministic and temporally spread.** It is what decides
   which frames a human spends hours on. If it drifted, a re-run would produce a
   different set and the two batches could not be pooled — and nothing would say
   so. The min-gap rule exists because labelled frames in this corpus arrive 4
   frames (24 ms) apart, so an unconstrained sample buys near-duplicate images.

2. **Interpolation is by node name and respects partial instances.** The new
   node sits at index 1, exactly where `tail` used to be. Anything reading by
   position gets a plausible-looking point on the fish rather than an error,
   which is the failure mode that survives review.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load():
    module_path = TOOLS / "s01_pectoral_seed.py"
    if not module_path.is_file():
        pytest.skip("tools/s01_pectoral_seed.py has been retired")
    name = "_s01_pectoral_seed"
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        # Register before executing: `@dataclass` resolves the defining module
        # through `sys.modules`, and fails on a module that is not there yet.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


s01 = _load()
sio = pytest.importorskip("sleap_io")


@pytest.fixture(scope="module")
def skeleton():
    return sio.Skeleton(
        nodes=list(s01.NODE_ORDER),
        edges=[list(e) for e in s01.EDGES],
        name=s01.SKELETON_NAME,
    )


def _instance(mouth, tail, *, mouth_visible=True, tail_visible=True):
    """A 2-node source instance, as the round-8 packages hold them."""
    old = sio.Skeleton(nodes=["mouth", "tail"], edges=[["mouth", "tail"]])
    points = np.array(
        [mouth if mouth is not None else [np.nan, np.nan],
         tail if tail is not None else [np.nan, np.nan]],
        dtype=float,
    )
    instance = sio.Instance.from_numpy(points, skeleton=old)
    instance.points["visible"][0] = mouth_visible and mouth is not None
    instance.points["visible"][1] = tail_visible and tail is not None
    return instance


# --------------------------------------------------------------------------
# Skeleton and interpolation
# --------------------------------------------------------------------------

def test_skeleton_is_a_chain_through_the_new_node():
    assert s01.NODE_ORDER == ("mouth", "pectoral", "tail")
    assert s01.EDGES == (("mouth", "pectoral"), ("pectoral", "tail"))


def test_pectoral_lands_at_t_along_the_axis(skeleton):
    converted, kind = s01.convert(_instance([0.0, 0.0], [100.0, 200.0]), skeleton, 0.30)
    assert kind == "interpolated"
    xy, visible = s01.node_state(converted, "pectoral")
    assert xy == pytest.approx([30.0, 60.0])
    assert visible


def test_mouth_and_tail_survive_by_name_not_position(skeleton):
    """The regression the node order creates: index 1 is no longer the tail."""
    converted, _ = s01.convert(_instance([1.0, 2.0], [3.0, 4.0]), skeleton, 0.5)
    assert s01.node_state(converted, "mouth")[0] == pytest.approx([1.0, 2.0])
    assert s01.node_state(converted, "tail")[0] == pytest.approx([3.0, 4.0])
    # Positional index 1 is the pectoral, and is *not* the tail.
    assert converted.points["name"][1] == "pectoral"
    assert s01.node_state(converted, "tail")[0] != pytest.approx(
        s01.node_state(converted, "pectoral")[0]
    )


@pytest.mark.parametrize(
    ("mouth", "tail", "expected"),
    [
        (None, [1.0, 1.0], "unseeded_no_mouth"),
        ([1.0, 1.0], None, "unseeded_no_tail"),
        (None, None, "unseeded_empty"),
    ],
)
def test_partial_instances_leave_the_pectoral_unplaced(skeleton, mouth, tail, expected):
    converted, kind = s01.convert(_instance(mouth, tail), skeleton, 0.30)
    assert kind == expected
    xy, visible = s01.node_state(converted, "pectoral")
    assert np.isnan(xy).all()
    assert not visible


def test_hidden_but_positioned_parents_still_interpolate(skeleton):
    """The experimenter's rule for partial instances.

    A point flagged not-visible but carrying coordinates must still drive the
    interpolation; the seeded node inherits the hidden flag rather than
    claiming a visibility its parents do not have. No such point exists in the
    round-8 packages — every point there is either finite-and-visible or
    NaN-and-invisible — so this is the only place the behaviour is pinned.
    """
    instance = _instance([0.0, 0.0], [10.0, 10.0], tail_visible=False)
    converted, kind = s01.convert(instance, skeleton, 0.30)
    assert kind == "interpolated_from_hidden"
    xy, visible = s01.node_state(converted, "pectoral")
    assert xy == pytest.approx([3.0, 3.0])
    assert not visible


def test_numpy_would_have_hidden_the_parent_coordinates(skeleton):
    """Why `node_state` reads `points`, not `numpy()`."""
    instance = _instance([0.0, 0.0], [10.0, 10.0], tail_visible=False)
    assert np.isnan(instance.numpy()[1]).all()          # masked away
    assert np.isfinite(instance.points["xy"][1]).all()  # still there


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _moments(experiment: str, frame_indices) -> dict:
    return {(experiment, int(i)): {v: None for v in s01.VIEWS} for i in frame_indices}


def test_selection_is_deterministic():
    moments = _moments("20251225_K_1_2_FCMC", range(0, 100_000, 211))
    first, _ = s01.select(moments, fraction=0.1, min_gap=166, tag=s01.HASH_TAG)
    second, _ = s01.select(
        dict(reversed(list(moments.items()))), fraction=0.1, min_gap=166, tag=s01.HASH_TAG
    )
    assert first == second, "selection must not depend on input ordering"


def test_selection_respects_the_minimum_gap():
    # Labelled frames as they actually occur: runs spaced 4 frames apart.
    runs = [start + offset for start in range(0, 200_000, 5_000) for offset in range(0, 20, 4)]
    moments = _moments("20251225_K_1_2_FCMC", runs)
    chosen, _ = s01.select(moments, fraction=0.5, min_gap=166, tag=s01.HASH_TAG)
    indices = sorted(i for _, i in chosen)
    gaps = np.diff(indices)
    assert gaps.size and gaps.min() >= 166, f"min gap violated: {gaps.min()}"


def test_incomplete_moments_are_excluded():
    moments = {
        ("E", 1): {"top_cam": None, "side1_cam": None, "side2_cam": None},
        ("E", 5_000): {"top_cam": None, "side1_cam": None},  # a view is missing
    }
    chosen, meta = s01.select(moments, fraction=1.0, min_gap=1, tag=s01.HASH_TAG)
    assert meta["complete_moments"] == 1
    assert meta["incomplete_moments"] == 1
    assert chosen == [("E", 1)]


def test_each_experiment_is_sampled_in_proportion():
    moments = {}
    moments.update(_moments("20251225_K_1_1_FCMC", range(0, 400_000, 4_000)))   # 100
    moments.update(_moments("20251225_K_1_2_FCMC", range(0, 800_000, 1_000)))   # 800
    _, meta = s01.select(moments, fraction=0.10, min_gap=166, tag=s01.HASH_TAG)
    assert meta["per_experiment"]["20251225_K_1_1_FCMC"]["selected"] == 10
    assert meta["per_experiment"]["20251225_K_1_2_FCMC"]["selected"] == 80


def test_a_short_experiment_is_reported_rather_than_padded():
    """min-gap can make the target unreachable. That must surface, not pass."""
    moments = _moments("20251225_K_1_1_FCMC", range(0, 100, 4))  # 25 within 100 frames
    _, meta = s01.select(moments, fraction=1.0, min_gap=166, tag=s01.HASH_TAG)
    report = meta["per_experiment"]["20251225_K_1_1_FCMC"]
    assert report["selected"] == 1
    assert report["short_by"] == 24


def test_stable_rank_is_stable():
    assert s01.stable_rank("E", 42, "t") == s01.stable_rank("E", 42, "t")
    assert s01.stable_rank("E", 42, "t") != s01.stable_rank("E", 43, "t")
    assert s01.stable_rank("E", 42, "t") != s01.stable_rank("E", 42, "other")


# --------------------------------------------------------------------------
# Source path parsing
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("path", "experiment", "view"),
    [
        ("//bucket/StephensU/medaka/medaka_202512/20251225_K_1_2_FCMC/top_cam/cam0_top.mp4",
         "20251225_K_1_2_FCMC", "top_cam"),
        ("Z:/StephensU/medaka/medaka_202512/20251225_K_1_1_FCMC/side2_cam/cam2_side.mp4",
         "20251225_K_1_1_FCMC", "side2_cam"),
        ("/somewhere/unrelated.mp4", None, None),
    ],
)
def test_source_paths_resolve(path, experiment, view):
    assert s01.experiment_of(path) == experiment
    assert s01.view_of(path) == view
