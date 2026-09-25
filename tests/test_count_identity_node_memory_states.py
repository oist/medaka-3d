"""Independent tiny full-path oracle and K1 frozen-counter compatibility."""

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work/three_body"))
import count_identity_memory_states as old  # noqa: E402
import count_identity_node_memory_states as mod  # noqa: E402


def exhaustive_layers(direct):
    """Enumerate complete permutation histories, then search backward per cell."""
    permutations = list(itertools.permutations(range(3)))
    layers = []
    for t in range(len(direct)):
        layer = set()
        for suffix in itertools.product(range(6), repeat=t):
            path = (0, *suffix)
            events = []
            for identity, node in itertools.product(range(3), range(direct.shape[2])):
                latest = next(
                    j for j in range(t, -1, -1) if direct[j, permutations[path[j]][identity], node]
                )
                events.append(3 * latest + permutations[path[latest]][identity])
            layer.add((path[-1], tuple(events)))
        layers.append(frozenset(layer))
    return tuple(layers)


@pytest.mark.parametrize("seed", range(6))
def test_two_node_counts_against_full_permutation_histories(seed):
    direct = np.random.default_rng(seed).random((5, 3, 2)) > 0.5
    direct[0] = True
    before = direct.tobytes()
    frames = np.arange(90, 95, dtype=np.int64)
    actual = mod.reachable_counts(frames, direct, retain_layers=True)
    expected = exhaustive_layers(direct)
    assert actual.layers == expected
    np.testing.assert_array_equal(actual.state_counts, [len(layer) for layer in expected])
    np.testing.assert_array_equal(actual.edge_counts, [6 * len(layer) for layer in expected[:-1]])
    assert actual.total_states == sum(map(len, expected)) and direct.tobytes() == before


@pytest.mark.parametrize(
    "caps",
    [old.Caps(), old.Caps(15, 1000, 10000), old.Caps(1000, 30, 10000), old.Caps(1000, 1000, 20)],
)
def test_K1_exact_frozen_counter_including_cap_stop(caps):
    direct = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    frames = np.arange(5, dtype=np.int64)
    before = old.reachable_counts(frames, direct, caps, retain_layers=True)
    after = mod.reachable_counts(frames, direct[..., None], caps, retain_layers=True)
    for key in ("completed", "total_states", "total_edges", "stop", "layers"):
        assert getattr(before, key) == getattr(after, key)
    for key in ("frame_idx", "state_counts", "edge_counts"):
        a, b = getattr(before, key), getattr(after, key)
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_shared_permutation_and_ID_major_memory():
    direct = np.ones((3, 3, 2), bool)
    result = mod.reachable_counts(np.arange(3, dtype=np.int64), direct, retain_layers=True)
    np.testing.assert_array_equal(result.state_counts, [1, 6, 6])
    for p, events in result.layers[1]:
        slots = old.PERMUTATIONS[p]
        assert events == tuple(3 + slot for slot in slots for _ in range(2))
    direct[1:, :, 1] = False
    held = mod.reachable_counts(np.arange(3, dtype=np.int64), direct, retain_layers=True)
    assert held.layers == exhaustive_layers(direct)
    for _, events in held.layers[2]:
        assert events[1::2] == (0, 1, 2)


@pytest.mark.parametrize(
    "fault", ["K3", "float_support", "initial_mouth", "float_frame", "gap", "retention"]
)
def test_bad_shapes_initialization_and_retention(fault):
    frames = np.arange(3, dtype=np.int64)
    direct = np.ones((3, 3, 2), bool)
    if fault == "K3":
        direct = np.ones((3, 3, 3), bool)
    elif fault == "float_support":
        direct = direct.astype(float)
    elif fault == "initial_mouth":
        direct[0, 1, 1] = False
    elif fault == "float_frame":
        frames = frames.astype(float)
    elif fault == "gap":
        frames[-1] += 1
    else:
        frames = np.arange(9, dtype=np.int64)
        direct = np.ones((9, 3, 2), bool)
    with pytest.raises(ValueError):
        mod.reachable_counts(frames, direct, retain_layers=True)
