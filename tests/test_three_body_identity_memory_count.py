"""Compare the support-only counter with independently enumerated full P paths."""

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "work/three_body/count_identity_memory_states.py"
spec = importlib.util.spec_from_file_location("_identity_memory_count_test", PATH)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

# Explicit order, independent of the counter's implementation.
PERMS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


def enumerate_full_paths(direct):
    layers = [set() for _ in direct]
    for path in itertools.product(range(6), repeat=len(direct) - 1):
        history = (0, *path)
        last = [(0, 0), (0, 1), (0, 2)]
        for time, permutation in enumerate(history):
            for identity in range(3):
                slot = PERMS[permutation][identity]
                if direct[time][slot]:
                    last[identity] = (time, slot)
            encoded = tuple(3 * frame + slot for frame, slot in last)
            layers[time].add((permutation, encoded))
    return layers


@pytest.mark.parametrize(
    "pattern",
    [
        [[1, 1, 1]] * 5,
        [[1, 1, 1]] + [[1, 1, 0]] * 5,
        [[1, 1, 1]] + [[0, 1, 0]] * 5,
        [[1, 1, 1]] + [[0, 0, 0]] * 3,
        [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0], [1, 1, 0]],
    ],
)
def test_exact_layer_sets_equal_independent_full_path_enumeration(pattern):
    direct = np.asarray(pattern, bool)
    frames = np.arange(100, 100 + len(pattern), dtype=np.int64)
    expected = enumerate_full_paths(direct)
    result = m.reachable_counts(frames, direct, retain_layers=True)
    assert result.completed
    assert list(result.layers) == expected
    assert result.state_counts.tolist() == [len(layer) for layer in expected]
    assert result.edge_counts.tolist() == [6 * len(layer) for layer in expected[:-1]]
    assert result.total_states == sum(map(len, expected))
    assert result.total_edges == sum(6 * len(layer) for layer in expected[:-1])


def test_current_permutations_stay_distinct_when_observation_memory_is_identical():
    result = m.reachable_counts(
        np.arange(3, dtype=np.int64),
        np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], bool),
        retain_layers=True,
    )
    assert result.state_counts.tolist() == [1, 6, 6]
    assert result.edge_counts.tolist() == [6, 36]
    assert len({events for _p, events in result.layers[-1]}) == 1


def test_same_current_permutation_with_different_last_observations_is_not_merged():
    direct = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]], bool)
    result = m.reachable_counts(np.arange(3, dtype=np.int64), direct, retain_layers=True)
    assert len([key for key in result.layers[-1] if key[0] == 0]) > 1


def test_all_direct_collapse_uses_current_events_and_still_counts_every_previous_edge():
    direct = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1], [1, 1, 1]], bool)
    result = m.reachable_counts(np.arange(4, dtype=np.int64), direct, retain_layers=True)
    assert result.state_counts[-1] == 6
    assert result.edge_counts[-1] == 6 * result.state_counts[-2]
    assert result.layers[-1] == {
        (p, tuple(9 + slot for slot in perm)) for p, perm in enumerate(PERMS)
    }


@pytest.mark.parametrize(
    "cap,reason",
    [
        (m.Caps(maximum_layer_states=5), "maximum_layer_states"),
        (m.Caps(maximum_total_states=6), "maximum_total_states"),
        (m.Caps(maximum_total_edges=5), "maximum_total_edges"),
    ],
)
def test_cap_stops_without_approximating_or_accepting_partial_layer(cap, reason):
    result = m.reachable_counts(np.array([50, 51], np.int64), np.ones((2, 3), bool), cap)
    assert not result.completed
    assert result.frame_idx.tolist() == [50]
    assert result.state_counts.tolist() == [1]
    assert result.total_states == 1 and result.total_edges == 0
    assert result.stop["reason"] == reason and result.stop["attempted_frame"] == 51
    assert result.stop["required_total_edges"] == 6
    assert result.stop["attempted_layer_states_lower_bound"] == (
        None if reason == "maximum_total_edges" else 6
    )


def test_exact_cap_boundary_is_allowed():
    caps = m.Caps(maximum_layer_states=6, maximum_total_states=7, maximum_total_edges=6)
    result = m.reachable_counts(np.array([50, 51], np.int64), np.ones((2, 3), bool), caps)
    assert result.completed and result.total_states == 7 and result.total_edges == 6


@pytest.mark.parametrize("bad", ["seed_missing", "frame_gap", "frame_dtype", "support_dtype"])
def test_invalid_native_or_initial_observation_inputs_rejected(bad):
    frames, direct = np.arange(3, dtype=np.int64), np.ones((3, 3), bool)
    if bad == "seed_missing":
        direct[0, 0] = False
    elif bad == "frame_gap":
        frames[-1] += 1
    elif bad == "frame_dtype":
        frames = frames.astype(np.float64)
    else:
        direct = direct.astype(np.int64)
    with pytest.raises(ValueError):
        m.reachable_counts(frames, direct)
