"""Actual C25 loops with synthetic costs only; no recorded arrays or rays."""

import classical_viterbi_checked_cache as checked
import numpy as np
import pytest

original = checked.original


@pytest.fixture(autouse=True)
def no_recorded_inputs(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("recorded data/rays must never be read")

    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(original.c01, "load", forbidden)
    monkeypatch.setattr(original, "compact_rays", forbidden)


def case(counts, n=4):
    frames, base = np.arange(n), 6
    table = original.state_table(counts)
    labels, k = {}, {}
    for view, count in zip(original.VIEWS, counts, strict=True):
        labels[view] = np.full((n, original.K_MAX), -1, np.int32)
        labels[view][:, :count] = np.arange(count)
        k[view] = np.full(n, count, np.int8)
    packed = original.slot_identities(table, labels, np.array([0]), base)[0]

    def orbit(side1):
        target = np.stack(([0, 1, 2], side1, [0, 1, 2]), axis=1)
        row = np.flatnonzero(np.all(table == target, axis=(1, 2)))
        assert row.size == 1
        mask = np.all(np.sort(packed, axis=1) == np.sort(packed[row[0]]), axis=1)
        assert mask.sum() == 6
        return mask

    baseline, swap, cycle = orbit([0, 1, 2]), orbit([0, 2, 1]), orbit([1, 2, 0])
    costs = np.full((n, len(table)), 20.0)
    costs[:2, baseline], costs[:2, swap] = 0.0, 6.0
    costs[2:, baseline], costs[2:, cycle] = 2.5, 0.0
    costs[2, baseline], costs[2, cycle] = 0.0, 6.0
    return table, packed, costs, labels, k, frames, base


def install_costs(monkeypatch, table, costs):
    def synthetic(rays, nodes, actual_table, frames, mu):
        assert rays is None and nodes == ("pectoral", "mouth") and mu == 4.0
        assert np.array_equal(actual_table, table)
        return costs[frames].copy()

    monkeypatch.setattr(original, "state_costs", synthetic)
    monkeypatch.setattr(checked, "state_costs", synthetic)


def ledger(chosen, costs, packed, lam):
    return float(
        costs[np.arange(len(chosen)), chosen].sum()
        + lam * (packed[chosen][1:] != packed[chosen][:-1]).sum()
    )


def test_equal_size_pruned_lists_fix_old_wrong_objective_and_suboptimal_path(monkeypatch):
    table, packed, costs, labels, k, frames, base = case((4, 4, 3))
    install_costs(monkeypatch, table, costs)
    assert len(table) == 3456 > checked.PRUNE_ABOVE == original.PRUNE_ABOVE == 2000
    args = (None, ("pectoral", "mouth"), labels, k, frames, 1.0, 4.0, base)
    old = original.solve(*args)
    diagnostics = {}
    actual = checked.solve(*args, diagnostics=diagnostics)
    assert old[1] == 2.0 and ledger(old[0], costs, packed, 1.0) == 3.0
    assert actual[1] == ledger(actual[0], costs, packed, 1.0) == 2.5
    assert not np.array_equal(old[0], actual[0])
    assert old[5] == actual[5] == dict.fromkeys(range(4), (3456, 12))
    assert np.array_equal(old[2], actual[2]) and np.array_equal(old[4], actual[4])
    assert all(np.array_equal(a, b) for a, b in zip(old[3], actual[3], strict=True))
    assert diagnostics == dict(cache_replacements=1, numerically_different_cache_replacements=1)


@pytest.mark.parametrize("lam", [0.0, 0.767, 2.0])
def test_unpruned_original_outputs_all_agree(monkeypatch, lam):
    table, packed, costs, labels, k, frames, base = case((3, 3, 3))
    install_costs(monkeypatch, table, costs)
    assert len(table) == 216 < original.PRUNE_ABOVE
    args = (None, ("pectoral", "mouth"), labels, k, frames, lam, 4.0, base)
    old, new = original.solve(*args), checked.solve(*args)
    assert old[1] == new[1] == ledger(new[0], costs, packed, lam)
    for index in (0, 2, 4):
        assert np.array_equal(old[index], new[index])
    assert all(np.array_equal(a, b) for a, b in zip(old[3], new[3], strict=True))
    assert old[5] == new[5] == {}


def test_repeat_same_pruned_list_reuses_checked_matrix(monkeypatch):
    table, _, costs, labels, k, frames, base = case((4, 4, 3), n=5)
    install_costs(monkeypatch, table, costs)
    calls = []
    original_moved = checked.slots_moved

    def counted(before, here):
        calls.append(here.copy())
        return original_moved(before, here)

    monkeypatch.setattr(checked, "slots_moved", counted)
    checked.solve(None, ("pectoral", "mouth"), labels, k, frames, 1.0, 4.0, base)
    assert len(calls) == 2  # A/A builds A, B/B replaces it, following B/B reuses it.
    assert calls[0].shape == calls[1].shape == (12, 3)
    assert not np.array_equal(calls[0], calls[1])


def test_original_source_pin_checked_before_any_geometry(monkeypatch):
    monkeypatch.setattr(checked, "ORIGINAL_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen original"):
        checked.solve(None, None, None, None, None, 1.0, 4.0, 6)


def test_diagnostics_cannot_overwrite_existing_caller_evidence():
    diagnostics = dict(existing="keep")
    with pytest.raises(ValueError, match="empty output dict"):
        checked.solve(None, None, None, None, None, 1.0, 4.0, 6, diagnostics=diagnostics)
    assert diagnostics == dict(existing="keep")
