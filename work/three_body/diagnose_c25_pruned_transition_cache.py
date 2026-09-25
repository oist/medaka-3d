"""Synthetic four-frame regression of C25's equal-size pruned-state cache key.

Uses the real 3,456-state table and unchanged default pruning threshold. Only
state_costs is replaced with explicitly synthetic unaries. No rays, recordings,
registration payloads, checkpoints or GPU calculations are consumed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
sys.path.insert(0, str(REPO / "tools"))

import c25_variable_cardinality as c25  # noqa: E402

C25_SHA = "1b7662bd5c1235165f0aa9665ba899ee45c8a1b9acbfb27d629f5b63b5dfe7ad"
OUTPUT = WORK / "c25_pruned_transition_cache_synthetic_regression_v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment():
    frames, base, lam = np.arange(4), 6, 1.0
    counts = (4, 4, 3)
    table = c25.state_table(counts)
    assert len(table) == 3456 > c25.PRUNE_ABOVE == 2000
    labels = {}
    k = {}
    for view, count in zip(c25.VIEWS, counts, strict=True):
        labels[view] = np.full((4, c25.K_MAX), -1, np.int32)
        labels[view][:, :count] = np.arange(count)
        k[view] = np.full(4, count, np.int8)
    packed = c25.slot_identities(table, labels, np.array([0]), base)[0]

    def orbit(side1):
        target = np.stack(([0, 1, 2], side1, [0, 1, 2]), axis=1)
        index = np.flatnonzero(np.all(table == target, axis=(1, 2)))
        assert index.size == 1
        result = np.all(np.sort(packed, axis=1) == np.sort(packed[index[0]]), axis=1)
        assert result.sum() == 6
        return result

    baseline, swap, cycle = orbit([0, 1, 2]), orbit([0, 2, 1]), orbit([1, 2, 0])
    costs = np.full((4, len(table)), 20.0)
    costs[:2, baseline], costs[:2, swap] = 0.0, 6.0
    costs[2, baseline], costs[2, cycle] = 0.0, 6.0
    costs[3, baseline], costs[3, cycle] = 2.5, 0.0
    _, _, kept = c25.prune(costs, np.broadcast_to(packed, (4, *packed.shape)), lam)
    assert [len(x) for x in kept] == [12] * 4
    assert np.array_equal(kept[0], kept[1]) and np.array_equal(kept[2], kept[3])
    assert not np.array_equal(kept[0], kept[2])
    old_matrix = c25.slots_moved(packed[kept[0]], packed[kept[0]])
    correct_matrix = c25.slots_moved(packed[kept[2]], packed[kept[2]])
    assert not np.array_equal(old_matrix, correct_matrix)

    def synthetic_costs(rays, nodes, actual_table, requested, mu):
        assert rays is None and nodes == ("pectoral", "mouth") and mu == 4.0
        assert np.array_equal(actual_table, table)
        return costs[requested].copy()

    original_step = c25.viterbi_step

    def fresh_step(value, previous, here, local_cost, penalty, moved=None):
        return original_step(value, previous, here, local_cost, penalty, moved=None)

    args = (None, ("pectoral", "mouth"), labels, k, frames, lam, 4.0, base)
    with patch.object(c25, "state_costs", synthetic_costs):
        old = c25.solve(*args)
        with patch.object(c25, "viterbi_step", fresh_step):
            fresh = c25.solve(*args)

    def ledger(chosen):
        unary = costs[np.arange(4), chosen]
        state = packed[chosen]
        changes = np.sum(state[1:] != state[:-1], axis=1)
        return dict(
            unary=unary.tolist(),
            slots_changed=changes.tolist(),
            total=float(unary.sum() + lam * changes.sum()),
        )

    old_ledger, fresh_ledger = ledger(old[0]), ledger(fresh[0])
    assert old[1] == 2.0 and old_ledger["total"] == 3.0
    assert fresh[1] == fresh_ledger["total"] == 2.5
    assert old_ledger["total"] > fresh_ledger["total"]
    result = dict(
        schema="c25_pruned_cache_synthetic_regression_v1",
        status="ok",
        all_inputs_synthetic=True,
        synthetic_local_frames=frames.tolist(),
        counts_per_view=list(counts),
        states_before_prune=len(table),
        original_prune_above=c25.PRUNE_ABOVE,
        original_transition_limit=c25.TRANSITION_LIMIT,
        retained_states=[len(x) for x in kept],
        lambda_mm=lam,
        cached_transition_entries_wrong=int(np.count_nonzero(old_matrix != correct_matrix)),
        old_reported_objective=old[1],
        old_chosen_state=old[0].tolist(),
        old_path_independent_ledger=old_ledger,
        fresh_reported_objective=fresh[1],
        fresh_chosen_state=fresh[0].tolist(),
        fresh_path_independent_ledger=fresh_ledger,
        old_path_excess_cost=old_ledger["total"] - fresh_ledger["total"],
        frozen_solve_calls_with_synthetic_unaries=2,
        fresh_reference_change="discard supplied cached moved; use real slots_moved at every edge",
        source_files_modified=False,
        real_data_or_rays_read=False,
        GPU_used=False,
        real_recording_affected_frames_established=False,
        conclusion="equal-sized different pruned subsets can produce a wrong objective and suboptimal path",
    )
    arrays = dict(
        table=table,
        packed=packed,
        synthetic_costs=costs,
        kept_A=kept[0],
        kept_B=kept[2],
        cached_moved_A=old_matrix,
        correct_moved_B=correct_matrix,
        old_chosen=old[0],
        fresh_chosen=fresh[0],
    )
    return result, arrays


def main():
    source = Path(c25.__file__).resolve()
    assert sha(source) == C25_SHA
    assert not OUTPUT.exists(), "fresh synthetic result required"
    assert "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized()
    result, arrays = experiment()
    assert sha(source) == C25_SHA
    assert "torch" not in sys.modules or not sys.modules["torch"].cuda.is_initialized()
    OUTPUT.mkdir()
    np.savez_compressed(OUTPUT / "synthetic_case.npz", **arrays)
    result["input_source"] = dict(path=str(source), sha256=C25_SHA)
    result["diagnostic_source"] = dict(
        path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))
    )
    result["synthetic_case"] = dict(
        path=str(OUTPUT / "synthetic_case.npz"), sha256=sha(OUTPUT / "synthetic_case.npz")
    )
    (OUTPUT / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
