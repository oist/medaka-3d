"""Independent arithmetic on three completed first-minute output archives only."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK / "nodewise_body_row_first_minute_comparison_v1"
SEAL_SHA = "7d3a205eecf1a83b19d39418a2a7e7e286df323eb4b82854655f479251217604"
QA = WORK / "NODEWISE_FIRST_MINUTE_OUTPUT_QA_V1.json"
PEAKS = WORK / "NODEWISE_FIRST_MINUTE_GEOMETRY_PEAKS_V1.json"
START, STOP = 66400, 76360
ARMS = ("tail", "nodewise")
TOL = 1e-12


def sha(data):
    return hashlib.sha256(data).hexdigest()


def signature(array):
    return dict(dtype=array.dtype.str, shape=list(array.shape), sha256=sha(array.tobytes()))


def numeric_equal(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=TOL, equal_nan=False)


def longest_missing(observed):
    result = np.zeros((3, 3), dtype=np.int64)
    for body, node in itertools.product(range(3), repeat=2):
        flags = np.r_[False, ~observed[:, body, node], False].astype(np.int8)
        edges = np.diff(flags)
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        result[body, node] = max(lengths, default=0)
    return result


def method_metrics(data):
    pose, observed = data["pose_xyz"], data["observed"]
    delta = pose[1:] - pose[:-1]
    step = np.sqrt(np.einsum("tbnx,tbnx->tbn", delta, delta))
    both = np.logical_and(observed[:-1], observed[1:])
    maxima = [
        [float(step[:, b, n][both[:, b, n]].max()) if both[:, b, n].any() else None
         for n in range(3)] for b in range(3)
    ]
    events = [dict(previous_frame=START + int(t), frame_idx=START + int(t) + 1,
                   raw_slot=int(b), node=int(n), distance_mm=float(step[t, b, n]))
              for t, b, n in np.argwhere(both & (step > 10))]
    return dict(
        interval=[START, STOP], frames=STOP - START,
        adjacent_pairs_per_slot_node=STOP - START - 1,
        directly_observed_by_slot_node=np.count_nonzero(observed, axis=0).tolist(),
        missing_by_slot_node=np.count_nonzero(~observed, axis=0).tolist(),
        longest_missing_run_by_slot_node=longest_missing(observed).tolist(),
        all_three_chests_direct_frames=int(np.count_nonzero(np.all(observed[:, :, 1], axis=1))),
        directly_supported_adjacent_pairs_by_slot_node=np.count_nonzero(both, axis=0).tolist(),
        direct_max_step_by_slot_node_mm=maxima,
        all_estimated_max_step_by_slot_node_mm=step.max(axis=0).tolist(),
        direct_adjacent_steps_over_10mm=events,
        nonfinite_pose_count=int(np.count_nonzero(~np.isfinite(pose))),
    ), step


def direct_context(observed, index):
    result = []
    for body, node in itertools.product(range(3), repeat=2):
        column = observed[:, body, node]
        before, after = np.flatnonzero(column[:index]), np.flatnonzero(column[index + 1:])
        result.append(dict(raw_slot=body, node=node, direct_at_peak=bool(column[index]),
                           last_strictly_previous_direct_frame=(START + int(before[-1])) if len(before) else None,
                           first_strictly_subsequent_direct_frame=(START + index + 1 + int(after[0])) if len(after) else None))
    return result


def main():
    assert not QA.exists() and not PEAKS.exists(), "fresh QA artifacts required"
    raw_seal = (ROOT / "comparison.provenance.json").read_bytes()
    assert sha(raw_seal) == SEAL_SHA
    seal = json.loads(raw_seal)
    assert seal["status"] == "ok" and seal["stage"] == ROOT.name
    expected = {"science/tail.npz", "science/nodewise.npz", "science_signatures.json",
                "geometric_permutations.npz", "geometric_permutation_signatures.json",
                "plan.json", "summary.json"}
    assert set(seal["outputs"]) == expected
    actual_members = {p.relative_to(ROOT).as_posix() for p in ROOT.iterdir() if p.is_file()}
    actual_members.update("science/" + p.name for p in (ROOT / "science").iterdir() if p.is_file())
    assert actual_members == expected | {"comparison.provenance.json"}
    captured, output_checks = {}, {}
    for name, ref in seal["outputs"].items():
        path = ROOT / name
        assert path.resolve() == Path(ref["path"]).resolve()
        content = path.read_bytes()
        assert len(content) == ref["size_bytes"] and sha(content) == ref["sha256"]
        captured[name] = content
        output_checks[name] = dict(sha256=sha(content), size_bytes=len(content))
    summary, plan = (json.loads(captured[n]) for n in ("summary.json", "plan.json"))
    for key, value in plan.items():
        assert summary[key] == value, key
    assert summary["native_interval"] == [START, STOP]
    assert summary["terminal_horizon"] == STOP and summary["whole_five_minute_completion_claimed"] is False
    sources = {}
    for name, digest in {
        "compare_nodewise_body_row_first_minute.py": "705ede6ebeee394b7dfa7fc43697c62d0dd9fa23f4208173737ac4a75f170d98",
        "compare_nodewise_body_row_five_minutes.py": "512c46a8aee7b5ce7182fc86c282033ad34408509c2a32315447620c930f921e",
    }.items():
        path = WORK / name
        assert sha(path.read_bytes()) == summary["source_sha256"][str(path)] == digest
        sources[name] = digest
    declarations = json.loads(captured["science_signatures.json"])
    assert declarations == plan["science_signatures"] == summary["science_signatures"]
    data, actual_signatures = {}, {}
    for arm in ARMS:
        with np.load(io.BytesIO(captured[f"science/{arm}.npz"]), allow_pickle=False) as bank:
            assert len(bank.files) == 3 and set(bank.files) == {"frame_idx", "pose_xyz", "observed"}
            data[arm] = {name: bank[name] for name in bank.files}
        actual_signatures[arm] = {k: signature(v) for k, v in data[arm].items()}
        assert actual_signatures[arm] == declarations[arm]
        np.testing.assert_array_equal(data[arm]["frame_idx"], np.arange(START, STOP, dtype=np.int64))
        assert data[arm]["pose_xyz"].dtype == np.float64 and data[arm]["pose_xyz"].shape == (9960, 3, 3, 3)
        assert data[arm]["observed"].dtype == bool and data[arm]["observed"].shape == (9960, 3, 3)
        assert np.isfinite(data[arm]["pose_xyz"]).all()
    geo_signatures = json.loads(captured["geometric_permutation_signatures.json"])
    with np.load(io.BytesIO(captured["geometric_permutations.npz"]), allow_pickle=False) as bank:
        assert len(bank.files) == 7 and set(bank.files) == set(geo_signatures)
        saved_geo = {name: bank[name] for name in bank.files}
    assert {k: signature(v) for k, v in saved_geo.items()} == geo_signatures
    methods, steps = {}, {}
    for arm in ARMS:
        methods[arm], steps[arm] = method_metrics(data[arm])
        for key, value in methods[arm].items():
            if key.endswith("_mm"):
                numeric_equal(value, summary["methods"][arm][key])
            else:
                assert value == summary["methods"][arm][key], (arm, key)
    permutations = np.array(list(itertools.permutations(range(3))), dtype=np.int64)
    frames = data["tail"]["frame_idx"]
    np.testing.assert_array_equal(saved_geo["frame_idx"], frames)
    np.testing.assert_array_equal(saved_geo["permutations"], permutations)
    chest, all_node = np.empty((9960, 6)), np.empty((9960, 6))
    direct_counts = np.zeros((9960, 6), dtype=np.int64)
    for p_index, permutation in enumerate(permutations):
        distance_sq = np.zeros((9960, 3, 3))
        for a_body, b_body in enumerate(permutation):
            for node in range(3):
                delta = data["tail"]["pose_xyz"][:, a_body, node] - data["nodewise"]["pose_xyz"][:, b_body, node]
                distance_sq[:, a_body, node] = np.einsum("tx,tx->t", delta, delta)
                direct_counts[:, p_index] += (data["tail"]["observed"][:, a_body, node] & data["nodewise"]["observed"][:, b_body, node])
        chest[:, p_index] = np.sqrt(np.sum(distance_sq[:, :, 1], axis=1) / 3)
        all_node[:, p_index] = np.sqrt(np.sum(distance_sq.reshape(9960, 9), axis=1) / 9)
    np.testing.assert_array_equal(direct_counts, saved_geo["both_direct_node_counts_by_permutation"])
    reports, peaks, peak_indices, numeric_error = {}, {}, set(), {}
    for label, costs in (("chest", chest), ("all_node", all_node)):
        numeric_equal(costs, saved_geo[label + "_rms_mm"])
        numeric_error[label] = float(np.max(np.abs(costs - saved_geo[label + "_rms_mm"])))
        minimum = np.min(costs, axis=1)
        ties = np.abs(costs - minimum[:, None]) <= TOL
        np.testing.assert_array_equal(ties, saved_geo[label + "_optimal_permutation_mask"])
        unique = np.count_nonzero(ties, axis=1) == 1
        winner = np.argmin(costs, axis=1)
        reports[label] = dict(
            raw_identity_rms_median_mm=float(np.median(costs[:, 0])), raw_identity_rms_maximum_mm=float(np.max(costs[:, 0])),
            minimum_rms_median_mm=float(np.median(minimum)), minimum_rms_maximum_mm=float(np.max(minimum)),
            unique_nonidentity_frames=int(np.count_nonzero(unique & (winner != 0))), tied_frames=int(np.count_nonzero(~unique)),
            unique_winner_counts=[int(np.count_nonzero(unique & (winner == p))) for p in range(6)],
            both_direct_node_pairs_on_unique_frames=int(np.sum(direct_counts[np.flatnonzero(unique), winner[unique]])),
        )
        for key, value in reports[label].items():
            if key.endswith("_mm"):
                numeric_equal(value, summary["geometric_comparison"][label][key])
            else:
                assert value == summary["geometric_comparison"][label][key], (label, key)
        index = int(np.argmax(minimum))
        peak_indices.add(index)
        order = np.argsort(costs[index], kind="stable")
        peaks[label] = dict(frame_idx=int(frames[index]), minimum_rms_mm=float(minimum[index]),
                            all_six_rms_mm=costs[index].tolist(), best_permutation=permutations[order[0]].tolist(),
                            second_permutation=permutations[order[1]].tolist(), second_minus_best_mm=float(costs[index, order[1]] - costs[index, order[0]]))
    selected = sorted({j for index in peak_indices for j in range(max(0, index - 2), min(9960, index + 3))})
    peak_details = {}
    for index in sorted(peak_indices):
        delta = data["tail"]["pose_xyz"][index] - data["nodewise"]["pose_xyz"][index]
        peak_details[str(int(frames[index]))] = dict(
            raw_identity_distance_by_slot_node_mm=np.sqrt(np.einsum("bnx,bnx->bn", delta, delta)).tolist(),
            both_direct_by_slot_node=(data["tail"]["observed"][index] & data["nodewise"]["observed"][index]).tolist(),
            direct_context={arm: direct_context(data[arm]["observed"], index) for arm in ARMS},
            adjacent_steps={arm: [dict(previous_frame=START + t, frame_idx=START + t + 1,
                                      distance_by_slot_node_mm=steps[arm][t].tolist(),
                                      both_direct=(data[arm]["observed"][t] & data[arm]["observed"][t + 1]).tolist())
                                 for t in (index - 1, index) if 0 <= t < 9959] for arm in ARMS},
        )
    peak_artifact = dict(schema="nodewise_first_minute_geometry_peaks_v1", status="ok", comparison_seal_sha256=SEAL_SHA,
                         selection="Global maxima of minimum all-six-permutation RMS; development cases, not held-out or physical-ID truth.",
                         node_order=["mouth", "chest", "tail"], peaks=peaks, details=peak_details,
                         selected_frames=frames[selected].tolist(),
                         samples={arm: {k: v[selected].tolist() for k, v in data[arm].items()} for arm in ARMS},
                         all_science_geometry_finite=True, original_candidate_checkpoint_SL_mask_image_GPU_reads=0)
    peak_bytes = (json.dumps(peak_artifact, indent=2, allow_nan=False) + "\n").encode()
    report = dict(schema="nodewise_first_minute_output_qa_v1", status="ok", comparison_seal_sha256=SEAL_SHA,
                  qa_source=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__).read_bytes())),
                  source_pins_verified=sources, outputs_verified=output_checks, output_count=len(output_checks),
                  archive_decodes={"science/tail.npz": 1, "science/nodewise.npz": 1, "geometric_permutations.npz": 1},
                  science_signatures_verified=actual_signatures, geometric_signatures_verified=geo_signatures,
                  independent_methods=methods, independent_geometric_comparison=reports,
                  geometric_max_absolute_difference_mm=numeric_error, absolute_numeric_tolerance_mm=TOL,
                  peak_artifact=dict(path=str(PEAKS), sha256=sha(peak_bytes)), peak_summary=peaks,
                  scope=dict(native_interval=[START, STOP], adjacent_pairs_per_slot_node=9959,
                             all_six_whole_body_permutations_per_frame=True, saved_support_recounted=True,
                             support_from_original_SL_recalculated=False, source_row_mixing_and_proposal_kind_reproved=False,
                             pre_science_prefix_difference_reproved=False, review_flags_reproved=False,
                             lineage_and_unread_fields_attributed_to_sealed_comparison=True,
                             framewise_correspondence_is_not_temporal_or_physical_ID=True,
                             original_candidate_checkpoint_SL_mask_image_GPU_reads=0))
    assert sha((ROOT / "comparison.provenance.json").read_bytes()) == SEAL_SHA
    with PEAKS.open("xb") as stream:
        stream.write(peak_bytes)
    with QA.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="ok", output_count=len(output_checks), peaks=peaks,
                          details=peak_details, qa_sha256=sha(QA.read_bytes()), peaks_sha256=sha(peak_bytes)), indent=2))


if __name__ == "__main__":
    main()
