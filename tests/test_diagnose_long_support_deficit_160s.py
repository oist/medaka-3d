"""Anonymous support-set diagnostics, no archived input or identity truth."""

import importlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
m = importlib.import_module("diagnose_long_support_deficit_160s")


def fixture():
    methods = {}
    for name in m.comparison.METHODS:
        methods[name] = dict(
            frame_idx=np.arange(10, 16),
            pose_xyz=np.zeros((6, 3, 3, 3)),
            sources=np.broadcast_to(np.arange(3)[None, :, None, None], (6, 3, 3, 3)).copy(),
            observed=np.ones((6, 3, 3), bool),
            original_identity_review=np.zeros((6, 3), bool),
        )
    details = dict(
        proposal_kind=np.full(6, "predict"),
        body_supported=np.ones((6, 3), bool),
        mask_supported=np.ones((6, 3), bool),
    )
    return methods, details


def test_fixed_every_baseline_condition_not_majority_or_slot_agreement():
    methods, details = fixture()
    methods["new"]["observed"][1:4, 0, 1] = False
    methods["new"]["sources"][1:4, 0, :, 1] = -1
    methods[m.comparison.METHODS[-1]]["observed"][2, 2, 1] = False
    report, selected, _ = m.analyze(methods, details)
    assert selected["qualifying_frame"].tolist() == [False, True, False, True, False, False]
    assert report["qualifying_missing_chest_fish_frames"] == 2
    assert [x["frame_range"] for x in report["all_episodes"]] == [[11, 12], [13, 14]]


def test_no_selected_frames_is_valid_empty_result():
    methods, details = fixture()
    report, selected, matching = m.analyze(methods, details)
    assert report["no_qualifying_frames"] and not selected["qualifying_frame"].any()
    assert report["all_episodes"] == report["largest_five_episodes"] == []
    assert matching["frame_idx"].shape == (0,)


def test_episode_splits_native_gaps_and_missing_selection():
    frames = np.array([10, 11, 13, 14, 15, 16])
    spans = m.episodes(frames, np.array([1, 1, 1, 0, 1, 1], bool))
    assert [x["frame_range"] for x in spans] == [[10, 12], [13, 14], [15, 17]]


def test_source_matching_no_elimination_ambiguity_and_slot_reorder():
    pose = np.zeros((1, 3, 3, 3))
    pose[0, :, :, 0] = np.array([0, 10, 20])[:, None]
    original = np.broadcast_to(np.arange(3)[None, :, None, None], (1, 3, 3, 3)).copy()
    query = original.copy()
    query[0, 0, :, 1] = -1  # no source: never infer remaining fish by elimination
    query[0, 1, :, 1] = [1, 2, -1]  # contradictory within-frame overlap
    perm = [2, 0, 1]
    overlap, distance, source, nearest = m.within_frame_comparison(
        pose, query, pose[:, perm], original[:, perm]
    )
    assert source.tolist() == [[-1, -1, 0]]
    assert nearest.tolist() == [[1, 2, 0]]
    assert overlap[0, 1].tolist() == [1, 0, 1]
    assert distance[0, 0].tolist() == [20, 0, 10]


def test_spatial_ties_not_arbitrary_identity_assignment():
    pose = np.zeros((1, 3, 3, 3))
    sources = np.full(pose.shape, -1, np.int64)
    _, _, source, nearest = m.within_frame_comparison(pose, sources, pose, sources)
    assert np.all(source == -1) and np.all(nearest == -1)
