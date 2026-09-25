"""Metric safeguards: ID permutation, abstention, and geometric support."""

import numpy as np

import evaluate_resumed_nodewise_beam_female57 as evaluation


def example():
    n = 3
    sources = np.full((n, 3, 3, 3), -1, dtype=np.int32)
    sources[:, 1, 1:, 0] = 0
    pose = np.zeros((n, 3, 3, 3))
    pose[:, 1, 0] = [2., 3., 4.]
    data = {
        "frame_idx": np.arange(evaluation.START, evaluation.START + n),
        "pose_xyz": pose,
        "observed": np.ones((n, 3, 3), dtype=bool),
        "sources": sources,
        "identity_review": np.tile([True, False, True], (n, 1)),
    }
    anchors = [evaluation.MouthAnchor(int(frame), {"side1_cam": 0, "side2_cam": 0},
               {"side1_cam": [1., 3.], "side2_cam": [2., 1.]}, {})
               for frame in data["frame_idx"]]
    return data, anchors


def test_time_varying_permutation_changes_endpoint_ids_but_not_geometry():
    raw, anchors = example()
    permutation = np.array([[0, 1, 2], [0, 2, 1], [1, 2, 0]])
    beam = {"frame_idx": raw["frame_idx"]}
    for key in evaluation.FIELDS[1:]:
        beam[key] = raw[key][np.arange(3)[:, None], permutation]
    raw_score = evaluation.score_stream(raw, anchors)
    beam_score = evaluation.score_stream(beam, anchors)
    assert raw_score["interval_counts"] == {"preserved": 2, "swap": 0, "unscored": 0}
    assert beam_score["interval_counts"] == {"preserved": 0, "swap": 2, "unscored": 0}

    class Camera:
        def project(self, point):
            return point[:2]

    class Rig:
        def camera(self, view):
            assert view in ("side1_cam", "side2_cam")
            return Camera()

    raw_errors = evaluation.reprojection_rows(raw, anchors, raw_score, Rig())
    beam_errors = evaluation.reprojection_rows(beam, anchors, beam_score, Rig())
    assert raw_errors == beam_errors
    assert len(raw_errors) == 6
    assert sorted(raw_errors.values()) == [1., 1., 1., 2., 2., 2.]


def test_review_abstention_never_bridges_an_unavailable_anchor():
    data, anchors = example()
    data["identity_review"][1, 1] = True
    score = evaluation.score_stream(data, anchors, data["identity_review"])
    assert score["available_anchors"] == 2
    assert score["requested_anchors"] == 3
    assert score["interval_counts"] == {"preserved": 0, "swap": 0, "unscored": 2}


def test_observed_coverage_and_accepted_identity_have_distinct_denominators():
    data, _ = example()
    data["observed"][1, 1, 0] = False
    metrics = evaluation.coverage(data)
    assert metrics["total_fish_frames"] == 9
    assert metrics["total_node_frames"] == 27
    assert metrics["accepted_identity_frames"] == 3
    assert metrics["observed_node_frames"] == 26
    assert metrics["accepted_identity_fraction"] == 1 / 3
    assert metrics["accepted_observed_mouth_fraction"] == 2 / 9
    assert metrics["meets_identity_coverage_thresholds"] is False
