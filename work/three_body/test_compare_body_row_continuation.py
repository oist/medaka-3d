"""Boundary/support interpretation checks for the rollout comparison."""

import unittest

import compare_body_row_continuation as comparison
import numpy as np


def fixture():
    frames = np.arange(comparison.START - 1, comparison.STOP + 1)
    n = len(frames)
    return dict(
        frame_idx=frames,
        pose_xyz=np.zeros((n, 3, 3, 3)),
        observed=np.ones((n, 3, 3), bool),
        sources=np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy(),
        proposal_kind=np.full(n, "predict", dtype="U24"),
    )


class ComparisonTests(unittest.TestCase):
    def test_outside_interval_does_not_count(self):
        data = fixture()
        data["pose_xyz"][[0, -1]] = 1000
        result = comparison.summarize(data)
        self.assertEqual(result["frames"], 1660)
        self.assertEqual(result["direct_adjacent_steps_over_10mm"], [])
        self.assertEqual(result["directly_supported_adjacent_pairs_by_slot_node"], [[1659] * 3] * 3)

    def test_missing_endpoints_remove_both_step_claims(self):
        data = fixture()
        data["pose_xyz"][4, 2, 2, 0] = 11
        result = comparison.summarize(data)
        self.assertEqual(
            [e["frame_idx"] for e in result["direct_adjacent_steps_over_10mm"]],
            [comparison.START + 3, comparison.START + 4],
        )
        data["observed"][4, 2, 2] = False
        self.assertEqual(comparison.summarize(data)["direct_adjacent_steps_over_10mm"], [])

    def test_missing_source_is_not_a_different_row(self):
        data = fixture()
        data["sources"][1, 0, 0, 0] = -1
        self.assertEqual(comparison.summarize(data)["body_frames_with_any_mixed_source_rows"], 0)
        data["sources"][1, 0, 0, 2] = 4
        result = comparison.summarize(data)
        self.assertEqual(result["mixed_source_rows_by_camera"], [1, 0, 0])
        self.assertEqual(result["body_frames_with_any_mixed_source_rows"], 1)

    def test_whole_row_output_cannot_silently_change_rows(self):
        data = fixture()
        data["proposal_kind"][2] = "body_row"
        self.assertEqual(comparison.summarize(data)["final_backtrace_body_row_frames"], 1)
        data["sources"][2, 0, 2, 2] = 7
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            comparison.summarize(data)

    def test_longest_missing_run_keeps_slot_node_axes(self):
        values = np.zeros((7, 2, 2), bool)
        values[:3, 0, 0] = True
        values[5:, 0, 0] = True
        values[3:, 1, 1] = True
        np.testing.assert_array_equal(comparison.longest_true(values), [[3, 0], [0, 4]])


if __name__ == "__main__":
    unittest.main()
