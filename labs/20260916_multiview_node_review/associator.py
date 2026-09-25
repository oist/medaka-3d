"""Memoryless diagnostic wrapper around an explicitly supplied Associator.

The baseline's instances are returned by identity, without any refinement.
Records contain only this frame's geometric evidence. The separate review
runner owns candidate searches that require frozen C34 slots or motion flanks;
those inputs do not belong to the Associator protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from geometry import geometry_metrics

from medaka3d.stages import Associator
from medaka3d.types import VIEWS, Detection2D, Instance3D

NODES = ("mouth", "pectoral", "tail")


class MultiviewNodeDiagnosticAssociator:
    """Record geometry for three baseline instances without changing their output.

    ``node_order`` declares the order in each input detection's keypoints;
    Detection2D itself does not carry node names. Source indices refer directly
    to each supplied view's detection sequence. An absent source or keypoint
    remains NaN. Unsupported counts and malformed source data decline only the
    diagnostic, preserving the baseline result.

    Prior diagnostic records are never used by ``associate``. Recording them
    is an audit trail, not a temporal model or a biological identity claim.
    """

    def __init__(self, baseline: Associator, rig, *, node_order=NODES):
        if not isinstance(baseline, Associator):
            raise TypeError("An explicit baseline Associator is required")
        node_order = tuple(node_order)
        if len(node_order) != len(NODES) or set(node_order) != set(NODES):
            raise ValueError("node_order must name mouth, pectoral and tail exactly once")
        if tuple(rig.views) != VIEWS or getattr(rig, "world_units", "mm") != "mm":
            raise ValueError("Rig must use the three canonical views and millimetres")
        self.baseline = baseline
        self.rig = rig
        self.node_order = node_order
        self._node_indices = [node_order.index(node) for node in NODES]
        self.name = f"{baseline.name}+multiview-node-diagnostic"
        self.diagnostics: list[dict] = []

    def associate(
        self, detections: dict[str, Sequence[Detection2D]], frame: int
    ) -> list[Instance3D]:
        instances = self.baseline.associate(detections, frame)
        record = {
            "frame": int(frame),
            "baseline": self.baseline.name,
            "instance_count": len(instances),
            "slots": [instance.slot for instance in instances],
            "status": "declined_diagnostic",
            "baseline_instances_unchanged": True,
            "accepted_corrections": 0,
            "biological_identity_claim": False,
            "evidence_scope": "single_frame_geometric_consistency_only",
            "node_order": NODES,
            "view_order": VIEWS,
        }
        if len(instances) != 3:
            record["reason"] = "requires_exactly_three_baseline_instances"
            self.diagnostics.append(record)
            return instances
        if len(set(record["slots"])) != 3 or any(i.frame != frame for i in instances):
            record["reason"] = "baseline_instances_have_duplicate_slots_or_wrong_frame"
            self.diagnostics.append(record)
            return instances

        uv = np.full((3, 3, 3, 2), np.nan)
        source_rows = np.full((3, 3), -1, dtype=np.int64)
        problem = None
        for slot_index, instance in enumerate(instances):
            if set(instance.sources) - set(VIEWS):
                problem = "source_contains_unknown_view"
                break
            for view_index, view in enumerate(VIEWS):
                row = instance.sources.get(view, -1)
                if not isinstance(row, (int, np.integer)) or isinstance(row, (bool, np.bool_)):
                    problem = "source_row_is_not_an_integer"
                    break
                if row == -1:
                    continue
                view_detections = detections.get(view, ())
                if row < 0 or row >= len(view_detections):
                    problem = "source_row_out_of_range"
                    break
                detection = view_detections[row]
                if detection.view != view or detection.frame != frame:
                    problem = "source_detection_view_or_frame_mismatch"
                    break
                source_rows[slot_index, view_index] = row
                if detection.keypoints is None:
                    continue
                points = np.asarray(detection.keypoints, dtype=float)
                if points.shape != (3, 2):
                    problem = "source_keypoints_shape_is_not_three_nodes_xy"
                    break
                uv[slot_index, view_index] = points[self._node_indices]
            if problem is not None:
                break
        if problem is None:
            for rows in source_rows.T:
                present = rows[rows >= 0]
                if len(present) != len(np.unique(present)):
                    problem = "baseline_reuses_detection_row_within_view"
                    break
        if problem is not None:
            record["reason"] = problem
            self.diagnostics.append(record)
            return instances

        record.update(
            status="recorded_baseline_geometry",
            source_rows=source_rows,
            uv=uv,
            metrics=geometry_metrics(self.rig, uv),
        )
        self.diagnostics.append(record)
        return instances
