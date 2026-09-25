"""Select actual, initially supported SLEAP node pixels for a future point tracker.

This pure selector does not establish geometry or identity: its caller supplies
sources already qualified by the frozen refractive support checks. No projected
latent node, mask centroid, replacement query, model, or image is used here.
"""

from __future__ import annotations

import numpy as np

VIEWS = ("top_cam", "side1_cam", "side2_cam")
NODES = ("mouth", "pectoral", "tail")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def select_queries(points, qualified_sources, observed, *, native_frame, image_hw=(1200, 1200)):
    """Return owned fixed query arrays, all27 slot records and all9 node-set records.

    Source layout is body/view/node, observed is body/node. A node must already
    be supported by two views, and each selected source pixel must also be finite
    and inside its image. Exact duplicate pixels are retained with flags; distinct
    semantic labels are never silently merged. These labels are conditional body
    slots, not animal-ID ground truth.
    """
    require(type(native_frame) is int and 0 <= native_frame < 2**63, "native frame required")
    require(
        type(image_hw) is tuple
        and len(image_hw) == 2
        and all(type(n) is int and n > 0 for n in image_hw),
        "positive image height/width required",
    )
    require(set(points) == set(VIEWS), "exact three views required")
    require(
        qualified_sources.shape == (3, 3, 3)
        and qualified_sources.dtype == np.int32
        and np.all(qualified_sources >= -1),
        "qualified source schema differs",
    )
    require(observed.shape == (3, 3) and observed.dtype == bool, "observed schema differs")
    require(
        np.array_equal(observed, (qualified_sources >= 0).sum(axis=1) >= 2),
        "observed must match qualified two-view support",
    )
    for vi, view in enumerate(VIEWS):
        a = points[view]
        require(
            isinstance(a, np.ndarray)
            and a.ndim == 3
            and a.shape[1:] == (3, 2)
            and a.dtype == np.float32,
            "native float32 SLEAP rows required",
        )
        require(np.all(qualified_sources[:, vi] < len(a)), "source row outside detector data")
        for node in range(3):
            rows = qualified_sources[:, vi, node]
            rows = rows[rows >= 0]
            require(len(np.unique(rows)) == len(rows), "source reused by competing bodies")

    views, slots = {}, []
    width_height = np.asarray(image_hw[::-1], np.float32)
    for vi, view in enumerate(VIEWS):
        chosen = []
        indices = []
        for body in range(3):
            for node in range(3):
                source = int(qualified_sources[body, vi, node])
                reasons = []
                if not observed[body, node]:
                    reasons.append("node_lacks_initial_two_view_support")
                xy = None
                if source < 0:
                    reasons.append("no_qualified_source_in_view")
                else:
                    xy = points[view][source, node]
                    if not np.isfinite(xy).all():
                        reasons.append("source_pixel_nonfinite")
                    elif not ((xy >= 0) & (xy < width_height)).all():
                        reasons.append("source_pixel_outside_image")
                index = len(chosen) if not reasons else -1
                if index >= 0:
                    chosen.append(xy.copy())
                    indices.append((body, node, source))
                slots.append(
                    dict(
                        view=view,
                        body_slot=body,
                        node_index=node,
                        node_name=NODES[node],
                        source_row=source,
                        initial_observed=bool(observed[body, node]),
                        query_index=index,
                        selected=index >= 0,
                        reasons=reasons,
                    )
                )
        xy = np.asarray(chosen, dtype=np.float32).reshape(-1, 2)
        duplicate = np.zeros(len(xy), bool)
        for i in range(len(xy)):
            duplicate[i] = np.count_nonzero((xy == xy[i]).all(axis=1)) > 1
        label = np.asarray(indices, np.int32).reshape(-1, 3)
        views[view] = dict(
            frame_idx=np.asarray(native_frame, np.int64),
            xy=xy,
            body_slot=label[:, 0].copy(),
            node_index=label[:, 1].copy(),
            source_row=label[:, 2].copy(),
            duplicate_pixel=duplicate,
        )
    node_sets = []
    for body in range(3):
        for node in range(3):
            rows = [r for r in slots if r["body_slot"] == body and r["node_index"] == node]
            selected_views = [r["view"] for r in rows if r["selected"]]
            node_sets.append(
                dict(
                    body_slot=body,
                    node_index=node,
                    node_name=NODES[node],
                    selected_views=selected_views,
                    selected_view_count=len(selected_views),
                    two_view_query_set_available=len(selected_views) >= 2,
                )
            )
    return dict(
        schema="semantic_node_query_selection_v1",
        native_frame=native_frame,
        views=views,
        slots=slots,
        node_sets=node_sets,
        scheduled_slots=27,
        scheduled_node_sets=9,
        geometry_established_by_selector=False,
        biological_ID_truth=False,
        missing_queries_synthesized=False,
    )
