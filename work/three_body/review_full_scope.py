"""Pure, same-origin expansion of extracted slice constraints to full geometry.

This adapter verifies in-memory hashes and exact slices. It does not read or
verify review bytes, files, calibration, physical projections or source rays.
Those checks belong to the original extraction and the caller's source proof.
No origin migration, matching, ID naming, dense correction or file I/O occurs.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parents[2] / "labs/20260909_three_body"
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))
import review_constraints as constraints  # noqa: E402

from medaka3d.eval.encounter_review import CONTINUITY_STATES  # noqa: E402

KEYS = ("frame_idx", "pose_xyz", "sources", "observed", "identity_review")
PHASES = ("entry", "after_reported_event", "exit")
ADAPTER = {
    "name": "bind_slice_constraints_to_full",
    "version": 1,
    "contract": "same-origin exact five-array/projection slice and nine endpoint states; "
    "six original local links unchanged; sparse projection outside slice; no file verification",
}
ADAPTER_DIGEST = constraints._digest(ADAPTER)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def exact_array(left, right, name):
    require(
        isinstance(left, np.ndarray)
        and isinstance(right, np.ndarray)
        and left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C"),
        "exact dtype/shape/bytes slice mismatch: " + name,
    )


def geometry_arrays(geometry):
    require(isinstance(geometry, Mapping), "geometry mapping required")
    require(
        all(isinstance(geometry.get(k), np.ndarray) for k in KEYS), "five numpy arrays required"
    )
    frames = geometry["frame_idx"]
    require(
        frames.ndim == 1
        and len(frames) > 0
        and frames.dtype.kind in "iu"
        and np.all(frames >= 0)
        and np.all(frames <= np.iinfo(np.int64).max),
        "nonnegative int64-compatible native frames required",
    )
    require(np.all(np.diff(frames.astype(np.int64)) == 1), "contiguous native frames required")
    constraints.validate_candidate(geometry, frames)
    require(np.isfinite(geometry["pose_xyz"]).all(), "full persistent latent poses must be finite")
    require(np.all(geometry["sources"] <= np.iinfo(np.int64).max), "source rows exceed int64")
    for a, b in ((0, 1), (0, 2), (1, 2)):
        require(
            not np.any(
                (geometry["sources"][:, a] >= 0)
                & (geometry["sources"][:, a] == geometry["sources"][:, b])
            ),
            "source rows must be exclusive per view and node",
        )
    return {k: constraints._array_digest(geometry[k]) for k in KEYS}


def projected_array(projected, count):
    require(
        isinstance(projected, np.ndarray)
        and projected.shape == (count, 3, 3, 2)
        and projected.dtype.kind == "f"
        and not np.isinf(projected).any(),
        "projected mouths must be floating (T,3,3,2), with NaN allowed but no infinity",
    )
    return constraints._array_digest(projected)


def verify_endpoints(bundle, geometry, projected):
    endpoints = bundle["endpoints"]
    require(
        bundle["requested_endpoints"] == 9 and len(endpoints) == 9, "all nine endpoints required"
    )
    anchors = []
    positions = {int(frame): i for i, frame in enumerate(geometry["frame_idx"])}
    for probe in range(3):
        frame = endpoints[probe * 3]["frame"]
        require(type(frame) is int and frame in positions, "endpoint frame absent from slice")
        anchors.append(frame)
        t = positions[frame]
        state = constraints._digest(
            {
                **{k: constraints._array_digest(geometry[k][t]) for k in KEYS[1:]},
                "projected_mouth": constraints._array_digest(projected[t]),
            }
        )
        for animal, label in enumerate(constraints.ANIMALS):
            endpoint = endpoints[3 * probe + animal]
            require(
                endpoint["frame"] == frame
                and endpoint["probe_index"] == probe
                and endpoint["phase"] == PHASES[probe]
                and endpoint["anonymous_label"] == label
                and endpoint["clip_scope_id"] == bundle["clip_scope_id"]
                and endpoint["source_state_sha256"] == state,
                "endpoint native identity/source state differs",
            )
            slot, raw = endpoint["raw_slot"], endpoint["raw_match"]
            require(slot is None or (type(slot) is int and 0 <= slot < 3), "invalid endpoint slot")
            require(raw["slot"] == slot, "endpoint raw match differs from slot")
            expected_id = (
                None
                if slot is None
                else constraints._digest(
                    {
                        "origin": bundle["origin"],
                        "frame": frame,
                        "slot": slot,
                    }
                )
            )
            require(endpoint["endpoint_id"] == expected_id, "endpoint ID differs from origin/state")
            expected_accepted = copy.deepcopy(raw)
            if slot is None:
                require(
                    endpoint["raw_identity_review"] is None
                    and endpoint["mouth_source_rows"] is None
                    and endpoint["mouth_direct_view_count"] is None,
                    "unresolved endpoint must retain no slot-specific state",
                )
            else:
                rows = geometry["sources"][t, slot, :, 0]
                flag = bool(geometry["identity_review"][t, slot])
                require(
                    bool(geometry["observed"][t, slot, 0])
                    and endpoint["mouth_source_rows"] == rows.tolist()
                    and type(endpoint["mouth_direct_view_count"]) is int
                    and endpoint["mouth_direct_view_count"] == int((rows >= 0).sum())
                    and type(endpoint["raw_identity_review"]) is bool
                    and endpoint["raw_identity_review"] == flag,
                    "resolved endpoint source/observed/review differs",
                )
                if flag:
                    expected_accepted.update(slot=None, reason="identity requires review")
            require(
                endpoint["accepted_match"] == expected_accepted, "accepted endpoint review differs"
            )
    require(anchors[0] < anchors[1] < anchors[2], "ordered distinct review anchors required")
    require(bundle["requested_links"] == 6 and len(bundle["links"]) == 6, "all six links required")
    review_sha = bundle["provenance"]["review_sha256"]
    for animal, label in enumerate(constraints.ANIMALS):
        previous_confirmed = True
        for edge, edge_name in enumerate(constraints.EDGE_NAMES):
            link = bundle["links"][2 * animal + edge]
            left, right = endpoints[3 * edge + animal], endpoints[3 * (edge + 1) + animal]
            require(link["human_status"] in CONTINUITY_STATES, "invalid human link status")
            status = (
                "human_unconfirmed"
                if link["human_status"] != "confirmed"
                else (
                    "endpoint_unresolved"
                    if left["endpoint_id"] is None or right["endpoint_id"] is None
                    else "confirmed_endpoint_link"
                )
            )
            previous_confirmed = previous_confirmed and status == "confirmed_endpoint_link"
            expected = {
                "link_id": constraints._digest(
                    {
                        "review": review_sha,
                        "origin": bundle["origin"],
                        "animal": label,
                        "edge": edge_name,
                    }
                ),
                "clip_scope_id": bundle["clip_scope_id"],
                "anonymous_label": label,
                "edge": edge_name,
                "status": status,
                "start_frame": left["frame"],
                "stop_frame": right["frame"],
                "left_endpoint_id": left["endpoint_id"],
                "right_endpoint_id": right["endpoint_id"],
                "left_raw_slot": left["raw_slot"],
                "right_raw_slot": right["raw_slot"],
                "left_reason": left["raw_match"]["reason"],
                "right_reason": right["raw_match"]["reason"],
                "chain_to_entry_confirmed": bool(previous_confirmed),
                "interior_identity_certified": False,
                "switch_time_known": False,
                "raw_review_cleared": False,
                "global_identity_assigned": False,
            }
            require(
                all(link.get(k) == v for k, v in expected.items()), "link endpoint/scope differs"
            )
    return anchors


def bind_slice_constraints_to_full(
    slice_bundle,
    full_geometry,
    slice_geometry,
    slice_projected_mouth,
    full_projected_mouth,
    slice_indices,
    *,
    candidate_sha256,
    calibration_sha256,
    projection_origin_sha256,
    source_proof_sha256,
):
    """Return a newly content-hashed core-compatible bundle and independent proof.

    The original extraction must already declare this SAME final candidate SHA.
    `slice_indices` is a contiguous native subset; both five-array and projected
    slices must be byte-identical. Full projection rows outside that slice must
    be NaN (uncomputed), not fabricated geometry. Every anchor must occur in the
    slice, including endpoints left unresolved by matching or direct support.
    The caller must separately seal original extraction/source-proof files and
    this adapter's actual source version. This pure function reads no files.
    """
    require(isinstance(slice_bundle, Mapping), "extracted constraint mapping required")
    original = copy.deepcopy(dict(slice_bundle))
    original_digest = original.pop("content_sha256", None)
    require(
        original.get("schema") == constraints.SCHEMA
        and original_digest == constraints._digest(original),
        "original extracted content hash/schema differs",
    )
    for value, name in (
        (candidate_sha256, "candidate"),
        (projection_origin_sha256, "projector"),
        (source_proof_sha256, "source proof"),
    ):
        constraints._sha(value, name)
    require(
        isinstance(calibration_sha256, Mapping) and calibration_sha256,
        "nonempty calibration digest mapping required",
    )
    calibration = {}
    for name, digest in calibration_sha256.items():
        require(isinstance(name, str) and bool(name), "calibration names must be nonempty strings")
        calibration[name] = constraints._sha(digest, "calibration")
    origin = original["origin"]
    require(
        origin["candidate_sha256"] == candidate_sha256
        and origin["calibration_sha256"] == calibration
        and origin["projection_origin_sha256"] == projection_origin_sha256
        and origin["view_order"] == list(constraints.VIEWS),
        "same final candidate/calibration/projector origin required; migration prohibited",
    )
    provenance = original["provenance"]
    require(
        provenance["review_and_manifest_bytes_verified_here"] is True
        and provenance["candidate_calibration_files_verified_here"] is False
        and "scope_binding" not in provenance,
        "original extractor provenance required; rebinding an adapted bundle is prohibited",
    )
    for name in ("review_sha256", "source_manifest_sha256"):
        constraints._sha(provenance[name], name)
    require(
        original["clip_scope_id"]
        == constraints._digest(
            {k: provenance[k] for k in ("review_sha256", "source_manifest_sha256")}
        ),
        "original review/manifest clip scope differs",
    )
    full_digests, slice_digests = geometry_arrays(full_geometry), geometry_arrays(slice_geometry)
    require(
        provenance["candidate_arrays_sha256"] == slice_digests,
        "extracted slice array digests differ",
    )
    full_projected_digest = projected_array(full_projected_mouth, len(full_geometry["frame_idx"]))
    slice_projected_digest = projected_array(
        slice_projected_mouth, len(slice_geometry["frame_idx"])
    )
    require(
        provenance["projected_mouth_array_sha256"] == slice_projected_digest,
        "extracted slice projection digest differs",
    )
    indices = np.asarray(slice_indices)
    require(
        indices.ndim == 1
        and len(indices) == len(slice_geometry["frame_idx"])
        and indices.dtype.kind in "iu"
        and np.all(indices >= 0)
        and np.all(indices < len(full_geometry["frame_idx"])),
        "slice indices must be in-range integer native positions",
    )
    index = indices.astype(np.int64, copy=True)
    require(np.all(np.diff(index) == 1), "slice indices must be consecutive and ordered")
    for key in KEYS:
        exact_array(slice_geometry[key], full_geometry[key][index], key)
    exact_array(slice_projected_mouth, full_projected_mouth[index], "projected_mouth")
    require(
        np.isnan(full_projected_mouth[: index[0]]).all()
        and np.isnan(full_projected_mouth[index[-1] + 1 :]).all(),
        "outside-slice full projections must remain uncomputed NaN",
    )
    anchors = verify_endpoints(original, slice_geometry, slice_projected_mouth)
    require(
        verify_endpoints(original, full_geometry, full_projected_mouth) == anchors,
        "full endpoint state changed",
    )
    proof = {
        "schema": "review_slice_to_full_scope_proof_v1",
        "adapter": copy.deepcopy(ADAPTER),
        "adapter_version_digest": ADAPTER_DIGEST,
        "original_bundle_content_sha256": original_digest,
        "origin": copy.deepcopy(origin),
        "source_proof_sha256": source_proof_sha256,
        "slice_indices": index.tolist(),
        "slice_indices_array_sha256": constraints._array_digest(indices),
        "slice_native_range": [
            int(slice_geometry["frame_idx"][0]),
            int(slice_geometry["frame_idx"][-1]) + 1,
        ],
        "full_native_range": [
            int(full_geometry["frame_idx"][0]),
            int(full_geometry["frame_idx"][-1]) + 1,
        ],
        "slice_arrays_sha256": slice_digests,
        "full_arrays_sha256": full_digests,
        "slice_projected_mouth_sha256": slice_projected_digest,
        "full_projected_mouth_sha256": full_projected_digest,
        "anchor_frames": anchors,
        "verified_endpoint_count": 9,
        "preserved_link_count": 6,
        "source_proof_file_verified_here": False,
        "review_and_manifest_bytes_verified_here": False,
        "candidate_calibration_files_verified_here": False,
        "projection_geometry_recomputed_here": False,
        "ray_pixel_support_recomputed_here": False,
        "endpoint_content_unchanged": True,
        "human_status_unchanged": True,
        "origin_changed": False,
        "dense_geometry_changed": False,
        "actual_adapter_source_binding_required_of_caller": True,
    }
    proof["content_sha256"] = constraints._digest(proof)
    result = copy.deepcopy(original)
    result["provenance"].update(
        candidate_arrays_sha256=copy.deepcopy(full_digests),
        projected_mouth_array_sha256=full_projected_digest,
        original_extraction_provenance=copy.deepcopy(provenance),
        review_and_manifest_bytes_verified_here=False,
        scope_binding={
            "proof_content_sha256": proof["content_sha256"],
            "original_bundle_content_sha256": original_digest,
            "adapter_version_digest": ADAPTER_DIGEST,
            "source_proof_sha256": source_proof_sha256,
        },
        trusted_loader_required="Caller must bind original extraction, source proof, completed "
        "candidate/calibration/projector files and adapter source. This adapter verifies only "
        "in-memory scope expansion; original byte verification is recorded separately.",
    )
    result["content_sha256"] = constraints._digest(result)
    # Compatibility check uses the frozen content/graph validator, not any new matching.
    constraints.combine_constraints([result])
    return result, proof
