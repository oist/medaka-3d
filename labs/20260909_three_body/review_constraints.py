"""Provenance-scoped human endpoint constraints; no dense correction or ID naming.

This pure helper verifies review/manifest bytes itself and reuses the existing
review validation and shared raw matching. A trusted caller must verify completed
candidate/calibration files and the projection implementation against its supplied
digests before calling. Declared file digests are scope keys, not proof that this
helper has opened or verified those files. No files are read or written here.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping

import numpy as np
from evaluate_review import validate_candidate, validate_inputs

from medaka3d.eval.encounter_review import ANIMALS, MATCH_RADIUS_PX, score_encounter_review
from medaka3d.types import VIEWS

SCHEMA = "human_endpoint_constraints_v1"
EDGE_NAMES = ("entry_to_after", "after_to_exit")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _array_digest(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(json.dumps([array.dtype.str, list(array.shape)]).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sha(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _document(value, name):
    if not isinstance(value, bytes):
        raise ValueError(f"{name} must be the original JSON bytes")
    try:
        result = json.loads(value.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {name} JSON") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return result, hashlib.sha256(value).hexdigest()


def extract_review_constraints(
    review_bytes, sequence_manifest_bytes, candidate_arrays, projected_mouth, *,
    candidate_sha256, calibration_sha256, projection_origin_sha256,
):
    """Return all six requested links, retaining every unresolved endpoint.

    Projections have the shared scorer's (T,3,V,2) shape and contain ALL latent
    mouths. The supplied candidate arrays follow the completed runner schema.
    Anonymous A/B/C scope includes the review's exact bytes and source manifest.
    Confirmed after->exit is a local constraint even if entry->after is ambiguous;
    no chain to entry or global biological identity is invented.
    """
    review, review_sha = _document(review_bytes, "review")
    sequence, manifest_sha = _document(sequence_manifest_bytes, "sequence manifest")
    frames, anchors, human_points, continuity = validate_inputs(review, sequence, manifest_sha)
    validate_candidate(candidate_arrays, frames)
    _sha(candidate_sha256, "candidate_sha256")
    _sha(projection_origin_sha256, "projection_origin_sha256")
    if not isinstance(calibration_sha256, Mapping) or not calibration_sha256:
        raise ValueError("nonempty calibration file digest mapping required")
    calibration = {}
    for name, digest in calibration_sha256.items():
        if not isinstance(name, str) or not name:
            raise ValueError("calibration digest names must be nonempty strings")
        calibration[name] = _sha(digest, f"calibration {name}")
    projected = np.asarray(projected_mouth)
    if projected.shape != (len(candidate_arrays["frame_idx"]), 3, len(VIEWS), 2):
        raise ValueError("projected mouths must contain all three canonical views and latent slots")
    source = candidate_arrays["sources"]
    for fish in range(3):
        for other in range(fish):
            if np.any((source[:, fish] >= 0) & (source[:, fish] == source[:, other])):
                raise ValueError("candidate per-view/node source rows must be exclusive")
    scored = score_encounter_review(
        candidate_arrays["frame_idx"], projected, candidate_arrays["observed"][:, :, 0],
        candidate_arrays["identity_review"], anchors, human_points, continuity,
    )
    origin = {
        "experiment": sequence["experiment"],
        "candidate_sha256": candidate_sha256,
        "calibration_sha256": calibration,
        "projection_origin_sha256": projection_origin_sha256,
        "view_order": list(VIEWS),
    }
    clip_scope = _digest({"review_sha256": review_sha, "source_manifest_sha256": manifest_sha})
    index = {int(frame): i for i, frame in enumerate(candidate_arrays["frame_idx"])}
    endpoints = []
    for probe, anchor in enumerate(scored["raw"]["anchors"]):
        frame = anchor["frame"]
        t = index.get(frame)
        state_sha = None if t is None else _digest({
            key: _array_digest(value[t])
            for key, value in {
                **{key: candidate_arrays[key] for key in
                   ("pose_xyz", "sources", "observed", "identity_review")},
                "projected_mouth": projected,
            }.items()
        })
        for animal, match in enumerate(anchor["animals"]):
            slot = match["slot"]
            endpoint_id = None if slot is None else _digest({
                "origin": origin, "frame": frame, "slot": slot,
            })
            endpoints.append({
                "probe_index": probe, "phase": review["anchors"][probe]["phase"],
                "frame": frame, "anonymous_label": ANIMALS[animal], "clip_scope_id": clip_scope,
                "endpoint_id": endpoint_id, "raw_slot": slot,
                "raw_match": copy.deepcopy(match),
                "accepted_match": copy.deepcopy(scored["accepted"]["anchors"][probe]
                                                ["animals"][animal]),
                "source_state_sha256": state_sha,
                "raw_identity_review": None if slot is None
                else bool(candidate_arrays["identity_review"][t, slot]),
                "mouth_source_rows": None if slot is None else source[t, slot, :, 0].tolist(),
                "mouth_direct_view_count": None if slot is None
                else int((source[t, slot, :, 0] >= 0).sum()),
                "human_visibility": review["anchors"][probe]["animals"][ANIMALS[animal]]
                ["visibility"],
            })
    links = []
    for animal, label in enumerate(ANIMALS):
        for edge, name in enumerate(EDGE_NAMES):
            left, right = endpoints[edge * 3 + animal], endpoints[(edge + 1) * 3 + animal]
            human_status = str(continuity[animal, edge])
            status = "confirmed_endpoint_link"
            if human_status != "confirmed":
                status = "human_unconfirmed"
            elif left["endpoint_id"] is None or right["endpoint_id"] is None:
                status = "endpoint_unresolved"
            links.append({
                "link_id": _digest({"review": review_sha, "origin": origin,
                                    "animal": label, "edge": name}),
                "clip_scope_id": clip_scope, "anonymous_label": label, "edge": name,
                "human_status": human_status, "status": status,
                "basis": review["continuity"][label]["basis"],
                "start_frame": left["frame"], "stop_frame": right["frame"],
                "left_endpoint_id": left["endpoint_id"], "right_endpoint_id": right["endpoint_id"],
                "left_raw_slot": left["raw_slot"], "right_raw_slot": right["raw_slot"],
                "left_reason": left["raw_match"]["reason"],
                "right_reason": right["raw_match"]["reason"],
                "chain_to_entry_confirmed": bool(
                    status == "confirmed_endpoint_link" and endpoints[animal]["endpoint_id"]
                    and np.all(continuity[animal, :edge+1] == "confirmed")
                    and all(endpoints[p*3+animal]["endpoint_id"] for p in range(edge+2))
                ),
                "interior_identity_certified": False, "switch_time_known": False,
                "raw_review_cleared": False, "global_identity_assigned": False,
            })
    result = {
        "schema": SCHEMA, "origin": origin,
        "provenance": {
            "review_sha256": review_sha, "source_manifest_sha256": manifest_sha,
            "candidate_arrays_sha256": {key: _array_digest(candidate_arrays[key])
                                        for key in ("frame_idx", "pose_xyz", "sources", "observed",
                                                    "identity_review")},
            "projected_mouth_array_sha256": _array_digest(projected),
            "review_and_manifest_bytes_verified_here": True,
            "candidate_calibration_files_verified_here": False,
            "trusted_loader_required": (
                "Completed candidate, calibration and projector digests must be verified by caller"
            ),
        },
        "human_review_metadata": {
            "status": copy.deepcopy(review["status"]),
            "reviewer": copy.deepcopy(review["reviewer"]),
            "biological_naming_preserved_not_applied": copy.deepcopy(
                review.get("biological_naming")
            ),
        },
        "clip_scope_id": clip_scope, "endpoints": endpoints, "links": links,
        "requested_endpoints": 9, "requested_links": 6,
        "match_radius_px": MATCH_RADIUS_PX,
        "shared_raw_interval_counts_unchanged": scored["raw"]["interval_counts"],
        "dense_trajectory_relabelled": False, "review_flags_cleared": False,
        "identity_by_elimination": False,
    }
    result["content_sha256"] = _digest(result)
    return result


def combine_constraints(bundles):
    """Combine compatible endpoint graphs, withholding whole conflicting components.

    Different candidate/calibration/projector origins are rejected. Anonymous
    names never connect two documents: only identical scoped physical endpoints
    or explicit confirmed links connect nodes. Unresolved links are retained but
    do not add edges. This transitive closure covers endpoints only, never frames
    between probes, risk resolution, or automatic tracklet propagation.
    """
    bundles = list(bundles)
    if not bundles:
        raise ValueError("at least one extracted constraint bundle required")
    origin = None
    endpoints, links, documents = {}, {}, {}
    state_conflicts = set()
    for bundle in bundles:
        copied = copy.deepcopy(bundle)
        digest = copied.pop("content_sha256", None)
        if copied.get("schema") != SCHEMA or digest != _digest(copied):
            raise ValueError("constraint bundle content/schema changed after extraction")
        if origin is None:
            origin = copied["origin"]
        elif origin != copied["origin"]:
            raise ValueError("candidate/calibration/projection origin mismatch; cannot merge")
        documents[digest] = {
            "provenance": copied["provenance"],
            "human_review_metadata": copied["human_review_metadata"],
            "clip_scope_id": copied["clip_scope_id"],
        }
        for endpoint in copied["endpoints"]:
            node = endpoint["endpoint_id"]
            if node is None:
                continue
            if node in endpoints:
                if endpoints[node]["source_state_sha256"] != endpoint["source_state_sha256"]:
                    state_conflicts.add(node)
            else:
                endpoints[node] = endpoint
        for link in copied["links"]:
            identifier = link["link_id"]
            if identifier in links and links[identifier] != link:
                raise ValueError("one review link has conflicting extracted content")
            links[identifier] = link
    neighbors = defaultdict(set)
    for link in links.values():
        if link["status"] == "confirmed_endpoint_link":
            left, right = link["left_endpoint_id"], link["right_endpoint_id"]
            neighbors[left].add(right)
            neighbors[right].add(left)
    components, visited = [], set()
    for initial in sorted(neighbors):
        if initial in visited:
            continue
        pending, nodes = [initial], set()
        while pending:
            node = pending.pop()
            if node in nodes:
                continue
            nodes.add(node)
            pending.extend(neighbors[node] - nodes)
        visited |= nodes
        frames = defaultdict(set)
        for node in nodes:
            frames[endpoints[node]["frame"]].add(endpoints[node]["raw_slot"])
        conflicts = [
            {"kind": "one_component_has_multiple_slots_at_one_native_frame",
             "frame": frame, "slots": sorted(slots)}
            for frame, slots in sorted(frames.items()) if len(slots) > 1
        ]
        conflicts += [
            {"kind": "endpoint_source_or_projection_content_disagrees", "endpoint_id": node}
            for node in sorted(nodes & state_conflicts)
        ]
        component_id = _digest(sorted(nodes))
        dependent = []
        for link in links.values():
            if link["status"] not in ("confirmed_endpoint_link", "conflict"):
                continue
            if link["left_endpoint_id"] in nodes or link["right_endpoint_id"] in nodes:
                dependent.append(link["link_id"])
                link["component_id"] = component_id
                if conflicts:
                    link["status"] = "conflict"
        components.append({
            "component_id": component_id, "endpoint_ids": sorted(nodes),
            "dependent_link_ids": sorted(dependent), "conflicts": conflicts,
            "status": "conflict" if conflicts else "consistent_endpoint_constraints",
            "global_identity_assigned": False, "dense_interval_certified": False,
        })
    result = {
        "schema": "combined_human_endpoint_constraints_v1", "origin": origin,
        "document_provenance": documents, "links": list(links.values()),
        "components": components,
        "endpoint_observations": [copy.deepcopy(endpoint)
            for bundle in bundles for endpoint in bundle["endpoints"]],
        "unresolved_endpoint_observations": [
            copy.deepcopy(endpoint) for bundle in bundles for endpoint in bundle["endpoints"]
            if endpoint["endpoint_id"] is None
        ],
        "counts": {
            "distinct_documents": len(documents), "requested_links": len(links),
            **{status: sum(link["status"] == status for link in links.values())
               for status in ("confirmed_endpoint_link", "human_unconfirmed", "endpoint_unresolved",
                              "conflict")},
            "conflicting_components": sum(bool(c["conflicts"]) for c in components),
        },
        "anonymous_names_merged_across_documents": False,
        "dense_trajectory_relabelled": False, "review_flags_cleared": False,
        "identity_by_elimination": False,
    }
    result["content_sha256"] = _digest(result)
    return result
