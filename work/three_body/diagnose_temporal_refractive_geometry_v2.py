"""Fixed two64-window conditional body/mask evidence, saved inputs and CPU only.

No raw RGB, model construction/forward, fit, history reload, GPU or ID changes.
Historical proof headers are reused; only consumed payloads are rehashed.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import diagnose_temporal_mask_votes as votes
import numpy as np
import temporal_body_mask_evidence as arithmetic

REPO, WORK, base = votes.REPO, votes.WORK, votes.base
sys.path[:0] = [str(REPO / "labs/20260909_three_body"), str(REPO / "src")]
import diagnose_identity_components as identity
from node_geometry import RayObservations, assigned_rays, supported_nodes
from observations import CachedNodes

from medaka3d.calib import load_rig

coh, VIEWS, ARMS = identity.coh, votes.VIEWS, votes.ARMS
require, same = base.require, base.same
STAGE = "temporal_refractive_geometry_diagnosis_v2"
PLAN = WORK / "TEMPORAL_REFRACTIVE_GEOMETRY_PLAN.md"
PLAN_SHA = "f125cae41c5ae1260271144bc022a56b92b73fca87531228dc03a1eecd950c4c"
VOTE_ROOT = WORK / "temporal_mask_vote_diagnosis_v1"
VOTE_SHA = "353919dd9ba67cd243fe6b463826334ec6119bf4288be45285903ecd1772cefb"
PINS = {
    WORK / "diagnose_temporal_mask_votes.py": (
        "8ef632464d1d82c52ed17235a53d278629556d85fcc9c79169bf2b720d29276b"
    ),
    WORK / "diagnose_instance_mask_coherence_v2.py": (
        "39aa2b6bbcd24200b4c1326ec923c06f3e331a20d4624687630bdbbb1dd487fa"
    ),
    WORK / "diagnose_identity_components.py": (
        "eba8c6d64da9f797afd1a08c72bb5886b7c7fd85af93e3c67845101c16d04040"
    ),
    WORK / "diagnose_temporal_refractive_geometry.py": (
        "4d8d4dfe9fc1f7d6f1be91aed8fab14504074c9ab2293100ef1bf19859d8a000"
    ),
    WORK / "temporal_body_mask_evidence.py": (
        "3e1ae044ee73972dfda2c4250aa50a6370e60181cd37ec49b0fec084fdd6d216"
    ),
    WORK / "temporal_point_core.py": votes.CORE_SHA,
    REPO
    / "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
}
REFERENCES = (
    dict(
        root="long_remaining_supervisor_v1/phases/f066400-076360/tracking",
        filename="continuation.provenance.json",
        stage="three_body_long_continuation",
        seal="95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382",
        candidate="870836f9e086db6d1675dd283846e758038fb942fb6e8df6296f63aad98a5877",
        metadata="f0de69186dab5ea82b678d2673b6b9e778fe872a7e20e5916a441500edcb739a",
        start=49800,
        stop=76360,
    ),
    dict(
        root="continuation_transient_200_concurrent_v1",
        filename="continuation.provenance.json",
        stage="three_body_persistent_continuation",
        seal="163135a1f70c25bcd8251b5f5077c4b6eecc5b707e6df09d44466d4b8b5c05ce",
        candidate="c36e9279317749224ec73b254e72d4282f65d1d818195f52e1374a6637a11447",
        metadata="7c089f5b3da8239bde679cb159c71866260f6fe8e290e7035bdfd7c2b154e7d1",
        start=333519,
        stop=333719,
    ),
)
FIELDS = {
    "frame_idx": (),
    "pose_xyz": (3, 3, 3),
    "observed": (3, 3),
    "sources": (3, 3, 3),
    "identity_review": (3,),
    "mask_supported": (3,),
    "body_supported": (3,),
    "missing": (3,),
    "costs": (),
    "path_costs": (),
    "loss_parts": (5,),
    "score_margin": (),
    "proposal_kind": (),
    "hypotheses": (),
    "proposal_flow_nodes": (3, 3),
    "proposal_flow_parent": (),
    "node_missing": (3, 3),
    "direct_pectoral_age": (3,),
    "xyz": (3, 3),
    "accepted_xyz": (3, 3),
    "body_xyz": (3, 3),
}


def input_sha(seal, path):
    path = str(Path(path).resolve())
    value = seal["inputs"].get(path)
    require(value is not None, "missing historical input: " + path)
    if isinstance(value, dict):
        require(str(Path(value["path"]).resolve()) == path, "input role/path mismatch")
        return value["sha256"]
    return value


def output_members(seal, root):
    """Allow identical legacy alias keys; reject conflicting aliases, no payload reads."""
    root, result = Path(root).resolve(), {}
    for value in seal["outputs"].values():
        path = Path(value["path"]).resolve()
        require(path.is_relative_to(root), "historical output outside producer root")
        row = dict(
            path=str(path),
            sha256=value["sha256"],
            size_bytes=value.get("size_bytes", value.get("size")),
        )
        require(type(row["size_bytes"]) is int and row["size_bytes"] >= 0, "output size missing")
        require(str(path) not in result or result[str(path)] == row, "conflicting output aliases")
        result[str(path)] = row
    return result


def header(root, filename, stage, digest, evidence):
    path = root / filename
    evidence.add(path, digest)
    seal = base.read_json(path)
    require(
        seal["status"] == "ok" and seal["stage"] == stage and seal.get("finished_at"),
        "completed historical stage required",
    )
    return seal, output_members(seal, root)


def payload(root, members, name, evidence, expected=None):
    path = votes.bound_path(root, members, name, evidence)
    require(expected is None or evidence.paths[str(path)] == expected, "fixed payload SHA differs")
    return path


def json_payload(root, members, name, evidence, expected=None):
    return base.read_json(payload(root, members, name, evidence, expected))


def select_candidate(arrays, first, stop, selected_first):
    require(set(arrays) == set(FIELDS), "exact21 candidate fields required")
    for key, shape in FIELDS.items():
        require(arrays[key].shape == (stop - first, *shape), "candidate shape: " + key)
    require(arrays["frame_idx"].dtype == np.int64, "native index dtype")
    same(arrays["frame_idx"], np.arange(first, stop, dtype=np.int64), "candidate native interval")
    require(first <= selected_first and selected_first + 64 <= stop, "selected native scope")
    for key in (
        "observed",
        "identity_review",
        "mask_supported",
        "body_supported",
        "proposal_flow_nodes",
    ):
        require(arrays[key].dtype == bool, "boolean candidate field: " + key)
    for key in (
        "sources",
        "missing",
        "hypotheses",
        "proposal_flow_parent",
        "node_missing",
        "direct_pectoral_age",
    ):
        require(arrays[key].dtype.kind in "iu", "integer candidate field: " + key)
    require(
        arrays["sources"].dtype.kind == "i" and np.all(arrays["sources"] >= -1), "source sentinel"
    )
    require(
        arrays["pose_xyz"].dtype.kind == "f" and np.isfinite(arrays["pose_xyz"]).all(),
        "finite latent pose",
    )
    require(arrays["proposal_kind"].dtype.kind == "U", "proposal kind dtype")
    xyz = arrays["pose_xyz"][:, :, 1]
    for name, valid in (
        ("xyz", arrays["observed"][:, :, 1]),
        ("accepted_xyz", arrays["observed"][:, :, 1] & ~arrays["identity_review"]),
        ("body_xyz", arrays["body_supported"]),
    ):
        same(arrays[name], np.where(valid[..., None], xyz, np.nan), "candidate derived " + name)
    offset = selected_first - first
    return {k: a[offset : offset + 64].copy() for k, a in arrays.items()}


def context_geometry(metadata, rig):
    """Use the frozen dtype/shape/SHA camera serializer, including ndarray fields."""
    coh.validate_rig_context(metadata["context"], rig)


def load_context(spec, reference, evidence):
    root, prep = WORK / spec["root"], WORK / spec["preparation"]
    seal, mem = header(root, "comparison.provenance.json", spec["stage"], spec["seal"], evidence)
    summary = json_payload(root, mem, "summary.json", evidence)
    require(
        summary["status"] == "ok"
        and summary["schema"] == spec["stage"]
        and summary["frames"] == list(range(spec["first"], spec["first"] + 64))
        and summary["query_count"] == spec["queries"]
        and summary["model_forward_calls"] == 21
        and summary["model_initializations"] == 3,
        "fixed completed point comparison",
    )
    evidence.add(WORK / spec["producer"], spec["producer_sha"])
    require(
        input_sha(seal, WORK / spec["producer"]) == spec["producer_sha"], "point producer binding"
    )
    ps, pm = header(
        prep, "preparation.provenance.json", spec["preparation"], spec["prep_seal"], evidence
    )
    require(
        input_sha(seal, prep / "preparation.provenance.json") == spec["prep_seal"],
        "point/prep edge",
    )
    plan = json_payload(prep, pm, "plan.json", evidence)
    require(
        plan["frames"] == summary["frames"] and plan["views"] == list(VIEWS),
        "prepared native scope",
    )
    query_path = payload(prep, pm, "query_rows.json", evidence)
    require(
        input_sha(seal, query_path) == evidence.paths[str(query_path)], "prepared query rows edge"
    )
    query_rows = base.read_json(query_path)
    require(
        [q["view"] for q in query_rows] == list(VIEWS)
        and sum(len(q["rows"]) for q in query_rows) == spec["groups"],
        "all initial groups",
    )
    native_rows = json_payload(prep, pm, "native_evidence.json", evidence)
    require(
        [(r["frame_idx"], r["view"]) for r in native_rows]
        == [(t, v) for t in summary["frames"] for v in VIEWS],
        "native/view membership and order",
    )
    if spec["name"] == "known_gap":
        json_payload(
            prep,
            pm,
            "legacy_detector_condition.json",
            evidence,
            "2a20fe43f9ec8ff896159939cccddae1a9a44b57ab21f92638502d0665a35c25",
        )
    shared = plan["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7 and sum(Path(p).suffix == ".npz" for p in shared) == 3,
        "shared SL3/calibration4 required",
    )
    rr = WORK / reference["root"]
    rs, rm = header(rr, reference["filename"], reference["stage"], reference["seal"], evidence)
    for p, h in shared.items():
        require(input_sha(ps, p) == input_sha(rs, p) == h, "same registered source/calibration")
        evidence.add(p, h)
    calibration = next(
        Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"
    )
    manifest = base.read_json(calibration / "calibration_manifest.json")
    require(
        {str((calibration / r["path"]).resolve()) for r in manifest["model_files"]}
        | {str(calibration / "calibration_manifest.json")}
        == {p for p in shared if Path(p).suffix == ".json"},
        "rig exact four-file closure",
    )
    rig = load_rig(calibration)
    metadata = json_payload(rr, rm, "checkpoint/metadata.json", evidence, reference["metadata"])
    context_geometry(metadata, rig)
    candidate = payload(rr, rm, "candidate.npz", evidence, reference["candidate"])
    if spec["name"] == "initial":
        proof_root = WORK / "tail_one_minute_comparison_v1"
        proof, proof_mem = header(
            proof_root,
            "comparison.provenance.json",
            "tail_one_minute_comparison_v1",
            "8a3a77326f34e77801f834b529fef79626116fba63a1f99155ec33f2cea3e557",
            evidence,
        )
        require(
            input_sha(proof, candidate) == reference["candidate"]
            and input_sha(proof, rr / reference["filename"]) == reference["seal"],
            "original21 proof edge",
        )
        proof_summary = json_payload(proof_root, proof_mem, "summary.json", evidence)
        require(
            proof_summary["all21_arrays_exact_backtrace"] is True
            and proof_summary["horizon_stop"] == 76360
            and proof_summary["candidate_sha256"]["original"] == reference["candidate"]
            and proof_summary["shared_SL_calibration_sha256"] == shared,
            "original proof scope",
        )
    else:
        old_root = WORK / "persistent_mouth_transient_200_v1"
        old_sha = "7084161d4e43fbfdb1e5621afbd53246f1371a57748d912f77fa78d4b3ca18e1"
        _old, om = header(
            old_root,
            "experiment.provenance.json",
            "three_body_persistent_node_recovery_algorithm_arm",
            old_sha,
            evidence,
        )
        require(
            input_sha(rs, old_root / "experiment.provenance.json") == old_sha,
            "transient original edge",
        )
        require(
            om[str(old_root / "candidate.npz")]["sha256"] == reference["candidate"],
            "original/transient candidate identity",
        )
        ss = json_payload(
            old_root,
            om,
            "summary.json",
            evidence,
            "11e520643ce3e9da351431e119c94f052d4c3c5b0fb074e778a73e74d95f5f8c",
        )
        require(
            ss["frames"] == [333519, 333719] and ss["seed"]["later_baseline_input"] is False,
            "fixed optimistic seed scope",
        )
        cs = json_payload(rr, rm, "summary.json", evidence)
        require(cs["original_candidate_21_arrays_byte_exact"] is True, "transient21 equivalence")
    arrays = votes.load_npz(candidate)
    signatures = {k: base.signature(a) for k, a in arrays.items()}
    selected = select_candidate(arrays, reference["start"], reference["stop"], spec["first"])
    del arrays
    node_path = Path(next(p for p in shared if Path(p).suffix == ".npz"))
    input_root = node_path.parents[3]
    nodes = CachedNodes(
        input_root, "20251225_K_1_2_FCMC", np.arange(spec["first"], spec["first"] + 64)
    )
    require(
        {str(p.resolve()) for p in nodes.paths.values()}
        == {p for p in shared if Path(p).suffix == ".npz"},
        "SL loader source closure",
    )
    return dict(
        spec=spec,
        root=root,
        prep=prep,
        members=mem,
        prep_members=pm,
        seal=seal,
        prep_seal=ps,
        reference_seal=rs,
        queries=query_rows,
        native_rows=native_rows,
        shared=shared,
        rig=rig,
        candidate=selected,
        nodes=nodes,
        candidate_signatures=signatures,
        reference=reference,
    )


def strict_frame(context, t):
    a, rig = context["candidate"], context["rig"]
    native = int(a["frame_idx"][t])
    points = {v: context["nodes"].at(v, native).copy() for v in VIEWS}
    before = {v: base.signature(x) for v, x in points.items()}
    pose, sources, observed = a["pose_xyz"][t], a["sources"][t], a["observed"][t]
    for vi, view in enumerate(VIEWS):
        for node in range(3):
            ix = sources[:, vi, node]
            held = ix[ix >= 0]
            require(
                np.all(held < len(points[view])) and len(held) == len(set(held.tolist())),
                "invalid/reused saved source row",
            )
            require(np.isfinite(points[view][held, node]).all(), "assigned source nonfinite")
    rays = RayObservations.from_points(rig, points)
    replay, flags, distance = supported_nodes(pose, sources, rays, rig)
    same(replay, sources, "saved strict sources")
    same(flags, observed, "saved observed support")
    origin, direction, held = assigned_rays(rays, sources[None])
    pixel = np.full((3, 3, 3), np.nan)
    arrays = dict(
        frame_idx=np.asarray(native, np.int64),
        ray_distance_mm=distance,
        assigned_origin_f32=origin[0],
        assigned_direction_f32=direction[0],
        assigned=held[0],
        observed=observed.copy(),
        sources=sources.copy(),
    )
    for vi, view in enumerate(VIEWS):
        uv = np.asarray(rig.camera(view).project(pose.copy()))
        for b, n in np.argwhere(sources[:, vi] >= 0):
            pixel[b, vi, n] = np.linalg.norm(uv[b, n] - points[view][sources[b, vi, n], n])
        arrays[view + "_points"] = points[view]
        arrays[view + "_origins"] = rays.origins[vi]
        arrays[view + "_directions"] = rays.directions[vi]
        arrays[view + "_finite_node_rows"] = np.isfinite(points[view]).all(-1).sum(0)
    arrays["pixel_error_px"] = pixel
    require({v: base.signature(x) for v, x in points.items()} == before, "SL inputs mutated")
    return arrays


def node_inventory(context, support):
    a = context["candidate"]
    rows, sets = [], []
    for b in range(3):
        for n in range(3):
            views = []
            for vi, v in enumerate(VIEWS):
                r = int(a["sources"][0, b, vi, n])
                observed = bool(a["observed"][0, b, n])
                eligible = r >= 0 and observed
                reasons = []
                if r < 0:
                    reasons.append("no_assigned_current_source")
                if not observed:
                    reasons.append("node_not_supported_in_two_views")
                if not support[v + "_finite_node_rows"][n]:
                    reasons.append("no_finite_SL_node_rows")
                xy = support[v + "_points"][r, n].copy() if r >= 0 else np.full(2, np.nan)
                rows.append(
                    dict(
                        window=context["spec"]["name"],
                        native=int(a["frame_idx"][0]),
                        body=b,
                        node=n,
                        view=v,
                        source_row=r,
                        observed=observed,
                        eligible=eligible,
                        reasons=reasons,
                        source_xy=xy,
                        source_xy_dtype=xy.dtype.str,
                        source_xy_sha256=base.signature(xy)["sha256"],
                        source_archive=(
                            f"{context['spec']['name']}/sources/native_{int(a['frame_idx'][0])}.npz"
                        ),
                        source_array_key=v + "_points",
                        source_array_indices=[r, n] if r >= 0 else None,
                        ray_distance_mm=support["ray_distance_mm"][b, vi, n] if r >= 0 else np.nan,
                        pixel_error_px=support["pixel_error_px"][b, vi, n],
                        identity_review=bool(a["identity_review"][0, b]),
                    )
                )
                if eligible:
                    views.append(v)
            sets.append(
                dict(
                    window=context["spec"]["name"],
                    body=b,
                    node=n,
                    qualified_views=views,
                    count=len(views),
                    at_least_two=len(views) >= 2,
                    reasons=[] if len(views) >= 2 else ["fewer_than_two_qualified_views"],
                )
            )
    return rows, sets


def observation(context, native_row, evidence):
    path = payload(
        context["prep"],
        context["prep_members"],
        native_row["observation_path"],
        evidence,
        native_row["observation_sha256"],
    )
    require(
        input_sha(context["seal"], path) == native_row["observation_sha256"],
        "comparison/observation mask origin differs",
    )
    a = votes.load_npz(path)
    require(
        {k: base.signature(x) for k, x in a.items()} == native_row["observation_array_signatures"],
        "prepared observation array bytes",
    )
    require(
        a["frame_idx"].shape == ()
        and a["frame_idx"].dtype == np.int64
        and int(a["frame_idx"]) == native_row["frame_idx"]
        and a["mask_available"].shape == ()
        and a["mask_available"].dtype == bool,
        "observation native/availability",
    )
    masks, flag = a["masks"], bool(a["mask_available"])
    require(
        flag == native_row["mask_available"]
        and masks.dtype == bool
        and masks.shape == (native_row["mask_count"] or 0, 1200, 1200)
        and base.signature(masks)["sha256"] == native_row["mask_sha256"],
        "mask identity",
    )
    return masks, flag, {k: base.signature(x) for k, x in a.items()}, a


def geometry_frame(pose, observed, sources, masks, available, camera):
    before = [base.signature(a) for a in (pose, observed, sources, masks)]
    bodies = [coh.project_body(p, camera) for p in pose]
    for p, b in zip(pose, bodies, strict=True):
        repeat = coh.project_body(p, camera)
        for k in b:
            same(b[k], repeat[k], "body33 repeated projection")
    xyz, uv = (np.stack([b[k] for b in bodies]) for k in ("points_xyz", "projected_uv"))
    same(xyz[:, [0, 16, 32]], pose, "body endpoint world bytes")
    scene = np.asarray(camera.project(pose.copy()))
    same(scene, np.asarray(camera.project(pose.copy())), "scene9 repeated projection")
    # One EDT per mask for all three bodies and endpoints; no geometry change.
    joined = coh.sample_mask_fields(
        masks, available, np.concatenate((uv.reshape(-1, 2), scene.reshape(-1, 2)))
    )
    point_keys = {"valid", "finite", "pixel_xy"}

    def portion(start, end):
        return {
            k: v.copy()
            if k == "mask_status"
            else v[start:end].copy()
            if k in point_keys
            else v[:, start:end].copy()
            for k, v in joined.items()
        }

    fields, endpoints = portion(0, 99), portion(99, 108)
    metrics = [coh.coherence_metrics(fields, np.arange(33) + b * 33) for b in range(3)]
    reference = identity.geometry_reference(masks, available, scene, observed, sources, metrics)
    current = single_current_view_reference(masks, available, scene, sources, metrics, observed)
    complete = fields["valid"].reshape(3, 33).all(-1)
    coverage = np.full((len(masks), 3), np.nan)
    for b in np.flatnonzero(complete):
        if available:
            coverage[:, b] = (fields["membership"][: len(masks), b * 33 : (b + 1) * 33] == 1).mean(
                1
            )
    scene_member = endpoints["membership"][: len(masks)].reshape(len(masks), 3, 3)
    collision = np.stack(
        [(scene_member[:, np.arange(3) != b, :2] == 1).any((1, 2)) for b in range(3)], axis=1
    )
    extra = {
        name: [coh.coherence_metrics(fields, np.asarray(ix) + b * 33) for b in range(3)]
        for name, ix in coh.PARTS.items()
        if name != "body"
    }
    data = dict(
        points_xyz=xyz,
        projected_uv=uv,
        scene_projected_uv=scene,
        endpoint_batch_delta_uv=scene - uv[:, [0, 16, 32]],
        coverage=coverage,
        complete=complete,
        collision=collision,
        **fields,
        **{"endpoint_" + k: v for k, v in endpoints.items()},
    )
    data["continuous_inside"] = (
        np.isfinite(uv).all(-1)
        & (uv[..., 0] >= 0)
        & (uv[..., 0] < masks.shape[2])
        & (uv[..., 1] >= 0)
        & (uv[..., 1] < masks.shape[1])
    )
    require(
        [base.signature(a) for a in (pose, observed, sources, masks)] == before,
        "geometry input mutation",
    )
    return data, dict(reference=reference, current_single_view=current, parts=extra)


def single_current_view_reference(masks, available, projected, sources, metrics, observed):
    """Current source rows have ALREADY passed strict_frame; preserve global observed.

    Reuse frozen assignment and coherence arithmetic, with a single-view source
    gate explicitly distinct from the global two-view flag used for initial links.
    """
    before = [base.signature(a) for a in (masks, projected, sources, observed)]
    direct = (sources[:, :2] >= 0).all(-1)
    assigned, reasons = identity.appearance.assign_instances(
        masks if available else None, projected[:, :2].copy(), direct.copy()
    )
    eligible, rows = np.zeros(3, bool), []
    for b, body in enumerate(metrics):
        why = []
        if reasons[b] != "usable":
            why.append(str(reasons[b]))
        if body["requested_points"] != 33 or body["valid_points"] != 33:
            why.append("incomplete_33_projection")
        best = body["best_mask_rows"]
        if len(best) != 1:
            why.append("coherence_no_unique_minimum")
        elif assigned[b] >= 0 and int(assigned[b]) != best[0]:
            why.append("assignment_coherence_disagree")
        if not np.isfinite(body["C_single"]):
            why.append("coherence_nonfinite")
        eligible[b] = not why
        rows.append(
            dict(
                body=b,
                assigned_mask_row=int(assigned[b]),
                eligible=bool(eligible[b]),
                view_sources_mouth_chest=sources[b, :2].tolist(),
                global_observed=observed[b].tolist(),
                reasons=why,
            )
        )
    require(
        [base.signature(a) for a in (masks, projected, sources, observed)] == before,
        "current-view evidence mutated global support",
    )
    return dict(
        scope="single_current_view_geometry",
        strict_source_gate_preverified=True,
        global_two_view_observed_imputed=False,
        eligible=eligible.tolist(),
        rows=rows,
    )


def write_json(output, name, value):
    return output.write(name, coh.encode_nonfinite(value))


def source_snapshot(evidence, output, captured):
    required = set(PINS) | {PLAN, Path(__file__).resolve(), Path(arithmetic.__file__).resolve()}
    paths, inventory = base.source_inventory(dict(sys.modules), required)
    output.write(f"source_inventory/{len(captured):03d}.json", inventory)
    for path in sorted(paths):
        digest = evidence.add(path, PINS.get(path))
        if str(path) in captured:
            require(captured[str(path)] == digest, "loaded source drift")
            continue
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == digest, "snapshot read changed")
        output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), data)
        captured[str(path)] = digest


def geometry_pass(context, evidence, output, initial_rows, node_rows, node_sets):
    name, a = context["spec"]["name"], context["candidate"]
    before = {k: base.signature(x) for k, x in a.items()}
    output.write(name + "/reference_selected21.npz", a)
    write_json(
        output,
        name + "/reference_origin.json",
        dict(
            reference=context["reference"],
            full_candidate_array_signatures=context["candidate_signatures"],
            selected_array_signatures=before,
            retrospective_reference=True,
            biological_ID_truth=False,
        ),
    )
    cells, all_refs, total_masks = {}, [], 0
    for t in range(64):
        native = context["spec"]["first"] + t
        strict = strict_frame(context, t)
        output.write(f"{name}/sources/native_{native}.npz", strict)
        if t == 0:
            nr, ns = node_inventory(context, strict)
            node_rows.extend(nr)
            node_sets.extend(ns)
        for vi, view in enumerate(VIEWS):
            row = context["native_rows"][t * 3 + vi]
            masks, available, before_obs, obs = observation(context, row, evidence)
            data, ref = geometry_frame(
                a["pose_xyz"][t],
                a["observed"][t],
                a["sources"][t, :, vi],
                masks,
                available,
                context["rig"].camera(view),
            )
            require(
                {k: base.signature(x) for k, x in obs.items()} == before_obs,
                "packed observation mutated",
            )
            output.write(f"{name}/geometry/{native}_{view}.npz", data)
            ref.update(
                window=name,
                native_frame=native,
                view=view,
                identity_review=a["identity_review"][t].tolist(),
                missing=a["missing"][t].tolist(),
                node_missing=a["node_missing"][t].tolist(),
                direct_pectoral_age=a["direct_pectoral_age"][t].tolist(),
            )
            all_refs.append(ref)
            if t == 0:
                links = arithmetic.initial_link_matrix(ref["reference"], len(masks))
                for g in range(len(masks)):
                    for body in range(3):
                        initial_rows.append(
                            dict(
                                window=name,
                                view=view,
                                initial_mask_row=g,
                                body=body,
                                linked=bool(links[g, body]),
                                cell_reasons=(
                                    ref["reference"]["slots"][body]["reasons"]
                                    + (
                                        ["different_initial_assigned_mask"]
                                        if ref["reference"]["slots"][body]["assigned_mask_row"] != g
                                        else []
                                    )
                                ),
                                coverage=data["coverage"][g, body],
                                geometry_reference=ref["reference"]["slots"][body],
                            )
                        )
                context.setdefault("initial_links", {})[view] = links
            cells[(t, view)] = dict(
                coverage=data["coverage"],
                complete=data["complete"],
                collision=data["collision"],
                available=available,
                direct=np.asarray(ref["current_single_view"]["eligible"], bool),
            )
            total_masks += len(masks)
    require({k: base.signature(x) for k, x in a.items()} == before, "reference candidate mutated")
    write_json(output, name + "/geometry_references.json", all_refs)
    context["geometry"] = cells
    return total_masks


def join_view(context, view, evidence, output, saved_rows, vote_members, joined):
    query, qrow, metrics = votes.load_view(context, view, evidence)
    origin, groups = query["mask_index"], len(qrow["rows"])
    require(context["initial_links"][view].shape == (groups, 3), "frame0 group/mask geometry scope")
    name, first = context["spec"]["name"], context["spec"]["first"]
    for t in range(64):
        native = first + t
        # Independent mask pixels are reopened only from the owned sealed preparation,
        # not raw videos. One stack is live for this native/view.
        row = context["native_rows"][t * 3 + VIEWS.index(view)]
        masks, available, before, obs = observation(context, row, evidence)
        memberships = {}
        for arm in ARMS:
            p = payload(
                context["root"],
                context["members"],
                f"{view}/{arm}/membership_{native}.npz",
                evidence,
            )
            saved = votes.load_npz(p)
            votes.verify_membership(saved, metrics[arm], origin, groups, t, native)
            pixel = base.core.mask_memberships(
                metrics[arm]["xy"][t], masks if available else None, (1200, 1200)
            )
            for k in ("finite", "in_image", "sampled", "pixel_xy", "membership"):
                same(saved[k], pixel[k], "prepared pixel membership " + k)
            memberships[arm] = saved
        require(
            {k: base.signature(x) for k, x in obs.items()} == before, "mask pixel recount mutation"
        )
        old = votes.load_npz(
            payload(VOTE_ROOT, vote_members, f"{name}/{view}/native_{native}.npz", evidence)
        )
        same(old["initial_mask_index"], origin, "vote group identity")
        require(
            int(old["native_frame"]) == native and len(old["row_id"]) == groups * 11,
            "vote row scope",
        )
        gdata = context["geometry"][(t, view)]
        results, local_index = [], 0
        for g in range(groups):
            for comparison, subset, arm, chosen in votes.selections(metrics, t):
                selected = chosen & (origin == g)
                same(selected, old["selected"][local_index], "saved vote subset bits")
                row_id = int(old["row_id"][local_index])
                old_row = saved_rows[row_id]
                require(
                    (
                        old_row["row_id"],
                        old_row["window"],
                        old_row["view"],
                        old_row["native_frame"],
                        old_row["initial_mask_row"],
                        old_row["comparison"],
                        old_row["subset"],
                        old_row["arm"],
                    )
                    == (row_id, name, view, native, g, comparison, subset, arm),
                    "all vote row mapping",
                )
                result = arithmetic.score_votes(
                    gdata["coverage"],
                    gdata["complete"],
                    memberships[arm]["membership"],
                    selected,
                    context["initial_links"][view][g],
                    gdata["collision"],
                    available=available,
                )
                for key in ("votes", "single_votes"):
                    same(result[key], old[key][local_index], "saved vote count " + key)
                require(
                    result["selected_count"] == old_row["selected_count"]
                    and (result["single_count"] if available else None)
                    == old_row["single_membership_count"],
                    "vote denominator differs",
                )
                meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                joined.append(
                    dict(
                        row_id=row_id,
                        window=name,
                        view=view,
                        native_frame=native,
                        relative_frame=t,
                        initial_mask_row=g,
                        comparison=comparison,
                        subset=subset,
                        arm=arm,
                        linked=context["initial_links"][view][g].tolist(),
                        scores=result["scores"].tolist(),
                        available_body=result["available_body"].tolist(),
                        maximum_ties=result["maximum_ties"].tolist(),
                        **meta,
                    )
                )
                results.append(result)
                if comparison in ARMS[1:] and subset in ("ct_only", "lk_only"):
                    key = (comparison, subset, t)
                    bucket = context.setdefault("availability", {}).setdefault(
                        key, np.zeros((3, 3), bool)
                    )
                    bucket[VIEWS.index(view)] |= result["available_body"]
                local_index += 1
        require(local_index == len(old["row_id"]), "lost vote rows")
        arrays = {
            k: np.stack([r[k] for r in results])
            for k, v in results[0].items()
            if isinstance(v, np.ndarray)
        }
        arrays["row_id"] = old["row_id"].copy()
        output.write(f"{name}/scores/{native}_{view}.npz", arrays)


def cases(context):
    rows = []
    for t in range(64):
        direct = np.stack([context["geometry"][(t, v)]["direct"] for v in VIEWS])
        base_row = [
            dict(
                window=context["spec"]["name"],
                native_frame=context["spec"]["first"] + t,
                relative_frame=t,
                body=b,
                identity_review=bool(context["candidate"]["identity_review"][t, b]),
            )
            for b in range(3)
        ]
        for comparison in ARMS[1:]:
            for subset in ("ct_only", "lk_only"):
                value = arithmetic.availability_cases(
                    context["availability"][(comparison, subset, t)],
                    direct,
                    context["candidate"]["observed"][t],
                    initial_frame=t == 0,
                )
                for b, r in enumerate(value):
                    base_row[b][comparison + "__" + subset] = r
        rows.extend(base_row)
    return rows


def summaries(rows, body_rows):
    out = {}
    for spec in votes.DATASETS:
        name = spec["name"]
        local = [r for r in body_rows if r["window"] == name and r["relative_frame"] > 0]
        record = dict(requested_body_times=189, comparisons={})
        for ct in ARMS[1:]:
            for subset in ("ct_only", "lk_only"):
                key = ct + "__" + subset
                record["comparisons"][key] = {
                    field: sum(int(r[key][field]) for r in local)
                    for field in ("case_A", "case_B", "case_A_and_B", "case_A_or_B")
                }
        record["views"] = {}
        for view in VIEWS:
            vr = [
                r
                for r in rows
                if r["window"] == name and r["view"] == view and r["relative_frame"] > 0
            ]
            stats = {}
            for ct in ARMS[1:]:
                own = {
                    (r["native_frame"], r["initial_mask_row"], r["arm"]): r
                    for r in vr
                    if r["comparison"] == ct and r["subset"] == "common"
                }
                delta, linked_delta = [], []
                for key, r in own.items():
                    if key[2] != ct:
                        continue
                    old = own[(key[0], key[1], "lk")]
                    require(r["selected_count"] == old["selected_count"], "common count drift")
                    a, b = np.asarray(r["scores"]), np.asarray(old["scores"])
                    finite = np.isfinite(a) & np.isfinite(b)
                    delta.extend((a - b)[finite].tolist())
                    require(r["linked"] == old["linked"], "common initial link drift")
                    linked_delta.extend((a - b)[finite & np.asarray(r["linked"], bool)].tolist())
                stats[ct] = dict(
                    finite_common_body_score_differences=len(delta),
                    median_CT_minus_LK=float(np.median(delta)) if delta else None,
                    all_body_score_scope="all3 including unlinked competitors; descriptive",
                    strict_initial_link_common_score_count=len(linked_delta),
                    strict_initial_link_median_CT_minus_LK=(
                        float(np.median(linked_delta)) if linked_delta else None
                    ),
                    independent_encounters=False,
                )
            record["views"][view] = stats
        out[name] = record
    return out


def plots(body_rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for spec in votes.DATASETS:
        fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        for b, ax in enumerate(axes):
            rows = [r for r in body_rows if r["window"] == spec["name"] and r["body"] == b]
            for key, color, offset in (("case_A", "tab:blue", 0), ("case_B", "tab:orange", 0.05)):
                ax.step(
                    [r["native_frame"] for r in rows],
                    [int(r["ct_first__ct_only"][key]) + offset for r in rows],
                    where="mid",
                    label="CT-first " + key,
                    color=color,
                )
            ax.set_ylim(-0.1, 1.2)
            ax.set_ylabel(f"Body {b}\navailability")
            ax.grid(alpha=0.2)
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("Native frame (frame0 excluded from temporal counts)")
        fig.suptitle(spec["name"] + ": conditional geometry, not identity truth")
        fig.tight_layout()
        stream = io.BytesIO()
        fig.savefig(stream, format="png", dpi=130)
        plt.close(fig)
        output.write("figures/" + spec["name"] + ".png", stream.getvalue())


def retry_binding(evidence):
    """Prior failure is provenance, never numerical input or a partial success."""
    root = WORK / "temporal_refractive_geometry_diagnosis_v1"
    seal_sha = "a6b6addc9ae39252184cb566f5644ff130d5a372c7ab9d41c78b8aa36a1fb9aa"
    evidence.add(root / "diagnosis.provenance.json", seal_sha)
    failed = base.read_json(root / "diagnosis.provenance.json")
    source = WORK / "diagnose_temporal_refractive_geometry.py"
    require(
        failed["status"] == "failed"
        and failed["stage"] == "temporal_refractive_geometry_diagnosis_v1"
        and failed["error"] == "registered refractive context differs"
        and input_sha(failed, source) == PINS[source]
        and not any(Path(p).name == "candidate.npz" for p in failed["inputs"]),
        "fixed pre-candidate V1 failure evidence differs",
    )
    return dict(
        failed_stage=failed["stage"],
        failed_seal_sha256=seal_sha,
        failed_source_sha256=PINS[source],
        prior_partial_is_success=False,
        correction="frozen camera ndarray dtype/shape/SHA serializer, not array.tolist",
        same_candidate_calibration_and_numerical_policy=True,
        full_two_window_reexecution_planned=True,
        prior_numerical_outputs_reused=False,
    )


def run(destination):
    destination = Path(destination).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    require(PLAN_SHA != "TO_BE_FROZEN", "reviewed plan pin required")
    evidence, output = base.Evidence(), base.Output(destination)
    started, clock = base.utc(), time.perf_counter()
    captured, timings = {}, {}
    try:
        evidence.add(PLAN, PLAN_SHA)
        source_snapshot(evidence, output, captured)
        write_json(output, "retry.json", retry_binding(evidence))
        write_json(
            output,
            "plan.json",
            dict(
                schema=STAGE,
                plan_sha256=PLAN_SHA,
                datasets=votes.DATASETS,
                references=REFERENCES,
                primary="LK vs CT-first",
                supplementary="CT-final",
                score="sum(single votes/all selected * full33 centerline mask coverage)",
                frames_per_window=64,
                initial_groups=23,
                current_masks=1314,
                future_reference_horizons=[76360, 333719],
                model_calls=0,
                GPU=False,
                physical_ID_truth=False,
                expected_initial_node_view_slots=54,
                expected_initial_node_sets=18,
            ),
        )
        vs, vm = header(VOTE_ROOT, "diagnosis.provenance.json", votes.STAGE, VOTE_SHA, evidence)
        vsummary = json_payload(VOTE_ROOT, vm, "summary.json", evidence)
        require(
            vsummary["total_rows"] == 16192 and vsummary["source_membership_recount_exact"] is True,
            "completed all-group votes required",
        )
        saved_rows = json_payload(VOTE_ROOT, vm, "group_time_rows.json", evidence)
        require(
            len(saved_rows) == 16192 and [r["row_id"] for r in saved_rows] == list(range(16192)),
            "ordered vote rows",
        )
        contexts = []
        stage = time.perf_counter()
        for spec, ref in zip(votes.DATASETS, REFERENCES, strict=True):
            require(
                input_sha(vs, WORK / spec["root"] / "comparison.provenance.json") == spec["seal"],
                "vote/comparison edge",
            )
            contexts.append(load_context(spec, ref, evidence))
        require(
            contexts[0]["shared"] == contexts[1]["shared"], "cross-window calibration and SL differ"
        )
        for context in contexts:
            for path, digest in captured.items():
                for seal in (context["prep_seal"], context["reference_seal"], context["seal"]):
                    if path in seal["inputs"]:
                        require(input_sha(seal, path) == digest, "shared historical source drift")
        timings["input_validation_and_selected_array_load_seconds"] = time.perf_counter() - stage
        initial, node_rows, node_sets, mask_count = [], [], [], 0
        stage = time.perf_counter()
        for context in contexts:
            mask_count += geometry_pass(context, evidence, output, initial, node_rows, node_sets)
            print(
                base.json_bytes(
                    dict(event="geometry_saved", window=context["spec"]["name"])
                ).decode(),
                flush=True,
            )
        require(
            (mask_count, len(initial), len(node_rows), len(node_sets)) == (1314, 69, 54, 18),
            "geometry denominators differ",
        )
        write_json(output, "initial_group_body_links.json", initial)
        write_json(output, "initial_node_query_qualification.json", node_rows)
        write_json(output, "initial_node_sets.json", node_sets)
        timings["geometry_support_and_save_including_SHA_seconds"] = time.perf_counter() - stage
        # Only now join stored arm values to the already-saved geometry.
        stage = time.perf_counter()
        joined = []
        for context in contexts:
            for view in VIEWS:
                join_view(context, view, evidence, output, saved_rows, vm, joined)
                print(
                    base.json_bytes(
                        dict(event="votes_joined", window=context["spec"]["name"], view=view)
                    ).decode(),
                    flush=True,
                )
        require(
            len(joined) == 16192 and [r["row_id"] for r in joined] == list(range(16192)),
            "joined full row order",
        )
        body_rows = [r for c in contexts for r in cases(c)]
        require(len(body_rows) == 384, "all body-times required")
        write_json(output, "group_body_scores.json", joined)
        write_json(output, "body_time_availability.json", body_rows)
        timings["membership_recount_join_save_including_SHA_seconds"] = time.perf_counter() - stage
        plots(body_rows, output)
        source_snapshot(evidence, output, captured)
        require(
            not any(k.startswith("cotracker") for k in sys.modules),
            "temporal model unexpectedly imported",
        )
        torch = sys.modules.get("torch")
        require(torch is None or not torch.cuda.is_initialized(), "CUDA unexpectedly initialized")
        summary = dict(
            schema=STAGE,
            status="ok",
            initial_groups=23,
            initial_group_body_cells=69,
            current_masks=1314,
            body_view_times=1152,
            projected_body_points=38016,
            individual_mask_body_cells=3942,
            group_score_rows=len(joined),
            body_score_cells=48576,
            body_times=384,
            later_body_times=378,
            initial_node_view_slots=len(node_rows),
            initial_node_sets=len(node_sets),
            source_support_frames_verified=128,
            prepared_pixel_memberships_verified=1152,
            all_saved_vote_counts_exact=True,
            statistics=summaries(joined, body_rows),
            timings=timings,
            runtime_before_final_guard_seconds=time.perf_counter() - clock,
            saved_only=True,
            model_calls=0,
            GPU=False,
            raw_image_reads=0,
            fit_calls=0,
            physical_ID_truth=False,
            ID_or_review_changed=False,
            limitations=[
                "retrospective fixed references and optimistic C34 seed",
                "shared D2/SL/LK evidence is not independent truth",
                "centerline coverage is not anatomical silhouette likelihood",
                "reflection and initial ambiguity retained; no200 run",
            ],
        )
        write_json(output, "summary.json", summary)
        evidence.final_guard()
        output.final_guard()
        seal = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records.copy(),
        )
    except BaseException as exc:
        output.write(
            "diagnosis.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=base.utc(),
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=evidence.paths,
                outputs=output.records.copy(),
            ),
        )
        raise
    output.write("diagnosis.provenance.json", seal)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute", action="store_true", help="execute reviewed saved-array CPU diagnostic"
    )
    args = parser.parse_args()
    require(args.execute, "explicit reviewed execution required")
    run(args.output)


if __name__ == "__main__":
    main()
