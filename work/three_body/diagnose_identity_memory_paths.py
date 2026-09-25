"""Audit observed-node reassignment and chosen-path costs on frozen minute geometry."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import shutil
from pathlib import Path

import compare_identity_memory_minute as joined
import numpy as np

from medaka3d.provenance import provenance

REPO = joined.REPO
PERMUTATIONS = np.array(list(itertools.permutations(range(3))), dtype=np.int64)


def sequential_links(geometry, mapping, scale, weight):
    """Recompute direct-event motion from the chosen mapping, without the DP graph."""
    frames, observed, points = geometry["frame_idx"], geometry["observed"], geometry["pose_xyz"]
    if not observed[0, mapping[0], 1].all():
        raise ValueError("first direct observations are required")
    n = len(frames)
    previous_frame = np.full((n, 3), -1, dtype=np.int64)
    previous_slot = np.full((n, 3), -1, dtype=np.int64)
    gap = np.zeros((n, 3), dtype=np.int64)
    displacement = np.full((n, 3, 3), np.nan)
    cost = np.zeros((n, 3), dtype=np.float64)
    applied = np.zeros((n, 3), dtype=bool)
    last_frame = np.full(3, frames[0], dtype=np.int64)
    last_slot = mapping[0].copy()
    last_point = points[0, last_slot, 1].copy()
    for t in range(1, n):
        previous_frame[t] = last_frame
        previous_slot[t] = last_slot
        for fish, slot in enumerate(mapping[t]):
            if not observed[t, slot, 1]:
                continue
            dt = frames[t] - last_frame[fish]
            dx = points[t, slot, 1] - last_point[fish]
            previous_frame[t, fish] = last_frame[fish]
            previous_slot[t, fish] = last_slot[fish]
            gap[t, fish], displacement[t, fish] = dt, dx
            cost[t, fish] = weight / 3 * (
                np.sum(dx * dx) / (scale * scale * dt) + 3 * np.log(dt)
            )
            applied[t, fish] = True
            last_frame[fish], last_slot[fish] = frames[t], slot
            last_point[fish] = points[t, slot, 1]
    return dict(
        previous_direct_frame=previous_frame, previous_direct_raw_slot=previous_slot,
        gap_frames_when_reobserved=gap, displacement_mm_when_reobserved=displacement,
        motion_cost_applied=applied, brownian_motion_cost_by_id=cost,
    )


def path_costs(geometry, mapping, unary, config):
    matched = np.all(mapping[:, None] == PERMUTATIONS[None], axis=2)
    if not np.all(matched.sum(1) == 1) or not np.array_equal(mapping[0], [0, 1, 2]):
        raise ValueError("chosen paths must contain complete permutations and fixed first IDs")
    state = matched.argmax(1)
    links = sequential_links(
        geometry, mapping, config["motion_scale_mm"], config["motion_weight"]
    )
    color = unary[np.arange(len(mapping)), state]
    motion = links["brownian_motion_cost_by_id"].sum(1)
    return links, dict(
        path_state=state, motion_by_frame=motion, color_by_frame=color,
        joint_by_frame=motion + color,
    )


def reassignment_rows(geometry, mapping):
    """A raw supported node moves only if its raw slot maps to a different ID."""
    rows = []
    for t in np.flatnonzero(np.any(mapping != [0, 1, 2], axis=1)):
        for slot in np.flatnonzero(mapping[t] != [0, 1, 2]):
            new_id = int(np.flatnonzero(mapping[t] == slot)[0])
            support = geometry["observed"][t, slot]
            sources = geometry["sources"][t, slot]
            rows.append(dict(
                frame=int(geometry["frame_idx"][t]), raw_slot=int(slot),
                fixed_identity=int(slot), proposed_identity=new_id,
                observed_nodes_mouth_pectoral_tail=support.tolist(),
                supported_view_counts=(sources >= 0).sum(0).tolist(),
                source_rows_view_by_node=sources.tolist(),
                directly_observed_nodes_reassigned=int(support.sum()),
                latent_mapping_only=not bool(support.any()),
                prior_identity_review=bool(geometry["identity_review"][t, slot]),
            ))
    return rows


def summarize_reassignment(rows):
    return dict(
        nonidentity_frames=len({r["frame"] for r in rows}),
        affected_raw_fish_frames=len(rows),
        supported_node_reassignments=sum(r["directly_observed_nodes_reassigned"] for r in rows),
        supported_node_reassignments_by_node=(
            np.array([r["observed_nodes_mouth_pectoral_tail"] for r in rows], dtype=bool)
            .reshape(-1, 3).sum(0).tolist()
        ),
        frames_reassigning_any_observed_node=sorted({
            r["frame"] for r in rows if not r["latent_mapping_only"]
        }),
        latent_only_affected_fish_frames=sum(r["latent_mapping_only"] for r in rows),
        biological_identity_verdict="unverified",
    )


def link_rows(geometry, mapping, links, fixed_links):
    n = len(mapping)
    changed = np.any(mapping != [0, 1, 2], axis=1)[:, None].repeat(3, axis=1)
    for key in ("previous_direct_frame", "previous_direct_raw_slot", "motion_cost_applied"):
        changed |= links[key] != fixed_links[key]
    changed |= ~np.isclose(links["brownian_motion_cost_by_id"],
                          fixed_links["brownian_motion_cost_by_id"], rtol=1e-12, atol=1e-12)
    result = []
    for t, fish in zip(*np.nonzero(changed), strict=True):
        if not 0 <= t < n:
            raise ValueError("invalid native offset")
        item = dict(frame=int(geometry["frame_idx"][t]), identity=int(fish),
                    current_raw_slot=int(mapping[t, fish]))
        for prefix, value in (("path", links), ("fixed", fixed_links)):
            applied = bool(value["motion_cost_applied"][t, fish])
            item[prefix] = dict(
                previous_frame=int(value["previous_direct_frame"][t, fish]),
                previous_raw_slot=int(value["previous_direct_raw_slot"][t, fish]),
                current_direct_pectoral=applied,
                elapsed_native_frames=int(value["gap_frames_when_reobserved"][t, fish]),
                displacement_mm=(value["displacement_mm_when_reobserved"][t, fish].tolist()
                                 if applied else None),
                motion_cost=float(value["brownian_motion_cost_by_id"][t, fish]),
            )
        result.append(item)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists():
        raise FileExistsError("a new workspace output directory is required")
    output.mkdir(parents=True)
    with provenance(
        "memory_chosen_path_observation_diagnostic", output, name="diagnostic"
    ) as record:
        record.params_from(args)
        own_code = {Path(__file__).resolve(), Path(joined.__file__).resolve(),
                    joined.LAB / "identity_output.py", joined.LAB / "run_experiment.py",
                    REPO / "src/medaka3d/provenance.py"}
        hashes = {str(p): joined.base.sha256(p) for p in own_code}
        root = joined.local_path(args.joined)
        seal = joined.verify_record(
            root, "comparison.provenance.json", "sealed_memory_minute_comparison_join",
            {"summary": "summary.json", "path_differences": "path_differences.npz"}, hashes,
        )
        if seal["params"].get("final_inputs_sources_outputs_unchanged") is not True:
            raise ValueError("joined comparison did not seal final evidence")
        producer_root = joined.local_path(seal["params"]["run"])
        evaluation_root = joined.local_path(seal["params"]["comparison"])

        def bound(path):
            path = joined.local_path(path)
            if str(path) not in hashes:
                raise ValueError("diagnostic input was not bound by the completed join")
            return path

        producer = joined.read_json(bound(producer_root / "experiment.provenance.json"))
        summary = joined.read_json(bound(producer_root / "summary.json"))
        comparison = joined.read_json(bound(evaluation_root / "comparison.json"))
        geometry_root = joined.local_path(producer["params"]["geometry_run"])
        geometry = joined.npz(bound(geometry_root / "candidate.npz"))
        joined.exact_array(geometry["frame_idx"], joined.FRAMES, "native minute")
        scores = joined.npz(bound(producer_root / "scores.npz"))
        joint = joined.npz(bound(producer_root / "identity.npz"))
        motion = joined.npz(bound(evaluation_root / "motion_only_identity.npz"))
        for value in (joint, motion):
            joined.check_export(geometry, value)
        mappings = dict(
            fixed=np.tile(np.arange(3), (len(joined.FRAMES), 1)),
            joint=joint["identity_to_raw_slot"], motion_only=motion["identity_to_raw_slot"],
        )
        paths, all_arrays = {}, {"frame_idx": joined.FRAMES}
        for name, mapping in mappings.items():
            links, values = path_costs(geometry, mapping, scores["unary"], summary["config"])
            rows = reassignment_rows(geometry, mapping)
            paths[name] = dict(
                links=links, values=values, observed_reassignment_rows=rows,
                observations=summarize_reassignment(rows),
                costs={key: float(values[key + "_by_frame"].sum())
                       for key in ("motion", "color", "joint")},
            )
            for key, value in {**values, **links, "mapping": mapping}.items():
                all_arrays[name + "_" + key] = value
        for name, location, filename in (
            ("joint", producer_root, "memory_links.npz"),
            ("motion_only", evaluation_root, "motion_only_links.npz"),
        ):
            saved = joined.npz(bound(location / filename))
            for key, value in paths[name]["links"].items():
                if value.dtype.kind == "f":
                    if not np.allclose(saved[key], value, rtol=1e-12, atol=1e-12, equal_nan=True):
                        raise ValueError("independent sequential link differs: " + name + "/" + key)
                else:
                    joined.exact_array(saved[key], value, name + "/" + key)
        for name, objective, expected in (
            ("joint", "joint", summary["dp"]["path_cost"]),
            ("motion_only", "motion", comparison["motion_only_reference"]["dp"]["path_cost"]),
        ):
            if not np.isclose(paths[name]["costs"][objective], expected, rtol=1e-11, atol=1e-10):
                raise ValueError("independent path total differs from the sealed objective")
        report = dict(
            frames=[76620, 86580], path_objectives_recomputed_without_search=True,
            registered_config=summary["config"],
            node_order=["mouth", "pectoral", "tail"],
            mapping_convention="identity_to_raw_slot[identity]=raw geometry slot",
            paths={},
            limitations=[
                "Direct support validates a detector-to-geometry association; "
                "biological identity remains unverified.",
                "Raw S0/S1/S2 are geometry slots, not independently verified fish labels.",
                "Motion-only color is evaluated retrospectively; "
                "it was absent from that optimization.",
                "Costs are uncalibrated model values, not likelihood confidence or ID correctness.",
                "No GT labels are parsed and no images are decoded or optimized here.",
                "No review is cleared and no geometry is modified.",
            ],
        )
        for name, path in paths.items():
            deltas = {key: path["costs"][key] - paths["fixed"]["costs"][key]
                      for key in ("motion", "color", "joint")}
            changed_cost_frames = np.flatnonzero(
                ~np.isclose(path["values"]["joint_by_frame"],
                            paths["fixed"]["values"]["joint_by_frame"],
                            rtol=1e-12, atol=1e-12)
            )
            report["paths"][name] = dict(
                observations=path["observations"], costs=path["costs"],
                cost_delta_from_fixed=deltas,
                observed_reassignment_rows=path["observed_reassignment_rows"],
                changed_links=link_rows(
                    geometry, mappings[name], path["links"], paths["fixed"]["links"]
                ),
                changed_joint_cost_native_frames=joined.FRAMES[changed_cost_frames].tolist(),
                changed_cost_frames=[dict(
                    frame=int(joined.FRAMES[t]),
                    **{key: float(path["values"][key + "_by_frame"][t])
                       for key in ("motion", "color", "joint")},
                    fixed={key: float(paths["fixed"]["values"][key + "_by_frame"][t])
                           for key in ("motion", "color", "joint")},
                ) for t in changed_cost_frames],
            )
        rows_path = output / "node_reassignments.csv"
        with rows_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["path", "frame", "raw_slot", "fixed_id", "proposed_id", "mouth",
                             "pectoral", "tail", "latent_only", "prior_review"])
            for name, value in paths.items():
                for row in value["observed_reassignment_rows"]:
                    writer.writerow([name, row["frame"], row["raw_slot"], row["fixed_identity"],
                                     row["proposed_identity"],
                                     *row["observed_nodes_mouth_pectoral_tail"],
                                     row["latent_mapping_only"], row["prior_identity_review"]])
        record.output("nodes", rows_path, sha256=joined.base.sha256(rows_path))
        arrays_path = output / "path_costs_and_links.npz"
        np.savez_compressed(arrays_path, **all_arrays)
        record.output("arrays", arrays_path, sha256=joined.base.sha256(arrays_path))
        report_path = output / "summary.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", report_path, sha256=joined.base.sha256(report_path))
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for path in own_code:
            target = output / "code_snapshot" / path.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            record.output("code:" + str(path.relative_to(REPO)), target, sha256=hashes[str(path)])
        for path, digest in {
            **hashes, **{x["path"]: x["sha256"] for x in record.outputs.values()}
        }.items():
            if joined.base.sha256(path) != digest:
                raise RuntimeError("path diagnostic input/source/output changed")
        record.param("final_inputs_sources_outputs_unchanged", True)
    print(json.dumps(dict(output=str(output))), flush=True)
    return report


if __name__ == "__main__":
    main()
