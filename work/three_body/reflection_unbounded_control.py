"""Post-hoc algebraic control on saved reflection alpha; no rays/images/fit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path[:0] = [str(LAB), str(REPO / "src")]
from prepare_sequence import decode_frame_masks, sha256  # noqa: E402
from reflection_probe import FRAMES, combine_branches  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

SOURCE = REPO / "work/three_body/reflection_probe_13frames_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")
NAMES = {
    "finite_direct": "inherited_unbounded",
    "inner_unit": "unbounded_inner_unit",
    "outer_unit": "unbounded_outer_unit",
    "both_unit": "unbounded_both_unit",
    "inner_fresnel": "unbounded_inner_fresnel",
    "outer_fresnel": "unbounded_outer_fresnel",
    "both_fresnel": "unbounded_both_fresnel",
}


def npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def prepare_plan():
    seal_path = SOURCE / "probe.provenance.json"
    seal = json.loads(seal_path.read_text())
    if seal["status"] != "ok":
        raise ValueError("primary probe is incomplete")
    hashes = {str(seal_path): sha256(seal_path)}
    for entry in seal["outputs"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError("primary probe output was modified")
        hashes[entry["path"]] = entry["sha256"]
    primary = json.loads((SOURCE / "plan.json").read_text())
    run = Path(primary["run"])
    provenance_path = run / "experiment.provenance.json"
    if sha256(provenance_path) != primary["input_hashes"][str(provenance_path)]:
        raise ValueError("geometry provenance is not bound to the primary probe")
    producer = json.loads(provenance_path.read_text())
    sequence = Path(producer["params"]["sequence"]).resolve(strict=True)
    manifest_path = sequence / "manifest.json"
    if sha256(manifest_path) != primary["input_hashes"][str(manifest_path)]:
        raise ValueError("selector manifest differs from the primary probe")
    manifest = json.loads(manifest_path.read_text())
    selectors = {}
    for view in VIEWS:
        entry = manifest["masks"][view]
        path = (sequence / entry["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(sequence)
            or sha256(path) != entry["sha256"]
            or entry["sha256"] != primary["input_hashes"][str(path)]
        ):
            raise ValueError("D2 selector is not the unchanged primary input")
        selectors[view] = str(path)
        hashes[str(path)] = entry["sha256"]
    hashes.update(
        {str(manifest_path): sha256(manifest_path), str(provenance_path): sha256(provenance_path)}
    )
    code_paths = [
        Path(__file__).resolve(),
        LAB / "reflection_probe.py",
        LAB / "prepare_sequence.py",
        REPO / "src/medaka3d/provenance.py",
    ]
    code = {str(p): sha256(p) for p in code_paths}
    return dict(
        schema="posthoc_inherited_unbounded_reflection_control_v1",
        frames=list(FRAMES),
        primary_probe=str(SOURCE),
        input_hashes=hashes,
        code_hashes=code,
        selectors=selectors,
        formula="Use saved unbounded_alpha for direct; unchanged finite-valid inner/outer "
        "alpha and weights; U=1-product_(body,branch)(1-weight*alpha).",
        contrasts=list(NAMES.values()),
        primary_frame=79381,
        primary_view="top_cam",
        primary_instance_row=3,
        geometry_fitting=False,
        ray_or_gaussian_rendering=False,
        raw_bgr_reading=False,
        gt_reading=False,
        selector_role="unchanged D2/core pixel labels only",
        evaluation="same saved fit-excluded foreground/ring/global targets and fixed weights; "
        "same calibrated-ray denominator, all13frames/all3views/allinstances",
        limitation="Post-hoc opened development diagnostic. Inherited unbounded direct crosses "
        "nominal walls; combined scene is not fully physical. No ID or generalization conclusion.",
        packages=dict(python=sys.version, numpy=np.__version__, opencv=cv2.__version__),
    )


def control(saved):
    values = combine_branches(
        saved["unbounded_alpha"],
        saved["inner_alpha"],
        saved["outer_alpha"],
        saved["inner_weight"],
        saved["outer_weight"],
    )
    result = {NAMES[key]: value for key, value in values.items()}
    if not np.allclose(
        result["inherited_unbounded"], saved["unbounded_direct"], rtol=0, atol=1e-12
    ):
        raise AssertionError("saved unbounded baseline changed")
    if np.any(result["unbounded_both_fresnel"] + 1e-14 < result["inherited_unbounded"]) or np.any(
        result["unbounded_both_fresnel"] > result["unbounded_both_unit"] + 1e-14
    ):
        raise AssertionError("branch response bound failed")
    return result


def region(values, view_index, selected):
    n = int(selected.sum())
    mass = {k: float(value[view_index, selected].sum()) for k, value in values.items()}
    return dict(
        pixels=n,
        mass=mass,
        mean={k: x / n if n else None for k, x in mass.items()},
        added_mass={k: x - mass["inherited_unbounded"] for k, x in mass.items()},
    )


def compute(plan, output, record):
    primary = json.loads((SOURCE / "summary.json").read_text())
    reference = {r["frame_idx"]: r for r in primary["frame_results"]}
    masks = {v: npz(path) for v, path in plan["selectors"].items()}
    if any(
        not np.array_equal(data["frame_idx"], np.arange(76620, 86580)) for data in masks.values()
    ):
        raise ValueError("selector native frame range differs")
    rows, all_scores, csv_rows = [], [], []
    for frame in FRAMES:
        sample = npz(SOURCE / f"f{frame}_pixels.npz")
        saved = npz(SOURCE / f"f{frame}_evaluation_response.npz")
        values = control(saved)
        scores, stratum_mass = {}, {}
        for mode, union in values.items():
            error = (union - sample["eval_target"]) ** 2 * sample["eval_weights"]
            error *= saved["calibrated_valid"]
            parts = np.stack(
                [np.sum(error * (sample["eval_strata"] == k), axis=-1) for k in range(3)], axis=-1
            )
            scores[mode] = parts.tolist()
            for v, view in enumerate(VIEWS):
                csv_rows.append(
                    dict(
                        frame_idx=frame,
                        view=view,
                        mode=mode,
                        foreground_loss=parts[v, 0],
                        ring_loss=parts[v, 1],
                        global_background_loss=parts[v, 2],
                        total=parts[v].sum(),
                    )
                )
        for v, view in enumerate(VIEWS):
            stratum_mass[view] = {
                str(k): region(values, v, sample["eval_strata"][v] == k) for k in range(3)
            }
        path = output / f"f{frame}_evaluation_control.npz"
        np.savez_compressed(path, frame_idx=np.array(frame), **values)
        record.output(path.name, path, sha256=sha256(path))
        foreground = npz(SOURCE / f"f{frame}_foreground_response.npz")
        dense = control(foreground)
        regions = {}
        for v, view in enumerate(VIEWS):
            uv = foreground[view + "_pixels"].astype(int)
            n = len(uv)
            original_mask = decode_frame_masks(masks[view], frame - 76620)
            selected = np.arange(dense["inherited_unbounded"].shape[1]) < n
            rows_for_view = dict(union=region(dense, v, selected), instances=[], cores=[])
            for i, mask in enumerate(original_mask):
                active = np.zeros_like(selected)
                active[:n] = mask[uv[:, 1], uv[:, 0]]
                row = region(dense, v, active)
                expected = reference[frame]["foreground"][view]["instances"][i]
                if row["pixels"] != expected["pixels"]:
                    raise AssertionError("primary instance denominator changed")
                rows_for_view["instances"].append(dict(instance_row=i, **row))
                core = cv2.erode(
                    mask.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                    borderType=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(bool)
                active[:n] = core[uv[:, 1], uv[:, 0]]
                row = region(dense, v, active)
                if row["pixels"] != reference[frame]["core"][view][i]["pixels"]:
                    raise AssertionError("primary core denominator changed")
                rows_for_view["cores"].append(dict(instance_row=i, **row))
            regions[view] = rows_for_view
        rows.append(
            dict(
                frame_idx=frame,
                scores_by_view_stratum=scores,
                eval_region_response=stratum_mass,
                foreground=regions,
            )
        )
        all_scores.append(scores)
        path = output / f"f{frame}_foreground_control.npz"
        np.savez_compressed(path, frame_idx=np.array(frame), **dense)
        record.output(path.name, path, sha256=sha256(path))
    summaries = {}
    for mode in NAMES.values():
        values = np.array([row[mode] for row in all_scores])
        summaries[mode] = dict(
            mean=float(values.sum(-1).mean()),
            by_view=values.sum(-1).mean(0).tolist(),
            by_stratum=values.mean((0, 1)).tolist(),
        )
    result = dict(
        frame_results=rows,
        aggregate=summaries,
        no_rerender=True,
        no_raw_bgr_read=True,
        no_fit=True,
        no_gt=True,
        posthoc_development_only=True,
        inherited_unbounded_direct_not_fully_physical=True,
    )
    path = output / "summary.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    record.output(path.name, path, sha256=sha256(path))
    path = output / "evaluation.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    record.output(path.name, path, sha256=sha256(path))
    print(json.dumps(summaries), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(REPO):
        raise ValueError("output must be new and inside the workspace")
    plan = prepare_plan()
    hashes = {**plan["input_hashes"], **plan["code_hashes"]}
    if not args.register_only:
        if args.registration is None:
            raise ValueError("sealed pre-computation plan required")
        root = args.registration.resolve(strict=True)
        seal_path, plan_path = root / "registration.provenance.json", root / "plan.json"
        seal = json.loads(seal_path.read_text())
        if (
            seal["status"] != "ok"
            or sha256(plan_path) != seal["outputs"]["plan"]["sha256"]
            or json.loads(plan_path.read_text()) != plan
        ):
            raise ValueError("registered plan differs")
        hashes.update({str(seal_path): sha256(seal_path), str(plan_path): sha256(plan_path)})
    output.mkdir(parents=True)
    name = "registration" if args.register_only else "control"
    with provenance("posthoc_unbounded_reflection_" + name, output, name=name) as record:
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        for source, digest in plan["code_hashes"].items():
            target = output / "code_snapshot" / Path(source).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(source).read_bytes())
            record.output(target.relative_to(output).as_posix(), target, sha256=digest)
        path = output / "plan.json"
        path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
        record.output("plan", path, sha256=sha256(path))
        if not args.register_only:
            compute(plan, output, record)
        for path, digest in hashes.items():
            if sha256(path) != digest:
                raise AssertionError("registered input/source changed")
        for entry in record.outputs.values():
            if sha256(entry["path"]) != entry["sha256"]:
                raise AssertionError("output/snapshot changed")


if __name__ == "__main__":
    main()
