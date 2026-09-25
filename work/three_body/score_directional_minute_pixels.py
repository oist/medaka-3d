"""Fixed uniform-time, spatially unused silhouette assay for the completed minute.

Preparation never reads pose values. Evaluation requires the successful full
minute comparison and checks regenerated fit targets against saved diagnostics.
These are the same development images, not temporal holdout or identity truth.
"""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import fit_chest_recovery_candidates as base
import numpy as np
import torch
from observations import sample_pixels
from prepare_sequence import decode_frame_masks
from renderer import RefractiveFishRenderer, RendererConfig
from score_temporal_disjoint_pixels import COUNTS, SEED_OFFSET, unused_samples

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks = base.evidence_tools
REPO = base.REPO
START, STOP, CHUNK = 66400, 76360, 1660
SCORED = np.arange(START + 83, STOP, 166, dtype=np.int64)
DIAGNOSTIC = np.r_[np.arange(66656, 66706), np.arange(75047, 75163)]
PREPARED = np.union1d(SCORED, DIAGNOSTIC)
PREP_STAGE = "directional_minute_unused_pixel_preparation"
MINUTE_CODE_SHA = "066f702d0c64ddf048fe5d9c65e6add64f92daf40025e90b50a698355e6707c3"


def require_member(root, name, hashes):
    path = (root / name).resolve()
    checks.require(str(path) in hashes, "required output not sealed: " + str(path))
    return path


def new_output(path):
    path = path.resolve()
    checks.require(
        path.is_relative_to(REPO / "work") and not path.exists(), "new work output required"
    )
    return path


def finish(record, output, hashes, code):
    checks.require(code == base.local_code(), "loaded local source set changed")
    for path, digest in hashes.items():
        checks.require(checks.sha(path) == digest, "bound file changed: " + path)
        record.input(path, path, sha256=digest)
    for path in output.iterdir():
        if path.is_file() and not path.name.endswith("provenance.json"):
            record.output(path.name, path, sha256=checks.sha(path))
    record.param("all_bound_files_unchanged", True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def fit_arrays(frame, config):
    targets = sample_pixels(
        frame,
        "detectron2",
        config["positive_samples"],
        config["near_samples"],
        config["global_samples"],
    )
    return dict(
        fit_pixels=np.stack([targets.pixels[v] for v in VIEWS]),
        fit_target=targets.target,
        fit_weights=targets.weights,
        fit_foreground=targets.foreground,
    )


def validate_diagnostic(prepared, diagnostic):
    for name in ("fit_target", "fit_weights", "fit_foreground"):
        checks.require(
            checks.equal_bytes(prepared[name], diagnostic[name]),
            "regenerated fit evidence differs: " + name,
        )
    for vi, view in enumerate(VIEWS):
        checks.require(
            checks.equal_bytes(prepared["fit_pixels"][vi], diagnostic[view + "_fit_pixels"]),
            "regenerated fitted pixels differ: " + view,
        )


def validate_preparation(prepared, seal, code, config):
    checks.require(
        prepared["scored_frames"] == SCORED.tolist()
        and prepared["prepared_frames"] == PREPARED.tolist()
        and prepared["diagnostic_frames"] == DIAGNOSTIC.tolist(),
        "sampling schedule changed",
    )
    checks.require(
        prepared["counts"] == list(COUNTS)
        and prepared["seed_offset"] == SEED_OFFSET
        and prepared["sampler_config"] == config
        and config["mode"] == "detectron2"
        and config["width_ratio"] == 0.06,
        "prepared sampler configuration changed",
    )
    for path, digest in code.items():
        checks.require(
            path in seal["inputs"] and seal["inputs"][path]["sha256"] == digest,
            "prepared implementation differs: " + path,
        )


def prepare(original, output):
    code = base.local_code()
    hashes = dict(code)
    seal = checks.verify(
        original, "continuation.provenance.json", "three_body_long_continuation", hashes
    )
    checks.require(seal["params"]["through_frame"] == STOP, "wrong original horizon")
    meta_path = require_member(original, "checkpoint/metadata.json", hashes)
    config = checks.read(meta_path)["context"]["tracker_config"]
    checks.require(
        config["mode"] == "detectron2" and config["width_ratio"] == 0.06,
        "unexpected renderer / sampling protocol",
    )
    binding = seal["params"]["bindings"]
    for path, digest in binding["code_sha256"].items():
        checks.require(checks.sha(path) == digest, "original source changed")
        hashes[path] = digest
    shared = binding["inputs"]["seed"]["SL_and_calibration_sha256"]
    calibration = [Path(p).parent for p in shared if Path(p).name == "calibration_manifest.json"]
    checks.require(len(calibration) == 1, "unique calibration required")
    for path, digest in shared.items():
        checks.require(checks.sha(path) == digest, "calibration / SL binding changed")
        hashes[path] = digest
    output.mkdir(parents=True)
    membership, mask_bindings = [], {}
    with provenance(PREP_STAGE, output, name="preparation") as record:
        for first in range(START, STOP, CHUNK):
            selected = PREPARED[(first <= PREPARED) & (first + CHUNK > PREPARED)]
            masks = {}
            for view in VIEWS:
                paths = [
                    Path(item["path"])
                    for item in seal["inputs"].values()
                    if f"f{first:06d}-{first + CHUNK:06d}" in item["path"]
                    and Path(item["path"]).name == view + ".npz"
                ]
                checks.require(len(paths) == 1, "unique consumed mask chunk required")
                path = paths[0]
                digest = seal["inputs"][str(path)]["sha256"]
                hashes[str(path)] = digest
                mask_bindings[str(path)] = digest
                masks[view] = base.load_arrays(path, hashes)
                checks.require(
                    np.array_equal(masks[view]["frame_idx"], np.arange(first, first + CHUNK)),
                    "mask native coverage differs",
                )
                checks.require(
                    np.array_equal(masks[view]["image_shape_hw"], [1200, 1200]),
                    "mask native dimensions differ",
                )
            for native in selected:
                values = {}
                for view, packed in masks.items():
                    index = int(native - first)
                    decoded = decode_frame_masks(packed, index)
                    checks.require(
                        hashlib.sha256(np.ascontiguousarray(decoded).tobytes()).hexdigest()
                        == bytes(packed["frame_mask_sha256"][index]).decode(),
                        "native decoded mask changed",
                    )
                    values[view] = decoded if packed["view_available"][index] else None
                frame = SimpleNamespace(
                    frame_idx=int(native),
                    masks=values,
                    images_bgr={v: SimpleNamespace(shape=(1200, 1200, 3)) for v in VIEWS},
                )
                arrays = fit_arrays(frame, config)
                if native in SCORED:
                    pixels, target, weights, overlap, audit = unused_samples(
                        frame, arrays["fit_pixels"]
                    )
                    checks.require(not overlap.any(), "fitted coordinate reused")
                    arrays.update(
                        pixels=pixels,
                        target=target,
                        weights=weights,
                        reused_fitted_coordinate=overlap,
                    )
                    membership.extend(audit)
                np.savez_compressed(output / f"frame_{native:06d}.npz", **arrays)
            print(
                json.dumps(dict(event="pixel_samples_prepared", through=first + CHUNK)), flush=True
            )
        summary = dict(
            status="ok",
            native_interval=[START, STOP],
            scored_frames=SCORED.tolist(),
            diagnostic_frames=DIAGNOSTIC.tolist(),
            prepared_frames=PREPARED.tolist(),
            sampler_config=config,
            counts=COUNTS,
            seed_offset=SEED_OFFSET,
            calibration=str(calibration[0]),
            source_mask_sha256=mask_bindings,
            native_decoded_mask_checks=len(PREPARED) * 3,
            fitted_coordinate_overlap=0,
            pose_values_read=False,
            temporal_holdout=False,
            identity_truth=False,
            original_seal_sha256=hashes[str(original / "continuation.provenance.json")],
        )
        write_json(output / "summary.json", summary)
        write_json(output / "membership.json", membership)
        finish(record, output, hashes, code)
    print(
        json.dumps(
            {k: summary[k] for k in ("status", "native_decoded_mask_checks", "pose_values_read")}
        ),
        flush=True,
    )


def evaluate(original, branch, comparison, prepared, original_replay, output):
    code = base.local_code()
    hashes = dict(code)
    cmp_seal = checks.verify(
        comparison, "comparison.provenance.json", "directional_one_minute_comparison", hashes
    )
    minute_code = str(Path(__file__).with_name("compare_directional_one_minute.py"))
    checks.require(
        cmp_seal["inputs"][minute_code]["sha256"] == MINUTE_CODE_SHA
        and checks.sha(minute_code) == MINUTE_CODE_SHA,
        "reviewed minute comparison implementation required",
    )
    hashes[minute_code] = MINUTE_CODE_SHA
    require_member(comparison, "summary.json", hashes)
    cmp_summary = checks.read(comparison / "summary.json")
    checks.require(
        cmp_summary["status"] == "ok"
        and cmp_summary["native_frames"] == 9960
        and cmp_summary["restores_during_one_minute"] == 0,
        "full minute comparison required",
    )
    prep_seal = checks.verify(prepared, "preparation.provenance.json", PREP_STAGE, hashes)
    require_member(prepared, "summary.json", hashes)
    prep = checks.read(prepared / "summary.json")
    for root, filename, stage in (
        (original, "continuation.provenance.json", "three_body_long_continuation"),
        (branch, "tracking.provenance.json", "directional_one_minute_continuous"),
    ):
        checks.verify(root, filename, stage, hashes)
        path = str(root / filename)
        checks.require(
            cmp_seal["inputs"][path]["sha256"] == hashes[path], "comparison trajectory differs"
        )
    metadata = checks.read(require_member(original, "checkpoint/metadata.json", hashes))
    validate_preparation(prep, prep_seal, code, metadata["context"]["tracker_config"])
    checks.require(
        prep["original_seal_sha256"] == hashes[str(original / "continuation.provenance.json")],
        "sample preparation reference differs",
    )
    branch_seal = checks.read(branch / "tracking.provenance.json")
    for path, digest in prep["source_mask_sha256"].items():
        checks.require(
            branch_seal["inputs"][path]["sha256"] == digest and checks.sha(path) == digest,
            "branch consumed different segmentation",
        )
        hashes[path] = digest
    replay = checks.verify(
        original_replay, "replay.provenance.json", "chest_deficit_original_beam_replay_v1", hashes
    )
    checks.require(
        replay["inputs"][str(original / "continuation.provenance.json")]["sha256"]
        == hashes[str(original / "continuation.provenance.json")],
        "diagnostic reference differs",
    )
    for native in DIAGNOSTIC:
        arrays = base.load_arrays(
            require_member(prepared, f"frame_{native:06d}.npz", hashes), hashes
        )
        for root in [branch, original_replay] if native < 66706 else [branch]:
            path = require_member(root, f"diagnostics/frame_{native:06d}/arrays.npz", hashes)
            validate_diagnostic(arrays, base.load_arrays(path, hashes))
    candidates = []
    for root in (original, branch):
        candidate = base.load_arrays(require_member(root, "candidate.npz", hashes), hashes)
        checks.require(
            np.array_equal(candidate["frame_idx"], np.arange(49800, STOP)),
            "candidate native horizon differs",
        )
        path = str(root / "candidate.npz")
        checks.require(
            cmp_seal["inputs"][path]["sha256"] == hashes[path], "comparison candidate differs"
        )
        candidates.append(candidate)
    calibration = Path(prep["calibration"])
    prep_seal = checks.read(prepared / "preparation.provenance.json")
    for path in calibration.iterdir():
        if path.is_file():
            digest = prep_seal["inputs"][str(path)]["sha256"]
            checks.require(checks.sha(path) == digest, "prepared calibration changed")
            hashes[str(path)] = digest
    rig = load_rig(calibration)
    torch.set_num_threads(2)
    rows = []
    output.mkdir(parents=True)
    with provenance(
        "directional_minute_unused_pixel_evaluation", output, name="evaluation"
    ) as record:
        for native in SCORED:
            arrays = base.load_arrays(
                require_member(prepared, f"frame_{native:06d}.npz", hashes), hashes
            )
            checks.require(
                not arrays["reused_fitted_coordinate"].any(), "fitted coordinate overlap"
            )
            poses = np.stack([candidate["pose_xyz"][native - 49800] for candidate in candidates])
            renderer = RefractiveFishRenderer(
                rig,
                {v: arrays["pixels"][vi] for vi, v in enumerate(VIEWS)},
                RendererConfig(width_ratio=0.06),
                device="cpu",
                dtype=torch.float64,
            )
            with torch.no_grad():
                alpha = renderer.render(poses).union_alpha.numpy()
            valid = renderer.ray_valid.numpy()
            error = ((alpha - arrays["target"]) ** 2 * arrays["weights"] * valid).sum(-1)
            rows.append(
                dict(
                    frame_idx=int(native),
                    original_error=error[0].tolist(),
                    directional_error=error[1].tolist(),
                    mean_difference=float((error[1] - error[0]).mean()),
                )
            )
            np.savez_compressed(
                output / f"frame_{native:06d}.npz",
                poses=poses,
                union_alpha=alpha,
                ray_valid=valid,
                per_view_error=error,
                **arrays,
            )
        errors = np.array([[r["original_error"], r["directional_error"]] for r in rows])
        means = errors.mean((0, 2))
        summary = dict(
            status="ok",
            native_interval=[START, STOP],
            scored_native_frames=SCORED.tolist(),
            scored_frame_count=len(rows),
            regenerated_diagnostic_checks=216,
            original_mean_error=float(means[0]),
            directional_mean_error=float(means[1]),
            relative_error_reduction_percent=float(100 * (means[0] - means[1]) / means[0]),
            mean_difference_by_view=(errors[:, 1] - errors[:, 0]).mean(0).tolist(),
            frames_lower_error=sum(r["mean_difference"] < 0 for r in rows),
            frames_higher_error=sum(r["mean_difference"] > 0 for r in rows),
            fitted_coordinate_overlap=0,
            temporal_holdout=False,
            all_three_ID_truth=False,
            candidate_reselected=False,
            adopted=False,
        )
        write_json(output / "summary.json", summary)
        write_json(output / "frames.json", rows)
        finish(record, output, hashes, code)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "evaluate"))
    for name in ("original", "output", "branch", "comparison", "prepared", "original-replay"):
        parser.add_argument("--" + name, type=Path, required=name in ("original", "output"))
    args = parser.parse_args()
    output = new_output(args.output)
    if args.mode == "prepare":
        prepare(args.original.resolve(), output)
    else:
        checks.require(
            all(
                getattr(args, k) is not None
                for k in ("branch", "comparison", "prepared", "original_replay")
            ),
            "evaluation requires branch/comparison/prepared/original-replay",
        )
        evaluate(
            *(
                getattr(args, k).resolve()
                for k in ("original", "branch", "comparison", "prepared", "original_replay")
            ),
            output,
        )
