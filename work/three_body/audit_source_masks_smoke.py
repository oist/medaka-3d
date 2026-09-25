"""CPU exact-field audit of the authorized four-frame fresh source-mask smoke."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from observations import SequenceReader  # noqa: E402
from prepare_sequence import VIEWS, decode_frame_masks, sha256  # noqa: E402
from prepare_source_masks import verify_completion  # noqa: E402
from source_sequence import SourceSequenceReader  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

FRESH = REPO / "work/three_body/inputs/transient_source_fresh_333519-333523"
PARENT = REPO / "work/three_body/inputs/transient_333519-333719"
NODES = REPO / "work/generative_tracking_inputs"
OUTPUT = REPO / "work/three_body/source_masks_smoke_qa"
EXPECTED_CLI = "b6e78bc70ef4191fe57dc4a5ec391a42f2060c51aae62e2e715412f3ae344262"


def compare_array(left, right):
    shape = left.shape == right.shape
    result = dict(
        exact=bool(shape and left.dtype == right.dtype and np.array_equal(left, right)),
        new_shape=list(left.shape),
        old_shape=list(right.shape),
        new_dtype=str(left.dtype),
        old_dtype=str(right.dtype),
    )
    if shape:
        result["differing_elements"] = int(np.count_nonzero(left != right))
        if left.dtype.kind in "iuf":
            result["max_absolute_difference"] = float(
                np.max(np.abs(left.astype(float) - right.astype(float)), initial=0)
            )
    return result


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("QA destination exists; no overwrite")
    if sha256(LAB / "prepare_source_masks.py") != EXPECTED_CLI:
        raise ValueError("the source-mask CLI changed after the authorized freeze")
    fresh_manifest = verify_completion(FRESH)
    old_manifest = json.loads((PARENT / "manifest.json").read_text(encoding="utf-8"))
    old_seal = json.loads((PARENT / "staging.provenance.json").read_text(encoding="utf-8"))
    if old_seal["status"] != "ok" or old_seal["outputs"]["manifest"]["sha256"] != sha256(
        PARENT / "manifest.json"
    ):
        raise ValueError("parent staging manifest is not sealed")
    if fresh_manifest["frames"] != list(range(333519, 333523)):
        raise ValueError("fresh inference must contain exactly the authorized four frames")
    if fresh_manifest["runtime"]["batch_size"] != 1:
        raise ValueError("fresh inference changed batch size")
    started = time.perf_counter()
    input_paths = {
        FRESH / "manifest.json",
        FRESH / "staging.provenance.json",
        PARENT / "manifest.json",
        PARENT / "staging.provenance.json",
    }
    # Reader initialization separately verifies all original-video full hashes.
    for root, manifest in ((FRESH, fresh_manifest), (PARENT, old_manifest)):
        for category in ("masks", "backgrounds"):
            input_paths.update(root / info["path"] for info in manifest[category].values())
        for kind, model in manifest["models"].items():
            input_paths.add(root / model["effective_config"])
            input_paths.add(root / f"{kind}_source_config.yaml")
    input_paths.update(PARENT / info["path"] for info in old_manifest["raw"].values())
    code_paths = {
        Path(__file__),
        LAB / "prepare_source_masks.py",
        LAB / "prepare_sequence.py",
        LAB / "source_sequence.py",
        LAB / "observations.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/types.py",
    }
    inputs_before = {str(p): sha256(p) for p in sorted(input_paths)}
    code_before = {str(p): sha256(p) for p in sorted(code_paths)}
    OUTPUT.mkdir(parents=True, exist_ok=False)
    with provenance("audit_source_mask_four_frame_parity", OUTPUT, name="audit") as record:
        record.param("gpu_inference_images", 12)
        record.param("audit_gpu_used", False)
        for path, digest in inputs_before.items():
            record.input(path, path, sha256=digest)
        for path, digest in code_before.items():
            destination = OUTPUT / "code_snapshots" / Path(path).relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            if sha256(destination) != digest:
                raise ValueError("QA code snapshot differs from frozen source")
            record.input("code:" + path, path, sha256=digest)
        current = SourceSequenceReader(FRESH, NODES)
        old = SequenceReader(PARENT, NODES)
        nodes_before = {str(p): sha256(p) for p in current.nodes.paths.values()}
        for path, digest in nodes_before.items():
            record.input("untracked_nodes:" + path, path, sha256=digest)
        results = {view: dict(frames=[]) for view in VIEWS}
        models = {}
        for kind in ("top", "side"):
            new_model, old_model = fresh_manifest["models"][kind], old_manifest["models"][kind]
            new_cfg = yaml.safe_load((FRESH / new_model["effective_config"]).read_text())
            old_cfg = yaml.safe_load((PARENT / old_model["effective_config"]).read_text())
            paths = dict(new=new_cfg["MODEL"]["WEIGHTS"], old=old_cfg["MODEL"]["WEIGHTS"])
            new_cfg["MODEL"]["WEIGHTS"] = old_cfg["MODEL"]["WEIGHTS"] = (
                "<same verified weights bytes>"
            )
            models[kind] = dict(
                weights_sha_exact=new_model["weights"]["sha256"] == old_model["weights"]["sha256"],
                source_config_sha_exact=new_model["config"]["sha256"]
                == old_model["config"]["sha256"],
                effective_config_exact_except_verified_weights_path=new_cfg == old_cfg,
                effective_weights_paths=paths,
            )
        for view in VIEWS:
            nr, pr = fresh_manifest["raw"][view], old_manifest["raw"][view]
            original_file = Path(fresh_manifest["source_videos"][view]["path"])
            if not os.path.samefile(FRESH / nr["path"], original_file):
                raise ValueError("source-video hardlink identity failed")
            results[view].update(
                raw_bgr_sha_columns_exact=nr["frame_bgr_sha256"] == pr["frame_bgr_sha256"][:4],
                hardlink_verified=True,
                background_frames_exact=(
                    fresh_manifest["backgrounds"][view]["frames"]
                    == old_manifest["backgrounds"][view]["frames"]
                ),
                background_bgr=compare_array(current.backgrounds[view], old.backgrounds[view]),
            )
            record.input(view + "_original_video", original_file, sha256=nr["sha256"])
        left, right = iter(current), iter(old)
        decode_start = time.perf_counter()
        try:
            for index, frame in enumerate(fresh_manifest["frames"]):
                new_frame, old_frame = next(left), next(right)
                if new_frame.frame_idx != frame or old_frame.frame_idx != frame:
                    raise ValueError("native frame alignment differs during audit")
                for view in VIEWS:
                    na, oa = current.masks[view], old.masks[view]
                    nlo, nhi = na["frame_offsets"][index : index + 2]
                    olo, ohi = oa["frame_offsets"][index : index + 2]
                    fields = {
                        key: compare_array(na[key][nlo:nhi], oa[key][olo:ohi])
                        for key in (
                            "scores",
                            "pred_boxes",
                            "mask_boxes",
                            "classes",
                            "instance_id",
                            "source_track_id",
                        )
                    }
                    fields["masks_in_original_instance_order"] = compare_array(
                        decode_frame_masks(na, index), decode_frame_masks(oa, index)
                    )
                    fields["available"] = compare_array(
                        na["view_available"][index : index + 1],
                        oa["view_available"][index : index + 1],
                    )
                    fields["frame_mask_sha256"] = compare_array(
                        na["frame_mask_sha256"][index : index + 1],
                        oa["frame_mask_sha256"][index : index + 1],
                    )
                    fields["raw_bgr"] = compare_array(
                        new_frame.images_bgr[view], old_frame.images_bgr[view]
                    )
                    fields["untracked_sleap_rows"] = dict(
                        exact=bool(
                            np.array_equal(
                                new_frame.detections[view],
                                old_frame.detections[view],
                                equal_nan=True,
                            )
                        )
                    )
                    results[view]["frames"].append(
                        dict(
                            frame_idx=frame,
                            new_instances=int(nhi - nlo),
                            old_instances=int(ohi - olo),
                            fields=fields,
                        )
                    )
            sentinel = object()
            if next(left, sentinel) is not sentinel:
                raise ValueError("new descriptor emitted frames beyond the authorized four")
        finally:
            left.close()
            right.close()
        decode_seconds = time.perf_counter() - decode_start
        if inputs_before != {path: sha256(path) for path in inputs_before}:
            raise ValueError("saved parent/fresh input changed during audit")
        if code_before != {path: sha256(path) for path in code_before}:
            raise ValueError("frozen source changed during audit")
        if nodes_before != {path: sha256(path) for path in nodes_before}:
            raise ValueError("untracked node cache changed during audit")
        all_fields_exact = all(
            field["exact"]
            for view in results.values()
            for frame in view["frames"]
            for field in frame["fields"].values()
        )
        summary = dict(
            protocol="four native frames, fresh batch1 masks versus existing first four frames",
            native_frames=4,
            views=3,
            images=12,
            all_fields_exact=all_fields_exact,
            all_backgrounds_exact=all(r["background_bgr"]["exact"] for r in results.values()),
            raw_bgr_sha_columns_exact=all(r["raw_bgr_sha_columns_exact"] for r in results.values()),
            comparison=results,
            model_config_comparison=models,
            new_runtime=fresh_manifest["runtime"],
            parent_runtime=old_manifest["runtime"],
            audit_runtime=dict(
                python=sys.version, numpy=np.__version__, opencv=cv2.__version__, gpu_used=False
            ),
            staging_elapsed_s=fresh_manifest["staging_elapsed_s"],
            predictor_seconds=sum(
                sum(fresh_manifest["masks"][view]["inference_s_by_frame"]) for view in VIEWS
            ),
            cpu_pair_decode_and_compare_s=decode_seconds,
            audit_total_s=time.perf_counter() - started,
            timing_scope=(
                "reference only: concurrent Detectron2 staging; no throughput conclusion"
            ),
            video_payload_bytes_copied=fresh_manifest["video_payload_bytes_copied"],
            source_video_hashes_verified_by_reader=True,
            staging_inputs_before_after_equal=(
                fresh_manifest["input_sha256_before"] == fresh_manifest["input_sha256_after"]
            ),
            input_sha256_before=inputs_before,
            input_sha256_after=inputs_before,
            code_sha256_before=code_before,
            code_sha256_after=code_before,
            code_and_inputs_unchanged=True,
            limitations=[
                "Only this four-frame window is tested for numerical parity.",
                "No detection ordering changes or tolerances are hidden in exact-equality flags.",
                "Empty/available semantics are compared; zero-detection counts are explicit.",
                "Backgrounds match here; long windows can fail the fixed-nine rule.",
                "This is storage/detector parity, not biological identity or tracking validation.",
            ],
        )
        path = OUTPUT / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "images",
                        "all_fields_exact",
                        "all_backgrounds_exact",
                        "raw_bgr_sha_columns_exact",
                        "predictor_seconds",
                        "audit_total_s",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
