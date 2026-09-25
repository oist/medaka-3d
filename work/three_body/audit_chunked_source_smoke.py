"""CPU-only evidence audit after the three registered six-frame smoke invocations."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(LAB))
from chunk_masks import validate_packed_masks  # noqa: E402
from observations import SequenceReader  # noqa: E402
from prepare_chunked_source_masks import exact_contract, validate_protocol  # noqa: E402
from prepare_sequence import VIEWS, array_sha256, sha256  # noqa: E402
from source_sequence import SourceSequenceReader  # noqa: E402

from medaka3d.provenance import provenance  # noqa: E402

DEFAULT_RUN = REPO / "work/three_body/inputs/chunked_source_smoke_333519-333525_v1"
DEFAULT_OUT = REPO / "work/three_body/chunked_source_smoke_qa_v1"
PARENT = REPO / "work/three_body/inputs/transient_333519-333719"
BACKGROUND = REPO / "work/three_body/long_background_audit"
NODES = REPO / "work/generative_tracking_inputs"
EXPECTED = {
    "prepare_chunked_source_masks.py": (
        "a081c47e6ed67b4b61c02635e00c97c967803b3528fe107f016ab23b5ae34f16"
    ),
    "chunk_masks.py": "9e08248ea593b4ccb105a6ff351fd85c6db24aa6754413b2df766404cdfe4b81",
    "prepare_source_masks.py": "b6e78bc70ef4191fe57dc4a5ec391a42f2060c51aae62e2e715412f3ae344262",
    "prepare_sequence.py": "959e862cc58c5840a00a763d8a63b635de134492a637ac2f1e8a7131f25178b8",
    "source_sequence.py": "9db0e992768d54369d1431f82280c1b1b22d324d7a66daa35aa94dc4296a5cfe",
}
FIRST_CHUNK_SAVED_SHA = {
    "manifest_sha256": "b21ca56798e9f8418426ef5fa1482ab06b958b544db8dbeda41c9656713f5ac6",
    "provenance_sha256": "8f7b7009ac558cab3a7f673c46dd271a654c904f1479bda4527c7deb3a6b2e7e",
}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix_masks(data, count):
    """Take a prefix without changing within-frame detection order or field values."""
    validate_packed_masks(data)
    instances = int(data["frame_offsets"][count])
    byte_end = int(data["mask_byte_offsets"][instances])
    result = {}
    for key, array in data.items():
        if key == "image_shape_hw":
            result[key] = array.copy()
        elif key == "frame_offsets":
            result[key] = array[: count + 1].copy()
        elif key == "mask_byte_offsets":
            result[key] = array[: instances + 1].copy()
        elif key == "mask_bits":
            result[key] = array[:byte_end].copy()
        elif key in {"frame_idx", "view_available", "frame_mask_sha256"}:
            result[key] = array[:count].copy()
        else:
            result[key] = array[:instances].copy()
    validate_packed_masks(result)
    return result


def compare_array(left, right):
    matching_shape = left.shape == right.shape
    matching_layout = matching_shape and left.dtype == right.dtype
    bitwise = bool(matching_layout and left.tobytes(order="C") == right.tobytes(order="C"))
    return dict(
        exact=bitwise,
        bitwise_exact=bitwise,
        new_shape=list(left.shape),
        old_shape=list(right.shape),
        new_dtype=str(left.dtype),
        old_dtype=str(right.dtype),
        differing_elements=int(np.count_nonzero(left != right)) if matching_shape else None,
    )


class EvidenceHashes:
    """Hash each unchanged inode once per phase, including shared MP4 hardlinks."""

    def __init__(self):
        self.memo, self.paths = {}, {}

    def digest(self, path):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(REPO), "audit input escaped local workspace")
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in self.memo:
            self.memo[key] = sha256(path)
        self.paths[str(path)] = self.memo[key]
        return self.memo[key]

    def check(self, path, expected):
        require(self.digest(path) == expected, f"changed evidence checksum: {path}")

    def verify_after(self):
        after = EvidenceHashes()
        for path, digest in self.paths.items():
            after.check(path, digest)
        return after.paths


def require_bound_outputs(seal, root, relative_paths):
    outputs = {Path(info["path"]).resolve(): info for info in seal["outputs"].values()}
    for relative in relative_paths:
        require(
            (root / relative).resolve(strict=True) in outputs,
            f"required output is unsealed: {root / relative}",
        )


def verify_seal(path, hashes, expected_status="ok", check_inputs=False, required=()):
    path = Path(path)
    seal = load(path)
    hashes.digest(path)
    require(seal["status"] == expected_status, f"unexpected completion status: {path}")
    for info in seal["outputs"].values():
        output = Path(info["path"]).resolve(strict=True)
        require(output.is_relative_to(path.parent.resolve()), "sealed output escaped its directory")
        hashes.check(output, info["sha256"])
    if check_inputs:
        for info in seal["inputs"].values():
            hashes.check(info["path"], info["sha256"])
    require_bound_outputs(seal, path.parent, required)
    return seal


def verify_history(root, hashes):
    directories = [root / "runs" / f"run_{number:04d}" for number in (1, 2, 3)]
    seals = [
        verify_seal(
            directory / "run.provenance.json",
            hashes,
            "failed" if i == 0 else "ok",
            required=("incomplete.json" if i == 0 else "summary.json",),
        )
        for i, directory in enumerate(directories)
    ]
    incomplete = load(directories[0] / "incomplete.json")
    second, third = (load(directory / "summary.json") for directory in directories[1:])
    require(incomplete["status"] == "incomplete", "first invocation was not explicitly incomplete")
    require(
        incomplete["reason"] == "requested_max_new_chunks",
        "first invocation failed for another reason",
    )
    require(incomplete["final_sequence_path"] is None, "paused invocation advertised an assembly")
    require(
        seals[0]["params"]["global_preparation_complete"] is False,
        "paused seal advertised completion",
    )
    require(
        not (directories[0] / "summary.json").exists(), "paused invocation wrote a complete summary"
    )
    require(len(incomplete["chunks"]) == 1, "first invocation did not stop after one chunk")
    require(
        [entry["reused"] for entry in second["chunks"]] == [True, False, False],
        "resume did not reuse only the first chunk",
    )
    require(
        [entry["reused"] for entry in third["chunks"]] == [True, True, True],
        "third invocation was not full reuse",
    )
    require(second["status"] == third["status"] == "complete", "final invocations incomplete")
    for key in ("final_sequence_path", "final_manifest_sha256", "final_provenance_sha256"):
        require(second[key] == third[key], "third invocation changed the assembled result")
    original = incomplete["chunks"][0]
    for key, digest in FIRST_CHUNK_SAVED_SHA.items():
        require(original[key] == digest, "first chunk differs from the hash recorded before resume")
    for key in ("attempt_path", "manifest_sha256", "provenance_sha256"):
        require(
            original[key] == second["chunks"][0][key] == third["chunks"][0][key],
            "first sealed chunk was replaced during resume",
        )
    assembly = (root / third["final_sequence_path"]).resolve(strict=True)
    require(assembly.is_relative_to(root.resolve()), "assembly escaped smoke root")
    assembly_seal = verify_seal(
        assembly / "staging.provenance.json",
        hashes,
        check_inputs=True,
        required=("manifest.json", "protocol.json", "chunks.json"),
    )
    assembly_manifest = load(assembly / "manifest.json")
    require_bound_outputs(
        assembly_seal,
        assembly,
        [
            info["path"]
            for category in ("raw", "masks", "backgrounds")
            for info in assembly_manifest[category].values()
        ],
    )
    index = load(assembly / "chunks.json")
    expected_entries = [
        {
            key: entry[key]
            for key in ("start", "stop", "attempt_path", "manifest_sha256", "provenance_sha256")
        }
        for entry in third["chunks"]
    ]
    require(
        exact_contract(index["chunks"], expected_entries)
        and index["schema"] == "medaka_mask_chunk_index_v1"
        and exact_contract(index["source_frame_range"], [333519, 333525])
        and exact_contract(index["chunk_frames"], 2)
        and index["protocol_sha256"] == third["protocol_sha256"]
        and Path(index["attempt_paths_relative_to"]).resolve() == root.resolve(),
        "sealed chunk index differs from run history/protocol",
    )
    hashes.check(assembly / "chunks.json", assembly_manifest["chunk_index_sha256"])
    require(
        datetime.fromisoformat(assembly_seal["started_at"])
        > datetime.fromisoformat(seals[0]["finished_at"]),
        "assembly does not postdate the incomplete invocation",
    )
    hashes.check(assembly / "manifest.json", third["final_manifest_sha256"])
    hashes.check(assembly / "staging.provenance.json", third["final_provenance_sha256"])
    manifests = []
    require(
        [(entry["start"], entry["stop"]) for entry in third["chunks"]]
        == [(333519, 333521), (333521, 333523), (333523, 333525)],
        "smoke chunk ranges are not the registered three native pairs",
    )
    for entry in third["chunks"]:
        folder = (root / entry["attempt_path"]).resolve(strict=True)
        require(folder.is_relative_to(root.resolve()), "chunk escaped smoke root")
        chunk_seal = verify_seal(
            folder / "staging.provenance.json",
            hashes,
            check_inputs=True,
            required=("manifest.json", "protocol.json"),
        )
        hashes.check(folder / "manifest.json", entry["manifest_sha256"])
        hashes.check(folder / "staging.provenance.json", entry["provenance_sha256"])
        require(
            len(list(folder.parent.glob("attempt_*"))) == 1,
            "real smoke unexpectedly needed another chunk attempt",
        )
        chunk_manifest = load(folder / "manifest.json")
        require_bound_outputs(
            chunk_seal,
            folder,
            [
                info["path"]
                for category in ("raw", "masks", "backgrounds")
                for info in chunk_manifest[category].values()
            ],
        )
        require(
            chunk_manifest["frames"] == list(range(entry["start"], entry["stop"])),
            "chunk manifest range disagrees with its run index",
        )
        manifests.append((folder, chunk_manifest))
    require(
        len(list((root / "assemblies").glob("attempt_*"))) == 1,
        "full reuse created another assembly",
    )
    return (
        assembly,
        load(assembly / "manifest.json"),
        manifests,
        dict(
            first_intentionally_incomplete=True,
            first_chunk_hash_preserved=True,
            first_run_has_no_complete_summary_or_assembly_reference=True,
            assembly_started_after_first_invocation=True,
            first_wrapper_reported_exit_code=1,
            first_native_exit_code_measured=None,
            first_cli_intended_exit_code=2,
            second_reused=[True, False, False],
            third_reused=[True, True, True],
            final_manifest_sha256=third["final_manifest_sha256"],
            final_provenance_sha256=third["final_provenance_sha256"],
        ),
    )


def audit(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    require(output.is_relative_to(REPO), "audit output must stay in workspace")
    if output.exists() or output.is_symlink():
        raise FileExistsError("audit output already exists; no overwrite")
    for name, digest in EXPECTED.items():
        require(sha256(LAB / name) == digest, f"frozen source changed: {name}")
    # Fail before creating an audit directory unless all three root invocations finished.
    require(
        (root / "runs/run_0003/summary.json").exists(), "wait for the third full-reuse invocation"
    )
    started = time.perf_counter()
    hashes = EvidenceHashes()
    code = [
        Path(__file__),
        *(LAB / name for name in EXPECTED),
        LAB / "observations.py",
        REPO / "src/medaka3d/provenance.py",
        REPO / "src/medaka3d/types.py",
    ]
    code_before = {str(path): hashes.digest(path) for path in code}
    output.mkdir(parents=True, exist_ok=False)
    with provenance("audit_chunked_six_frame_resume_parity", output, name="audit") as record:
        record.param("gpu_used", False)
        for path, digest in code_before.items():
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha256(target) == digest, "audit source snapshot mismatch")
            record.output("code:" + path, target, sha256=digest)
        assembly, manifest, chunks, history = verify_history(root, hashes)
        require(
            manifest["frames"] == list(range(333519, 333525)),
            "unexpected real smoke frame interval",
        )
        require(manifest["video_payload_bytes_copied"] == 0, "stager copied video payload")
        protocol = validate_protocol(load(assembly / "protocol.json"))
        require(
            protocol["source_frame_range"] == [333519, 333525] and protocol["chunk_frames"] == 2,
            "smoke protocol differs from registered three-chunk window",
        )
        require(
            manifest["binding"]["protocol_sha256"] == hashes.digest(assembly / "protocol.json"),
            "assembly protocol binding mismatch",
        )
        require(
            exact_contract(manifest["runtime"]["detector_contract"], protocol["detector"]),
            "effective detector contract mismatch",
        )
        verify_seal(PARENT / "staging.provenance.json", hashes, required=("manifest.json",))
        parent_manifest = load(PARENT / "manifest.json")
        background_seal = verify_seal(
            BACKGROUND / "audit.provenance.json",
            hashes,
            required=("summary.json", *(f"background_{view}.png" for view in VIEWS)),
        )
        background_summary = load(BACKGROUND / "summary.json")
        require(background_summary["status"] == "ok", "prefix background audit incomplete")
        for view in VIEWS:
            inputs = [
                item
                for item in background_seal["inputs"].values()
                if Path(item["path"]).parent.name == view and Path(item["path"]).suffix == ".mp4"
            ]
            require(
                len(inputs) == 1, "background audit does not identify exactly one source per view"
            )
            require(
                inputs[0]["sha256"] == protocol["source_videos_sha256"][view],
                "background source differs from smoke protocol",
            )
            hashes.check(inputs[0]["path"], inputs[0]["sha256"])
        results, models = {}, {}
        for kind in ("top", "side"):
            new, old = manifest["models"][kind], parent_manifest["models"][kind]
            newer = yaml.safe_load((assembly / new["effective_config"]).read_text())
            older = yaml.safe_load((PARENT / old["effective_config"]).read_text())
            weights_paths = dict(new=newer["MODEL"]["WEIGHTS"], old=older["MODEL"]["WEIGHTS"])
            newer["MODEL"]["WEIGHTS"] = older["MODEL"]["WEIGHTS"] = "<verified same weights bytes>"
            models[kind] = dict(
                weights_sha_exact=new["weights"]["sha256"] == old["weights"]["sha256"],
                source_config_sha_exact=new["config"]["sha256"] == old["config"]["sha256"],
                config_exact_except_weights_path=newer == older,
                effective_weights_paths=weights_paths,
            )
            for folder, data in ((assembly, new), (PARENT, old)):
                hashes.check(folder / data["effective_config"], data["effective_config_sha256"])
                hashes.check(folder / f"{kind}_source_config.yaml", data["config"]["sha256"])
        current, old = SourceSequenceReader(assembly, NODES), SequenceReader(PARENT, NODES)
        for path in current.nodes.paths.values():
            hashes.digest(path)
        for view in VIEWS:
            expected_arrays = prefix_masks(old.masks[view], 6)
            validate_packed_masks(current.masks[view])
            require(set(current.masks[view]) == set(expected_arrays), "packed mask schema changed")
            packed = {
                name: compare_array(current.masks[view][name], expected_arrays[name])
                for name in expected_arrays
            }
            raw = manifest["raw"][view]
            originals = [
                Path(load(folder / "staging.provenance.json")["inputs"]["video:" + view]["path"])
                for folder, _ in chunks
            ]
            hardlinks = all(
                os.path.samefile(assembly / raw["path"], source) for source in originals
            )
            hardlinks &= all(
                os.path.samefile(folder / data["raw"][view]["path"], originals[0])
                for folder, data in chunks
            )
            prefix_png = cv2.imread(str(BACKGROUND / f"background_{view}.png"))
            require(prefix_png is not None, "missing registered prefix background")
            require(
                [frame["frame"] for frame in background_summary["views"][view]["frames"]]
                == protocol["background_frames"],
                "registered background frames differ",
            )
            require(
                array_sha256(prefix_png) == background_summary["views"][view]["median_bgr_sha256"],
                "prefix background pixels differ from audit",
            )
            bg = compare_array(current.backgrounds[view], prefix_png)
            old_bg = compare_array(current.backgrounds[view], old.backgrounds[view])
            chunk_background_hashes = [data["backgrounds"][view]["sha256"] for _, data in chunks]
            results[view] = dict(
                packed_arrays=packed,
                hardlinks_verified=bool(hardlinks),
                raw_bgr_sha_columns_exact=raw["frame_bgr_sha256"]
                == parent_manifest["raw"][view]["frame_bgr_sha256"][:6],
                instances_per_frame=np.diff(current.masks[view]["frame_offsets"]).tolist(),
                zero_detection_native_frames=current.frames[
                    np.diff(current.masks[view]["frame_offsets"]) == 0
                ].tolist(),
                all_native_frames_available=bool(current.masks[view]["view_available"].all()),
                registered_background_bgr=bg,
                old_background_bgr=old_bg,
                chunk_background_sha_identical=len(set(chunk_background_hashes)) == 1,
                frames=[],
            )
        left, right = iter(current), iter(old)
        decode_started = time.perf_counter()
        try:
            for frame in manifest["frames"]:
                new_frame, old_frame = next(left), next(right)
                require(
                    new_frame.frame_idx == old_frame.frame_idx == frame,
                    "native decoder frame alignment differs",
                )
                for view in VIEWS:
                    results[view]["frames"].append(
                        dict(
                            frame_idx=frame,
                            raw_bgr=compare_array(
                                new_frame.images_bgr[view], old_frame.images_bgr[view]
                            ),
                            raw_bgr_sha256=array_sha256(new_frame.images_bgr[view]),
                            masks=compare_array(new_frame.masks[view], old_frame.masks[view]),
                            untracked_sleap_rows_exact=bool(
                                np.array_equal(
                                    new_frame.detections[view],
                                    old_frame.detections[view],
                                    equal_nan=True,
                                )
                            ),
                        )
                    )
            sentinel = object()
            require(
                next(left, sentinel) is sentinel,
                "source reader did not stop at six declared frames",
            )
        finally:
            left.close()
            right.close()
        decode_seconds = time.perf_counter() - decode_started
        before = dict(hashes.paths)
        after = hashes.verify_after()
        for path, digest in before.items():
            record.input(path, path, sha256=digest)
        all_exact = all(
            item["exact"]
            for result in results.values()
            for item in result["packed_arrays"].values()
        )
        pixels_exact = all(
            frame["raw_bgr"]["exact"]
            and frame["masks"]["exact"]
            and frame["untracked_sleap_rows_exact"]
            for result in results.values()
            for frame in result["frames"]
        )
        backgrounds_exact = all(
            result["registered_background_bgr"]["exact"]
            and result["chunk_background_sha_identical"]
            for result in results.values()
        )
        model_exact = all(
            item[key]
            for item in models.values()
            for key in (
                "weights_sha_exact",
                "source_config_sha_exact",
                "config_exact_except_weights_path",
            )
        )
        identity_links = all(
            result["hardlinks_verified"]
            and result["raw_bgr_sha_columns_exact"]
            and result["all_native_frames_available"]
            for result in results.values()
        )
        passed = bool(
            all_exact and pixels_exact and backgrounds_exact and model_exact and identity_links
        )
        summary = dict(
            status="passed" if passed else "failed",
            native_frame_range=[333519, 333525],
            native_frames=6,
            views=3,
            decoded_images=18,
            all_13_mask_arrays_exact=all_exact,
            all_raw_masks_and_untracked_rows_exact=pixels_exact,
            registered_backgrounds_exact=backgrounds_exact,
            registered_background_source_video_hashes_exact=True,
            all_old_backgrounds_differ=all(
                not result["old_background_bgr"]["exact"] for result in results.values()
            ),
            model_configs_exact_except_verified_weights_path=model_exact,
            hardlinks_and_native_sha_columns_verified=identity_links,
            video_payload_bytes_copied=0,
            resume_history=history,
            comparison=results,
            model_comparison=models,
            input_sha256_before=before,
            input_sha256_after=after,
            code_sha256=code_before,
            all_inputs_unchanged=before == after,
            cpu_decode_compare_s=decode_seconds,
            audit_elapsed_s=time.perf_counter() - started,
            audit_runtime=dict(
                python=sys.version, numpy=np.__version__, opencv=cv2.__version__, gpu_used=False
            ),
            detector_runtime=manifest["runtime"],
            limitations=[
                "Only six native frames are tested; this does not validate identity "
                "or full-window detector behavior.",
                "The newly registered prefix-nine background is deliberately different "
                "from the old stage background.",
                "SLEAP is read only after detector staging, solely to verify "
                "existing reader compatibility.",
                "No GPU inference or tracking runs during this audit; saved detector runtime "
                "is provenance, not the audit runtime.",
                "Concurrent long staging and repeated full-source hash checks make "
                "timing unsuitable as detector throughput.",
                "First-run incompleteness is established from its immutable seal/history "
                "and assembly creation chronology.",
            ],
        )
        path = output / "summary.json"
        path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", path, sha256=sha256(path))
        require(passed, "six-frame input parity audit failed; inspect summary")
        print(
            json.dumps(
                {
                    key: summary[key]
                    for key in (
                        "status",
                        "decoded_images",
                        "all_13_mask_arrays_exact",
                        "registered_backgrounds_exact",
                        "all_old_backgrounds_differ",
                        "audit_elapsed_s",
                    )
                }
            ),
            flush=True,
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    audit(args.run, args.output)


if __name__ == "__main__":
    main()
