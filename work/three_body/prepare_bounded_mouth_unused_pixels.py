"""Fix spatially unused pixels for all33 local packets before opening fit results.

Only packet fit-sample fields are deserialized. Images are represented by their
registered shape because the frozen D2 sampler uses mask pixels, not BGR values.
This is development-image spatial resampling, not a temporal holdout or ID test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import prepare_bounded_mouth_fit_packets as packets
import score_temporal_disjoint_pixels as sampler
from observations import sample_pixels
from prepare_sequence import decode_frame_masks

from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = packets.checks, packets.REPO
STAGE = "bounded_mouth_unused_pixel_preparation_v1"
PACKET_ROOT = REPO / "work/three_body/bounded_mouth_local_fit_packets_33_v1"
PACKET_SEAL_SHA = "c1b4fc4b0f369f7f4a376f8bcd4a5b1736ad42efdcc0101118f9a407243da080"
PACKET_SOURCE_SHA = "5094ce9b8ba5b5a7bcbcb31abd838340cfc374709d3c14c8a15e976bdf336da3"
SAMPLER_SHA = "8a86b3942d93b5ed506ac85cb3520fa9ccd51a2a11c55201fd1884e4e5c31d47"
FIT_KEYS = ("fit_pixels", "fit_target", "fit_weights", "fit_foreground")
FRAMES = np.arange(67663, 67696, dtype=np.int64)


def regenerated_fit(frame, config, saved):
    checks.require(config["mode"] == "detectron2", "mask-only D2 sampling required")
    targets = sample_pixels(
        frame,
        config["mode"],
        config["positive_samples"],
        config["near_samples"],
        config["global_samples"],
    )
    result = dict(
        fit_pixels=np.stack([targets.pixels[view] for view in VIEWS]),
        fit_target=targets.target,
        fit_weights=targets.weights,
        fit_foreground=targets.foreground,
    )
    for key in FIT_KEYS:
        checks.require(
            checks.equal_bytes(result[key], saved[key]), "packet fit bytes differ: " + key
        )
    return result


def prepare_frame(frame, config, saved):
    fitted = regenerated_fit(frame, config, saved)
    pixels, target, weights, overlaps, membership = sampler.unused_samples(
        frame, fitted["fit_pixels"]
    )
    checks.require(not overlaps.any(), "fitted coordinate reused")
    return dict(
        **fitted, pixels=pixels, target=target, weights=weights, reused_fitted_coordinate=overlaps
    ), membership


def bound_input(path, expected, seal, hashes):
    path = Path(path).resolve(strict=True)
    item = seal["inputs"].get(str(path))
    checks.require(
        item is not None and Path(item["path"]).resolve() == path and item["sha256"] == expected,
        "file was not an exact packet preparation input: " + str(path),
    )
    packets.prior.bind(path, expected, hashes)
    return path


def run(output):
    output = Path(output).resolve()
    checks.require(output.is_relative_to(REPO / "work") and not output.exists(), "new work output")
    code = packets.comparison.source_registry.local_code()
    hashes, outputs = dict(code), {}
    checks.require(checks.sha(packets.__file__) == PACKET_SOURCE_SHA, "packet producer changed")
    checks.require(checks.sha(sampler.__file__) == SAMPLER_SHA, "frozen spatial sampler changed")
    checks.require(
        sampler.COUNTS == (96, 96, 64) and sampler.SEED_OFFSET == 8803010,
        "fixed sampling constants changed",
    )
    output.mkdir(parents=True)
    with provenance(STAGE, output, name="preparation") as record:

        def emit(name, value=None):
            path = output / name
            if value is not None:
                path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
            digest = checks.sha(path)
            outputs[str(path)] = digest
            record.output(name, path, sha256=digest)

        packets.prior.bind(PACKET_ROOT / "preparation.provenance.json", PACKET_SEAL_SHA, hashes)
        seal = checks.verify(PACKET_ROOT, "preparation.provenance.json", packets.STAGE, hashes)
        for path, digest in code.items():
            if path in seal["inputs"]:
                checks.require(seal["inputs"][path]["sha256"] == digest, "shared code changed")
            target = output / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(path).read_bytes())
            checks.require(checks.sha(target) == digest, "source copy changed")
            emit(str(target.relative_to(output)))
        required = ["metadata.json", "summary.json"] + [f"f{f}_packet.npz" for f in FRAMES]
        for name in required:
            path = PACKET_ROOT / name
            checks.require(
                str(path) in hashes
                and name in seal["outputs"]
                and Path(seal["outputs"][name]["path"]).resolve() == path,
                "required packet output role missing",
            )
        meta = checks.read(PACKET_ROOT / "metadata.json")
        summary = checks.read(PACKET_ROOT / "summary.json")
        checks.require(
            meta["frames"] == FRAMES.tolist()
            and meta["native_interval"] == [67663, 67696]
            and summary["status"] == "ok"
            and summary["packets"] == 33
            and summary["fit_calls"] == summary["CUDA_calls"] == 0,
            "wrong or fitted packet preparation",
        )
        config = meta["tracker_config"]
        checks.require(
            config == meta["original_context"]["tracker_config"], "sampler config differs"
        )
        shared = meta["checkpoint_binding"]["original_binding"]["inputs"]["seed"][
            "SL_and_calibration_sha256"
        ]
        checks.require(len(shared) == 7, "same SL3/calibration4 required")
        for path, digest in shared.items():
            bound_input(path, digest, seal, hashes)
        chunk = meta["source_chunk"]
        checks.require(
            [chunk["start"], chunk["stop"], chunk["index"]] == [66400, 68060, 10],
            "wrong source chunk",
        )
        stage = Path(meta["checkpoint_binding"]["original_binding"]["inputs"]["input_stage_root"])
        root = (stage / chunk["attempt_path"]).resolve(strict=True)
        checks.require(root.is_relative_to(stage.resolve()), "chunk escaped stage")
        manifest_path = bound_input(root / "manifest.json", chunk["manifest_sha256"], seal, hashes)
        bound_input(root / "staging.provenance.json", chunk["provenance_sha256"], seal, hashes)
        manifest = checks.read(manifest_path)
        masks, mask_hashes = {}, {}
        for view in VIEWS:
            member = chunk["output_hashes"][view + "_masks"]
            path = bound_input(member["path"], member["sha256"], seal, hashes)
            info = manifest["masks"][view]
            checks.require(
                (root / info["path"]).resolve() == path and info["sha256"] == member["sha256"],
                "manifest mask differs",
            )
            with np.load(path, allow_pickle=False) as archive:
                masks[view] = {k: archive[k].copy() for k in archive.files}
            packed = masks[view]
            checks.require(
                np.array_equal(packed["frame_idx"], np.arange(66400, 68060))
                and np.array_equal(packed["image_shape_hw"], [1200, 1200]),
                "native mask layout differs",
            )
            mask_hashes[str(path)] = member["sha256"]
        frames, membership, mask_evidence = [], [], []
        for native in FRAMES:
            decoded = {}
            t = int(native) - chunk["start"]
            for view in VIEWS:
                packed = masks[view]
                value = decode_frame_masks(packed, t)
                digest = packets.array_sha256(value)
                checks.require(
                    digest == bytes(packed["frame_mask_sha256"][t]).decode(),
                    "decoded native mask differs",
                )
                decoded[view] = value if packed["view_available"][t] else None
                mask_evidence.append(
                    dict(
                        frame_idx=int(native),
                        view=view,
                        available=bool(packed["view_available"][t]),
                        decoded_mask_sha256=digest,
                    )
                )
            frame = SimpleNamespace(
                frame_idx=int(native),
                masks=decoded,
                images_bgr={v: SimpleNamespace(shape=(1200, 1200, 3)) for v in VIEWS},
            )
            # Lazy NPZ: access these four fields only, never proposal/retained poses.
            with np.load(PACKET_ROOT / f"f{native}_packet.npz", allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in FIT_KEYS}
            values, audit = prepare_frame(frame, config, saved)
            frames.append(values)
            membership.extend(audit)
        arrays = dict(
            frame_idx=FRAMES.copy(),
            views=np.asarray(VIEWS),
            **{key: np.stack([row[key] for row in frames]) for key in frames[0]},
        )
        np.savez_compressed(output / "samples.npz", **arrays)
        emit("samples.npz")
        emit("arrays.json", packets.signature(arrays))
        emit("membership.json", membership)
        emit("native_mask_evidence.json", mask_evidence)
        emit(
            "metadata.json",
            dict(
                schema=STAGE,
                frames=FRAMES.tolist(),
                native_interval=[67663, 67696],
                counts=list(sampler.COUNTS),
                seed_offset=sampler.SEED_OFFSET,
                packet_root=str(PACKET_ROOT),
                packet_seal_sha256=PACKET_SEAL_SHA,
                packet_metadata_sha256=hashes[str(PACKET_ROOT / "metadata.json")],
                sampler_config=config,
                calibration_directory=meta["calibration_directory"],
                SL_and_calibration_sha256=shared,
                mask_sha256=mask_hashes,
                stratum_order=["foreground", "near-background", "global-background"],
                scope=(
                    "fixed spatially unused pixels in the same development frames; "
                    "no temporal holdout"
                ),
            ),
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[67663, 67696],
            frames=33,
            views=3,
            samples_per_view=sum(sampler.COUNTS),
            total_samples=33 * 3 * sum(sampler.COUNTS),
            regenerated_fit_fields_byte_exact=True,
            fitted_pixel_overlap=0,
            decoded_masks=99,
            raw_images_decoded=0,
            pose_fields_read=False,
            fitted_results_read=False,
            render_calls=0,
            fit_calls=0,
            CUDA_calls=0,
            temporal_holdout=False,
            biological_ID_truth=False,
        )
        emit("summary.json", summary)
        (output / "RESULT.md").write_text(
            "# 局所fit33時刻の固定未使用画素\n\n"
            "全33時刻・3視点に各256画素、合計25,344画素を固定した。"
            "元fitの4配列をD2 maskから再生成してbyte一致を確認し、その全座標を除外した。\n\n"
            "COUNTS=(96,96,64)、seed=frame*7+view+8803010。"
            "マスク99枚のみ復元し、原画decode・poseフィールド・fit結果の読込・描画は行っていない。"
            "同じ開発フレームの空間再標本化であり、時間holdoutやID精度評価ではない。\n",
            encoding="utf-8",
        )
        emit("RESULT.md")
        checks.require(
            code == packets.comparison.source_registry.local_code(), "loaded code set changed"
        )
        for path, digest in hashes.items():
            record.input(path, path, sha256=digest)
        packets.final_guard(hashes, outputs)
        record.param("all_source_input_snapshot_output_SHA_unchanged", True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))
