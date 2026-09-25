"""Fix spatially unused pixels for all301 local packets before opening fit results.

Only packet fit-sample fields are deserialized. Images are represented by their
registered shape because the frozen D2 sampler uses mask pixels, not BGR values.
This is development-image spatial resampling, not a temporal holdout or ID test.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import prepare_bounded_mouth_fit_packets as packets
import prepare_bounded_mouth_unused_pixels as prior
import prepare_tail_fit_packets as producer
import score_temporal_disjoint_pixels as sampler
from prepare_sequence import decode_frame_masks

from medaka3d.calib import load_rig
from medaka3d.provenance import provenance
from medaka3d.types import VIEWS

checks, REPO = packets.checks, packets.REPO
STAGE = "tail_unused_pixel_preparation_v1"
PACKET_ROOT = REPO / "work/three_body/tail_local_fit_packets_301_v1"
PACKET_SEAL_SHA = "6e0e3a8d7d006a9f3afc673af829f54de77567d1346a7cb613179114f1769cdc"
PACKET_SOURCE_SHA = "900183e8978f150cf9f4a7fa09c2de1799a17a556ade06da8af57d516479d7f3"
SAMPLER_SHA = "8a86b3942d93b5ed506ac85cb3520fa9ccd51a2a11c55201fd1884e4e5c31d47"
FIT_KEYS = ("fit_pixels", "fit_target", "fit_weights", "fit_foreground")
FRAMES = np.r_[np.array([66524], np.int64), np.arange(66620, 66920, dtype=np.int64)]
PRIOR_SHA = "5aa21f5cfc25accd074c67037f72ad6f264d7c8cd8c3fdcef7bee3808abf5a6f"


# Frozen arithmetic is reused without changing its globals or inputs.
prepare_frame = prior.prepare_frame


def coordinate_audit(values, membership, *, shape):
    """Keep draw counts separate from unique coordinates, including cross-stratum reuse."""
    pixels, fitted = values["pixels"], values["fit_pixels"]
    checks.require(
        pixels.shape == (3, 256, 2)
        and fitted.shape[0] == 3
        and fitted.ndim == 3
        and fitted.shape[2] == 2,
        "native sample shape differs",
    )
    result = []
    cuts = np.r_[0, np.cumsum(sampler.COUNTS)]
    for vi, view in enumerate(VIEWS):
        uv, fit_uv = pixels[vi], fitted[vi]
        for a in (uv, fit_uv):
            checks.require(
                np.isfinite(a).all()
                and np.array_equal(a, np.rint(a))
                and (a >= 0).all()
                and (a[:, 0] < shape[1]).all()
                and (a[:, 1] < shape[0]).all(),
                "native pixel bounds/integrality differ",
            )
        unique = len(np.unique(uv, axis=0))
        fit_set = set(map(tuple, fit_uv))
        checks.require(not (set(map(tuple, uv)) & fit_set), "strict fit/evaluation overlap")
        row = dict(membership[vi])
        checks.require(row["view"] == view, "view membership order differs")
        by_class = [len(np.unique(uv[cuts[j] : cuts[j + 1]], axis=0)) for j in range(3)]
        checks.require(by_class == row["selected_unique_by_class"], "draw membership differs")
        row.update(
            selected_draw_count=256,
            selected_unique_all=unique,
            duplicate_draws_including_cross_class=256 - unique,
            selected_draws_by_class=list(sampler.COUNTS),
            duplicate_draws_within_class=[
                n - u for n, u in zip(sampler.COUNTS, by_class, strict=True)
            ],
            fit_draw_count=len(fit_uv),
            fitted_unique_pixels=len(fit_set),
            strict_fit_evaluation_overlap=0,
        )
        result.append(row)
    return result


def sampled_rays(rig, pixels):
    """Same normalized water rays and invalid placeholders as frozen renderer initialization."""
    checks.require(
        pixels.shape == (3, 256, 2) and np.isfinite(pixels).all(),
        "finite native rays require3x256 pixels",
    )
    origins = np.zeros((3, 256, 3), np.float64)
    directions = np.zeros_like(origins)
    directions[..., 2] = 1
    valid = np.zeros((3, 256), bool)
    for vi, view in enumerate(VIEWS):
        ray_o, ray_d = rig.camera(view).rays(pixels[vi])
        length = np.linalg.norm(ray_d, axis=1)
        good = np.isfinite(ray_o).all(axis=1) & np.isfinite(ray_d).all(axis=1)
        good &= length > 1e-12
        origins[vi, good] = ray_o[good]
        directions[vi, good] = ray_d[good] / length[good, None]
        valid[vi, good] = True
    return dict(ray_origins=origins, ray_directions=directions, ray_valid=valid)


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
    checks.require(checks.sha(producer.__file__) == PACKET_SOURCE_SHA, "packet producer changed")
    checks.require(checks.sha(sampler.__file__) == SAMPLER_SHA, "frozen spatial sampler changed")
    checks.require(
        sampler.COUNTS == (96, 96, 64) and sampler.SEED_OFFSET == 8803010,
        "fixed sampling constants changed",
    )
    checks.require(checks.sha(prior.__file__) == PRIOR_SHA, "frozen preparation math changed")
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
        seal = checks.verify(PACKET_ROOT, "preparation.provenance.json", producer.STAGE, hashes)
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
            and meta["native_interval"] == [66524, 66920]
            and meta["allow_bounded"] is False
            and summary["status"] == "ok"
            and summary["packets"] == 301
            and summary["originals"] == 4273
            and summary["added"] == 676
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
        rig = load_rig(meta["calibration_directory"])
        normalized = packets.comparison.support_audit.COMMON.normalize(
            [dataclasses.asdict(camera) for camera in rig.cameras]
        )
        checks.require(
            meta["original_context"]["cameras"] == normalized
            and list(rig.views) == list(VIEWS)
            and rig.world_units == "mm",
            "actual calibration differs from packet context",
        )
        checks.require(
            sum(config[k] for k in ("positive_samples", "near_samples", "global_samples")) == 1024,
            "original 1024 fit samples required",
        )
        chunk = meta["source_chunk"]
        checks.require(
            [chunk["start"], chunk["stop"], chunk["index"]] == [66400, 68060, 10],
            "wrong source chunk",
        )
        stage = Path(meta["checkpoint_binding"]["original_binding"]["inputs"]["input_stage_root"])
        root = (stage / chunk["attempt_path"]).resolve(strict=True)
        checks.require(root.is_relative_to(stage.resolve()), "chunk escaped stage")
        manifest_path = bound_input(root / "manifest.json", chunk["manifest_sha256"], seal, hashes)
        staging_path = bound_input(
            root / "staging.provenance.json", chunk["provenance_sha256"], seal, hashes
        )
        staging = checks.read(staging_path)
        checks.require(
            staging["status"] == "ok" and staging["stage"] == "prepare_independent_mask_chunk",
            "successful fixed mask chunk required",
        )
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
            checks.require(
                Path(staging["outputs"][view + "_masks"]["path"]).resolve() == path
                and staging["outputs"][view + "_masks"]["sha256"] == member["sha256"],
                "sealed mask output role differs",
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
            # Lazy NPZ: native scalar and four fit fields; no proposal/retained poses.
            with np.load(PACKET_ROOT / f"f{native}_packet.npz", allow_pickle=False) as archive:
                checks.require(
                    archive["frame_idx"].shape == () and int(archive["frame_idx"]) == int(native),
                    "native packet index differs",
                )
                saved = {key: archive[key].copy() for key in FIT_KEYS}
            values, audit = prepare_frame(frame, config, saved)
            audit = coordinate_audit(values, audit, shape=(1200, 1200))
            values.update(sampled_rays(rig, values["pixels"]))
            frames.append(values)
            membership.extend(audit)
        arrays = dict(
            frame_idx=FRAMES.copy(),
            views=np.asarray(VIEWS),
            strata=np.broadcast_to(
                np.repeat(np.arange(3, dtype=np.int8), sampler.COUNTS), (301, 3, 256)
            ).copy(),
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
                native_interval=[66524, 66920],
                counts=list(sampler.COUNTS),
                seed_offset=sampler.SEED_OFFSET,
                packet_root=str(PACKET_ROOT),
                packet_seal_sha256=PACKET_SEAL_SHA,
                packet_metadata_sha256=hashes[str(PACKET_ROOT / "metadata.json")],
                sampler_config=config,
                calibration_directory=meta["calibration_directory"],
                SL_and_calibration_sha256=shared,
                mask_sha256=mask_hashes,
                selection="{66524} union [66620,66920)",
                sample_units="draws; duplicates retained according to frozen sampler",
                ray_schema=(
                    "float64 normalized water ray; invalid origin0/direction001 and ray_valid=False"
                ),
                sampler_source_sha256=SAMPLER_SHA,
                original_prepare_arithmetic_sha256=PRIOR_SHA,
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
            native_interval=[66524, 66920],
            frames=301,
            views=3,
            samples_per_view=sum(sampler.COUNTS),
            total_samples=301 * 3 * sum(sampler.COUNTS),
            regenerated_fit_fields_byte_exact=True,
            fitted_pixel_overlap=0,
            strict_fit_evaluation_overlap=0,
            selected_unique_sum_across_frame_views=sum(
                r["selected_unique_all"] for r in membership
            ),
            duplicate_draws_sum_across_frame_views=sum(
                r["duplicate_draws_including_cross_class"] for r in membership
            ),
            valid_water_rays=int(arrays["ray_valid"].sum()),
            decoded_masks=903,
            source_packet_raw_native_frames=303,
            source_packet_raw_images=909,
            masks_cover_current301_frames_only=True,
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
            "# 局所fit301時刻の固定未使用画素\n\n"
            "全301時刻・3視点に各256 draws、合計231,168 drawsを固定した。"
            "元fitの4配列をD2 maskから再生成してbyte一致を確認し、その全座標を除外した。\n\n"
            "COUNTS=(96,96,64)、seed=frame*7+view+8803010。"
            "現在301時刻のマスク903 view-frame stacksのみ復元し、"
            "元packetの303時刻/909BGRとは別の範囲である。"
            "原画decode・poseフィールド・fit結果の読込・描画は行っていない。"
            "drawとuniqueを区別し、同校正の水中ray/validを保存した。"
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
