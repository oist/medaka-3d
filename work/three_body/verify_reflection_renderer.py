"""Bounded CPU parity against sealed saved responses; no pose fit or raw decode."""

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/20260909_three_body"
sys.path.insert(0, str(LAB))
from reflection_renderer import ReflectionFishRenderer  # noqa: E402
from renderer import RendererConfig  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402

SOURCE = REPO / "work/three_body/reflection_probe_13frames_v1"
OUTPUT = REPO / "work/three_body/reflection_renderer_saved_parity_v1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(bindings):
    for path, expected in bindings.items():
        if sha(path) != expected:
            raise ValueError(f"source/input changed: {path}")


def main():
    codes = {str(Path(__file__).resolve()): sha(__file__)}
    for module in tuple(sys.modules.values()):
        value = getattr(module, "__file__", None)
        if value:
            path = Path(value).resolve()
            if path.suffix == ".py" and (
                path.is_relative_to(LAB) or path.is_relative_to(REPO / "src")
            ):
                codes[str(path)] = sha(path)
    OUTPUT.mkdir(exist_ok=False)
    with provenance("saved_reflection_renderer_cpu_parity", OUTPUT, name="audit") as record:
        old_seal_path = SOURCE / "probe.provenance.json"
        old_seal = read(old_seal_path)
        if old_seal["status"] != "ok":
            raise ValueError("primary response probe is not complete")
        inputs = {str(old_seal_path): sha(old_seal_path)}
        for entry in old_seal["outputs"].values():
            path = Path(entry["path"]).resolve(strict=True)
            if not path.is_relative_to(SOURCE):
                raise ValueError("foreign output in primary probe seal")
            inputs[str(path)] = entry["sha256"]
        check(inputs)
        plan = read(old_seal["outputs"]["plan"]["path"])
        calib = {}
        for path, expected in plan["input_hashes"].items():
            if "calibration_snapshot" in Path(path).parts:
                calib[path] = expected
        if len(calib) != 4:
            raise ValueError("expected four primary-bound calibration files")
        inputs.update(calib)
        for relative in (
            "labs/20260909_three_body/renderer.py",
            "labs/20260909_three_body/reflection_paths.py",
            "src/medaka3d/calib.py",
            "src/medaka3d/types.py",
        ):
            path = str(REPO / relative)
            if codes[path] != plan["code_hashes"][path]:
                raise ValueError("frozen optical/body source differs from primary probe")
        check(inputs)
        for path, expected in {**inputs, **codes}.items():
            record.input(path, path, sha256=expected)
        snapshots = {}
        for path, expected in codes.items():
            target = OUTPUT / "code_snapshot" / Path(path).relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            snapshots[str(target)] = expected
            record.output(str(target.relative_to(OUTPUT)), target, sha256=expected)
        check(snapshots)
        config = RendererConfig(**read(SOURCE / "summary.json")["renderer_config"])
        rig = load_rig(
            next(Path(p).parent for p in calib if Path(p).name == "calibration_manifest.json")
        )
        rows, saved_arrays = [], {}
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        begin = time.perf_counter()
        try:
            for frame in plan["frames"]:
                pose = np.asarray(read(SOURCE / f"f{frame}_summary.json")["pose_xyz"], float)
                with np.load(SOURCE / f"f{frame}_pixels.npz", allow_pickle=False) as data:
                    points = {k: data[k] for k in data.files}
                for label in ("fitting", "evaluation", "foreground"):
                    with np.load(
                        SOURCE / f"f{frame}_{label}_response.npz", allow_pickle=False
                    ) as data:
                        old = {k: data[k] for k in data.files}
                    if label == "fitting":
                        uv = {v: points["fit_uv"][i] for i, v in enumerate(rig.views)}
                    elif label == "evaluation":
                        uv = {v: points[v + "_eval_uv"] for v in rig.views}
                    else:
                        uv = {v: old[v + "_pixels"] for v in rig.views}
                    indices = {
                        v: np.linspace(0, len(x) - 1, min(32, len(x)), dtype=int)
                        for v, x in uv.items()
                    }
                    selected = {v: uv[v][indices[v]] for v in rig.views}
                    for v in rig.views:
                        saved_arrays[f"f{frame}_{label}_{v}_indices"] = indices[v]
                        saved_arrays[f"f{frame}_{label}_{v}_pixels"] = selected[v]
                    saved_arrays[f"f{frame}_pose_xyz"] = pose
                    for mode in ("finite", "unbounded"):
                        renderer = ReflectionFishRenderer(rig, selected, config, direct_mode=mode)
                        with torch.no_grad():
                            result = renderer.render(pose)
                        fields = (
                            "direct_alpha",
                            "inner_alpha",
                            "outer_alpha",
                            "direct_union",
                            "unit_union",
                            "fresnel_union",
                        )
                        errors = {field: 0.0 for field in fields}
                        for i, v in enumerate(rig.views):
                            ix = indices[v]
                            direct = old["direct_alpha" if mode == "finite" else "unbounded_alpha"][
                                i, ix
                            ]
                            inner, outer = old["inner_alpha"][i, ix], old["outer_alpha"][i, ix]
                            weighted = (
                                direct,
                                inner * old["inner_weight"][i, ix, None],
                                outer * old["outer_weight"][i, ix, None],
                            )
                            expected = dict(
                                direct_alpha=direct,
                                inner_alpha=inner,
                                outer_alpha=outer,
                                direct_union=1 - np.prod(1 - direct, axis=-1),
                                unit_union=1
                                - np.prod(1 - np.stack((direct, inner, outer), -1), axis=(-2, -1)),
                                fresnel_union=1
                                - np.prod(1 - np.stack(weighted, -1), axis=(-2, -1)),
                            )
                            for field, value in expected.items():
                                actual = getattr(result, field)[0, i, : len(ix)].numpy()
                                np.testing.assert_allclose(actual, value, rtol=0, atol=1e-12)
                                errors[field] = max(
                                    errors[field], float(np.max(np.abs(actual - value), initial=0))
                                )
                                saved_arrays[f"f{frame}_{label}_{mode}_{v}_{field}"] = actual
                            np.testing.assert_array_equal(
                                result.ray_valid[i, : len(ix)], old["calibrated_valid"][i, ix]
                            )
                            for branch in ("inner", "outer"):
                                np.testing.assert_array_equal(
                                    getattr(result, branch + "_valid")[i, : len(ix)],
                                    old[v + "_" + branch + "_valid"][ix],
                                )
                                np.testing.assert_array_equal(
                                    getattr(result, branch + "_weight")[i, : len(ix)],
                                    old[branch + "_weight"][i, ix],
                                )
                        rows.append(
                            dict(
                                frame_idx=frame,
                                pixel_group=label,
                                direct_mode=mode,
                                pixels_by_view={v: len(x) for v, x in indices.items()},
                                maximum_absolute_difference=errors,
                            )
                        )
        finally:
            torch.set_num_threads(threads)
        elapsed = time.perf_counter() - begin
        summary = dict(
            status="passed",
            device="cpu",
            dtype="float64",
            frames=plan["frames"],
            selected_pixels_per_view_per_group_max=32,
            primary_outputs_verified=len(old_seal["outputs"]),
            selection="equally spaced integer indices from each saved pixel group; no response selection",
            frame_group_mode_comparisons=len(rows),
            rows=rows,
            maximum_absolute_difference=max(
                v for r in rows for v in r["maximum_absolute_difference"].values()
            ),
            source_and_inputs_unchanged=True,
            raw_images_decoded=0,
            pose_fits_performed=0,
            gt_opened=False,
            id_decisions=False,
            elapsed_seconds=elapsed,
            versions=dict(numpy=np.__version__, torch=str(torch.__version__)),
            scope="Saved fixed-pose pixel response parity only; not whole-image or optimization parity.",
        )
        target = OUTPUT / "summary.json"
        target.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        record.output("summary", target, sha256=sha(target))
        target = OUTPUT / "selected_responses.npz"
        np.savez_compressed(target, **saved_arrays)
        record.output("selected_responses", target, sha256=sha(target))
        check(inputs)
        check(codes)
        check(snapshots)
        check({entry["path"]: entry["sha256"] for entry in record.outputs.values()})
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
