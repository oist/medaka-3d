"""Explicit CPU one-crop smoke after reviewed official asset acquisition.

This file does not fetch assets or run on import. A caller must supply the exact
completed acquisition seal and a float32 normalized CHW NPY crop SHA. No Torch Hub,
package installation, remote imports, training, GPU or identity experiment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import fetch_dinov2_assets as assets
import numpy as np

STAGE = "official_dinov2_vits14_cpu_one_crop_smoke_v1"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def verify_assets(root, expected_seal_sha256):
    root = Path(root).resolve(strict=True)
    if not re.fullmatch("[0-9a-f]{64}", expected_seal_sha256):
        raise ValueError("explicit acquisition seal SHA256 required")
    seal_path = root / "acquisition.provenance.json"
    if assets.sha(seal_path) != expected_seal_sha256:
        raise ValueError("acquisition seal SHA differs")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("stage") != assets.STAGE
        or seal.get("status") != "ok"
        or not seal.get("finished_at")
    ):
        raise ValueError("successful official acquisition required")
    names = {x["path"] for x in assets.requests()} | {
        "assets.manifest.json",
        "acquisition_plan.json",
        "fetch_dinov2_assets.py",
    }
    output_roles = {Path(name).as_posix(): item for name, item in seal["outputs"].items()}
    if len(output_roles) != len(seal["outputs"]) or set(output_roles) != names:
        raise ValueError("exact official asset output membership required")
    source_files = {
        p.relative_to(root / "source").as_posix()
        for p in (root / "source").rglob("*")
        if p.is_file()
    }
    if source_files != set(assets.SOURCE_PATHS):
        raise ValueError("unexpected source or bytecode file in official source tree")
    evidence = {str(seal_path): expected_seal_sha256}
    for name, item in output_roles.items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or Path(item["path"]).resolve() != path:
            raise ValueError("asset output escaped or changed role")
        if path.stat().st_size != item["size_bytes"] or assets.sha(path) != item["sha256"]:
            raise ValueError("asset bytes changed: " + name)
        evidence[str(path)] = item["sha256"]
    manifest = json.loads((root / "assets.manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "official_dinov2_vits14_assets_v1"
        or manifest.get("status") != "ok"
        or manifest.get("commit") != assets.COMMIT
        or manifest.get("model") != assets.MODEL
        or manifest.get("with_registers") is not False
        or manifest.get("license") != "Apache-2.0"
    ):
        raise ValueError("registered standard DINOv2 ViT-S/14 specification differs")
    if len(manifest["assets"]) != len(assets.requests()):
        raise ValueError("asset list membership differs")
    for item, request in zip(manifest["assets"], assets.requests(), strict=True):
        if any(item[k] != request[k] for k in request):
            raise ValueError("official origin/path/order/cap changed")
        bound = output_roles[item["path"]]
        if item["sha256"] != bound["sha256"] or item["size_bytes"] != bound["size_bytes"]:
            raise ValueError("manifest and acquisition output bytes disagree")
    source_sha = assets.sha(assets.__file__)
    if (
        manifest["acquisition_source_sha256"] != source_sha
        or seal["inputs"] != {str(Path(assets.__file__).resolve()): source_sha}
        or evidence[str(root / "fetch_dinov2_assets.py")] != source_sha
    ):
        raise ValueError("acquisition implementation binding differs")
    evidence[str(Path(assets.__file__).resolve())] = source_sha
    return manifest, evidence


def prepare_input(chw):
    """Preserve the crop utility's exact normalized values; add only batch axis."""
    if (
        not isinstance(chw, np.ndarray)
        or chw.dtype != np.float32
        or chw.shape != (3, 224, 224)
        or not np.isfinite(chw).all()
    ):
        raise ValueError("finite pre-normalized float32[3,224,224] crop required")
    return np.array(chw[None], dtype=np.float32, order="C", copy=True)


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(assets.REPO / "work/three_body") or output.exists():
        raise ValueError("new workspace smoke output required")
    manifest, evidence = verify_assets(args.assets, args.expected_acquisition_sha256)
    crop = args.crop_normalized.resolve(strict=True)
    if not re.fullmatch("[0-9a-f]{64}", args.expected_crop_sha256):
        raise ValueError("explicit original crop file SHA required")
    if assets.sha(crop) != args.expected_crop_sha256:
        raise ValueError("crop file SHA differs")
    evidence[str(crop)] = args.expected_crop_sha256
    values = prepare_input(np.load(crop, allow_pickle=False))
    if any(name == "dinov2" or name.startswith("dinov2.") for name in sys.modules):
        raise ValueError("fresh process without previously imported DINOv2 required")
    source_root = (args.assets.resolve() / "source").resolve(strict=True)
    evidence[str(Path(__file__).resolve())] = assets.sha(__file__)
    output.mkdir(parents=True)
    outputs = {}
    started = assets.now()
    try:
        # These settings affect only this explicitly invoked short-lived process.
        os.environ["XFORMERS_DISABLED"] = "1"
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source_root))
        import torch

        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(0)
        if torch.cuda.is_initialized():
            raise ValueError("CPU smoke must not initialize CUDA")
        backbones = importlib.import_module("dinov2.hub.backbones")
        flags = {}
        for name in ("attention", "block", "swiglu_ffn"):
            module = importlib.import_module("dinov2.layers." + name)
            flags[name] = dict(enabled=module.XFORMERS_ENABLED, available=module.XFORMERS_AVAILABLE)
            if module.XFORMERS_ENABLED or module.XFORMERS_AVAILABLE:
                raise ValueError("official non-xFormers CPU path required")
        model = backbones.dinov2_vits14(pretrained=False)
        state = torch.load(
            args.assets / "weights/dinov2_vits14_pretrain.pth",
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, dict) or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) and v.device.type == "cpu"
            for k, v in state.items()
        ):
            raise ValueError("plain CPU tensor state dictionary required; no pickle fallback")
        model.load_state_dict(state, strict=True)
        del state
        model.eval().requires_grad_(False)
        model.to(device="cpu", dtype=torch.float32)
        tensor = torch.from_numpy(values)
        elapsed, tokens = [], []
        with torch.inference_mode():
            for _ in range(2):
                begin = time.perf_counter()
                result = model.forward_features(tensor)
                elapsed.append(time.perf_counter() - begin)
                tokens.append(result["x_norm_patchtokens"].detach().cpu().clone())
        first, second = (token.numpy() for token in tokens)
        if (
            tokens[0].shape != (1, 256, 384)
            or not torch.isfinite(tokens[0]).all()
            or first.dtype != second.dtype
            or first.shape != second.shape
            or first.tobytes() != second.tobytes()
        ):
            raise ValueError("CPU patch output shape/finite/repeatability check failed")
        norms = torch.linalg.vector_norm(tokens[0], dim=-1, keepdim=True)
        if not (norms > 0).all() or torch.cuda.is_initialized():
            raise ValueError("zero descriptor or unexpected CUDA initialization")
        loaded = {}
        for name, module in list(sys.modules.items()):
            if name == "dinov2" or name.startswith("dinov2."):
                path = Path(module.__file__).resolve(strict=True)
                if not path.is_relative_to(source_root) or str(path) not in evidence:
                    raise ValueError("unregistered DINOv2 implementation was imported")
                loaded[name] = dict(path=str(path), sha256=assets.sha(path))
        np.savez_compressed(
            output / "patch_tokens.npz",
            normalized_input=values,
            final_layernorm_patch_tokens=tokens[0].numpy(),
            unit_patch_tokens=(tokens[0] / norms).numpy(),
        )
        summary = dict(
            schema=STAGE,
            status="ok",
            model=assets.MODEL,
            commit=assets.COMMIT,
            asset_manifest_sha256=assets.sha(args.assets / "assets.manifest.json"),
            crop_sha256=args.expected_crop_sha256,
            crop_count=1,
            forward_calls=2,
            repeat_same_crop_byte_exact=True,
            patch_shape=[1, 256, 384],
            final_layernorm_not_global_pool=True,
            device="cpu",
            dtype="float32",
            torch_cpu_threads=2,
            XFORMERS_DISABLED=True,
            XFORMERS_module_flags=flags,
            torch_load_weights_only=True,
            strict_state_load=True,
            runtime=assets.runtime(),
            loaded_source=loaded,
            forward_seconds=elapsed,
            GPU_used=False,
            training=False,
            identity_experiment=False,
            biological_ID_truth=False,
            preprocessing=dict(
                mean=MEAN,
                std=STD,
                resize_in_this_runner=False,
                normalization_in_this_runner=False,
                original_crop_geometry_external=True,
            ),
            acquisition_runtime=manifest["runtime"],
        )
        assets.write_json(output / "summary.json", summary)
        for code in (Path(__file__), Path(assets.__file__)):
            shutil.copyfile(code, output / code.name)
            if assets.sha(output / code.name) != evidence[str(code.resolve())]:
                raise ValueError("smoke source snapshot changed")
        for path in output.iterdir():
            outputs[path.name] = dict(path=str(path), sha256=assets.sha(path))
        if any(assets.sha(path) != digest for path, digest in evidence.items()):
            raise ValueError("asset/input/loaded source changed during smoke")
        if any(assets.sha(v["path"]) != v["sha256"] for v in outputs.values()):
            raise ValueError("output changed before seal")
        assets.write_json(
            output / "smoke.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=assets.now(),
                inputs=evidence,
                outputs=outputs,
            ),
        )
    except BaseException as error:
        assets.write_json(
            output / "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error=repr(error),
                started_at=started,
                finished_at=assets.now(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--expected-acquisition-sha256", required=True)
    parser.add_argument("--crop-normalized", type=Path, required=True)
    parser.add_argument("--expected-crop-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
