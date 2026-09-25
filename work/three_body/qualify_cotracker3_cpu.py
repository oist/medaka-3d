"""Qualify pinned CoTracker3 CPU loading and query initialization, with zero forwards.

The CLI is a standalone CPU process; imports are delayed until all asset checks
pass. It never downloads, installs, decodes video, runs a model forward, or uses
CUDA. The reusable loader constructs on CPU and performs explicit local
weights-only strict loading; callers own their later execution/device policy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORK = REPO / "work/three_body"
ASSETS = WORK / "assets/cotracker3_online_82e02e8_v2"
ASSET_SEAL_SHA = "2c516b4d508f4de588f3ca23f83580194439d43e741463627e32da3a472985f8"
ASSET_MANIFEST_SHA = "befdf1b5ba9a376abfddd5b62afd89b7064661c39f28bdb73b55932c0e253ae8"
WEIGHT_SHA = "205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218"
WEIGHT_SIZE = 101_695_610
COMMIT = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
HF_REVISION = "bf55ea50d4390e1820a267f131cd6587240fb2c5"
STAGE = "cotracker3_cpu_load_qualification_v1"
MODULES = (
    "cotracker",
    "cotracker.models",
    "cotracker.models.build_cotracker",
    "cotracker.models.core",
    "cotracker.models.core.cotracker",
    "cotracker.models.core.cotracker.blocks",
    "cotracker.models.core.cotracker.cotracker",
    "cotracker.models.core.cotracker.cotracker3_offline",
    "cotracker.models.core.cotracker.cotracker3_online",
    "cotracker.models.core.embeddings",
    "cotracker.models.core.model_utils",
    "cotracker.predictor",
)


def now():
    return datetime.now(UTC).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_assets(root):
    root = Path(root).resolve(strict=True)
    require(root == ASSETS.resolve(), "the pinned workspace asset root is required")
    seal_path = root / "acquisition.provenance.json"
    require(sha(seal_path) == ASSET_SEAL_SHA, "asset seal SHA changed")
    seal = read_json(seal_path)
    require(
        seal.get("stage") == "official_cotracker3_online_asset_acquisition_v2"
        and seal.get("status") == "ok"
        and bool(seal.get("finished_at")),
        "completed successful acquisition required",
    )
    evidence = {str(seal_path): ASSET_SEAL_SHA}
    paths = set()
    require(len(seal["outputs"]) == 28, "acquisition output count changed")
    for relative, item in seal["outputs"].items():
        path = (root / relative).resolve(strict=True)
        require(
            path.is_relative_to(root) and Path(item["path"]).resolve() == path, "asset path changed"
        )
        require(
            path.stat().st_size == item["size_bytes"] and sha(path) == item["sha256"],
            "asset bytes changed: " + relative,
        )
        require(path not in paths, "duplicate asset output path")
        paths.add(path)
        evidence[str(path)] = item["sha256"]
    actual = {p.resolve() for p in root.rglob("*") if p.is_file()}
    require(actual == paths | {seal_path}, "asset membership changed, including bytecode")
    for path, expected in seal["inputs"].items():
        require(sha(path) == expected, "acquisition implementation changed")
        evidence[str(Path(path).resolve())] = expected
    require(
        evidence[str(root / "assets.manifest.json")] == ASSET_MANIFEST_SHA,
        "manifest binding changed",
    )
    manifest = read_json(root / "assets.manifest.json")
    require(
        manifest.get("schema") == "official_cotracker3_online_assets_v1"
        and manifest.get("status") == "ok"
        and manifest.get("github_commit") == COMMIT
        and manifest.get("huggingface_revision") == HF_REVISION
        and manifest.get("checkpoint") == "scaled_online.pth"
        and manifest.get("upstream_weight_sha256") == WEIGHT_SHA
        and manifest.get("upstream_weight_size_bytes") == WEIGHT_SIZE,
        "official model metadata changed",
    )
    return evidence


def expected_module_files(root):
    root = Path(root).resolve(strict=True)
    require(sha(root / "assets.manifest.json") == ASSET_MANIFEST_SHA, "asset manifest changed")
    records = {
        x["path"].replace("\\", "/"): x for x in read_json(root / "assets.manifest.json")["assets"]
    }
    files = {}
    for name in MODULES:
        relative = "source/" + name.replace(".", "/")
        relative = relative + ".py" if relative + ".py" in records else relative + "/__init__.py"
        require(relative in records, "required import closure member missing")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root / "source"), "source module escaped asset directory")
        require(sha(path) == records[relative]["sha256"], "official source bytes changed")
        files[name] = dict(path=str(path), sha256=records[relative]["sha256"])
    return files


def validate_loaded_modules(expected, modules):
    loaded = {
        name: module
        for name, module in modules.items()
        if name == "cotracker" or name.startswith("cotracker.")
    }
    require(set(loaded) == set(expected), "actual CoTracker import closure differs")
    for name, item in expected.items():
        require(
            getattr(loaded[name], "__file__", None) is not None, "loaded module lacks source path"
        )
        actual = Path(loaded[name].__file__).resolve(strict=True)
        require(
            actual == Path(item["path"]) and sha(actual) == item["sha256"],
            "loaded official module path/SHA changed: " + name,
        )
    return expected


def unwrap_checkpoint(state, torch_module):
    require(isinstance(state, dict), "checkpoint must be a state dictionary")
    unwrapped = "model" in state
    if unwrapped:
        state = state["model"]
    require(isinstance(state, dict) and bool(state), "nonempty model state dictionary required")
    require(
        all(isinstance(k, str) and isinstance(v, torch_module.Tensor) for k, v in state.items()),
        "all model state entries must be named tensors",
    )
    require(
        all(v.device.type == "cpu" for v in state.values()), "checkpoint tensors must load on CPU"
    )
    return state, unwrapped


def load_cpu_predictor(asset_root, *, torch_module):
    """Return (new CPU predictor, load report), with no forward or thread changes.

    A caller must verify the complete acquisition seal separately. This loader
    rechecks the fixed manifest, all 12 executable files and checkpoint bytes
    before and after loading. Existing modules are accepted only from this exact
    closure. All official imports happen here, followed by manual_seed(0).
    """
    torch = torch_module
    root = Path(asset_root).resolve(strict=True)
    require(root == ASSETS.resolve(), "pinned asset root required")
    require(sys.dont_write_bytecode, "use Python -B to preserve the sealed source directory")
    expected = expected_module_files(root)
    weight = root / "weights/scaled_online.pth"
    require(
        weight.stat().st_size == WEIGHT_SIZE and sha(weight) == WEIGHT_SHA,
        "checkpoint bytes differ",
    )
    preexisting = sorted(n for n in sys.modules if n == "cotracker" or n.startswith("cotracker."))
    if preexisting:
        validate_loaded_modules(expected, sys.modules)
    rng_before = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    sys.path.insert(0, str(root / "source"))
    try:
        official = importlib.import_module("cotracker.predictor")
    finally:
        del sys.path[0]
    loaded = validate_loaded_modules(expected, sys.modules)
    rng_after_import = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    torch.manual_seed(0)
    rng_after_seed = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
    predictor = official.CoTrackerOnlinePredictor(
        checkpoint=None, offline=False, v2=False, window_len=16
    ).eval()
    checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
    state, unwrapped = unwrap_checkpoint(checkpoint, torch)
    incompatible = predictor.model.load_state_dict(state, strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "strict state load reported mismatches",
    )
    loaded_state = predictor.model.state_dict()
    require(set(loaded_state) == set(state), "loaded state tensor names changed")
    for name, value in loaded_state.items():
        require(value.device.type == "cpu", "model state is not on CPU")
        require(
            value.dtype == state[name].dtype
            and tuple(value.shape) == tuple(state[name].shape)
            and torch.equal(value, state[name]),
            "loaded tensor differs from checkpoint: " + name,
        )
        if value.is_floating_point():
            require(value.dtype == torch.float32, "model floating state must remain float32")
    require(
        predictor.step == 8
        and predictor.model.window_len == 16
        and tuple(predictor.interp_shape) == (384, 512)
        and predictor.model.num_virtual_tracks == 64,
        "official online model configuration differs",
    )
    require(sha(weight) == WEIGHT_SHA, "checkpoint changed during loading")
    validate_loaded_modules(expected_module_files(root), sys.modules)
    parameters = list(predictor.model.named_parameters())
    buffers = list(predictor.model.named_buffers())
    report = dict(
        asset_seal_sha256=ASSET_SEAL_SHA,
        asset_manifest_sha256=ASSET_MANIFEST_SHA,
        checkpoint_sha256=WEIGHT_SHA,
        checkpoint_size_bytes=WEIGHT_SIZE,
        weights_only=True,
        strict_load=True,
        checkpoint_model_key_unwrapped=unwrapped,
        missing_keys=list(incompatible.missing_keys),
        unexpected_keys=list(incompatible.unexpected_keys),
        loaded_state_tensors_equal=True,
        state_tensor_count=len(state),
        parameter_tensor_count=len(parameters),
        parameter_count=sum(p.numel() for _, p in parameters),
        buffer_tensor_count=len(buffers),
        buffer_count=sum(b.numel() for _, b in buffers),
        state_dtypes=sorted({str(v.dtype) for v in loaded_state.values()}),
        state_devices=sorted({str(v.device) for v in loaded_state.values()}),
        loaded_modules=loaded,
        official_modules_preexisting=preexisting,
        import_seed_side_effect_source_modules=[
            "cotracker.models.core.cotracker." + x
            for x in ("cotracker", "cotracker3_online", "cotracker3_offline")
        ],
        rng_before_official_import_sha256=rng_before,
        rng_after_official_import_sha256=rng_after_import,
        manual_seed_after_import=0,
        rng_after_manual_seed_sha256=rng_after_seed,
        window_len=16,
        step=8,
        model_resolution=[384, 512],
        num_virtual_tracks=64,
        model_forward_calls=0,
    )
    return predictor, report


def check_initialization(predictor, *, torch_module):
    """One synthetic query initialization; a prehook rejects every model forward."""
    torch = torch_module
    video = torch.zeros((1, 1, 3, 1200, 1200), dtype=torch.float32, device="cpu")
    queries = torch.tensor(
        [[[0.0, 123.0, 456.0], [0.0, 1199.0, 1199.0]]], dtype=torch.float32, device="cpu"
    )
    before = queries.clone()
    expected = before.clone()
    expected[:, :, 1:] *= torch.tensor([511.0 / 1199.0, 383.0 / 1199.0], dtype=torch.float32)
    forward_calls = 0

    def forbidden_forward(module, args):
        nonlocal forward_calls
        forward_calls += 1
        raise ValueError("model forward forbidden in CPU loading qualification")

    hook = predictor.model.register_forward_pre_hook(forbidden_forward)
    try:
        result = predictor(
            video_chunk=video,
            is_first_step=True,
            queries=queries,
            grid_size=0,
            add_support_grid=False,
        )
    finally:
        hook.remove()
    require(
        isinstance(result, tuple) and len(result) == 2 and all(x is None for x in result),
        "official initialization return changed",
    )
    require(forward_calls == 0, "initialization invoked model forward")
    require(
        torch.equal(queries, before) and torch.count_nonzero(video).item() == 0,
        "initialization mutated synthetic input",
    )
    require(
        predictor.queries.data_ptr() != queries.data_ptr()
        and torch.equal(predictor.queries, expected),
        "query clone/coordinate scaling differs",
    )
    require(
        predictor.N == 2 and predictor.model.online_ind == 0, "initial query/online index differs"
    )
    require(
        predictor.model.online_track_feat == [None] * 4
        and predictor.model.online_track_support == [None] * 4,
        "initial feature cache differs",
    )
    require(
        all(
            getattr(predictor.model, name) is None
            for name in ("online_coords_predicted", "online_vis_predicted", "online_conf_predicted")
        ),
        "initial prediction cache is not empty",
    )
    report = dict(
        initialization_calls=1,
        initialization_checked=True,
        model_forward_calls=forward_calls,
        video_shape=list(video.shape),
        video_dtype=str(video.dtype),
        queries_count=2,
        queries_native_shape=list(queries.shape),
        query_times_relative=True,
        queries_input_unchanged=True,
        queries_cloned=True,
        queries_scaled_exact=True,
        video_input_unchanged=True,
        add_support_grid=False,
        public_return=[None, None],
        online_ind=0,
        empty_feature_cache_levels=4,
        empty_prediction_caches=True,
    )
    arrays = dict(
        queries_native=queries.numpy().copy(),
        queries_internal=predictor.queries.numpy().copy(),
        queries_expected=expected.numpy().copy(),
    )
    return report, arrays


def qualify(output):
    output = Path(output).resolve()
    require(
        output.parent == WORK.resolve() and not output.exists(),
        "new direct workspace output required",
    )
    require(
        sys.dont_write_bytecode and "torch" not in sys.modules,
        "fresh standalone Python -B process required",
    )
    require(
        not any(n == "cotracker" or n.startswith("cotracker.") for n in sys.modules),
        "CoTracker must not already be imported",
    )
    output.mkdir()
    started, start_time = now(), time.perf_counter()
    source = Path(__file__).resolve()
    inputs = {str(source): sha(source)}
    status, error, summary = "failed", None, None
    torch = None
    snapshot = output / "code_snapshot"
    snapshot.mkdir()
    shutil.copyfile(source, snapshot / source.name)
    snapshot_expected = {str(snapshot / source.name): inputs[str(source)]}
    try:
        inputs.update(verify_assets(ASSETS))
        for item in expected_module_files(ASSETS).values():
            path = Path(item["path"])
            target = snapshot / path.relative_to(ASSETS)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(sha(target) == item["sha256"], "official source snapshot changed")
            snapshot_expected[str(target)] = item["sha256"]
        env_file = REPO / ".venv/pyvenv.cfg"
        inputs[str(env_file)] = sha(env_file)
        shutil.copyfile(env_file, snapshot / "pyvenv.cfg")
        snapshot_expected[str(snapshot / "pyvenv.cfg")] = inputs[str(env_file)]
        previous_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch = importlib.import_module("torch")
        require(not torch.cuda.is_initialized(), "CUDA initialized during Torch import")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        require(torch.get_default_dtype() == torch.float32, "default Torch dtype differs")
        predictor, load_report = load_cpu_predictor(ASSETS, torch_module=torch)
        require(
            load_report["official_modules_preexisting"] == [], "standalone import was not fresh"
        )
        initialization, arrays = check_initialization(predictor, torch_module=torch)
        require(not torch.cuda.is_initialized(), "CUDA initialized during CPU qualification")
        require(
            torch.get_num_threads() == 2 and torch.get_num_interop_threads() == 1,
            "CPU thread settings changed",
        )
        numpy = importlib.import_module("numpy")
        with (output / "initialization.npz").open("xb") as stream:
            numpy.savez(stream, **arrays)
        with numpy.load(output / "initialization.npz", allow_pickle=False) as archive:
            require(set(archive.files) == set(arrays), "initialization output membership changed")
            for name, value in arrays.items():
                saved = archive[name]
                require(
                    saved.dtype == value.dtype
                    and saved.shape == value.shape
                    and saved.tobytes() == value.tobytes(),
                    "initialization output roundtrip differs",
                )
        require(
            verify_assets(ASSETS)
            == {k: v for k, v in inputs.items() if k not in (str(source), str(env_file))},
            "final acquisition binding changed",
        )
        for path, expected_sha in inputs.items():
            require(sha(path) == expected_sha, "input/source changed before qualification success")
        for path, expected_sha in snapshot_expected.items():
            require(sha(path) == expected_sha, "qualification code snapshot changed")
        summary = dict(
            schema=STAGE,
            status="ok",
            passed=True,
            asset_seal_sha256=ASSET_SEAL_SHA,
            asset_manifest_sha256=ASSET_MANIFEST_SHA,
            weights_only=True,
            strict_load=True,
            missing_keys=load_report["missing_keys"],
            unexpected_keys=load_report["unexpected_keys"],
            parameter_count=load_report["parameter_count"],
            buffer_count=load_report["buffer_count"],
            model_forward_calls=0,
            model_imported=True,
            checkpoint_deserialized_on_CPU=True,
            initialization_checked=True,
            initialization=initialization,
            load_report=load_report,
            torch_version=torch.__version__,
            numpy_version=numpy.__version__,
            python=sys.version,
            cpu_threads=2,
            cpu_interop_threads=1,
            cuda_initialized=False,
            CUDA_VISIBLE_DEVICES="",
            prior_CUDA_VISIBLE_DEVICES=previous_visible_devices,
            GPU_used=False,
            raw_images_read=False,
            videos_decoded=False,
            flow_computed=False,
            tracking_performed=False,
            model_inference_performed=False,
            packages_installed=False,
            physical_ID_qualified=False,
            wall_seconds=time.perf_counter() - start_time,
        )
        write_json(output / "summary.json", summary)
        status = "ok"
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        write_json(
            output / "failure.json",
            dict(
                schema=STAGE,
                status="failed",
                error=error,
                failed_at=now(),
                cuda_initialized=None if torch is None else torch.cuda.is_initialized(),
                successful_qualification=False,
            ),
        )
        raise
    finally:
        outputs = {
            str(p.relative_to(output)): dict(
                path=str(p), sha256=sha(p), size_bytes=p.stat().st_size
            )
            for p in sorted(output.rglob("*"))
            if p.is_file()
        }
        write_json(
            output / "qualification.provenance.json",
            dict(
                stage=STAGE,
                status=status,
                started_at=started,
                finished_at=now(),
                inputs=inputs,
                outputs=outputs,
                error=error,
                no_model_forward=True if status == "ok" else None,
                no_GPU=torch is None or not torch.cuda.is_initialized(),
                no_raw_decode=True,
                no_package_install=True,
            ),
        )
    print(
        json.dumps(
            dict(
                status="ok",
                output=str(output),
                parameter_count=summary["parameter_count"],
                buffer_count=summary["buffer_count"],
                qualification_seal_sha256=sha(output / "qualification.provenance.json"),
            )
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    qualify(parser.parse_args().output)
