"""CPU import metadata only: official CoTracker closure, no model construction."""

import importlib
import os
import sys
from pathlib import Path

import compare_temporal_point64_v2 as m
import qualify_cotracker3_cpu as qualification

STAGE = "temporal_point64_v2_official_source_inventory_preflight_v1"
DRIVER_SHA = "2461f383d7c69223c98cb9371058373361a7fb30a409a37d617b60fc2db53790"
QUALIFIER_SHA = "feb8b13adc59208bef6b56223c0bbdebd06f70d7ca7f2c13293d7c8e61eb84e0"


def main():
    output = m.Output(m.WORK / STAGE)
    evidence, captured = m.Evidence(), {}
    started = m.utc()
    status, error = "failed", None
    try:
        m.require(sys.dont_write_bytecode, "Python -B is required")
        own = Path(__file__).resolve()
        evidence.add(own)
        output.write("preflight_source.py", own.read_bytes())
        evidence.add(m.__file__, DRIVER_SHA)
        evidence.add(qualification.__file__, QUALIFIER_SHA)
        evidence.add(m.ASSETS / "acquisition.provenance.json", m.ASSET_SEAL_SHA)
        evidence.add(m.ASSETS / "assets.manifest.json", m.ASSET_MANIFEST_SHA)
        expected = qualification.expected_module_files(m.ASSETS)
        for item in expected.values():
            evidence.add(item["path"], item["sha256"])
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        m.require(not torch.cuda.is_initialized(), "CUDA was initialized")
        m.require(
            not any(k == "cotracker" or k.startswith("cotracker.") for k in sys.modules),
            "fresh official import required",
        )
        rng_before = m.hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        sys.path.insert(0, str(m.ASSETS / "source"))
        try:
            official = importlib.import_module("cotracker.predictor")
        finally:
            del sys.path[0]
        rng_after = m.hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        torch.manual_seed(0)
        loaded = qualification.validate_loaded_modules(expected, sys.modules)
        m.require(
            official.CoTrackerOnlinePredictor.__module__ == "cotracker.predictor",
            "official class origin differs",
        )
        m.loaded_sources(evidence, output, captured)
        inventory = m.read_json(output.root / "source_inventory/inventory_000.json")
        included = {row["module"]: row for row in inventory["included_modules"]}
        for name, item in loaded.items():
            row = included[name]
            m.require(
                row["preferred_source"] == item["path"]
                and {x["path"] for x in row["sources"]} == {item["path"]}
                and captured[item["path"]] == item["sha256"],
                "official module inventory differs: " + name,
            )
        dynamic = [
            row
            for row in inventory["skipped_modules"]
            if row["module"] in ("torch.classes", "torch.ops")
        ]
        m.require(
            {row["module"] for row in dynamic} == {"torch.classes", "torch.ops"},
            "dynamic Torch modules not recorded",
        )
        m.require(
            not any(Path(p).name in ("_classes.py", "_ops.py") for p in captured),
            "pseudo source was included",
        )
        m.require(not torch.cuda.is_initialized(), "CUDA was initialized by import")
        output.write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                passed=True,
                driver_sha256=DRIVER_SHA,
                asset_seal_sha256=m.ASSET_SEAL_SHA,
                asset_manifest_sha256=m.ASSET_MANIFEST_SHA,
                torch_version=torch.__version__,
                torch_threads=torch.get_num_threads(),
                torch_interop_threads=torch.get_num_interop_threads(),
                modules_seen=inventory["modules_seen"],
                local_source_count=len(captured),
                official_module_count=len(loaded),
                official_modules=loaded,
                official_module_sources_match_inventory=True,
                dynamic_modules=dynamic,
                rng_before_official_import_sha256=rng_before,
                rng_after_official_import_sha256=rng_after,
                manual_seed_after_import=0,
                failed_v1_binding_verified=True,
                official_code_imported=True,
                model_constructed=False,
                model_loaded=False,
                model_forward_calls=0,
                weights_read=False,
                raw_image_read=False,
                track_arrays_read=False,
                CUDA_initialized=torch.cuda.is_initialized(),
                GPU_used=False,
                scope="CPU Torch and pinned official source import metadata only",
            ),
        )
        evidence.final_guard()
        output.final_guard()
        status = "success"
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        output.write(
            "preflight.provenance.json",
            dict(
                stage=STAGE,
                status=status,
                started_at=started,
                finished_at=m.utc(),
                error=error,
                inputs=evidence.paths,
                outputs=output.records.copy(),
            ),
        )
        print(
            m.json_bytes(
                dict(status=status, seal_sha256=m.sha(output.root / "preflight.provenance.json"))
            ).decode()
        )


if __name__ == "__main__":
    main()
