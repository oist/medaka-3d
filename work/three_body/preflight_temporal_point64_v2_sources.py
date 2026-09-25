"""CPU Torch import/source inventory only; no model, images, forward or GPU use."""

import os
import sys
from pathlib import Path

import compare_temporal_point64_v2 as m


def main():
    output = m.Output(m.WORK / "temporal_point64_v2_source_inventory_preflight_v1")
    evidence = m.Evidence()
    captured = {}
    started = m.utc()
    status, error = "failed", None
    try:
        path = Path(__file__).resolve()
        evidence.add(path)
        output.write("preflight_source.py", path.read_bytes())
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import torch

        m.require(not torch.cuda.is_initialized(), "CUDA was initialized")
        m.require(not any(k.startswith("cotracker") for k in sys.modules), "model was imported")
        m.loaded_sources(evidence, output, captured)
        inventory = m.read_json(output.root / "source_inventory/inventory_000.json")
        dynamic = [
            row
            for row in inventory["skipped_modules"]
            if row["module"] in ("torch.classes", "torch.ops")
        ]
        m.require(
            {row["module"] for row in dynamic} == {"torch.classes", "torch.ops"},
            "dynamic modules absent",
        )
        m.require(
            not any(Path(p).name in ("_classes.py", "_ops.py") for p in captured),
            "pseudo source included",
        )
        output.write(
            "summary.json",
            dict(
                schema="temporal_point64_v2_source_inventory_preflight_v1",
                status="ok",
                passed=True,
                torch_version=torch.__version__,
                modules_seen=inventory["modules_seen"],
                local_source_count=len(captured),
                dynamic_modules=dynamic,
                current_driver_sha256=m.sha(m.__file__),
                failed_v1_binding_verified=True,
                CUDA_initialized=torch.cuda.is_initialized(),
                model_imported=False,
                model_loaded=False,
                model_forward_calls=0,
                raw_image_read=False,
                track_arrays_read=False,
                GPU_used=False,
                scope=(
                    "CPU Torch import and current source inventory; "
                    "future model imports not exercised"
                ),
            ),
        )
        evidence.final_guard()
        output.final_guard()
        status = "success"
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        seal = dict(
            stage="temporal_point64_v2_source_inventory_preflight_v1",
            status=status,
            started_at=started,
            finished_at=m.utc(),
            error=error,
            inputs=evidence.paths,
            outputs=output.records.copy(),
        )
        output.write("preflight.provenance.json", seal)
        print(
            m.json_bytes(
                dict(status=status, seal_sha256=m.sha(output.root / "preflight.provenance.json")),
            ).decode()
        )


if __name__ == "__main__":
    main()
