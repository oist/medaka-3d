"""Record deterministic recovery after isolated synthetic position disturbances."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from adapter import AdapterConfig, track

from medaka3d.provenance import provenance

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "work/sleap_pipeline_adapter_v1"


def main():
    cases = (
        ("constant_velocity", 0.0, AdapterConfig(), 1, 1, 1),
        ("single_9px_offset", 9.0, AdapterConfig(), 1, 1, 1),
        ("single_9px_offset_strict_margin", 9.0, replace(AdapterConfig(), margin=0.30), 3, 1, 7),
        ("single_80px_offset_birth", 80.0, AdapterConfig(), 2, 2, 6),
    )
    results = []
    with provenance("sleap_adapter_synthetic_recovery", OUT) as record:
        record.input("check_code", Path(__file__),
                     sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        source = HERE / "adapter.py"
        record.input("adapter", source, sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        for name, offset, config, expected_fragments, expected_histories, recovery in cases:
            frames = np.arange(120, dtype=np.int64)
            x = 0.2 * frames + np.where(frames >= 5, offset, 0.0)
            points = np.zeros((len(frames), 3, 2), dtype=float)
            points[:, :, 0] = x[:, None] + np.array([6.0, 0.0, -18.0])
            points[:, :, 1] = 100.0
            result = track(frames, points, np.ones(len(frames)), frames, "side1_cam", config=config)
            fragments = len(np.unique(result.labels))
            histories = len(np.unique(result.provisional_labels))
            accepted = {edge["to_frame"] for edge in result.edges}
            assert fragments == expected_fragments and histories == expected_histories
            assert all(frame in accepted for frame in range(recovery, 120))
            results.append(
                {
                    "case": name,
                    "config": asdict(config),
                    "frames": 120,
                    "detections_per_frame": 1,
                    "missing_nodes": 0,
                    "velocity_px_per_frame": 0.2,
                    "position_offset_at_f5_px": offset,
                    "confirmed_fragments": fragments,
                    "provisional_histories": histories,
                    "confirmed_adjacent_connections": len(result.edges),
                    "every_link_from_frame_confirmed": recovery,
                    "maximum_active_histories": result.metadata["max_active_tracks"],
                    "scope": "Synthetic engineering contract, not real-fish identity accuracy",
                }
            )
        path = OUT / "recovery_check.json"
        path.write_text(
            json.dumps({"status": "passed", "cases": results}, indent=2), encoding="utf-8"
        )
        record.output("recovery", path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    print("Four synthetic recovery cases passed; no persistent fragmentation")


if __name__ == "__main__":
    main()
