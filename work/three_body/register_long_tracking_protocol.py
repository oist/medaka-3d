"""Register fixed long-run settings before the first new 49800-frame fit."""

import hashlib
import json
from pathlib import Path

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / "labs/20260909_three_body"


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def main():
    source = ROOT / "work/three_body/persistent_mouth_continuous_one_minute_v1/experiment.provenance.json"
    old = json.loads(source.read_text(encoding="utf-8"))
    assert old["status"] == "ok"
    params = old["params"]
    seed = ROOT / "work/three_body/long_seed_registration_49800_v1/registration.json"
    predecessor = ROOT / "work/three_body/continuation_interval_666_single_v1/continuation.provenance.json"
    earlier = json.loads(predecessor.read_text(encoding="utf-8"))
    assert earlier["status"] == "ok"
    sources = earlier["params"]["bindings"]["code_sha256"]
    names = [
        "tracker.py", "persistent_node_tracker.py", "accelerated_tracker.py", "graph_fit.py",
        "graph_union.py", "continuous_tracker.py", "tracker_checkpoint.py",
        "resumable_observations.py", "source_sequence.py", "observations.py", "node_geometry.py",
        "node_recovery.py", "motion.py", "renderer.py", "fast_union.py",
    ]
    frozen = {}
    for path in [*(LAB / name for name in names), ROOT / "src/medaka3d/calib.py"]:
        digest = sha(path)
        assert sources[str(path)] == digest
        frozen[path.relative_to(ROOT).as_posix()] = digest
    values = {
        "schema": "three_body_long_tracking_protocol_v1",
        "experiment": "20251225_K_1_2_FCMC",
        "source_frame_range": [49800, 348600],
        "fps": 166,
        "seed_registration_sha256": sha(seed),
        "input_protocol_sha256": sha(LAB / "long_input_protocol.json"),
        "initial_operational_pilot_end_exclusive": 51460,
        "settings_source_provenance_sha256": sha(source),
        "frozen_tracker_sources_sha256": frozen,
        **{key: params[key] for key in (
            "tracker_config", "node_recovery_config", "persistence_config", "flow_config",
            "package_versions", "torch_cpu_threads",
        )},
        "limitations": [
            "The first 1660 frames are an operational pilot, not a completed thirty-minute run.",
            "One optimistic initial correspondence derives from a whole-recording C34 result; the long tracker reads only the registered first pose and no later baseline coordinates or human truth.",
            "The initial mouth at z=-0.413303 mm is inside the inherited engineering box [-6,186] in z; the physical tank floor is not newly surveyed or corrected.",
            "Nine fixed pre-window background frames differ from the old one-minute preparation, so input-dependent results are not an identical-input algorithm comparison.",
            "All ordered beams and their complete history are retained; checkpoint storage, validation and restoration grow with processed duration.",
            "A prefix backtrace may change when more observations arrive; a checkpoint retains all alternatives, not only its provisional winning path.",
            "Direct geometry and unflagged acceptance are engineering availability measures, not independent biological ID accuracy or a zero-swap guarantee.",
            "No all-three independent human identity truth has been supplied; ambiguous identities remain flagged and require the permitted human review.",
            "Tracking time excludes detector preprocessing, input hash validation, reader preparation, checkpoint serialization and final result validation.",
        ],
    }
    output = LAB / "long_tracking_protocol.json"
    assert not output.exists()
    seal_root = ROOT / "work/three_body/long_tracking_protocol_registration_v1"
    seal_root.mkdir(exist_ok=False)
    with provenance("register_long_tracking_protocol", seal_root, name="registration") as record:
        for path in [source, predecessor, seed, LAB / "long_input_protocol.json", Path(__file__),
                     *(ROOT / name for name in frozen)]:
            record.input(str(path), path, sha256=sha(path))
        output.write_text(json.dumps(values, indent=2, allow_nan=False), encoding="utf-8")
        record.output("protocol", output, sha256=sha(output))
        record.param("tracking_performed", False)
        record.param("after_initialization_baseline_coordinates_read", False)
    print(json.dumps({"protocol": str(output), "sha256": sha(output)}))


if __name__ == "__main__":
    main()
