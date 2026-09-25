"""Read four saved history arrays to diagnose pre-proposal mouth age eligibility.

This deliberately reads part of the new78020 checkpoint for a new question;
it is not a repeat of the producer's39-state roundtrip or an independent ID audit.
No restore, fitting, frame/mask decoding, or GPU operation is performed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
ROOT = WORK / "instance_rescore_ten_second_continuation_v1"
SEAL_SHA = "a71ddab8dfef8d97fecf68c1d5b7a5d7300fe21dfe49b48bc6f1df91c840196a"
FRAMES = (77709, 77723, 77880)
FIELDS = (
    "history_offsets",
    "history_frame_idx",
    "history_node_missing",
    "history_direct_pectoral_age",
)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert digest(ROOT / "tracking.provenance.json") == SEAL_SHA
    seal = json.loads((ROOT / "tracking.provenance.json").read_text("utf8"))
    assert seal["status"] == "ok"
    paths = {}
    for name in ("manifest.json", "metadata.json", "arrays.npz"):
        relative = "checkpoints/through_078020/" + name
        path = ROOT / relative
        ref = seal["outputs"][relative]
        assert path.resolve() == Path(ref["path"]).resolve()
        assert path.stat().st_size == ref["size_bytes"] and digest(path) == ref["sha256"]
        paths[name] = path
    manifest = json.loads(paths["manifest.json"].read_text("utf8"))
    metadata = json.loads(paths["metadata.json"].read_text("utf8"))
    config = metadata["context"]["persistence_config"]
    arrays = {}
    with np.load(paths["arrays.npz"], allow_pickle=False) as archive:
        for field in FIELDS:
            ref = manifest["arrays"][field]
            value = archive[ref["storage_key"]]
            assert list(value.shape) == ref["shape"] and value.dtype.str == ref["dtype"]["str"]
            assert hashlib.sha256(value.tobytes()).hexdigest() == ref["bytes_sha256"]
            arrays[field] = value
    records = []
    for frame in FRAMES:
        offset = frame - 1 - 49800
        start, stop = arrays["history_offsets"][offset : offset + 2]
        assert np.all(arrays["history_frame_idx"][start:stop] == frame - 1)
        missing = arrays["history_node_missing"][start:stop, :, 0]
        age = arrays["history_direct_pectoral_age"][start:stop]
        eligible = (missing >= config["minimum_missing_frames"]) & (
            age >= config["minimum_pectoral_frames"]
        )
        records.append(
            dict(
                native_frame=frame,
                parent_native_frame=frame - 1,
                mouth_missing_by_parent_slot=missing.tolist(),
                direct_chest_age_by_parent_slot=age.tolist(),
                age_eligible_by_parent_slot=eligible.tolist(),
                enabled_age_eligible_by_parent_slot=(eligible & config["enabled"]).tolist(),
                parent_zero_S2_age_eligible=bool(eligible[0, 2] and config["enabled"]),
            )
        )
    result = dict(
        schema="saved_mouth_age_gate_diagnosis_v1",
        status="ok",
        records=records,
        persistence_config=config,
        checkpoint_fields_read=list(FIELDS),
        complete39_checkpoint_decoded=False,
        restored=False,
        GPU_used=False,
        current_predicted_chest_support_tested=False,
        original_proposal_helpers_replayed=False,
        physical_ID_truth=False,
        interpretation=(
            "Age eligibility alone does not establish current predicted-pose support "
            "or admission of a mouth proposal."
        ),
        source_sha256=digest(__file__),
        producer_seal_sha256=SEAL_SHA,
        checkpoint_inputs={str(p): digest(p) for p in paths.values()},
        selected_array_signatures={k: manifest["arrays"][k] for k in FIELDS},
    )
    output = WORK / "SAVED_MOUTH_AGE_GATE_DIAGNOSIS.json"
    with output.open("x", encoding="utf8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(dict(status="ok", records=records, output=str(output)), ensure_ascii=False))


if __name__ == "__main__":
    main()
