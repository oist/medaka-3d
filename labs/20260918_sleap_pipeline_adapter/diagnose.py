"""Explain fixed-parameter effects from saved evidence; never retune the tracker."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from medaka3d.provenance import provenance

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work/sleap_pipeline_adapter_v1"
VIEWS = ("top_cam", "side1_cam", "side2_cam")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    methods = ("G0", "G_history3", "G_history9", "G_margin02")
    result = {
        "scope": "cg20s; core targets after first frame; association scores not detector scores",
        "identity_accuracy_claim": False,
        "methods": {},
        "example": {},
    }
    with provenance("sleap_adapter_parameter_diagnosis", OUT) as record:
        record.input("diagnosis_code", Path(__file__), sha256=sha(Path(__file__)))
        for method in methods:
            events, score_values = [], []
            for view in VIEWS:
                path = OUT / "cg20s" / f"evidence_{method}_{view}.json.gz"
                record.input(f"{method}_{view}", path, sha256=sha(path))
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    saved = json.load(stream)
                events.extend(event for event in saved["events"] if 75270 < event["frame"] < 78590)
                if view == "side1_cam" and method in ("G0", "G_history9"):
                    event = next(e for e in saved["events"] if e["global_row"] == 214502)
                    data_path = OUT / "cg20s" / f"tracking_{method}_{view}.npz"
                    with np.load(data_path, allow_pickle=False) as data:
                        row = int(np.flatnonzero(data["global_row"] == 214502)[0])
                        candidate = next(
                            c
                            for c in event["candidates"]
                            if c["provisional_id"] == event["provisional_id"]
                        )
                        history = np.isin(data["frame_idx"], candidate["history_frames"])
                        history &= data["provisional_labels"] == event["provisional_id"]
                        distances = np.linalg.norm(
                            data["points"][history, 1].astype(float) - data["points"][row, 1],
                            axis=1,
                        )
                        recomputed = float(np.exp(-(distances**2) / 50.0).mean())
                        np.testing.assert_allclose(recomputed, event["score"], rtol=1e-6)
                        peers = (data["frame_idx"] == 75286) & (data["global_row"] != 214502)
                        peer_distance = float(
                            np.linalg.norm(
                                data["points"][peers, 1] - data["points"][row, 1], axis=1
                            ).min()
                        )
                    result["example"][method] = {
                        "view": view,
                        "frame": 75286,
                        "global_row": 214502,
                        "event": event,
                        "distances_to_history_px": distances.tolist(),
                        "independently_recomputed_mean_oks": recomputed,
                        "nearest_other_current_detection_px": peer_distance,
                    }
            refused = [e for e in events if not e["confirmed"]]
            reasons = Counter(e["reason"] for e in refused)
            score_values = [e["score"] for e in events if e["score"] is not None]
            result["methods"][method] = {
                "eligible_observations": len(events),
                "unconnected_observations": len(refused),
                "refusal_reasons": dict(reasons),
                "selected_score_mean": float(np.mean(score_values)),
                "selected_score_p01": float(np.quantile(score_values, 0.01)),
                "score_scope": "Finite selected association scores; dummy refusals excluded",
                "margin_refusals_below_effective_score_floor": sum(
                    e["reason"] == "insufficient_similarity_margin"
                    and e["score"] < e["match_threshold"] + e["required_margin"]
                    for e in refused
                    if e["score"] is not None
                ),
            }
        result["interpretation"] = [
            "Dummy threshold is included in margin alternatives: G0 commit score must exceed 0.10.",
            "The verified example has the same immediate predecessor and overlapping history.",
            "A longer unshifted history dilutes its OKS mean as the animal moves in this example.",
            "This demonstrates a mechanism, not an attribution of every real-data refusal.",
            "History3 and margin02 are separate settings; their combination was not evaluated.",
            "Additional links need independent identity review before adoption.",
        ]
        path = OUT / "parameter_diagnosis.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        record.output("diagnosis", path, sha256=sha(path))
    print("Saved evidence-based parameter diagnosis and independently recomputed OKS example")


if __name__ == "__main__":
    main()
