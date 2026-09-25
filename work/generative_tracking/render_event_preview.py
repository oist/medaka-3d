"""Render selected frozen-result frames; no inference and no original output writes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

repository = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository / "labs/20260908_generative_tracking"))
from benchmark import CALIBRATION_DIR, open_videos, seek_read, write_previews
from medaka3d.calib import load_rig

run, out = Path(sys.argv[1]), Path(sys.argv[2])
chosen = list(map(int, sys.argv[3:]))
if out.exists():
    raise FileExistsError(out)
params = json.loads((run / "benchmark.provenance.json").read_text())["params"]
with np.load(run / "candidate.npz", allow_pickle=False) as arrays:
    result = SimpleNamespace(**{key: arrays[key] for key in arrays.files})
if not set(chosen).issubset(set(result.frame_idx.tolist())):
    raise ValueError("requested preview frame is outside frozen inference")
with np.load(params["seed"]["baseline"], allow_pickle=False) as arrays:
    xyz = np.where((arrays["slot_track"][result.frame_idx] >= 0)[..., None],
                   arrays["xyz"][result.frame_idx], np.nan)
rig = load_rig(Path(params["input_root"]) / CALIBRATION_DIR)
_, caps, metadata = open_videos(Path(params["raw_root"]), params["experiment"], 348600)
try:
    snapshots = {frame: {view: seek_read(cap, frame, view) for view, cap in caps.items()}
                 for frame in chosen}
finally:
    for cap in caps.values():
        cap.release()
out.mkdir(parents=True, exist_ok=False)
write_previews(out, snapshots, result, xyz, rig)
(out / "source.json").write_text(json.dumps({"run": str(run), "frames": chosen,
                                             "videos": metadata}, indent=2), encoding="utf-8")
print(out / "preview.html")
