"""One read-only live attachment check; no waiting, CUDA, or process control."""

import hashlib
import json
from pathlib import Path

from replay_gpu_window import ReplayGPUWindow

ROOT = Path(__file__).resolve().parent
output = ROOT / "replay_gpu_window_attach_probe_v1"
output.mkdir(exist_ok=False)
guard = ReplayGPUWindow(ROOT / "long_remaining_receipted_supervisor_v1", 9344, output)
try:
    guard.attach()
    print(json.dumps(guard.report), flush=True)
finally:
    guard.close()
summary = {
    "attachment_succeeded": guard.report.get("attached_at") is not None,
    "waited_for_exit": False, "tracking_executed": False,
    "window_report_sha256": hashlib.sha256((output / "gpu_window.json").read_bytes()).hexdigest(),
    "helper_sha256": hashlib.sha256((ROOT / "replay_gpu_window.py").read_bytes()).hexdigest(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
(output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary), flush=True)
