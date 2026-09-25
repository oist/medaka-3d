"""Single synthetic CPU/CUDA equivalence check; no tracking input or ID truth."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

repo = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "identity_emissions_fixture", repo / "tests/test_three_body_identity_emissions.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(2)
rig, pose, frame, templates = module.scene()
cpu = module.E.score_identity_permutations(
    rig, frame, pose, templates, np.ones(3, bool), device="cpu"
)
gpu = module.E.score_identity_permutations(
    rig, frame, pose, templates, np.ones(3, bool), device="cuda"
)
differences = {}
for name in ("costs", "per_view_costs", "visible_mass"):
    a, b = getattr(cpu, name), getattr(gpu, name)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    differences[name] = float(np.max(np.abs(a - b)))
for name in ("enabled_views", "sampled_pixels", "valid_pixels", "available_core_pixels"):
    np.testing.assert_array_equal(getattr(cpu, name), getattr(gpu, name))
print(json.dumps({
    "status": "synthetic_cpu_cuda_agreement", "device": torch.cuda.get_device_name(0),
    "maximum_absolute_differences": differences, "identity_correctness_test": False,
}), flush=True)
