"""Synthetic CPU/CUDA arithmetic parity; this is not real-image or ID evidence."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import run_curved_body_probe as m
from medaka3d.calib import Rig


class Camera:
    def __init__(self, view):
        self.view = view

    def rays(self, xy):
        origins = np.zeros((len(xy), 3))
        direction = np.zeros(3)
        if self.view == m.VIEWS[0]:
            origins[:, :2] = xy
            direction[2] = 1.
        elif self.view == m.VIEWS[1]:
            origins[:, [0, 2]] = xy
            direction[1] = 1.
        else:
            origins[:, 1:] = xy
            direction[0] = 1.
        return origins, np.broadcast_to(direction, origins.shape).copy()


torch.set_num_threads(2)
rig = Rig(tuple(Camera(v) for v in m.VIEWS), Path('synthetic-curve-gpu-parity'))
pose = torch.tensor([[[40., 40., 30.], [45., 40., 30.], [55., 40., 30.]]]*3,
                    dtype=torch.float64)
pose[:, :, 1] += torch.arange(3, dtype=torch.float64)[:, None] * 25
pixels = []
for fish in range(3):
    yy, xx = np.mgrid[38+fish*25:45+fish*25, 42:56]
    pixels.append(np.column_stack((xx.ravel(), yy.ravel())))
top_pixels = np.concatenate(pixels)
zz, xx = np.mgrid[26:38, 40:58]
side1_pixels = np.column_stack((xx.ravel(), zz.ravel()))
side2_pixels = []
for fish in range(3):
    zz, yy = np.mgrid[26:38, 37+fish*25:49+fish*25]
    side2_pixels.append(np.column_stack((yy.ravel(), zz.ravel())))
pixels = {m.VIEWS[0]: top_pixels, m.VIEWS[1]: side1_pixels,
          m.VIEWS[2]: np.concatenate(side2_pixels)}
hashes = {str(p): m.sha256(p) for p in m.all_source_paths()}
hashes[str(Path(__file__).resolve())] = m.sha256(__file__)
known = torch.tensor([[0., 3., 2.]]*3, dtype=torch.float64)
cpu = m.CurvedFishRenderer(rig, pixels, dtype=torch.float64, device='cpu')
with torch.no_grad():
    target = cpu.render(pose, tail_control_offset_mm=known).union_alpha[0].numpy()
targets = SimpleNamespace(target=target, weights=np.full(target.shape, 1/target.shape[1]))
results, gradients = [], []
for device in ('cpu', 'cuda'):
    engine = m.CurvedFishRenderer(rig, pixels, dtype=torch.float64, device=device)
    offset = torch.zeros((3, 3), dtype=torch.float64, device=device, requires_grad=True)
    rendered = engine.render(pose, tail_control_offset_mm=offset, sigma_reference_pose_xyz=pose)
    loss = m.image_loss(rendered.union_alpha, torch.as_tensor(target, device=device),
                        torch.as_tensor(targets.weights, device=device), rendered.ray_valid)[0]
    gradient, = torch.autograd.grad(loss, offset)
    gradients.append(gradient.detach().cpu().numpy())
    results.append(m.optimize_curve_offsets(engine, pose, targets, steps=12))
differences = {}
checks = {}
for key in ('offset_mm', 'loss_history', 'offset_history_mm'):
    checks[key] = bool(np.allclose(results[0][key], results[1][key], rtol=1e-11, atol=1e-11))
    differences[key] = float(np.max(np.abs(results[0][key] - results[1][key])))
checks['zero_offset_gradient'] = bool(np.allclose(gradients[0], gradients[1], rtol=1e-12, atol=1e-12))
differences['zero_offset_gradient'] = float(np.max(np.abs(gradients[0] - gradients[1])))
assert results[0]['best_iteration'] == results[1]['best_iteration']
assert all(m.sha256(p) == digest for p, digest in hashes.items())
output = Path(__file__).parent / 'curved_body_three_view_gpu_parity_v1'
output.mkdir(exist_ok=False)
summary = dict(status='passed' if all(checks.values()) else 'strict_parameter_parity_failed',
               synthetic_only=True, all_cameras_parallel=False,
               cuda_device=torch.cuda.get_device_name(), checks=checks,
               maximum_absolute_differences=differences, best_iteration=results[0]['best_iteration'],
               cpu_best_loss=results[0]['best_loss'], gpu_best_loss=results[1]['best_loss'],
               input_source_sha256=hashes)
(output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
np.savez_compressed(output / 'cpu_gpu.npz', cpu_offset=results[0]['offset_mm'],
                    gpu_offset=results[1]['offset_mm'], cpu_gradient=gradients[0],
                    gpu_gradient=gradients[1], cpu_losses=results[0]['loss_history'],
                    gpu_losses=results[1]['loss_history'])
print(json.dumps({k: v for k, v in summary.items() if k != 'input_source_sha256'}))
