"""K10 — what centroid-head capacity actually fits on this GPU. Measured, not guessed.

Round 8's **centroid** head is 7,853,169 parameters — `filters=16`,
`filters_rate=2.0`, `max_stride=32` — running on full 1200x1200 frames. Its
centered-instance head is **70,755,866** parameters on 512x512 crops.

The archive's oracle evaluation on the locked benchmark says every detection error
is centroid-stage: given ground-truth centroids, round 7 and round 8 both score
117/117 matched with 0 extra and 0 missed. So the stage that causes all of the
detection error carries **9x fewer parameters** than the stage that causes none.
Enriching it is the obvious move; this establishes what is affordable.

**Why the two knobs are not equivalent, and it is a memory argument.** With
`output_stride=2` the decoder returns to 600x600 carrying `filters` channels, so
that one tensor is `batch x filters x 600 x 600`:

* raising **`filters`** widens *every* level including the largest feature maps —
  parameters grow roughly linearly, activation memory grows linearly on the
  biggest tensors in the network;
* raising **`filters_rate`** widens only the *deep* levels, which are 32x smaller
  spatially — parameters grow fast, activation memory barely moves.

So `filters_rate` is the cheap way to buy parameters and `filters` is the
expensive one. This measures both rather than asserting it: each config gets a
real forward and backward pass at the production input size, and peak allocated
CUDA memory is recorded. Configs that OOM are reported as OOM, which is the
answer for those.

Run:
    .venv\\Scripts\\python.exe tools/k10_centroid_capacity_probe.py [--batch-size 8]
"""

from __future__ import annotations

import argparse
import json
import math

import torch
from omegaconf import OmegaConf
from sleap_nn.architectures.model import Model

from medaka3d import load_paths

#: Round-8 production values for the centroid head.
BASE = {"filters": 16, "filters_rate": 2.0, "max_stride": 32}
INPUT_HW = 1200
BATCH = 8

#: Candidates, ordered by parameter count. Chosen to separate the two knobs:
#: rows 2-4 raise `filters` alone, rows 5-7 raise `filters_rate` alone, and the
#: last few combine them or go deeper.
CANDIDATES = [
    (16, 2.0, 32, "round 8 PRODUCTION"),
    (24, 2.0, 32, "filters +50%"),
    (32, 2.0, 32, "filters x2"),
    (16, 2.5, 32, "rate 2.0 -> 2.5"),
    (16, 3.0, 32, "rate 2.0 -> 3.0"),
    (24, 2.5, 32, "both"),
]

#: Centered-instance candidates. The head that must resolve mouth-vs-tail across
#: +-180 deg of rotation, so this is where capacity is wanted. max_stride is 32
#: throughout: at crop 320 a receptive field beyond ~320 px buys nothing.
CI_CANDIDATES = [
    (24, 2.0, 64, "round 8 PRODUCTION (crop 512)"),
    (24, 2.0, 32, "ms 64 -> 32, filters unchanged"),
    (32, 2.0, 32, "k04 setting"),
    (48, 2.0, 32, "filters x2"),
    (64, 2.0, 32, "filters x2.7"),
    (96, 2.0, 32, "filters x4"),
    (24, 2.5, 32, "rate 2.0 -> 2.5"),
    (32, 2.5, 32, "wider + rate 2.5"),
    (48, 2.5, 32, "wider + rate 2.5, larger"),
    (32, 3.0, 32, "rate 3.0"),
    (48, 3.0, 32, "rate 3.0, wider"),
    (64, 2.5, 32, "big"),
]


def build(filters: float, rate: float, max_stride: int, head: str = "centroid"):
    backbone = OmegaConf.create({
        "in_channels": 3, "kernel_size": 3, "filters": int(filters),
        "filters_rate": float(rate), "max_stride": int(max_stride),
        "stem_stride": None, "middle_block": True, "up_interpolate": True,
        "stacks": 1, "convs_per_block": 2, "output_stride": 2,
    })
    if head == "centered_instance":
        head_cfg = OmegaConf.create({
            "confmaps": {"part_names": ["mouth", "tail"], "anchor_part": "mouth",
                         "sigma": 2.5, "output_stride": 2, "loss_weight": 1.0}
        })
    else:
        head_cfg = OmegaConf.create({
            "confmaps": {"anchor_part": "mouth", "sigma": 2.5,
                         "output_stride": 2, "loss_weight": 1.0}
        })
    return Model(backbone_type="unet", backbone_config=backbone,
                 head_configs=head_cfg, model_type=head)


def probe(filters, rate, max_stride, batch, device, input_hw, head) -> dict:
    """Params, and peak CUDA memory for one real training step."""
    model = build(filters, rate, max_stride, head)
    params = sum(p.numel() for p in model.parameters())
    result = {"params": params, "peak_gb": None, "status": "params only"}
    if device.type != "cuda":
        return result
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = model.to(device).train()
        x = torch.randn(batch, 3, input_hw, input_hw, device=device)
        out = model(x)
        tensor = out[0] if isinstance(out, (list, tuple)) else (
            next(iter(out.values())) if isinstance(out, dict) else out
        )
        loss = tensor.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        result["peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        result["status"] = "OK"
    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except RuntimeError as error:
        result["status"] = "OOM" if "out of memory" in str(error).lower() else f"ERR {error}"
    finally:
        del model
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--input", type=int, default=INPUT_HW)
    parser.add_argument("--head", choices=["centroid", "centered_instance"],
                        default="centroid")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = (torch.cuda.get_device_properties(0).total_memory / 1024**3
             if device.type == "cuda" else 0.0)
    print(__doc__.split("Run:")[0])
    print(f"device: {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}  "
          f"({total:.1f} GB)   input {args.input}x{args.input}   batch {args.batch_size}\n")

    print(f"{'filters':>7} {'rate':>5} {'stride':>6} {'channels':<34} {'params':>12} "
          f"{'x prod':>7} {'peak GB':>8}  note")
    rows = []
    grid = CI_CANDIDATES if args.head == 'centered_instance' else CANDIDATES
    reference = 70755866 if args.head == 'centered_instance' else 7853169
    for filters, rate, max_stride, note in grid:
        n = round(math.log2(max_stride))
        channels = [int(filters * (rate ** i)) for i in range(n + 1)]
        r = probe(filters, rate, max_stride, args.batch_size, device, args.input, args.head)
        ratio = r["params"] / reference
        peak = f"{r['peak_gb']:.2f}" if r["peak_gb"] else r["status"]
        print(f"{filters:>7} {rate:>5} {max_stride:>6} {channels!s:<34} "
              f"{r['params']:>12,d} {ratio:>6.1f}x {peak:>8}  {note}")
        rows.append({"filters": filters, "filters_rate": rate, "max_stride": max_stride,
                     "channels": channels, "note": note, **r})

    fits = [r for r in rows if r["status"] == "OK"]
    if fits:
        best = max(fits, key=lambda r: r["params"])
        print(f"\nLargest config that fits at batch {args.batch_size}: "
              f"filters={best['filters']}, filters_rate={best['filters_rate']}, "
              f"max_stride={best['max_stride']} — {best['params']:,d} params "
              f"({best['params'] / reference:.1f}x production), peak {best['peak_gb']:.2f} GB.")
    print("\nPeak memory here is one forward+backward on random input. A real run adds "
          "optimiser\nstate (Adam: 2 extra copies of every parameter) and the data "
          "pipeline, so leave headroom.")

    out = load_paths().work_dir / f"k10_capacity_probe.{args.head}.json"
    out.write_text(json.dumps({"device": torch.cuda.get_device_name(0)
                               if device.type == "cuda" else "cpu",
                               "total_gb": total, "input": args.input,
                               "batch_size": args.batch_size,
                               "head": args.head,
                               "production_params": reference,
                               "candidates": rows},
                              indent=2, default=float), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
