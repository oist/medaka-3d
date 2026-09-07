"""S15 — do the reviewed labels fix the 3-node models? Same frames, old vs new.

Four models were trained on the 543 moments reviewed on 2026-08-10 (`s14`):
top-down centered-instance and bottom-up, each for top and side. This scores
them the way `s10` scored the 2x2, on **the same 20,000 frames per view**, so
the new arms and the old ones are paired rather than merely both measured.

=============================================================================
WHAT IS AND IS NOT COMPARABLE HERE
=============================================================================

**`val/loss` is not the comparison, and cannot be.** Two reasons, both fatal:
the reviewed-only validation split shares only 9 of 30 moments with the split
`s07`/`s03` used, so old-vs-new val numbers describe different data; and
bottom-up's loss carries PAF terms that centered-instance has no analogue for,
so architecture-vs-architecture val numbers are not the same quantity. The
training curves are reported as convergence evidence and nothing more.

**The part-mixing metrics need no ground truth**, so none of that applies to
them. They are a property of the prediction on unlabelled video.

**Shared-node still does not compare architectures.** `s10` measured the
nearest tail-to-tail distance within a frame across 238,414 frames: top-down
reaches 0.01 px, bottom-up's 1st percentile is 22-24 px. A 5 px coincidence is
effectively unreachable for bottom-up, so its zero is a floor, not a merit.
Cross-architecture reading therefore rests on `tail_swap` (mouth-linked, the
only metric here that sees a permutation) and detection completeness.
Within an architecture, old vs new is a fair shared-node comparison.

Run::

    .venv\\Scripts\\python.exe tools/s15_compare_reviewed_models.py --view top
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import sleap_io as sio

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s06_locked_benchmark_3node import MODELS_BY_VIEW
from s10_shared_node_20k import (
    EXPERIMENTS,
    SHARED_NODE_PX,
    VIEW_CAM,
    VIEW_MODELS,
    block_frames,
    block_starts,
    cache_path,
    load_cache,
    median_span_px,
    predict_experiment,
    rates,
    release_gpu,
    save_cache,
    score_blocks,
    slug,
    tail_jump_threshold_px,
    two_proportion_z,
)

from medaka3d import load_catalog, load_paths
from medaka3d.provenance import provenance

NEW = Path(r"Z:\StephensU\medaka\sleap_models\retrain_20260810_3node_reviewed")

#: The four s14 arms, plus the s07/s03 arms they replace. The old entries are
#: the ones s10 already cached on these exact frames, so scoring them again is
#: free — `predict_experiment` is only called when a cache is missing.
def models_for(view: str) -> dict[str, list[Path]]:
    kind = VIEW_MODELS[view]
    old = MODELS_BY_VIEW[kind]
    centroid = next(m for m in old["topdown 2-node (round8, PRODUCTION)"] if "centroid" in m.name)
    new_td = NEW / kind / f"s14.topdown_3node.reviewed_{kind}"
    new_bu = next((NEW / kind).glob("s14.bottomup_3node.reviewed_*"), None)
    out: dict[str, list[Path]] = {}
    for name, paths_ in old.items():
        if "3-node" in name:
            out[f"OLD {name}"] = list(paths_)
    out["NEW topdown 3-node (reviewed)"] = [centroid, new_td]
    if new_bu is not None:
        out["NEW bottomup 3-node (reviewed)"] = [new_bu]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("=====")[0].strip())
    parser.add_argument("--view", choices=tuple(VIEW_CAM), default="top")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--block-len", type=int, default=1000)
    parser.add_argument("--experiments", nargs="*", default=list(EXPERIMENTS))
    args = parser.parse_args()

    paths = load_paths()
    catalog = load_catalog(paths.asset_root)
    out_dir = paths.work_dir / "s15_compare_reviewed"
    # Reuse s10's cache directory: the old arms were already run over exactly
    # these frames, so this measurement costs only the two new models.
    cache_dir = paths.work_dir / "s10_shared_node_20k" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = VIEW_CAM[args.view]
    kind = VIEW_MODELS[args.view]
    models = models_for(args.view)
    plan_id = f"b{args.blocks}x{args.block_len}"

    with provenance("compare_reviewed_models", out_dir, name=f"s15_{args.view}") as record:
        record.params_from(args)
        label_key = f"labels.round8.{kind}.train"
        span, _n_span = median_span_px(catalog.get(label_key))
        jump_px = tail_jump_threshold_px(span)
        print(f"view {args.view} ({cam})  span {span:.2f} px  tail_jump > {jump_px:.0f} px  "
              f"shared {SHARED_NODE_PX:.0f} px")
        record.param("median_span_px", round(span, 3))
        record.param("tail_jump_px", jump_px)

        plan = {}
        for experiment in args.experiments:
            n = int(sio.Video.from_filename(str(paths.video(experiment, cam))).shape[0])
            plan[experiment] = block_frames(
                block_starts(n, args.blocks, args.block_len), args.block_len)

        results: dict[str, dict] = {}
        for name, model_paths in models.items():
            missing = [m for m in model_paths if not (m / "best.ckpt").is_file()]
            if missing:
                print(f"  SKIP {name}: {missing}")
                continue
            print(f"\n--- {name}")
            record.param(f"model_{slug(name)}", [str(m) for m in model_paths])
            pooled: collections.Counter = collections.Counter()
            per_exp = {}
            for experiment in args.experiments:
                frames = [i for block in plan[experiment] for i in block]
                # Old arms keep s10's cache name so their cache is found.
                cache_name = name[4:] if name.startswith("OLD ") else name
                cache = cache_path(cache_dir, args.view, cache_name, experiment, plan_id)
                if cache.is_file():
                    cached, per_frame = load_cache(cache)
                    if cached != frames:
                        raise SystemExit(f"{cache} was built for other frames")
                    print(f"    {experiment}: cached")
                else:
                    per_frame = predict_experiment(
                        experiment, args.view, model_paths, frames,
                        args.device, args.batch_size)
                    save_cache(cache, frames, per_frame)
                    print(f"    {experiment}: predicted")
                counts = score_blocks(plan[experiment], per_frame, jump_px)
                per_exp[experiment] = rates(counts)
                pooled.update(counts)
            release_gpu()
            results[name] = {"pooled": rates(pooled), "per_experiment": per_exp}

        header = (f"{'model':<38}{'shared%':>9}{'/1krisk':>9}{'jump%':>8}"
                  f"{'swap%':>9}{'inst':>7}{'<3':>7}")
        print(f"\n{'=' * len(header)}\nPOOLED — {args.view}, "
              f"{args.blocks * args.block_len * len(args.experiments)} frames/model")
        print("=" * len(header))
        print(header)
        for name, r in results.items():
            p = r["pooled"]
            print(f"{name:<38}{100 * p['shared_frame_rate']:>9.3f}"
                  f"{p['shared_per_1k_pairs_at_risk']:>9.3f}"
                  f"{100 * p['tail_jump_rate']:>8.3f}{100 * p['tail_swap_rate']:>9.4f}"
                  f"{p['mean_instances']:>7.3f}{p['frames_lt3']:>7}")

        print("\nOLD vs NEW, within architecture (shared-node is fair here):")
        for arch in ("topdown", "bottomup"):
            old = next((k for k in results if k.startswith("OLD") and arch in k), None)
            new = next((k for k in results if k.startswith("NEW") and arch in k), None)
            if not old or not new:
                continue
            o, n = results[old]["pooled"], results[new]["pooled"]
            _z, p = two_proportion_z(n["frames_shared_any"], n["frames"],
                                    o["frames_shared_any"], o["frames"])
            rel = ((o["shared_frame_rate"] - n["shared_frame_rate"])
                   / o["shared_frame_rate"] if o["shared_frame_rate"] else float("nan"))
            _zj, pj = two_proportion_z(n["tail_swap_over"], max(n["tail_swap_n"], 1),
                                      o["tail_swap_over"], max(o["tail_swap_n"], 1))
            print(f"  {arch:<9} shared {100 * o['shared_frame_rate']:.3f} -> "
                  f"{100 * n['shared_frame_rate']:.3f}%  rel {100 * rel:+.1f}%  p={p:.3g}"
                  f"   | swap {100 * o['tail_swap_rate']:.4f} -> "
                  f"{100 * n['tail_swap_rate']:.4f}%  p={pj:.3g}"
                  f"   | <3 {o['frames_lt3']} -> {n['frames_lt3']}")

        record.param("results", results)
        out = out_dir / f"s15_results_{args.view}.json"
        out.write_text(json.dumps({"view": args.view, "results": results},
                                  indent=2, default=float), encoding="utf-8")
        record.output("results", out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
