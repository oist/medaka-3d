"""Fixed-bank post-fit endpoint diagnostics; no fitting or temporal replay.

Each original numeric cost remains exact. Compare its stable path order with
non-split-first priority, retaining split backups. This is not beam execution,
new support, a final trajectory, or physical identity evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import endpoint_mask_conflict_score as scorer
import numpy as np

from medaka3d import calib

WORK = Path(__file__).resolve().parent
REPO = WORK.parent.parent
STAGE = "saved_endpoint_mask_priority_v1"
FRAMES = (82720, 82727, 82728, 82750, 90141, 90258, 90259)
PRIMARY = FRAMES[:4]
VIEWS = ("top_cam", "side1_cam", "side2_cam")
ARMS = ("sparse_instance_rescore", "expanded_depth_instance_rescore", "nodewise_body_row")
OLD_ROOT = WORK / "partial_body_saved_bank_comparison_v1"
NEW_ROOT = WORK / "nodewise_body_row_ten_second_continuation_v1"
MASK_ROOT = WORK / "nodewise_body_mask_classification_all_cases_v1"
OLD_SHA = "10d29aad72ca98674abbcbaf386ce48244087cb4abae552b51e592b5dbb9910f"
NEW_SHA = "d6c1dc36ef1c79901fa47ce536585f3b7e0a4e35984fdcd7e12f765ccb0a0985"
MASK_SHA = "12ac9157d89c1ad46b9201e913fa39bc43a6d5f55361493fcfdfecba3f021189"
PINS = {
    "work/three_body/endpoint_mask_conflict_score.py": (
        "8739a48c7fa72fa2f58cb720051fcea4ab81da609612763f42e959cc7ad33d60"
    ),
    "work/three_body/sparse_instance_body_score.py": (
        "68c41510d64a41098c6c1daf5e98621723f37c2f0f35730a68bd0e896517a634"
    ),
    "work/three_body/instance_body_score.py": (
        "cb9e04a20160f2b681aa1b7cca46ed5b4017c7b05760f9dbbeff691bf058cb87"
    ),
    "src/medaka3d/calib.py": "60849004cb53febe69f437e7afa8b18cb2b65213ba76ce7c921f7022d2be4274",
    "src/medaka3d/types.py": "71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a",
}
FIELDS = (
    "candidate_index",
    "candidate_pose_xyz",
    "candidate_parent",
    "candidate_plan",
    "candidate_permutation",
    "base_local_cost",
    "original_five_loss_parts",
    "sixth_instance_term",
    "adjusted_local_cost",
    "parent_path_cost",
    "adjusted_path_cost",
    "capped_contributions_by_view",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(array):
    return dict(
        dtype=array.dtype.str,
        shape=list(array.shape),
        sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    )


def stamp(path):
    st = path.stat()
    return (st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino)


def capture(path, expected, inputs, guards):
    path = Path(path).resolve()
    require(path.is_relative_to(REPO), "input escaped workspace")
    before = stamp(path)
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected and stamp(path) == before, "input SHA/stat differs: " + str(path))
    require(str(path) not in inputs or inputs[str(path)] == digest, "conflicting input reference")
    inputs[str(path)], guards[str(path)] = digest, before
    return payload


def read_json(path, expected, inputs, guards):
    return json.loads(capture(path, expected, inputs, guards))


def member(root, seal, name):
    path = (root / name).resolve()
    ref = seal["outputs"][name]
    require(
        path.is_relative_to(root.resolve()) and str(path) == ref["path"],
        "output member path differs",
    )
    require(path.stat().st_size == ref["size_bytes"], "output member size differs")
    return ref


def successful(root, name, digest, inputs, guards):
    result = read_json(root / name, digest, inputs, guards)
    require(
        result["status"] == "ok" and result["stage"] == root.name,
        "successful exact producer required",
    )
    return result


def case_keys():
    return [(arm, frame) for frame in FRAMES for arm in (ARMS if frame in PRIMARY else ARMS[:1])]


def load_headers(inputs, guards):
    old = successful(OLD_ROOT, "comparison.provenance.json", OLD_SHA, inputs, guards)
    new = successful(NEW_ROOT, "tracking.provenance.json", NEW_SHA, inputs, guards)
    mask = successful(MASK_ROOT, "classification.provenance.json", MASK_SHA, inputs, guards)

    def document(root, seal, name):
        ref = member(root, seal, name)
        return read_json(ref["path"], ref["sha256"], inputs, guards)

    old_plan = document(OLD_ROOT, old, "plan.json")
    new_plan = document(NEW_ROOT, new, "plan.json")
    mask_plan = document(MASK_ROOT, mask, "plan.json")
    require(
        old_plan["schema"] == OLD_ROOT.name and old_plan["terminal_horizon"] == 116200,
        "old bank provenance differs",
    )
    require(
        new_plan["schema"] == NEW_ROOT.name
        and new_plan["process_interval"] == [81340, 83000]
        and new_plan["retained_interval"] == [49800, 83000],
        "new bank interval differs",
    )
    require(
        mask_plan["schema"] == MASK_ROOT.name
        and mask_plan["frames"] == list(FRAMES)
        and mask_plan["endpoint_tolerance_px"] == 6.0,
        "fixed masks/tolerance differ",
    )
    old_refs = old_plan["candidate_bank_references"]
    expected_old = {f"{a}/{f}" for a, f in case_keys() if a != ARMS[2]}
    require(set(old_refs) == expected_old, "all eleven original banks required")
    refs = {}
    for arm, frame in case_keys():
        key = f"{arm}/{frame}"
        if arm == ARMS[2]:
            refs[key] = dict(
                companion=member(NEW_ROOT, new, f"diagnostics/f{frame}.json"),
                archive=member(NEW_ROOT, new, f"diagnostics/f{frame}.npz"),
            )
        else:
            refs[key] = old_refs[key]
            for kind in ("companion", "archive"):
                ref = refs[key][kind]
                require(
                    old["inputs"].get(ref["path"]) == ref["sha256"],
                    "old bank not consumed by bound comparison",
                )
    rig_refs = mask_plan["registered_calibration"]
    require(len(rig_refs) == 4, "one registered four-file rig required")
    for path, digest in rig_refs.items():
        require(
            all(s["inputs"].get(path) == digest for s in (old, new, mask)),
            "bank/mask calibration lineage differs",
        )
    mask_signatures = document(MASK_ROOT, mask, "frame_masks_signatures.json")
    require(
        set(mask_signatures) == {f"f{f}_{v}" for f in FRAMES for v in VIEWS},
        "exact twenty-one saved mask keys required",
    )
    return dict(
        bank_references=refs,
        rig_refs=rig_refs,
        mask_signatures=mask_signatures,
        mask_archive=member(MASK_ROOT, mask, "frame_masks.npz"),
    )


def load_bank(ref, frame, inputs, guards):
    companion = ref["companion"]
    doc = read_json(companion["path"], companion["sha256"], inputs, guards)
    tree = doc["tree"]
    require(
        doc["archive"] == ref["archive"]
        and tree["schema"] == "instance_rescore_frame_diagnostics_v1"
        and tree["frame_idx"] == frame
        and tree["post_fit_only"] is True
        and tree["temporal_state"] is False,
        "fitted companion scope differs",
    )
    archive_ref = ref["archive"]
    require(
        Path(archive_ref["path"]).stat().st_size == archive_ref["size_bytes"], "bank size differs"
    )
    payload = capture(archive_ref["path"], archive_ref["sha256"], inputs, guards)
    arrays = {}
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "duplicate archive members")
        keys = []
        for field in FIELDS:
            descriptor = tree[field]
            require(set(descriptor) == {"__array__", "signature"}, "invalid selected descriptor")
            key = descriptor["__array__"]
            require(key not in keys, "repeated selected array descriptor")
            keys.append(key)
            value = archive[key]
            require(
                not value.dtype.hasobject and signature(value) == descriptor["signature"],
                "selected array signature differs",
            )
            if "arrays" in ref:
                require(
                    ref["arrays"][field] == descriptor["signature"],
                    "old comparison signature differs",
                )
            value.flags.writeable = False
            arrays[field] = value
    n = len(arrays["candidate_index"])
    require(
        n > 0
        and arrays["candidate_index"].dtype == np.int64
        and np.array_equal(arrays["candidate_index"], np.arange(n)),
        "stable candidate indices required",
    )
    shapes = {
        "candidate_pose_xyz": (n, 3, 3, 3),
        "candidate_plan": (n, 3, 3, 3),
        "candidate_parent": (n,),
        "candidate_permutation": (n, 3),
        "original_five_loss_parts": (n, 5),
        "capped_contributions_by_view": (n, 3),
    }
    for field in FIELDS[1:]:
        value = arrays[field]
        dtype = (
            np.int32
            if field == "candidate_plan"
            else np.int64
            if field in ("candidate_parent", "candidate_permutation")
            else np.float64
        )
        require(
            value.shape == shapes.get(field, (n,))
            and value.dtype == dtype
            and np.isfinite(value).all(),
            "finite fitted field shape/dtype differs",
        )
    require(
        np.array_equal(
            arrays["base_local_cost"] + arrays["sixth_instance_term"], arrays["adjusted_local_cost"]
        ),
        "saved base+sixth differs",
    )
    require(
        np.array_equal(
            arrays["parent_path_cost"] + arrays["adjusted_local_cost"], arrays["adjusted_path_cost"]
        ),
        "saved parent+(base+sixth) differs",
    )
    require(
        len(tree["candidate_kind"]) == n
        and all(isinstance(k, str) for k in tree["candidate_kind"]),
        "candidate kind differs",
    )
    return arrays, tree["candidate_kind"]


def source_code(inputs, guards):
    for relative, digest in PINS.items():
        capture(REPO / relative, digest, inputs, guards)
    require(
        Path(scorer.__file__).resolve() == REPO / "work/three_body/endpoint_mask_conflict_score.py"
        and Path(calib.__file__).resolve() == REPO / "src/medaka3d/calib.py",
        "imported science source path differs",
    )
    # Record all already imported repository sources without importing a tracker.
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if (
            path
            and Path(path).suffix == ".py"
            and Path(path).resolve().is_relative_to(REPO)
            and ".venv" not in Path(path).parts
        ):
            path = Path(path).resolve()
            capture(path, hashlib.sha256(path.read_bytes()).hexdigest(), inputs, guards)


def load_rig(context, inputs, guards):
    refs = context["rig_refs"]
    for path, digest in refs.items():
        capture(path, digest, inputs, guards)
    manifests = [Path(p) for p in refs if Path(p).name == "calibration_manifest.json"]
    require(len(manifests) == 1, "one calibration manifest required")
    path = manifests[0]
    manifest = json.loads(path.read_bytes())
    models = {(path.parent / r["path"]).resolve() for r in manifest["model_files"]}
    require(models | {path} == {Path(p) for p in refs}, "calibration model membership differs")
    rig = calib.load_rig(path.parent)
    require(
        tuple(rig.views) == VIEWS and rig.world_units == "mm",
        "calibrated camera order/units differ",
    )
    return rig


def priority_orders(path_cost, split_by_body):
    require(
        path_cost.ndim == 1 and len(path_cost) and np.isfinite(path_cost).all(),
        "finite path costs required",
    )
    require(
        split_by_body.shape == (len(path_cost), 3) and split_by_body.dtype == bool,
        "three body split flags required",
    )
    old = np.argsort(path_cost, kind="stable")
    has_split = split_by_body.any(axis=1)
    new = np.concatenate((old[~has_split[old]], old[has_split[old]]))
    return old, new, bool(has_split.all())


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def run(output):
    output = Path(output).resolve()
    require(
        output.is_relative_to(WORK) and output != WORK,
        "fresh output under work/three_body required",
    )
    output.mkdir(exist_ok=False, parents=True)
    inputs, guards, outputs = {}, {}, {}
    started = datetime.now(UTC).isoformat()

    def write(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".npz":
            with path.open("xb") as stream:
                np.savez_compressed(stream, **value)
        else:
            with path.open("x", encoding="utf-8") as stream:
                json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        data = path.read_bytes()
        record = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        outputs[name] = record
        return record

    try:
        source_code(inputs, guards)
        context = load_headers(inputs, guards)
        rig = load_rig(context, inputs, guards)
        write(
            "plan.json",
            dict(
                schema=STAGE,
                cases=case_keys(),
                endpoint_tolerance_px=6.0,
                inputs=dict(inputs),
                **context,
                same_fitted_banks=True,
                fixed_parent_history=True,
                numeric_costs_unchanged=True,
                fit_or_temporal_or_ID_solver_calls=0,
                raw_or_SL_or_original_mask_archive_reads=0,
                cross_arm_absolute_path_costs_comparable=False,
                physical_ID_truth=False,
            ),
        )
        mask_ref = context["mask_archive"]
        payload = capture(mask_ref["path"], mask_ref["sha256"], inputs, guards)
        summaries, render_cases, candidate_total = [], [], 0
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            require(
                len(archive.files) == 21 and set(archive.files) == set(context["mask_signatures"]),
                "saved mask archive membership differs",
            )
            for frame in FRAMES:
                observations = {}
                for view in VIEWS:
                    key = f"f{frame}_{view}"
                    ref = context["mask_signatures"][key]
                    masks = archive[key]
                    require(
                        ref["frame_idx"] == frame
                        and ref["view"] == view
                        and type(ref["available"]) is bool,
                        "native mask position/availability differs",
                    )
                    require(
                        masks.dtype == bool
                        and masks.ndim == 3
                        and masks.shape[1:] == (1200, 1200)
                        and signature(masks) == ref["signature"]
                        and ref["original_mask_sha256"] == ref["signature"]["sha256"],
                        "native mask snapshot signature differs",
                    )
                    masks.flags.writeable = False
                    observations[view] = scorer.prepare_endpoint_mask_observation(
                        masks if ref["available"] else None
                    )
                    del masks
                for arm, current in case_keys():
                    if current != frame:
                        continue
                    key = f"{arm}/{frame}"
                    bank, kinds = load_bank(context["bank_references"][key], frame, inputs, guards)
                    n = len(bank["candidate_index"])
                    candidate_total += n
                    views = {
                        v: scorer.score_endpoint_mask_conflicts(
                            bank["candidate_pose_xyz"], rig.camera(v).project, observations[v]
                        )
                        for v in VIEWS
                    }
                    split = np.logical_or.reduce([views[v]["split_by_body"] for v in VIEWS])
                    old, priority, all_split = priority_orders(bank["adjusted_path_cost"], split)
                    arrays = dict(
                        bank, split_by_body=split, old_path_order=old, priority_order=priority
                    )
                    for view, result in views.items():
                        arrays.update(
                            {
                                f"{view}__{k}": value
                                for k, value in result.items()
                                if isinstance(value, np.ndarray)
                            }
                        )
                    name = f"cases/{arm}_{frame}"
                    saved = write(name + ".npz", arrays)
                    # Verify the newly written small output, not the source archive again.
                    with np.load(saved["path"], allow_pickle=False) as restored:
                        require(
                            len(restored.files) == len(arrays)
                            and set(restored.files) == set(arrays),
                            "output array membership differs",
                        )
                        require(
                            all(signature(restored[k]) == signature(v) for k, v in arrays.items()),
                            "output array roundtrip differs",
                        )
                    selections = {}
                    for mode, order in (("original_path", old), ("nonsplit_priority", priority)):
                        i = int(order[0])
                        selections[mode] = dict(
                            candidate_index=i,
                            kind=kinds[i],
                            candidate_parent=int(bank["candidate_parent"][i]),
                            candidate_permutation=bank["candidate_permutation"][i],
                            candidate_plan=bank["candidate_plan"][i],
                            pose_xyz=bank["candidate_pose_xyz"][i],
                            projections={
                                v: views[v]["projected_xy"][i]
                                if views[v]["projected_xy"] is not None
                                else None
                                for v in VIEWS
                            },
                            classifications={
                                v: views[v]["classification_by_body"][i] for v in VIEWS
                            },
                            split_by_body=split[i],
                            path_cost=float(bank["adjusted_path_cost"][i]),
                            local_cost=float(bank["adjusted_local_cost"][i]),
                            parent_cost=float(bank["parent_path_cost"][i]),
                        )
                    counts = {
                        v: dict(Counter(views[v]["classification_by_body"].flat)) for v in VIEWS
                    }
                    summary = dict(
                        arm=arm,
                        frame_idx=frame,
                        candidates=n,
                        split_candidates=int(split.any(axis=1).sum()),
                        all_split_fallback=all_split,
                        old_best=int(old[0]),
                        priority_best=int(priority[0]),
                        best_changed=bool(old[0] != priority[0]),
                        selected_priority_path_cost_difference=float(
                            bank["adjusted_path_cost"][priority[0]]
                            - bank["adjusted_path_cost"][old[0]]
                        ),
                        classification_counts=counts,
                    )
                    write(
                        name + ".json",
                        dict(
                            schema=STAGE + "_case",
                            **summary,
                            input_reference=context["bank_references"][key],
                            arrays=saved,
                            array_signatures={k: signature(v) for k, v in arrays.items()},
                            candidate_kind=kinds,
                            observation_evidence={v: observations[v].evidence for v in VIEWS},
                            original_cost_tie_set=np.flatnonzero(
                                bank["adjusted_path_cost"] == bank["adjusted_path_cost"][old[0]]
                            ),
                            selections=selections,
                        ),
                    )
                    summaries.append(summary)
                    render_cases.append(dict(arm=arm, frame_idx=frame, selections=selections))
                del observations
        require(
            len(summaries) == 15 and candidate_total == 262,
            "fixed fifteen-bank candidate coverage differs",
        )
        write(
            "render_payload.json",
            dict(
                schema=STAGE + "_render_payload",
                cases=render_cases,
                proposed_geometry_only=True,
                strict_support_recomputed=False,
                final_trajectory_selection=False,
                physical_ID_truth=False,
            ),
        )
        write(
            "summary.json",
            dict(
                schema=STAGE,
                status="ok",
                cases=summaries,
                candidate_count=candidate_total,
                bank_captures=15,
                selected_bank_fields=list(FIELDS),
                mask_archive_captures=1,
                mask_frame_decodes=21,
                numeric_costs_unchanged=True,
                priority_only_not_beam_execution=True,
                fixed_parent_history=True,
                all_split_is_unresolved=True,
                nonsplit_is_not_body_truth=True,
                physical_ID_truth=False,
            ),
        )
        require(
            all(stamp(Path(p)) == before for p, before in guards.items()),
            "input final stat guard differs",
        )
        write(
            "scoring.provenance.json",
            dict(
                stage=STAGE,
                status="ok",
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
                inputs=inputs,
                outputs=dict(outputs),
                input_guard="SHA at capture; final stat only; no second source payload read",
            ),
        )
    except BaseException as exc:
        write(
            "failure.json",
            dict(
                stage=STAGE,
                status="failed",
                error_type=type(exc).__name__,
                error=str(exc),
                inputs=inputs,
                outputs=dict(outputs),
                started_at=started,
                finished_at=datetime.now(UTC).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
