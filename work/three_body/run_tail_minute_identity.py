"""Fixed Original/Tail minute: exact support-only count, then motion-only DirectMemory.

Preflight decompresses only frame_idx and observed; other candidate members are
inspected as NPY headers. Solve is a separate invocation requiring that successful
preflight. All original21 arrays and review remain unchanged in a separate raw
slice; proposed identity order is exported separately. No RGB, SL-array loading,
rendering, fitting, model, torch, GPU, thresholds or source changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
sys.path.insert(0, str(REPO / "labs/20260909_three_body"))
sys.path.insert(0, str(REPO / "src"))

import count_identity_memory_states as counter  # noqa: E402
import identity_memory_dp as memory  # noqa: E402
from identity_memory_links import chosen_memory_links  # noqa: E402
from identity_output import build_identity_output  # noqa: E402

from medaka3d.eval.generative import MouthAnchor, score_female_identity  # noqa: E402

FIRST, START, STOP = 49800, 66400, 76360
METHODS = ("original", "tail")
STAGES = dict(
    preflight="tail_minute_identity_reachability_v1", solve="tail_minute_identity_motion_v1"
)
COMPARISON = WORK / "tail_one_minute_comparison_v1"
COMPARISON_SHA = "8a3a77326f34e77801f834b529fef79626116fba63a1f99155ec33f2cea3e557"
FEMALE = WORK / "long_female_endpoint_160s_v1"
FEMALE_SHA = "e5894da372b2f1b577e8c0ac2d6930140231a6674db90594501dda58d1fba5b6"
GT_SHA = "ab0a173ac737a63311aec834c5832fc19475d4f7d1892b8193f8465bba67519f"
ANCHOR_FRAMES = (66889, 68186, 69363, 70234, 70835, 71288, 72371, 73027, 73953, 74786, 75987)
RUNS = dict(
    original=dict(
        root=WORK / "long_remaining_supervisor_v1/phases/f066400-076360/tracking",
        seal="continuation.provenance.json",
        stage="three_body_long_continuation",
        sha="95098e39bbd4d1e09d7ef98e3a7c244fe89ab2caf36a3650f360d35fc8289382",
        candidate="870836f9e086db6d1675dd283846e758038fb942fb6e8df6296f63aad98a5877",
    ),
    tail=dict(
        root=WORK / "tail_one_minute_66400-76360_v1",
        seal="tracking.provenance.json",
        stage="tail_one_minute_continuous",
        sha="50ccc8d2a4bbcc7e09aa2b512b70374fb9b37a4d4ece0d0939322c66bfa59416",
        candidate="4a86511a1bb01a28de7f4497e848ca0e9c71badefe61fe23722039a2e993f7f7",
    ),
)
DP_CONFIG = dict(
    scale_mm=0.8,
    weight=0.2,
    initial_state=0,
    max_layer_states=100000,
    max_total_states=1000000,
    max_total_edges=6000000,
    tie_atol=1e-10,
    tie_rtol=1e-12,
)
SOURCE_PINS = {
    "work/three_body/count_identity_memory_states.py": (
        "3ae488bf9155837e8c950ec178b3b2f404545bf8412b788e331f495e3ae90325"
    ),
    "labs/20260909_three_body/identity_memory_dp.py": (
        "2ef045165dd9f0342d234064335f0f8dd015ada5a864696cfab8fb5449da1adf"
    ),
    "labs/20260909_three_body/identity_dp.py": (
        "2e848b6c4b14eebcd3d4d3791a4f61aae3e8f4e9fd0413588bd6a1548121e354"
    ),
    "labs/20260909_three_body/identity_memory_links.py": (
        "672dcd2d3c0175863179fa9b21bffec4336ada76ffc59c0e67130566f964140b"
    ),
    "labs/20260909_three_body/identity_output.py": (
        "3e462fe7e3d7c9a3128f751c7e32c9530dbb3cad7ff9f9eb913a40731aaaa136"
    ),
    "work/three_body/compare_tail_one_minute.py": (
        "da5f50c1dcc1399464c9ca8edfd65287a9f037f78119eee17cef7c27be388737"
    ),
    "src/medaka3d/eval/generative.py": (
        "a0e66679588053ba81de4f2333f00a2d906b768b8cfe1db5ba2a196101a82157"
    ),
    "src/medaka3d/types.py": ("71b002e8dbfe7efb139520bc387c9cc0aa097ce55efb24aadd1a6080da71e33a"),
    "src/medaka3d/provenance.py": (
        "1e2c260c959be0f240f244d064ec0f2b4c7f82cd7ecf0d9f29a82bdb1b2150c1"
    ),
    "src/medaka3d/__init__.py": (
        "d363fc11f8c37b85d2dd1d95fc7f4ecf4d098fe81cd148e0cc5675d3ba1a0b93"
    ),
    "src/medaka3d/eval/__init__.py": (
        "32b4f710774e628b40ad00b4c4328fd66c64c5f95299c1a47d7c1834b91fe161"
    ),
}
# Frozen finish21 schema; dtypes are recorded exactly, never storage-normalized.
LAYOUT = dict(
    frame_idx=((), "i"),
    pose_xyz=((3, 3, 3), "f"),
    observed=((3, 3), "b"),
    sources=((3, 3, 3), "i"),
    identity_review=((3,), "b"),
    mask_supported=((3,), "b"),
    body_supported=((3,), "b"),
    missing=((3,), "i"),
    costs=((), "f"),
    path_costs=((), "f"),
    loss_parts=((5,), "f"),
    score_margin=((), "f"),
    proposal_kind=((), "U"),
    hypotheses=((), "i"),
    proposal_flow_nodes=((3, 3), "b"),
    proposal_flow_parent=((), "i"),
    node_missing=((3, 3), "i"),
    direct_pectoral_age=((3,), "i"),
    xyz=((3, 3), "f"),
    accepted_xyz=((3, 3), "f"),
    body_xyz=((3, 3), "f"),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def signature(a):
    return dict(dtype=a.dtype.str, shape=list(a.shape), sha256=sha(a.tobytes(order="C")))


class Evidence:
    def __init__(self):
        self.inputs = {}

    def read(self, path, digest):
        path = Path(path).resolve()
        raw = path.read_bytes()
        require(sha(raw) == digest, "consumed SHA differs: " + str(path))
        require(self.inputs.setdefault(str(path), digest) == digest, "conflicting input SHA")
        return raw

    def guard(self):
        for p, digest in self.inputs.items():
            require(file_sha(p) == digest, "input mutated: " + p)


class Output:
    def __init__(self, root):
        self.root = Path(root).resolve()
        require(self.root.is_relative_to(WORK), "new work/three_body output required")
        self.root.mkdir(exist_ok=False, parents=True)
        self.outputs = {}

    def save(self, role, value, *, arrays=False):
        p = self.root / role
        p.parent.mkdir(exist_ok=True, parents=True)
        require(role not in self.outputs, "duplicate output")
        if arrays:
            require(all(not a.dtype.hasobject for a in value.values()), "object output forbidden")
            with p.open("xb") as f:
                np.savez_compressed(f, **value)
        else:
            with p.open("x", encoding="utf-8", newline="\n") as f:
                json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
                f.write("\n")
        self.outputs[role] = dict(path=str(p), size_bytes=p.stat().st_size, sha256=file_sha(p))

    def guard(self):
        actual = {p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()}
        require(actual == set(self.outputs), "output membership differs")
        for item in self.outputs.values():
            require(
                Path(item["path"]).stat().st_size == item["size_bytes"]
                and file_sha(item["path"]) == item["sha256"],
                "output mutated",
            )


def input_member(seal, path, digest):
    value = seal["inputs"].get(str(Path(path).resolve()))
    actual = value.get("sha256") if isinstance(value, dict) else value
    require(actual == digest, "historical input membership differs: " + str(path))


def output_member(seal, root, relative):
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "output escaped root")
    matches = [v for v in seal["outputs"].values() if Path(v["path"]).resolve() == path]
    require(bool(matches), "named producer output required: " + relative)
    # Frozen Original writes a named role and an absolute-path alias. Accept
    # these only when every complete metadata record is identical.
    require(all(v == matches[0] for v in matches), "conflicting named output aliases")
    return matches[0]


def named_read(seal, root, relative, evidence):
    item = output_member(seal, root, relative)
    raw = evidence.read(item["path"], item["sha256"])
    require(len(raw) == item.get("size_bytes", item.get("size")), "output size differs")
    return raw


def seal_read(root, name, stage, digest, evidence):
    value = json.loads(evidence.read(Path(root) / name, digest))
    require(value["stage"] == stage and value["status"] == "ok", "successful sealed stage required")
    return value


def validate_comparison(s):
    require(
        s["schema"] == "tail_one_minute_comparison_v1"
        and s["status"] == "ok"
        and s["native_interval"] == [START, STOP]
        and s["native_frames"] == STOP - START
        and s["horizon_stop"] == STOP,
        "same fixed minute comparison required",
    )
    for k in (
        "same_parent_and_horizon",
        "all39_arrays_semantically_decoded",
        "all21_arrays_exact_backtrace",
        "original66400_parent_and_migration_verified",
        "tail_all_six_checkpoint_prefixes_exact",
        "old_three_checkpoint_prefix_proofs_reused",
    ):
        require(s[k] is True, "prior proof missing: " + k)
    require(
        (
            s["logical_initialization_count"],
            s["checkpoint_restore_count"],
            s["restores_during_one_minute"],
            s["external_pose_resets"],
        )
        == (1, 4, 0, 0),
        "lifecycle differs",
    )
    require(
        s["owned_process_completion"]["exit_code"] == 0, "owned successful Tail completion required"
    )
    require(
        s["independent_ID_truth"] is False and s["all_three_ID_truth"] is False,
        "ID truth scope changed",
    )


def load_inputs(evidence):
    """Small JSON and source only; no candidate/SL/calibration payload opened here."""
    pins = {**SOURCE_PINS, str(Path(__file__).relative_to(REPO)): file_sha(__file__)}
    for rel, digest in pins.items():
        evidence.read(REPO / rel, digest)
    comparison = seal_read(
        COMPARISON,
        "comparison.provenance.json",
        "tail_one_minute_comparison_v1",
        COMPARISON_SHA,
        evidence,
    )
    input_member(
        comparison,
        WORK / "compare_tail_one_minute.py",
        SOURCE_PINS["work/three_body/compare_tail_one_minute.py"],
    )
    summary = json.loads(named_read(comparison, COMPARISON, "summary.json", evidence))
    validate_comparison(summary)
    shared = summary["shared_SL_calibration_sha256"]
    require(len(shared) == 7, "shared SL3/calibration4 contract required")
    runs = {}
    for method, spec in RUNS.items():
        input_member(comparison, spec["root"] / spec["seal"], spec["sha"])
        sealed = seal_read(spec["root"], spec["seal"], spec["stage"], spec["sha"], evidence)
        path = spec["root"] / "candidate.npz"
        member = output_member(sealed, spec["root"], "candidate.npz")
        require(
            Path(summary["candidate_paths"][method]).resolve() == path.resolve()
            and summary["candidate_sha256"][method] == member["sha256"] == spec["candidate"],
            "candidate lineage differs",
        )
        input_member(comparison, path, spec["candidate"])
        for p, digest in shared.items():
            input_member(comparison, p, digest)
            input_member(sealed, p, digest)
        runs[method] = sealed
    female = seal_read(
        FEMALE,
        "evaluation.provenance.json",
        "long_female_endpoint_evaluation_160s",
        FEMALE_SHA,
        evidence,
    )
    input_member(comparison, FEMALE / "evaluation.provenance.json", FEMALE_SHA)
    for p, digest in shared.items():
        input_member(female, p, digest)
    gt = [
        p
        for p, v in female["inputs"].items()
        if p.endswith("female_sex_gt.csv") and v["sha256"] == GT_SHA
    ]
    require(len(gt) == 1, "original female GT lineage required; no rematching")
    anchor_path = FEMALE / "anchor_matches.json"
    anchor_member = output_member(female, FEMALE, "anchor_matches.json")
    input_member(comparison, anchor_path, anchor_member["sha256"])
    rows = json.loads(named_read(female, FEMALE, "anchor_matches.json", evidence))
    rows = [r for r in rows if START <= r["frame"] < STOP]
    require(
        tuple(r["frame"] for r in rows) == ANCHOR_FRAMES,
        "same 11 requested female anchors required",
    )
    binding = dict(
        native_interval=[START, STOP],
        candidate_horizon=STOP,
        methods=list(METHODS),
        candidate_references={
            m: dict(
                path=str(s["root"] / "candidate.npz"),
                sha256=s["candidate"],
                producer_sha256=s["sha"],
            )
            for m, s in RUNS.items()
        },
        comparison_sha256=COMPARISON_SHA,
        shared_SL_calibration_sha256=shared,
        female_sha256=FEMALE_SHA,
        female_anchor_sha256=anchor_member["sha256"],
        female_GT_sha256=GT_SHA,
        female_frames=list(ANCHOR_FRAMES),
        source_pins=pins,
        dp_config=DP_CONFIG,
        unary="float64 exact zero, all T*6",
        initial_identity="P012 at native66400; common parent numeric slots, not biological truth",
        reused_proofs=(
            "successful four-way terminal39/backtrace21/support/lifecycle; not reexecuted"
        ),
        SL_calibration_scope=(
            "historical manifest lineage only; no arrays/reprojection consumed here"
        ),
    )
    return binding, summary, runs, rows


def candidate_headers(raw, expected_count):
    result = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        require(
            len(names) == 21
            and len(set(names)) == 21
            and set(names) == {k + ".npy" for k in LAYOUT},
            "complete unique21 NPZ fields required",
        )
        for name, (shape_tail, kind) in LAYOUT.items():
            with archive.open(name + ".npy") as f:
                version = np.lib.format.read_magic(f)
                require(version in ((1, 0), (2, 0)), "known NPY header version required")
                shape, fortran, dtype = (
                    np.lib.format.read_array_header_1_0(f)
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0(f)
                )
            require(
                shape == (expected_count, *shape_tail)
                and dtype.kind == kind
                and not dtype.hasobject,
                "candidate header differs: " + name,
            )
            if name in ("frame_idx", "pose_xyz"):
                require(
                    dtype == np.dtype("int64" if name == "frame_idx" else "float64"),
                    "native dtype differs",
                )
            result[name] = dict(shape=list(shape), dtype=dtype.str, fortran_order=fortran)
    return result


def read_candidate(raw, *, support_only, first=FIRST, start=START, stop=STOP):
    headers = candidate_headers(raw, stop - first)
    with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
        frames = saved["frame_idx"]
        require(
            np.array_equal(frames, np.arange(first, stop, dtype=np.int64)),
            "complete native prefix differs",
        )
        selected = slice(start - first, stop - first)
        require(first <= start < stop, "native slice bounds differ")
        keys = ("frame_idx", "observed") if support_only else tuple(LAYOUT)
        arrays = {k: (frames if k == "frame_idx" else saved[k])[selected].copy() for k in keys}
    require(arrays["observed"][0, :, 1].all(), "all initial chest nodes must be direct")
    return arrays, headers


def intervals(frames, flags):
    """Maximal consecutive true runs, [start,stop), preserving each ID separately."""
    require(flags.shape == (len(frames), 3) and flags.dtype == bool, "interval axes differ")
    out = []
    for identity in range(3):
        edges = np.diff(np.r_[False, flags[:, identity], False].astype(np.int8))
        for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            out.append(
                dict(
                    identity=identity,
                    start=int(frames[lo]),
                    stop=int(frames[hi - 1]) + 1,
                    native_frames=int(hi - lo),
                )
            )
    return out


def mapping_comparison(frames, before, after, direct):
    changed = before != after
    direct_involved = changed & (
        np.take_along_axis(direct, before, axis=1) | np.take_along_axis(direct, after, axis=1)
    )
    latent = changed & ~direct_involved
    arrays = dict(
        frame_idx=frames, changed=changed, direct_involved=direct_involved, latent_only=latent
    )
    report = {
        k: dict(
            native_frames=int(v.any(axis=1).sum()),
            identity_frames=int(v.sum()),
            intervals=intervals(frames, v),
        )
        for k, v in arrays.items()
        if k != "frame_idx"
    }
    report["scope"] = (
        "Change of proposed initial ID->raw slot; direct means either old or new chest "
        "has support. Latent means neither has direct chest support. "
        "Not a correctness judgment."
    )
    return arrays, report


def female_scores(frames, sources, observed, review, rows):
    anchors = [MouthAnchor(**r) for r in rows]
    args = (frames, sources[:, :, :, 0], observed[:, :, 0], anchors)
    return dict(
        raw=score_female_identity(*args),
        with_original_review=score_female_identity(*args, review=review),
    )


def solve_one(geometry, counted, anchor_rows):
    """Thin original solver call and independent chosen-link arithmetic only."""
    signatures = {k: signature(v) for k, v in geometry.items()}
    frames, direct = geometry["frame_idx"], geometry["observed"][:, :, 1]
    result = memory.solve_identity_memory_dp(
        frames,
        np.zeros((len(frames), 6), np.float64),
        geometry["pose_xyz"][:, :, 1],
        direct,
        **DP_CONFIG,
    )
    require(
        result.status == "optimal" and not result.approximation_used, "exact feasible DP required"
    )
    require(
        np.array_equal(np.diff(result.layer_offsets), counted["state_counts"])
        and np.array_equal(np.diff(result.edge_offsets), counted["edge_counts"]),
        "solver graph differs from support-only counter",
    )
    identity = build_identity_output(geometry, result.permutation_by_id)
    links = chosen_memory_links(geometry, result.permutation_by_id, scale_mm=0.8, weight=0.2)
    link_cost = float(links["brownian_motion_cost_by_id"].sum())
    require(
        np.isclose(link_cost, result.path_cost, rtol=1e-11, atol=1e-10),
        "chosen links disagree with DP path cost",
    )
    for dp_key, link_key in (
        ("terminal_last_event_frame", "terminal_last_direct_frame"),
        ("terminal_last_event_slot", "terminal_last_direct_raw_slot"),
        ("terminal_age_frames", "terminal_unobserved_age_frames"),
        ("terminal_unsettled_gap", "terminal_unsettled_gap"),
    ):
        require(
            np.array_equal(getattr(result, dp_key), links[link_key]),
            "terminal gap evidence differs",
        )
    baseline = np.tile(np.arange(3, dtype=np.int64), (len(frames), 1))
    changes, change_summary = mapping_comparison(frames, baseline, result.permutation_by_id, direct)
    old_links = chosen_memory_links(geometry, baseline, scale_mm=0.8, weight=0.2)
    report = dict(
        path_cost=result.path_cost,
        independently_summed_link_cost=link_cost,
        P012_link_cost=float(old_links["brownian_motion_cost_by_id"].sum()),
        changes=change_summary,
        terminal_age_frames=result.terminal_age_frames.tolist(),
        terminal_unsettled_gap=result.terminal_unsettled_gap.tolist(),
        all_min_marginals_saved=True,
        biological_ID_accuracy_measured=False,
        female_raw_geometry=female_scores(
            frames,
            geometry["sources"],
            geometry["observed"],
            geometry["identity_review"],
            anchor_rows,
        ),
        female_proposed=female_scores(
            frames,
            identity["proposed_sources"],
            identity["proposed_observed"],
            identity["prior_review_for_proposed_identity"],
            anchor_rows,
        ),
    )
    require(
        signatures == {k: signature(v) for k, v in geometry.items()}, "raw candidate arrays mutated"
    )
    return result, identity, links, changes, report


def json_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite_float64": "nan" if np.isnan(value) else "+inf" if value > 0 else "-inf"}
    return value


def save_result(output, prefix, result):
    values = {f.name: getattr(result, f.name) for f in dataclasses.fields(result)}
    output.save(
        prefix + "/dp_arrays.npz",
        {k: v for k, v in values.items() if isinstance(v, np.ndarray)},
        arrays=True,
    )
    output.save(
        prefix + "/dp_scalars.json",
        {k: json_scalar(v) for k, v in values.items() if not isinstance(v, np.ndarray)},
    )


def validate_preflight(plan, summary, binding):
    require(
        plan["schema"] == summary["schema"] == STAGES["preflight"] and plan["binding"] == binding,
        "exact same source/input/support preflight required",
    )
    require(
        summary["status"] == "ok"
        and summary["all_within_fixed_caps"] is True
        and set(summary["methods"]) == set(METHODS)
        and all(summary["methods"][m]["completed"] is True for m in METHODS),
        "completed bounded preflight required before solve",
    )


def run(args):
    evidence, output = Evidence(), Output(args.output)
    stage, began = STAGES[args.mode], datetime.now(UTC).isoformat()
    try:
        binding, prior, runs, anchors = load_inputs(evidence)
        output.save(
            "plan.json",
            dict(
                schema=stage,
                binding=binding,
                mode=args.mode,
                same_geometry=True,
                independent_ID_truth=False,
                color_unary_reused=False,
                full_DP_executed=args.mode == "solve",
            ),
        )
        for rel, digest in binding["source_pins"].items():
            p = output.root / "code_snapshot" / rel
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_bytes(evidence.read(REPO / rel, digest))
            output.outputs[p.relative_to(output.root).as_posix()] = dict(
                path=str(p), size_bytes=p.stat().st_size, sha256=digest
            )
        pre = pre_summary = None
        if args.mode == "solve":
            require(
                args.preflight_root is not None and args.preflight_sha256 is not None,
                "sealed preflight CLI arguments required",
            )
            pre = seal_read(
                args.preflight_root,
                "preflight.provenance.json",
                STAGES["preflight"],
                args.preflight_sha256,
                evidence,
            )
            pre_plan = json.loads(named_read(pre, args.preflight_root, "plan.json", evidence))
            pre_summary = json.loads(named_read(pre, args.preflight_root, "summary.json", evidence))
            validate_preflight(pre_plan, pre_summary, binding)
        method_reports, solved = {}, {}
        for method in METHODS:
            raw = named_read(runs[method], RUNS[method]["root"], "candidate.npz", evidence)
            candidate, headers = read_candidate(raw, support_only=args.mode == "preflight")
            output.save(method + "/candidate_headers.json", headers)
            frames, direct = candidate["frame_idx"], candidate["observed"][:, :, 1].copy()
            output.save(
                method + "/support.npz", dict(frame_idx=frames, direct_pectoral=direct), arrays=True
            )
            if args.mode == "preflight":
                started = time.perf_counter()
                counted = counter.reachable_counts(frames, direct, counter.Caps())
                seconds = time.perf_counter() - started
                output.save(
                    method + "/counts.npz",
                    dict(
                        frame_idx=counted.frame_idx,
                        state_counts=counted.state_counts,
                        edge_counts=counted.edge_counts,
                    ),
                    arrays=True,
                )
                method_reports[method] = dict(
                    completed=counted.completed,
                    stop=counted.stop,
                    total_states=counted.total_states,
                    total_edges=counted.total_edges,
                    maximum_layer_states=int(counted.state_counts.max()),
                    counted_native_frames=len(counted.state_counts),
                    direct_chest_by_slot=direct.sum(axis=0).tolist(),
                    all_three_direct_frames=int(direct.all(axis=1).sum()),
                    missing_chest_fish_frames=int((~direct).sum()),
                    seconds=seconds,
                    decoded_candidate_fields=["frame_idx", "observed"],
                    graph_scored=False,
                )
            else:
                with np.load(
                    io.BytesIO(
                        named_read(pre, args.preflight_root, method + "/support.npz", evidence)
                    ),
                    allow_pickle=False,
                ) as saved:
                    require(
                        signature(saved["frame_idx"]) == signature(frames)
                        and signature(saved["direct_pectoral"]) == signature(direct),
                        "support bytes differ from preflight",
                    )
                with np.load(
                    io.BytesIO(
                        named_read(pre, args.preflight_root, method + "/counts.npz", evidence)
                    ),
                    allow_pickle=False,
                ) as saved:
                    counted_arrays = {k: saved[k].copy() for k in saved.files}
                require(
                    int(counted_arrays["state_counts"].sum())
                    == pre_summary["methods"][method]["total_states"]
                    and int(counted_arrays["edge_counts"].sum())
                    == pre_summary["methods"][method]["total_edges"],
                    "preflight count accounting differs",
                )
                output.save(method + "/raw_candidate21.npz", candidate, arrays=True)
                before = {k: signature(v) for k, v in candidate.items()}
                result, identity, links, changes, report = solve_one(
                    candidate, counted_arrays, anchors
                )
                require(
                    report["female_raw_geometry"] == prior["methods"][method]["female"],
                    "fixed raw female scores differ from successful comparison",
                )
                save_result(output, method, result)
                for name, value in (
                    ("identity_output", identity),
                    ("chosen_links", links),
                    ("mapping_changes", changes),
                ):
                    output.save(method + "/" + name + ".npz", value, arrays=True)
                output.save(method + "/raw_array_signatures.json", before)
                method_reports[method], solved[method] = (
                    report,
                    (frames, result.permutation_by_id.copy(), direct),
                )
                del result, identity, links, changes
            print(
                json.dumps(dict(method=method, mode=args.mode, report=method_reports[method])),
                flush=True,
            )
        if args.mode == "solve":
            f, before, od = solved["original"]
            _, after, td = solved["tail"]
            # Numeric initial IDs share the verified parent. Different geometries
            # retain separate supports; direct involvement can occur in either.
            changes = before != after
            involved = changes & (
                np.take_along_axis(od, before, axis=1) | np.take_along_axis(td, after, axis=1)
            )
            output.save(
                "between_geometries.npz",
                dict(
                    frame_idx=f,
                    original_mapping=before,
                    tail_mapping=after,
                    changed=changes,
                    direct_involved=involved,
                    latent_only=changes & ~involved,
                ),
                arrays=True,
            )
            output.save(
                "between_geometries.json",
                dict(
                    scope=(
                        "numeric initial-ID assignments only; "
                        "different fixed geometry, not ID truth"
                    ),
                    changed_intervals=intervals(f, changes),
                    direct_involved_intervals=intervals(f, involved),
                    latent_only_intervals=intervals(f, changes & ~involved),
                ),
            )
        summary = dict(
            schema=stage,
            status="ok",
            native_interval=[START, STOP],
            native_frames=STOP - START,
            methods=method_reports,
            all_within_fixed_caps=all(r["completed"] for r in method_reports.values())
            if args.mode == "preflight"
            else True,
            full_DP_executed=args.mode == "solve",
            raw_candidate21_modified=False,
            sources_support_review_modified=False,
            all_three_ID_truth=False,
            RGB_model_fit_GPU_executed=False,
            female_requested_anchors=11,
            female_requested_intervals=10,
            limitations=(
                "Offline full-minute motion assignment; no color unary or new ID truth. "
                "Terminal missing endpoints remain unsettled; low motion/endpoint preservation "
                "is not zero swaps or male-ID correctness."
            ),
        )
        output.save("summary.json", summary)
        evidence.guard()
        output.guard()
        record = dict(
            stage=stage,
            status="ok",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
        )
    except BaseException as error:
        record = dict(
            stage=stage,
            status="failed",
            started_at=began,
            finished_at=datetime.now(UTC).isoformat(),
            inputs=evidence.inputs,
            outputs=output.outputs,
            error=repr(error),
        )
        name = (
            "preflight.provenance.json" if args.mode == "preflight" else "identity.provenance.json"
        )
        (output.root / name).write_text(
            json.dumps(record, indent=2, allow_nan=False), encoding="utf-8"
        )
        raise
    name = "preflight.provenance.json" if args.mode == "preflight" else "identity.provenance.json"
    (output.root / name).write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            dict(status="ok", seal_sha256=file_sha(output.root / name), outputs=len(output.outputs))
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(STAGES), default="preflight")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-root", type=Path)
    parser.add_argument("--preflight-sha256")
    parser.add_argument(
        "--metadata-only", action="store_true", help="only sources and sealed small JSON; no NPZ"
    )
    args = parser.parse_args()
    if args.metadata_only:
        evidence = Evidence()
        binding, _, _, anchors = load_inputs(evidence)
        evidence.guard()
        print(
            json.dumps(
                dict(
                    metadata_only=True,
                    inputs=len(evidence.inputs),
                    anchors=len(anchors),
                    binding=binding,
                )
            )
        )
    else:
        parser.error("--output required") if args.output is None else run(args)


if __name__ == "__main__":
    main()
