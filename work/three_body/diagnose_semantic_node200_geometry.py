"""Saved semantic200 point geometry, conditional on original reference predecessors.

All199 later frames, all nine nodes, six segments and four ray subsets are kept.
LK uses the same comparison release as CT-first, while native availability is
recorded separately. CT-final is supplementary and uses the terminal horizon.
No model, image decode, current SL reassignment, fit or review update is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

import diagnose_temporal_refractive_geometry_v2 as inherited
import numpy as np
import semantic_node_observation as geometry
import semantic_node_queries as selector

base, WORK, REPO = inherited.base, inherited.WORK, inherited.REPO
require, same = base.require, base.same
FIRST, STOP, COUNT = 333519, 333719, 200
VIEWS = tuple(geometry.VIEWS)
METHODS = ("lk", "ct_first", "ct_final")
STAGE = "semantic_node200_geometry_diagnosis_v1"
TRACK_STAGE = "semantic_node200_tracking_v1"
PREP_STAGE = "semantic_node200_preparation_v2"
PREP = WORK / PREP_STAGE
PREP_SHA = "153c9f5f4a316f193b609afd2113b875cf1a62380dca4b80a6b27834b53afc15"
INITIAL_GEOMETRY = WORK / "temporal_refractive_geometry_diagnosis_v2"
INITIAL_SHA = "149d159aea1a1f370f912aebfc616c3bc6d31b24732e704bb6d8e557c36b1fb4"
REFERENCE = inherited.REFERENCES[1]
PLAN = WORK / "SEMANTIC_NODE_200_IMPLEMENTATION_PLAN.md"
PINS = {
    WORK / "track_semantic_node200.py": (
        "219703a8101aac1bc69e60ee061ffbb5875c41027df6ceb337e4a1f8f029bcca"
    ),
    WORK / "prepare_semantic_node200_v2.py": (
        "fb7ca834df1f358deec4c7d82756700c2c3fdfe9a4ad754a29ac51bc5473cbd0"
    ),
    PLAN: "82a196625e8fa80fd3f9867ed65e4428a55301e33c39d7d6620648dca701fc3a",
    Path(
        geometry.__file__
    ).resolve(): "c5e11b7d4b41616c56e3a7e79a973175ddf67f0f614a80e38e272cc49eec7efd",
    Path(
        selector.__file__
    ).resolve(): "ee181e77476161cf6c03f5642caa30a3b1b1e89a1e776205d200ba1e8a947be1",
    Path(
        inherited.__file__
    ).resolve(): "e4a97d589bfedfce24de063c0a35f46af57771e6562daff1841ed5d8d5636404",
    WORK
    / "temporal_point_core.py": "3ec308ca7877338f8677c0703d6f180a9d5658c4858b9868451e1152b7d4e25b",
}


def first_horizons():
    return np.r_[np.full(16, 15, np.int64), np.repeat(np.arange(23, COUNT, 8, dtype=np.int64), 8)]


def read_json(root, members, name, evidence, expected=None):
    return inherited.json_payload(root, members, name, evidence, expected)


def read_arrays(root, members, name, evidence, expected=None):
    return inherited.votes.load_npz(inherited.payload(root, members, name, evidence, expected))


def bind_source(root, seal, members, source, evidence, expected=None):
    digest = inherited.input_sha(seal, source)
    require(
        base.valid_sha(digest) and (expected is None or digest == expected),
        "source binding differs",
    )
    evidence.add(source, digest)
    role = "code_snapshot/" + source.relative_to(REPO).as_posix()
    inherited.payload(root, members, role, evidence, digest)


def bind_declared_pins(root, seal, members, plan, summary, evidence):
    """Declared dependencies need not be imported; loaded sources have snapshots."""
    captured = summary["source_hashes"]
    result = []
    for name, digest in plan["source_pins"].items():
        require(Path(name).name == name and base.valid_sha(digest), "producer source pin schema")
        source = WORK / name
        require(inherited.input_sha(seal, source) == digest, "declared source pin input differs")
        require(
            str(source) not in captured or captured[str(source)] == digest,
            "captured source digest differs",
        )
        evidence.add(source, digest)
        snapshot_path = root / "code_snapshot" / source.relative_to(REPO)
        if str(snapshot_path) in members:
            bind_source(root, seal, members, source, evidence, digest)
            snapshotted = True
        else:
            require(str(source) not in captured, "captured source snapshot missing")
            snapshotted = False
        result.append(
            dict(
                source=str(source),
                sha256=digest,
                snapshot_present=snapshotted,
                loaded_and_captured=str(source) in captured,
            )
        )
    return result


def validate_tracking(plan, summary):
    require(
        plan["schema"] == summary["schema"] == TRACK_STAGE and summary["status"] == "ok",
        "successful semantic200 tracking required",
    )
    require(
        plan["native_interval"] == summary["native_interval"] == [FIRST, STOP], "fixed200 interval"
    )
    require(
        plan["views"] == list(VIEWS) and [v["view"] for v in summary["views"]] == list(VIEWS),
        "ordered three views required",
    )
    require(plan["windows"] == [[i, i + 16] for i in range(0, 185, 8)], "24 full windows required")
    require(plan["initial_geometry_sha256"] == INITIAL_SHA, "initial qualification differs")
    require(
        summary["query_count"] == plan["initial_query_count"] == 27
        and summary["model_initializations"] == 3
        and summary["model_forward_calls"] == 72
        and summary["all200_query_denominator"] == 5400
        and summary["tracking199_query_denominator"] == 5373,
        "complete query/forward accounting",
    )
    require(
        all(
            plan[k] is True and summary[k] is True
            for k in ("no_reseed", "no_query_replacement", "first_emission_primary")
        ),
        "fixed query policy differs",
    )
    require(
        all(
            v["query_count"] == 9
            and v["model_forward_calls"] == 24
            and v["model_initializations"] == 1
            for v in summary["views"]
        ),
        "per-view scope differs",
    )


def reference_metadata(geometry_seal, evidence):
    """Read the fixed original200/continuation proof JSON only, no numeric payload."""
    rr = WORK / REFERENCE["root"]
    rs, rm = inherited.header(
        rr, REFERENCE["filename"], REFERENCE["stage"], REFERENCE["seal"], evidence
    )
    metadata = read_json(rr, rm, "checkpoint/metadata.json", evidence, REFERENCE["metadata"])
    reference_summary = read_json(rr, rm, "summary.json", evidence)
    require(
        reference_summary["original_candidate_21_arrays_byte_exact"] is True,
        "original21 reference proof",
    )
    original_root = WORK / "persistent_mouth_transient_200_v1"
    original_sha = "7084161d4e43fbfdb1e5621afbd53246f1371a57748d912f77fa78d4b3ca18e1"
    _, original_members = inherited.header(
        original_root,
        "experiment.provenance.json",
        "three_body_persistent_node_recovery_algorithm_arm",
        original_sha,
        evidence,
    )
    require(
        inherited.input_sha(rs, original_root / "experiment.provenance.json") == original_sha
        and original_members[str(original_root / "candidate.npz")]["sha256"]
        == REFERENCE["candidate"],
        "original/transient candidate proof differs",
    )
    original_summary = read_json(
        original_root,
        original_members,
        "summary.json",
        evidence,
        "11e520643ce3e9da351431e119c94f052d4c3c5b0fb074e778a73e74d95f5f8c",
    )
    require(
        original_summary["frames"] == [FIRST, STOP]
        and original_summary["seed"]["later_baseline_input"] is False,
        "fixed original seed scope differs",
    )
    require(
        inherited.input_sha(geometry_seal, rr / REFERENCE["filename"]) == REFERENCE["seal"]
        and inherited.input_sha(geometry_seal, rr / "candidate.npz") == REFERENCE["candidate"],
        "initial geometry/reference edge differs",
    )
    return rr, rs, rm, metadata


def metadata_inputs(tracking_root, tracking_sha, evidence):
    """Only source and small JSON reads; no candidate, SL, query or point NPZ."""
    root = Path(tracking_root).resolve()
    require(
        root.is_relative_to(WORK) and base.valid_sha(tracking_sha),
        "workspace root and seal SHA required",
    )
    ts, tm = inherited.header(root, "tracking.provenance.json", TRACK_STAGE, tracking_sha, evidence)
    plan = read_json(root, tm, "plan.json", evidence)
    summary = read_json(root, tm, "summary.json", evidence)
    validate_tracking(plan, summary)
    prep_sha = plan["preparation_sha256"]
    require(prep_sha == PREP_SHA, "fixed successful V2 preparation SHA required")
    ps, pm = inherited.header(PREP, "preparation.provenance.json", PREP_STAGE, prep_sha, evidence)
    require(
        inherited.input_sha(ts, PREP / "preparation.provenance.json") == prep_sha,
        "tracking/preparation edge differs",
    )
    pp = read_json(PREP, pm, "plan.json", evidence)
    require(
        pp["schema"] == PREP_STAGE
        and pp["native_interval"] == [FIRST, STOP]
        and pp["frames"] == list(range(FIRST, STOP))
        and pp["views"] == list(VIEWS)
        and pp["geometry_seal_sha256"] == INITIAL_SHA
        and pp["plan_sha256"] == PINS[PLAN],
        "prepared fixed native/query plan differs",
    )
    gs, gm = inherited.header(
        INITIAL_GEOMETRY,
        "diagnosis.provenance.json",
        "temporal_refractive_geometry_diagnosis_v2",
        INITIAL_SHA,
        evidence,
    )
    require(
        inherited.input_sha(ps, INITIAL_GEOMETRY / "diagnosis.provenance.json") == INITIAL_SHA,
        "preparation/initial geometry edge differs",
    )
    rr, rs, rm, metadata = reference_metadata(gs, evidence)
    shared = pp["shared_SL_calibration_sha256"]
    require(
        len(shared) == 7 and sum(Path(p).suffix == ".npz" for p in shared) == 3,
        "SL3/calibration4 required",
    )
    for p, digest in shared.items():
        require(
            inherited.input_sha(ps, p)
            == inherited.input_sha(gs, p)
            == inherited.input_sha(rs, p)
            == digest,
            "shared geometry input differs",
        )
    # Model stage need not directly consume SL/calibration; it binds the same prep.
    for old, seal, members, source in (
        (root, ts, tm, WORK / "track_semantic_node200.py"),
        (PREP, ps, pm, WORK / "prepare_semantic_node200_v2.py"),
    ):
        bind_source(old, seal, members, source, evidence, PINS[source])
    bind_source(
        root,
        ts,
        tm,
        WORK / "temporal_point_core.py",
        evidence,
        PINS[WORK / "temporal_point_core.py"],
    )
    bind_source(
        PREP,
        ps,
        pm,
        Path(selector.__file__).resolve(),
        evidence,
        PINS[Path(selector.__file__).resolve()],
    )
    declared_pins = bind_declared_pins(root, ts, tm, plan, summary, evidence)
    rows = read_json(PREP, pm, "query_rows.json", evidence)
    require(
        [r["view"] for r in rows] == list(VIEWS) and all(r["query_count"] == 9 for r in rows),
        "all27 ordered initial queries required",
    )
    return dict(
        root=root,
        tracking_sha=tracking_sha,
        ts=ts,
        tm=tm,
        plan=plan,
        summary=summary,
        ps=ps,
        pm=pm,
        prep_sha=prep_sha,
        pp=pp,
        gs=gs,
        gm=gm,
        rr=rr,
        rs=rs,
        rm=rm,
        metadata=metadata,
        shared=shared,
        rows=rows,
        declared_pins=declared_pins,
    )


def validate_queries(query, expected, row):
    require(set(query) == set(expected), "seven fixed query fields required")
    for name in expected:
        same(query[name], expected[name], "initial qualified query " + name)
    require(
        {k: base.signature(a) for k, a in query.items()} == row["array_signatures"],
        "query row signatures differ",
    )


def load_geometry_inputs(context, evidence):
    c = context
    a = read_arrays(c["rr"], c["rm"], "candidate.npz", evidence, REFERENCE["candidate"])
    # Reuse its full21 native/shape/derived-array validation; discard the returned64 copy.
    inherited.select_candidate(a, FIRST, STOP, FIRST)
    for v in range(3):
        for n in range(3):
            sources = a["sources"][:, :, v, n]
            for row in sources:
                held = row[row >= 0]
                require(len(held) == len(np.unique(held)), "reference source reuse")
    require(
        np.array_equal(a["observed"], (a["sources"] >= 0).sum(2) >= 2),
        "reference source/observed differs",
    )
    for p, digest in c["shared"].items():
        evidence.add(p, digest)
    calibration = next(
        Path(p).parent for p in c["shared"] if Path(p).name == "calibration_manifest.json"
    )
    manifest = base.read_json(calibration / "calibration_manifest.json")
    require(
        {str((calibration / r["path"]).resolve()) for r in manifest["model_files"]}
        | {str(calibration / "calibration_manifest.json")}
        == {p for p in c["shared"] if Path(p).suffix == ".json"},
        "rig four-file closure differs",
    )
    rig = inherited.load_rig(calibration)
    inherited.context_geometry(c["metadata"], rig)
    original_context = read_json(PREP, c["pm"], "original_context.json", evidence)
    require(original_context == c["metadata"]["context"], "prepared original context differs")
    first_reference = read_arrays(PREP, c["pm"], "initial_reference21.npz", evidence)
    require(set(first_reference) == set(a), "prepared initial21 fields differ")
    for key in a:
        same(first_reference[key], a[key][:1], "prepared initial reference " + key)
    source = read_arrays(PREP, c["pm"], "initial_source_archive.npz", evidence)
    original_source_role = f"known_gap/sources/native_{FIRST}.npz"
    original_source_path = INITIAL_GEOMETRY / original_source_role
    original_source_digest = c["gm"][str(original_source_path)]["sha256"]
    require(
        inherited.input_sha(c["ps"], original_source_path) == original_source_digest,
        "prepared initial source archive edge differs",
    )
    original_source = read_arrays(
        INITIAL_GEOMETRY,
        c["gm"],
        original_source_role,
        evidence,
        original_source_digest,
    )
    require(set(source) == set(original_source), "prepared source archive fields differ")
    for key in source:
        same(source[key], original_source[key], "prepared source archive " + key)
    same(source["sources"], a["sources"][0], "initial source rows")
    same(source["observed"], a["observed"][0], "initial source observed")
    require(int(source["frame_idx"]) == FIRST, "initial source native differs")
    points = {v: source[v + "_points"] for v in VIEWS}
    rays = inherited.RayObservations.from_points(rig, points)
    strict, observed, _ = inherited.supported_nodes(a["pose_xyz"][0], source["sources"], rays, rig)
    same(strict, source["sources"], "initial unchanged2mm12px sources")
    same(observed, source["observed"], "initial unchanged observed")
    selected = selector.select_queries(points, strict, observed, native_frame=FIRST)
    require(
        read_json(PREP, c["pm"], "query_slots.json", evidence) == selected["slots"]
        and read_json(PREP, c["pm"], "node_sets.json", evidence) == selected["node_sets"],
        "all27 slot/9set qualifier differs",
    )
    queries = {}
    for view, row in zip(VIEWS, c["rows"], strict=True):
        expected = dict(selected["views"][view], query_index=np.arange(9, dtype=np.int32))
        prepared = read_arrays(PREP, c["pm"], row["path"], evidence, row["sha256"])
        require(
            inherited.input_sha(c["ts"], PREP / row["path"]) == row["sha256"],
            "tracking initial query edge",
        )
        validate_queries(prepared, expected, row)
        actual = read_arrays(c["root"], c["tm"], view + "/queries.npz", evidence)
        validate_queries(actual, prepared, row)
        queries[view] = actual
    return a, rig, queries


def validate_point_arrays(query, lk, emissions, calls):
    q = len(query["xy"])
    require(q == 9 and len(calls) == 24, "nine queries and24 calls required")
    expected_horizon = first_horizons()
    same(emissions["frame_idx"], np.arange(FIRST, STOP, dtype=np.int64), "emission native")
    for key, expected in (
        ("first_horizon", expected_horizon),
        ("first_release_native", FIRST + expected_horizon),
        ("final_horizon", np.full(COUNT, COUNT - 1, np.int64)),
    ):
        same(emissions[key], expected, "emission horizon " + key)
    same(
        emissions["first_release_index"],
        np.r_[np.zeros(16, np.int64), np.repeat(np.arange(1, 24, dtype=np.int64), 8)],
        "first release call indices",
    )
    for index, call in enumerate(calls):
        lo, hi = 8 * index, 16 + 8 * index
        require(
            call["call_index"] == index
            and call["input_relative"] == [lo, hi]
            and call["input_native"] == [FIRST + lo, FIRST + hi]
            and call["native_origin"] == FIRST
            and call["maximum_seen_native"] == FIRST + hi - 1
            and call["output_prefix"] == hi
            and call["query_count"] == 9
            and call["query_signature"] == base.signature(query["xy"])
            and call["forward_called"] is True
            and call["status"] == "ok",
            "call native/query/release differs",
        )
    for key in ("first_xy", "final_xy"):
        require(
            emissions[key].shape == (COUNT, q, 2) and emissions[key].dtype == np.float32,
            "CT coordinate schema differs",
        )
    for key in ("first_visibility", "final_visibility"):
        require(
            emissions[key].shape == (COUNT, q) and emissions[key].dtype == bool,
            "CT report schema differs",
        )
    require(
        lk["xy"].shape == (COUNT, q, 2)
        and lk["xy"].dtype == np.float32
        and lk["valid"].shape == (COUNT, q)
        and lk["valid"].dtype == bool,
        "LK point schema differs",
    )
    same(lk["xy"][0], query["xy"], "initial LK coordinates")
    require(
        lk["valid"][0].all() and not np.any(lk["valid"][1:] & ~lk["valid"][:-1]),
        "absorbing LK query chain differs",
    )


def scatter_points(queries, payloads):
    coordinates = {method: np.full((COUNT, 3, 3, 3, 2), np.nan, np.float32) for method in METHODS}
    reported = {method: np.zeros((COUNT, 3, 3, 3), bool) for method in METHODS}
    initial = np.zeros((3, 3, 3), bool)
    for vi, view in enumerate(VIEWS):
        query, data = queries[view], payloads[view]
        require(
            [(int(b), int(n)) for b, n in zip(query["body_slot"], query["node_index"], strict=True)]
            == [(b, n) for b in range(3) for n in range(3)],
            "initial body/node order differs",
        )
        for method, xy, flags in (
            ("lk", data["lk"]["xy"], data["lk"]["valid"]),
            ("ct_first", data["emissions"]["first_xy"], data["emissions"]["first_visibility"]),
            ("ct_final", data["emissions"]["final_xy"], data["emissions"]["final_visibility"]),
        ):
            for q, (body, node) in enumerate(
                zip(query["body_slot"], query["node_index"], strict=True)
            ):
                coordinates[method][:, body, vi, node] = xy[:, q]
                reported[method][:, body, vi, node] = flags[:, q]
                initial[body, vi, node] = True
    require(initial.all(), "all27 qualified queries required")
    return coordinates, reported, initial


def measures(result, reference):
    xyz, valid = result["pose_xyz"], result["node_valid"]
    difference = np.linalg.norm(xyz - reference, axis=-1)
    difference[~valid] = np.nan
    delta = np.diff(xyz, axis=1)
    ref_delta = np.diff(reference, axis=1)
    length, ref_length = np.linalg.norm(delta, axis=-1), np.linalg.norm(ref_delta, axis=-1)
    segment_valid = valid[:, :-1] & valid[:, 1:]
    length[~segment_valid] = np.nan
    direction_valid = segment_valid & (length > 0) & (ref_length > 0)
    angle = np.full((3, 2), np.nan)
    cosine = np.sum(delta * ref_delta, axis=-1)
    angle[direction_valid] = np.degrees(
        np.arccos(
            np.clip(
                cosine[direction_valid] / (length[direction_valid] * ref_length[direction_valid]),
                -1,
                1,
            )
        )
    )
    return dict(
        node_reference_distance_mm=difference,
        segment_valid=segment_valid,
        segment_length_mm=length,
        reference_segment_length_mm=ref_length,
        segment_length_delta_mm=length - ref_length,
        segment_direction_valid=direction_valid,
        segment_reference_angle_degrees=angle,
    )


def availability_comparison(first, second, reference_observed):
    require(
        first.shape == second.shape == reference_observed.shape == (199, 3, 3)
        and first.dtype == second.dtype == reference_observed.dtype == bool,
        "199x9 availability schema",
    )
    deficit = ~reference_observed
    return dict(
        common=first & second,
        gained=second & ~first,
        lost=first & ~second,
        neither=~first & ~second,
        reference_deficit=deficit,
        deficit_common=deficit & first & second,
        deficit_gained=deficit & second & ~first,
        deficit_lost=deficit & first & ~second,
        deficit_first_available=deficit & first,
        deficit_second_available=deficit & second,
    )


def distribution(value):
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return dict(
        scheduled=int(array.size),
        finite=int(finite.size),
        missing=int(array.size - finite.size),
        median=float(np.median(finite)) if finite.size else None,
        p95=float(np.quantile(finite, 0.95)) if finite.size else None,
        maximum=float(finite.max()) if finite.size else None,
    )


def evaluate_frame(method, offset, uv, reported, initial, rig, reference_pose):
    """Use the preceding original-path native pose; frame zero has no geometry."""
    require(
        method in METHODS and type(offset) is int and 1 <= offset < COUNT,
        "one of the199 later frames and fixed method required",
    )
    require(reference_pose.shape == (COUNT, 3, 3, 3), "original reference200 pose schema")
    native = FIRST + offset
    release = STOP - 1 if method == "ct_final" else FIRST + int(first_horizons()[offset])
    result = geometry.build_observation(
        uv,
        reported,
        initial,
        rig,
        reference_pose[offset - 1],
        initial_frame_idx=FIRST,
        previous_frame_idx=native - 1,
        frame_idx=native,
        release_horizon=release,
    )
    result["metadata"].update(
        method=method,
        predecessor_scope="original saved reference",
        native_available_at=native if method == "lk" else release,
        comparison_release_horizon=release,
        reference_is_truth=False,
    )
    return result


def report_text(summary):
    lines = [
        "# Semantic node200 conditional geometry",
        "",
        "All199 later frames and all27 slots/36 subsets per method are retained. "
        "LK uses the common CT-first release; CT-final is supplementary. "
        "The 6mm step gate uses the retrospective original predecessor. "
        "Reference differences measure consistency, not accuracy or biological identity.",
        "Node availability uses the frozen1mm/3px/6mm observation gates. "
        "It does not establish a tracker proposal: whole-pose engineering-box "
        "and segment constraints are not evaluated here.",
        "",
        "| Method | Available nodes /1791 | Reference-distance median (mm) |",
        "| --- | ---: | ---: |",
    ]
    for method in METHODS:
        row = summary["geometry"][method]
        median = row["node_reference_distance_mm"]["median"]
        label = "unavailable" if median is None else f"{median:.6g}"
        lines.append(f"| {method} | {row['available_nodes']} | {label} |")
    lines += ["", "| Comparison with LK | Gained nodes | Lost nodes |", "| --- | ---: | ---: |"]
    for method, row in summary["comparisons"].items():
        lines.append(f"| {method} | {row['gained']['total']} | {row['lost']['total']} |")
    lines += [
        "",
        "Current SL support is inherited from the fixed original reference; "
        "only the initial query qualification is geometrically rechecked. "
        "No images, model, fit, source or review updates occurred. "
        "Full missing denominators, all nodes, segments and rejection reasons "
        "are retained in summary.json and the per-native arrays.",
    ]
    return ("\n".join(lines) + "\n").encode()


def plot_availability(results, reference, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    native = np.arange(FIRST + 1, STOP)
    for body, ax in enumerate(axes):
        for method in METHODS:
            ax.plot(native, results[method]["node_valid"][:, body].sum(-1), label=method, alpha=0.8)
        ax.plot(
            native, reference["observed"][1:, body].sum(-1), label="reference direct SL", ls=":"
        )
        ax.set_ylabel(f"Slot {body}: nodes")
        ax.set_ylim(-0.1, 3.2)
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=4)
    axes[-1].set_xlabel("Native frame; retrospective predecessor gate, not ID truth")
    fig.tight_layout()
    stream = io.BytesIO()
    fig.savefig(stream, format="png", dpi=130)
    plt.close(fig)
    output.write("availability.png", stream.getvalue())


def snapshot(evidence, output, captured):
    paths, inventory = base.source_inventory(
        dict(sys.modules), set(PINS) | {Path(__file__).resolve()}
    )
    output.write(f"source_inventory/{len(captured):03}.json", inventory)
    for path in sorted(paths):
        digest = evidence.add(path, PINS.get(path))
        if str(path) in captured:
            require(captured[str(path)] == digest, "loaded source drift")
        else:
            raw = path.read_bytes()
            require(hashlib.sha256(raw).hexdigest() == digest, "source snapshot read changed")
            output.write("code_snapshot/" + path.relative_to(REPO).as_posix(), raw)
            captured[str(path)] = digest


def run(tracking_root, tracking_sha, destination):
    destination = Path(destination).resolve()
    require(
        destination.is_relative_to(WORK) and not destination.exists(),
        "new workspace output required",
    )
    evidence, output, captured = base.Evidence(), base.Output(destination), {}
    started, tick = base.utc(), time.perf_counter()
    try:
        snapshot(evidence, output, captured)
        context = metadata_inputs(tracking_root, tracking_sha, evidence)
        reference, rig, queries = load_geometry_inputs(context, evidence)
        payloads = {}
        for view in VIEWS:
            lk = read_arrays(context["root"], context["tm"], view + "/lk.npz", evidence)
            emissions = read_arrays(
                context["root"], context["tm"], view + "/emissions.npz", evidence
            )
            calls = read_json(context["root"], context["tm"], view + "/calls.json", evidence)
            validate_point_arrays(queries[view], lk, emissions, calls)
            payloads[view] = dict(lk=lk, emissions=emissions)
        coordinates, reported, initial = scatter_points(queries, payloads)
        output.write("reference21.npz", reference)
        output.write(
            "plan.json",
            dict(
                schema=STAGE,
                tracking_root=str(context["root"]),
                tracking_sha256=tracking_sha,
                preparation_sha256=context["prep_sha"],
                initial_geometry_sha256=INITIAL_SHA,
                reference=REFERENCE,
                methods=list(METHODS),
                native_interval=[FIRST, STOP],
                anchor_geometry_skipped=True,
                predecessor_scope="original retrospective saved path, not a running tracker state",
                same_initial_query_bytes=True,
                current_SL_support_recomputed_only_at_anchor=True,
                tracker_proposal_admissibility_established=False,
                engineering_box_and_segment_gates_applied=False,
                source_pins={str(p): s for p, s in PINS.items()},
                producer_declared_source_pins=context["declared_pins"],
            ),
        )
        results = {}
        for method in METHODS:
            rows = []
            for t in range(1, COUNT):
                native = FIRST + t
                result = evaluate_frame(
                    method,
                    t,
                    coordinates[method][t],
                    reported[method][t],
                    initial,
                    rig,
                    reference["pose_xyz"],
                )
                stem = f"{method}/native_{native}"
                output.write(stem + ".npz", result["arrays"])
                output.write(stem + ".json", result["metadata"])
                rows.append(
                    dict(
                        node_valid=result["arrays"]["node_valid"],
                        chosen_subset=result["arrays"]["chosen_subset"],
                        **measures(result["arrays"], reference["pose_xyz"][t]),
                    )
                )
            table = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            table.update(
                frame_idx=np.arange(FIRST + 1, STOP, dtype=np.int64),
                reference_observed=reference["observed"][1:].copy(),
            )
            output.write(method + "/metrics.npz", table)
            results[method] = table
            print(
                base.json_bytes(
                    dict(method=method, native_frames=199, status="geometry_saved")
                ).decode(),
                flush=True,
            )
        comparisons = {}
        for ct in METHODS[1:]:
            pairs = availability_comparison(
                results["lk"]["node_valid"], results[ct]["node_valid"], reference["observed"][1:]
            )
            output.write(ct + "_versus_lk.npz", pairs)
            comparisons[ct] = {
                k: dict(total=int(v.sum()), by_slot_node=v.sum(0).tolist())
                for k, v in pairs.items()
            }
        summary = dict(
            schema=STAGE,
            status="ok",
            native_interval=[FIRST, STOP],
            anchor_excluded=True,
            frames_per_method=199,
            methods=list(METHODS),
            helper_calls=597,
            slot_records=16119,
            subset_records=21492,
            node_measurements_per_method=1791,
            segment_measurements_per_method=1194,
            comparisons=comparisons,
            geometry={
                method: dict(
                    available_nodes=int(a["node_valid"].sum()),
                    by_slot_node=a["node_valid"].sum(0).tolist(),
                    node_reference_distance_mm=distribution(a["node_reference_distance_mm"]),
                    segment_length_delta_mm=distribution(a["segment_length_delta_mm"]),
                    segment_reference_angle_degrees=distribution(
                        a["segment_reference_angle_degrees"]
                    ),
                )
                for method, a in results.items()
            },
            reference_step_gate_is_conditional=True,
            reference_horizon=STOP,
            direct_support_state_changed=False,
            ID_or_review_changed=False,
            biological_ID_truth=False,
            tracker_proposal_admissibility_established=False,
            engineering_box_and_segment_gates_applied=False,
            model_calls=0,
            fit_calls=0,
            raw_image_decodes=0,
            GPU=False,
            current_SL_support_claim=(
                "Inherited original candidate; only anchor strict geometry repeated"
            ),
            elapsed_before_final_guards_seconds=time.perf_counter() - tick,
        )
        output.write("summary.json", summary)
        plot_availability(results, reference, output)
        output.write("REPORT.md", report_text(summary))
        snapshot(evidence, output, captured)
        require(
            not any(k == "cotracker" or k.startswith("cotracker.") for k in sys.modules),
            "model imported",
        )
        torch = sys.modules.get("torch")
        require(torch is None or not torch.cuda.is_initialized(), "GPU initialized")
        evidence.final_guard()
        output.final_guard()
        status = dict(
            stage=STAGE,
            status="ok",
            started_at=started,
            finished_at=base.utc(),
            inputs=evidence.paths,
            outputs=output.records.copy(),
        )
    except BaseException as error:
        output.write(
            "diagnosis.provenance.json",
            dict(
                stage=STAGE,
                status="failed",
                started_at=started,
                finished_at=base.utc(),
                error_type=type(error).__name__,
                error=str(error),
                inputs=evidence.paths,
                outputs=output.records.copy(),
            ),
        )
        raise
    output.write("diagnosis.provenance.json", status)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-root", type=Path, required=True)
    parser.add_argument("--tracking-sha256", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    require(args.execute != args.metadata_only, "choose reviewed execute or metadata-only")
    if args.metadata_only:
        evidence = base.Evidence()
        context = metadata_inputs(args.tracking_root, args.tracking_sha256, evidence)
        evidence.final_guard()
        print(
            base.json_bytes(
                dict(
                    status="metadata_only",
                    numeric_payloads_opened=False,
                    preparation_sha256=context["prep_sha"],
                )
            ).decode()
        )
    else:
        require(args.output is not None, "new output required")
        run(args.tracking_root, args.tracking_sha256, args.output)


if __name__ == "__main__":
    main()
