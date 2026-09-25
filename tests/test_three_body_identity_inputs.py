"""Identity inputs retain causal enrollment and reject broken evidence before inference."""

import copy
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

LAB = Path(__file__).resolve().parents[1] / "labs/20260909_three_body"

prior_spec = importlib.util.spec_from_file_location(
    "_identity_input_prior_fixture",
    Path(__file__).with_name("test_three_body_appearance_minute_evaluation.py"),
)
prior = importlib.util.module_from_spec(prior_spec)
prior_spec.loader.exec_module(prior)


def load_module():
    names = (
        "prepare_sequence",
        "observations",
        "run_experiment",
        "run_persistent_node_recovery",
        "run_appearance_tracking",
        "appearance_diagnostic",
        "node_geometry",
        "audit_appearance_enrollment",
        "chunk_masks",
        "identity_inputs",
    )
    saved, old_path = {n: sys.modules.get(n) for n in names}, sys.path.copy()
    try:
        sys.path.insert(0, str(LAB))
        for name in names:
            spec = importlib.util.spec_from_file_location(
                f"_identity_input_test_{name}", LAB / f"{name}.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


m = load_module()


@pytest.fixture
def enrollment():
    frames = np.arange(76620, 77286, dtype=np.int64)
    result = prior.appearance_arrays(frames, training=166)
    snap, meta, _ = prior.enrollment_records(frames, result, training=166)
    snap["features"] /= 100
    aggregate = m.enrollment_audit.aggregate_descriptors(snap["features"], 5)
    snap.update(aggregate)
    meta.update(
        config=dict(training_frames=166, minimum_template_samples=5),
        counts=aggregate["counts"].tolist(),
        enabled_views=aggregate["enabled_views"].tolist(),
    )
    nodes = SimpleNamespace(at=lambda _v, _f: np.zeros((3, 3, 2), np.float64))
    return frames, result, snap, meta, nodes


def test_enrollment_does_not_depend_on_final_traceback_or_postprefix_color_use(enrollment):
    frames, final, snap, meta, nodes = enrollment
    before = {k: v.copy() for k, v in snap.items()}
    # All final identities and poses may differ due to future traceback.
    final["pose_xyz"] += 100
    final["sources"] = final["sources"][:, [2, 0, 1]]
    final["identity_review"][:] = True
    meta["sampled_pixels"] = [999999, 999999, 999999]  # Irrelevant to registration.
    m.validate_enrollment(snap, meta, frames, nodes)
    for name in before:
        m.enrollment_audit.exact_array(snap[name], before[name], name)


@pytest.mark.parametrize(
    "bad",
    [
        "review",
        "mouth_unsupported",
        "registering_view_missing",
        "duplicate_source",
        "bad_source",
        "duplicate_instance",
        "small_core",
        "partial_feature",
        "infinite_feature",
        "range_feature",
        "wrong_count",
        "changed_template",
        "enabled_missing_view",
        "wrong_prefix",
        "wrong_dtype",
        "not_frozen",
        "external_templates",
        "future_frame",
        "relabelled",
        "extra_array",
    ],
)
def test_invalid_causal_enrollment_rejected(enrollment, bad):
    frames, _result, snap, meta, nodes = enrollment
    if bad == "review":
        snap["online_identity_review"][0, 0] = True
    elif bad == "mouth_unsupported":
        snap["online_sources"][0, 0, 1:, 0] = -1
        snap["online_observed"][0, 0, 0] = False
    elif bad == "registering_view_missing":
        snap["online_sources"][0, 0, 0, 0] = -1
    elif bad == "duplicate_source":
        snap["online_sources"][0, 1, 0, 0] = 0
    elif bad == "bad_source":
        snap["online_sources"][0, 1, 0, 0] = 3
    elif bad == "duplicate_instance":
        snap["instance_rows"][0, 0, 1] = 0
    elif bad == "small_core":
        snap["core_pixels"][0, 0, 0] = 11
    elif bad in ("partial_feature", "infinite_feature", "range_feature"):
        snap["features"][0, 0, 0, 0] = {
            "partial_feature": np.nan,
            "infinite_feature": np.inf,
            "range_feature": 1.1,
        }[bad]
    elif bad == "wrong_count":
        snap["counts"][0, 0] -= 1
    elif bad == "changed_template":
        snap["templates"][0, 0, 0] += 0.1
    elif bad == "enabled_missing_view":
        snap["enabled_views"][2] = True
    elif bad == "wrong_prefix":
        snap["frame_idx"][3] += 1
    elif bad == "wrong_dtype":
        snap["online_sources"] = snap["online_sources"].astype(np.int32)
    elif bad == "not_frozen":
        meta["frozen"] = False
    elif bad == "external_templates":
        meta["external_templates_or_saved_poses_used"] = True
    elif bad == "future_frame":
        meta["last_training_frame_idx"] += 1
    elif bad == "relabelled":
        meta["retroactive_beam_relabeling"] = True
    else:
        snap["final_traceback"] = snap["online_pose_xyz"].copy()
    with pytest.raises((ValueError, AssertionError)):
        m.validate_enrollment(snap, meta, frames, nodes)


def packed(frames):
    encoder = sys.modules["_identity_input_test_prepare_sequence"].PackedMasks((6, 8))
    for index in range(len(frames)):
        image = np.zeros((2, 6, 8), bool)
        image[0, 1:3, 1:3] = True
        if index % 2:
            image[1, 3:5, 4:7] = True
        encoder.append(
            int(frames[index]),
            image,
            np.ones(2, np.float32),
            np.zeros((2, 4), np.float32),
            np.zeros(2, np.int64),
        )
    return encoder.arrays()


def test_packed_prefix_preserves_every_row_float_bit_and_empty_mask():
    frames = np.arange(10, 14, dtype=np.int64)
    data = packed(frames)
    data["pred_boxes"][0, 0] = -0.0
    result = m.packed_prefix(data, frames, count=2)
    assert result["frame_offsets"].tolist() == [0, 2, 4]
    assert np.signbit(result["pred_boxes"][0, 0])
    assert result["mask_boxes"][1].tolist() == [0, 0, 0, 0]
    m.validate_packed_masks(result)
    for key in ("scores", "classes", "instance_id", "source_track_id", "pred_boxes"):
        m.enrollment_audit.exact_array(result[key], data[key][:4], key)


@pytest.mark.parametrize("bad", ["payload", "hash", "order", "offset", "end", "dtype", "extra"])
def test_corrupt_packed_prefix_rejected(bad):
    frames = np.arange(10, 14, dtype=np.int64)
    data = packed(frames)
    if bad == "payload":
        data["mask_bits"][0] ^= 1
    elif bad == "hash":
        data["frame_mask_sha256"][0] = b"f" * 64
    elif bad == "order":
        data["instance_id"][:2] = [1, 0]
    elif bad == "offset":
        data["frame_offsets"][1] = -1
    elif bad == "end":
        data["mask_byte_offsets"][-1] += 1
    elif bad == "dtype":
        data["scores"] = data["scores"].astype(np.float64)
    else:
        data["identity"] = np.zeros(4)
    with pytest.raises(ValueError):
        m.packed_prefix(data, frames, count=2)


def run_pair(tmp_path, cross=False):
    frames = np.arange(76620, 77286, dtype=np.int64)
    params = {k: {"value": 1} for k in m.COMMON_CONFIGS}
    params.update(seed={"frame": 76620}, tracker_config={"mode": "detectron2"})
    manifest = {"raw": {}, "backgrounds": {}}
    for view in m.base.VIEWS:
        manifest["raw"][view] = {"frame_bgr_sha256": ["a" * 64] * 166}
        manifest["backgrounds"][view] = {"sha256": "b" * 64}
    run = m._Run(
        tmp_path,
        params,
        tmp_path / "sequence",
        tmp_path / "input",
        manifest,
        frames,
        np.zeros((3, 3, 3)),
        {str(tmp_path / "input/a"): "a" * 64},
        {"core.py": "c" * 64},
        {},
    )
    geometry = copy.deepcopy(run)
    if cross:
        geometry = replace(geometry, frames=np.arange(76620, 86580, dtype=np.int64))
        for view in m.base.VIEWS:
            geometry.manifest["backgrounds"][view]["sha256"] = "d" * 64
    return geometry, run


def test_crosswindow_d2_prefix_allows_registered_unused_background_difference(tmp_path):
    geometry, appearance = run_pair(tmp_path, cross=True)
    result = m.compare_runs(geometry, appearance)
    assert result["same_all_inference_paths_and_sha"] is False
    assert result["background_sha_equal"] is False
    assert result["native_prefix_frames"] == 166


@pytest.mark.parametrize(
    "bad",
    [
        "background_mode",
        "pixel",
        "seed",
        "flow",
        "core",
        "calibration",
        "unsupported_window",
        "samewindow_input",
    ],
)
def test_incompatible_geometry_enrollment_pair_rejected(tmp_path, bad):
    geometry, appearance = run_pair(tmp_path, cross=bad != "samewindow_input")
    if bad == "background_mode":
        for run in (geometry, appearance):
            run.params["tracker_config"]["mode"] = "top_contrast"
    elif bad == "pixel":
        geometry.manifest["raw"][m.base.VIEWS[0]]["frame_bgr_sha256"][165] = "f" * 64
    elif bad == "seed":
        geometry.params["seed"]["frame"] += 1
    elif bad == "flow":
        geometry.params["flow_config"]["value"] = 2
    elif bad == "core":
        geometry.code["core.py"] = "f" * 64
    elif bad in ("calibration", "samewindow_input"):
        geometry.inference[str(tmp_path / "input/a")] = "f" * 64
    else:
        geometry = replace(geometry, frames=np.arange(76620, 78000, dtype=np.int64))
    with pytest.raises(ValueError):
        m.compare_runs(geometry, appearance)


@pytest.fixture
def evidence(tmp_path, monkeypatch, enrollment):
    frames, result, snap, meta, _nodes = enrollment
    summary = dict(
        algorithm_arm=prior.evaluation.appearance_runner.ALGORITHM_ARM,
        enrollment_config=meta["config"],
        appearance_config={},
        pixel_config={},
    )
    e = prior.evidence.__wrapped__(tmp_path, monkeypatch, (frames, result, snap, meta, summary))
    monkeypatch.setattr(m, "REPO", e["repo"])
    monkeypatch.setattr(m, "base", prior.evaluation.base)
    monkeypatch.setattr(m, "appearance_runner", prior.evaluation.appearance_runner)
    monkeypatch.setattr(m.base, "REGISTRATION", e["repo"] / "registration.json")
    e["record"]["stage"] = "three_body_causal_appearance_algorithm_arm"
    e["record"]["params"].update(
        output=str(e["run"]), old_results_and_gt_opened_only_after_tracking=True
    )
    e["seal_again"]()
    return e


def test_producer_metadata_loader_never_deserializes_summary_or_final_appearance(
    evidence, monkeypatch
):
    e = evidence
    e["record"]["inputs"]["post_inference:poison"] = {
        "path": "this_label_file_does_not_exist",
        "sha256": "0" * 64,
    }
    e["seal_again"]()
    old_json, old_np_load = m._json, m.np.load

    def checked_json(path):
        assert Path(path).name != "summary.json"
        return old_json(path)

    def checked_load(path, **kwargs):
        assert Path(path).name != "candidate.npz"
        return old_np_load(path, **kwargs)

    monkeypatch.setattr(m, "_json", checked_json)
    monkeypatch.setattr(m.np, "load", checked_load)
    run = m._verify_run(e["run"], appearance=True)
    assert len(run.frames) == 666
    assert all("this_label" not in p for p in run.hashes)


@pytest.mark.parametrize(
    "bad",
    [
        "not_complete",
        "wrong_stage",
        "wrong_arm",
        "changed_output",
        "changed_code",
        "missing_input",
        "bad_alias",
        "bad_seed",
    ],
)
def test_producer_evidence_binding_rejects_corruption(evidence, bad):
    e = evidence
    if bad == "not_complete":
        e["record"]["status"] = "running"
    elif bad == "wrong_stage":
        e["record"]["stage"] = "something_else"
    elif bad == "wrong_arm":
        e["record"]["params"]["algorithm_arm"] = "unknown"
    elif bad == "changed_output":
        (e["run"] / "candidate.npz").write_bytes(b"changed")
        with pytest.raises(ValueError):
            m._verify_run(e["run"], appearance=True)
        return
    elif bad == "changed_code":
        e["source"].write_bytes(b"changed")
    elif bad == "missing_input":
        e["record"]["inputs"].pop(str(e["input_root"] / "staging_manifest.json"))
    elif bad == "bad_alias":
        key = str(e["input_root"] / "staging_manifest.json")
        e["record"]["inputs"]["alias"] = e["record"]["inputs"].pop(key)
    else:
        e["record"]["params"]["seed"]["frame"] += 1
    e["seal_again"]()
    with pytest.raises(ValueError):
        m._verify_run(e["run"], appearance=True)


def test_unknown_asset_refused_before_hash_read(evidence, monkeypatch):
    e = evidence
    forbidden = e["repo"] / "labels.bin"
    forbidden.write_bytes(b"must not be opened")
    e["record"]["inputs"][str(forbidden)] = dict(path=str(forbidden), sha256="0" * 64)
    e["seal_again"]()
    original = m.base.sha256

    def checked(path):
        assert Path(path) != forbidden
        return original(path)

    monkeypatch.setattr(m.base, "sha256", checked)
    with pytest.raises(ValueError, match="unexpected inference asset"):
        m._verify_run(e["run"], appearance=True)


@pytest.fixture
def raw_audit(evidence, enrollment, monkeypatch):
    e = evidence
    frames, _result, snap, _metadata, nodes = enrollment
    run = m._verify_run(e["run"], appearance=True)
    root = e["repo"] / "audit"
    root.mkdir()
    source = e["repo"] / "labs/audit_appearance_enrollment.py"
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(b"synthetic audited replay implementation")
    digest = m.base.sha256(source)
    monkeypatch.setattr(m, "AUDIT_SOURCE_SHA256", digest)
    monkeypatch.setattr(m.enrollment_audit, "audit_code_paths", lambda: [source])
    monkeypatch.setattr(m.enrollment_audit, "REPLAY_SOURCES", ())
    copied = root / "code_snapshot/labs/audit_appearance_enrollment.py"
    copied.parent.mkdir(parents=True)
    copied.write_bytes(source.read_bytes())
    replay = {k: snap[k] for k in (*m.enrollment_audit.PER_FRAME, *m.enrollment_audit.FINAL)}
    np.savez(root / "recomputed_enrollment.npz", **replay)
    masks = {}
    for view in m.base.VIEWS:
        # The audit gate needs frame metadata; actual decoding belongs to packed_prefix tests.
        masks[view] = dict(
            frame_offsets=np.arange(167, dtype=np.int64) * 3,
            view_available=np.ones(166, bool),
            frame_mask_sha256=np.full(166, b"a" * 64, "S64"),
        )
    summary = dict(
        status="complete_exact_cpu_replay",
        algorithm_arm=m.appearance_runner.ALGORITHM_ARM,
        run=str(run.root),
        enrollment_sha256=run.hashes[str(run.root / "appearance_enrollment.npz")],
        sequence_manifest_sha256=run.inference[str(run.sequence / "manifest.json")],
        frame_range=[76620, 76786],
        native_frames=166,
        native_fps=166,
        decoded_images=498,
        all_recomputed_arrays_bitwise_equal=True,
        raw_feature_and_mask_recomputation_performed=True,
        online_native_support_verified=True,
        online_states_compared_to_final_traceback=False,
        identity_labels_verified=False,
        counts=snap["counts"].tolist(),
        enabled_views=snap["enabled_views"].tolist(),
        views=list(m.base.VIEWS),
        feature_names=list(m.FEATURES),
        erosion_radius_px=2,
        minimum_core_pixels=12,
        array_comparisons={
            k: m.enrollment_audit.exact_array(v, snap[k], k) for k, v in replay.items()
        },
        frames=[],
    )
    for frame in frames[:166]:
        summary["frames"].append(
            dict(
                frame_idx=int(frame),
                views={
                    view: dict(
                        bgr_sha256="a" * 64,
                        sleap_points_sha256=m.array_sha256(nodes.at(view, int(frame))),
                        mask_available=True,
                        mask_count=3,
                        decoded_masks_sha256="a" * 64,
                    )
                    for view in m.base.VIEWS
                },
            )
        )
    inputs = {**run.hashes, str(source): digest}
    record = dict(
        stage="three_body_raw_appearance_enrollment_audit",
        status="ok",
        params=dict(
            run=str(run.root),
            output=str(root),
            device="cpu",
            ground_truth_and_completed_baseline_arrays_opened=False,
            reexecuted_beam_selection_or_tracking=False,
            inputs_and_sources_unchanged=True,
        ),
        inputs={p: dict(path=p, sha256=h) for p, h in inputs.items()},
        outputs={},
    )

    def seal():
        prior.write_json(root / "summary.json", summary)
        for key, path in (
            ("summary", root / "summary.json"),
            ("recomputed_enrollment", root / "recomputed_enrollment.npz"),
            ("code:labs/audit_appearance_enrollment.py", copied),
        ):
            record["outputs"][key] = dict(path=str(path), sha256=m.base.sha256(path))
        prior.write_json(root / "audit.provenance.json", record)

    seal()
    return dict(
        run=run,
        root=root,
        record=record,
        summary=summary,
        masks=masks,
        enrollment=snap,
        nodes=nodes,
        seal=seal,
        copied=copied,
    )


def test_raw_audit_binds_producer_enrollment_code_pixels_and_recomputed_bytes(raw_audit):
    e = raw_audit
    hashes = m._verify_audit(e["root"], e["run"], e["enrollment"], e["masks"], e["nodes"])
    assert str(e["root"] / "recomputed_enrollment.npz") in hashes
    assert str(e["root"] / "audit.provenance.json") in hashes


def test_unavailable_view_retains_audit_unknown_instead_of_observed_zero_masks(raw_audit):
    e = raw_audit
    view = m.base.VIEWS[2]
    e["masks"][view]["view_available"][0] = False
    e["masks"][view]["frame_offsets"][1:] -= 3
    item = e["summary"]["frames"][0]["views"][view]
    item.update(mask_available=False, mask_count=None, decoded_masks_sha256=None)
    e["seal"]()
    m._verify_audit(e["root"], e["run"], e["enrollment"], e["masks"], e["nodes"])


@pytest.mark.parametrize(
    "bad",
    [
        "incomplete",
        "wrong_run",
        "opened_gt",
        "missing_input",
        "wrong_enrollment",
        "pixel",
        "sleap",
        "mask",
        "availability",
        "row_count",
        "native_order",
        "comparison",
        "changed_replay",
        "changed_source",
        "missing_source",
    ],
)
def test_failed_or_differently_bound_raw_audit_is_rejected(raw_audit, bad):
    e = raw_audit
    if bad == "incomplete":
        e["record"]["status"] = "running"
    elif bad == "wrong_run":
        e["record"]["params"]["run"] = str(e["root"])
    elif bad == "opened_gt":
        e["record"]["params"]["ground_truth_and_completed_baseline_arrays_opened"] = True
    elif bad == "missing_input":
        e["record"]["inputs"].pop(str(e["run"].root / "appearance_enrollment.npz"))
    elif bad == "wrong_enrollment":
        e["summary"]["enrollment_sha256"] = "b" * 64
    elif bad in ("pixel", "sleap", "mask", "availability", "row_count"):
        key = dict(
            pixel="bgr_sha256",
            sleap="sleap_points_sha256",
            mask="decoded_masks_sha256",
            availability="mask_available",
            row_count="mask_count",
        )[bad]
        e["summary"]["frames"][80]["views"][m.base.VIEWS[0]][key] = {
            "availability": False,
            "row_count": 2,
        }.get(bad, "b" * 64)
    elif bad == "native_order":
        e["summary"]["frames"][80]["frame_idx"] += 1
    elif bad == "comparison":
        e["summary"]["array_comparisons"]["features"]["bitwise_equal"] = False
    elif bad == "changed_replay":
        replay = m._npz(e["root"] / "recomputed_enrollment.npz")
        replay["features"][0, 0, 0, 0] += 0.1
        np.savez(e["root"] / "recomputed_enrollment.npz", **replay)
    elif bad == "changed_source":
        e["copied"].write_bytes(b"another implementation")
    else:
        e["record"]["outputs"].pop("code:labs/audit_appearance_enrollment.py")
        prior.write_json(e["root"] / "audit.provenance.json", e["record"])
        with pytest.raises(ValueError):
            m._verify_audit(e["root"], e["run"], e["enrollment"], e["masks"], e["nodes"])
        return
    e["seal"]()
    with pytest.raises(ValueError):
        m._verify_audit(e["root"], e["run"], e["enrollment"], e["masks"], e["nodes"])


@pytest.mark.parametrize("bad", [None, "pixels", "fps", "count", "truncated"])
def test_changed_clip_requires_actual_prefix_pixels_and_releases_decoder(monkeypatch, bad):
    images = [np.full((4, 6, 3), i, np.uint8) for i in range(3)]
    expected = [m.array_sha256(a) for a in images]
    if bad == "pixels":
        images[1][0, 0, 0] += 1

    class Capture:
        released = False
        position = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == m.cv2.CAP_PROP_FPS:
                return 30 if bad == "fps" else 166
            return 9 if bad == "count" else 10

        def read(self):
            index, self.position = self.position, self.position + 1
            if bad == "truncated" and index == 2:
                return False, None
            return True, images[index]

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(m.cv2, "VideoCapture", lambda _path: capture)
    if bad is None:
        m.verify_raw_prefix("fake", expected, 10, image_shape=(4, 6))
        assert capture.position == 3
    else:
        with pytest.raises(ValueError):
            m.verify_raw_prefix("fake", expected, 10, image_shape=(4, 6))
    assert capture.released
