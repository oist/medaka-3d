"""Synthetic transformation/seal tests; no unfinished run or actual comparison."""

import copy
import json
from pathlib import Path

import compare_directional_minute_classical as m
import numpy as np
import pytest

from medaka3d.eval.generative import MouthAnchor


def block(start, *, forward=False):
    n = m.CHUNK
    pose = np.zeros((n, 3, 3, 3), np.float64)
    pose[..., 0] = np.arange(n)[:, None, None] * 0.01 + np.arange(3)[None, :, None] * 30
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (n, 3, 3, 3)).copy()
    labels = np.ones((n, 3), bool)
    return dict(
        frame_idx=np.arange(start, start + n, dtype=np.int64),
        pose_xyz=pose,
        sources=source.astype(np.int32 if forward else np.int64),
        observed=np.ones((n, 3, 3), bool),
        original_xyz=pose[:, :, 1].copy(),
        original_label_available=labels,
        original_identity_review=~labels if forward else np.zeros((n, 3), bool),
        slot_track=np.broadcast_to(np.arange(3), (n, 3)).astype(np.int32),
        valid_source=np.ones((n, 3, 3, 3), bool),
        ray_distance_mm=np.zeros((n, 3, 3, 3)),
        native_pixel_error=np.zeros((n, 3, 3, 3)),
    )


def methods_summary():
    methods = []
    for name in m.female.METHODS:
        blocks = []
        for a in range(m.female.FIRST, m.STOP, m.CHUNK):
            blocks.append(
                dict(
                    frame_range=[a, a + m.CHUNK],
                    start_index=a - m.female.FIRST,
                    stop_index=a + m.CHUNK - m.female.FIRST,
                    path=f"{name}/support_f{a:06d}-{a + m.CHUNK:06d}.npz",
                    sha256="a" * 64,
                    ray_gate_mm=2.0,
                    native_pixel_gate=12.0,
                    all_sources_and_observed_flags_pass=True,
                )
            )
        methods.append(dict(name=name, blocks=blocks))
    return dict(schema=m.female.COMPARE_STAGE + "_v1", methods=methods)


def test_all_seven_fixed_methods_keep_exact_last_six_native_blocks():
    selected = m.selected_blocks(methods_summary())
    assert [method["name"] for method, _ in selected] == list(m.female.METHODS)
    assert len(selected) == 8 and all(len(blocks) == 6 for _, blocks in selected)
    assert selected[0][1][0]["frame_range"] == [66400, 68060]
    assert selected[-1][1][-1]["frame_range"] == [74700, 76360]


@pytest.mark.parametrize(
    "fault", ["missing_lambda", "order", "partial", "native", "namespace", "gate"]
)
def test_method_selection_rejects_missing_or_misbound_block_metadata(fault):
    data = methods_summary()
    if fault == "missing_lambda":
        data["methods"].pop()
    elif fault == "order":
        data["methods"][1:3] = data["methods"][1:3][::-1]
    elif fault == "partial":
        data["methods"][1]["blocks"].pop()
    elif fault == "native":
        data["methods"][1]["blocks"][10]["start_index"] += 1
    elif fault == "namespace":
        data["methods"][1]["blocks"][10]["path"] = data["methods"][2]["blocks"][10]["path"]
    else:
        data["methods"][1]["blocks"][10]["native_pixel_gate"] = 13.0
    with pytest.raises(ValueError):
        m.selected_blocks(data)


def test_six_chunks_preserve_cross_boundary_adjacent_pairs_and_owned_arrays():
    blocks = [block(a, forward=True) for a in range(m.START, m.STOP, m.CHUNK)]
    result = m.concatenate_blocks(blocks, method="directional")
    assert result["sources"].dtype == np.int64
    metrics = m.measure(result, [], name="directional")
    assert metrics["fish_frame_denominator"] == 29880
    assert metrics["direct_adjacent_movement"]["scored_adjacent_steps"] == (9960 - 1) * 3
    result["sources"][:] = -1
    assert (blocks[0]["sources"] >= 0).all()


@pytest.mark.parametrize("fault", ["frame", "source", "chest", "label", "ray", "pixel", "flag"])
def test_support_payload_or_saved_chest_tampering_rejected(fault):
    data = block(m.START)
    if fault == "frame":
        data["frame_idx"][4] += 1
    elif fault == "source":
        data["sources"][0, 1, 0, 0] = 0
    elif fault == "chest":
        data["pose_xyz"][0, 0, 1, 0] += 1
    elif fault == "label":
        data["slot_track"][0, 0] = -1
    elif fault == "ray":
        data["ray_distance_mm"][0, 0, 0, 0] = 2.000001
    elif fault == "pixel":
        data["native_pixel_error"][0, 0, 0, 0] = 12.000001
    else:
        data["valid_source"][0, 0, 0, 0] = False
    with pytest.raises(ValueError):
        m.validate_support_block(data, m.START, m.START + m.CHUNK, method="c34_lambda_2")


def test_native_acceptance_differs_without_erasing_common_support_or_nan_denominator():
    data = block(m.START)
    data["original_label_available"][:, 1] = False
    data["slot_track"][:, 1] = -1
    data["original_xyz"][0, 0] = np.nan
    data["pose_xyz"][0, 0, 1] = np.nan
    data["sources"][0, 0, :, 1] = -1
    data["observed"][0, 0, 1] = False
    data["valid_source"][0, 0, :, 1] = False
    normalized = m.validate_support_block(data, m.START, m.START + m.CHUNK, method="c34_lambda_2")
    result = m.measure(normalized, [], name="c34_lambda_2")
    assert result["fish_frame_denominator"] == m.CHUNK * 3
    assert result["direct_chest_fish_frames"] == m.CHUNK * 3 - 1
    assert result["original_method_available_fish_frames"] == m.CHUNK * 2 - 1
    assert result["original_review_fish_frames"] is None
    assert "derived" in result["node_scope"]


def test_unscored_female_anchor_remains_in_two_intervals_and_latent_jump_excluded():
    data = block(m.START, forward=True)
    data["pose_xyz"][8, 0] += 500
    data["original_xyz"][8, 0] = data["pose_xyz"][8, 0, 1]
    data["observed"][8, 0] = False
    data["sources"][8, 0] = -1
    data["valid_source"][8, 0] = False
    normalized = m.validate_support_block(data, m.START, m.START + m.CHUNK, method="directional")
    anchors = [MouthAnchor(m.START + i, {"side1_cam": 0}, {}, {}) for i in (5, 8, 11)]
    metrics = m.measure(normalized, anchors, name="directional")
    female = metrics["female"]["strict_raw"]
    assert female["requested_anchors"] == 3 and female["requested_intervals"] == 2
    assert female["interval_counts"] == dict(preserved=0, swap=0, unscored=2)
    assert metrics["direct_adjacent_movement"]["steps_over_10mm"] == 0


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return dict(path=str(path), sha256=m.female.sha(path), size=path.stat().st_size)


def completed_fixture(root):
    support = dict(
        native_frames=9960,
        frame_range=[m.START, m.STOP],
        blocks=6,
        block_frames=m.CHUNK,
        ray_gate_mm=2.0,
        native_pixel_gate=12.0,
        all_sources_and_observed_flags_pass=True,
        temporal_subsampling=False,
    )
    summary = dict(
        schema="directional_one_minute_comparison_v1",
        status="ok",
        native_interval=[m.START, m.STOP],
        native_frames=9960,
        same_parent_and_horizon=True,
        all21_original_and_directional_arrays_exact_backtrace=True,
        original_history_and_all_six_checkpoint_prefixes_exact=True,
        checkpoint_snapshots_and_earlier_manifest_digests_verified=True,
        logical_initialization_count=1,
        restores_during_one_minute=0,
        external_pose_resets=0,
        methods={
            name: dict(all_native_source_geometry=copy.deepcopy(support))
            for name in ("original", "directional")
        },
    )
    outputs = {
        "summary": write_json(root / "summary.json", summary),
        "checkpoints": write_json(root / "checkpoint_audit.json", []),
        "frames": write_json(root / "frames.csv", {}),
    }
    path = Path(m.minute.__file__).resolve()
    record = dict(
        stage="directional_one_minute_comparison",
        status="ok",
        params=dict(all_bound_files_unchanged=True),
        inputs={str(path): dict(path=str(path), sha256=m.PINS[path])},
        outputs=outputs,
    )
    write_json(root / "comparison.provenance.json", record)
    return record, summary


def test_completed_minute_header_scope_accepted_without_opening_arrays(tmp_path, monkeypatch):
    completed_fixture(tmp_path)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("payload opened by header gate"))
    _, summary = m.completed_minute(tmp_path, m.female.Evidence())
    assert summary["native_frames"] == 9960


@pytest.mark.parametrize("fault", ["unfinished", "partial_support", "code", "summary_hash"])
def test_unfinished_or_changed_minute_blocks_comparison_before_legacy_payload(
    tmp_path, monkeypatch, fault
):
    record, summary = completed_fixture(tmp_path)
    if fault == "unfinished":
        record["status"] = "running"
    elif fault == "partial_support":
        summary["methods"]["directional"]["all_native_source_geometry"]["native_frames"] = 1660
        record["outputs"]["summary"] = write_json(tmp_path / "summary.json", summary)
    elif fault == "code":
        next(iter(record["inputs"].values()))["sha256"] = "0" * 64
    else:
        write_json(tmp_path / "summary.json", dict(summary, native_frames=1))
    write_json(tmp_path / "comparison.provenance.json", record)
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("unfinished comparison opened arrays")
    )
    with pytest.raises(ValueError):
        m.load_evidence(
            tmp_path, tmp_path / "absent_old", tmp_path / "absent_female", m.female.Evidence()
        )


@pytest.mark.parametrize("corrupt_snapshot", [False, True])
def test_synthetic_final_report_keeps_all9_and_checks_its_output_seal(
    tmp_path, monkeypatch, corrupt_snapshot
):
    source = tmp_path / "synthetic_source.py"
    source.write_text("# synthetic only\n")
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(m.classical, "code_paths", lambda: [source])
    output = tmp_path / "work/result"

    def snapshot(codes, root, record):
        target = root / "code_snapshot/synthetic_source.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(source.read_bytes())
        record.output("source_snapshot", target, sha256=m.female.sha(target))

    monkeypatch.setattr(m, "snapshot", snapshot)
    data = m.concatenate_blocks(
        [block(a) for a in range(m.START, m.STOP, m.CHUNK)], method="c34_lambda_0"
    )
    candidates = {name: data for name in m.METHODS}
    anchors = [MouthAnchor(m.START + i, {"side1_cam": 0}, {}, {}) for i in (5, 8, 11)]
    scored = m.measure(data, anchors, name="directional")["female"]
    previous = dict(
        methods={
            name: dict(
                female=dict(
                    raw=scored["strict_raw"],
                    with_original_review=scored["strict_with_original_acceptance"],
                )
            )
            for name in ("original", "directional")
        }
    )

    def prepared(*args):
        if corrupt_snapshot:
            (output / "code_snapshot/synthetic_source.py").write_text("changed after snapshot")
        return candidates, anchors, previous, {"synthetic_only": True}

    monkeypatch.setattr(m, "load_evidence", prepared)
    if corrupt_snapshot:
        with pytest.raises(ValueError):
            m.run(tmp_path / "synthetic_completed_comparison", output)
    else:
        m.run(tmp_path / "synthetic_completed_comparison", output)
    seal = json.loads((output / "comparison.provenance.json").read_text())
    assert seal["status"] == ("failed" if corrupt_snapshot else "ok")
    if not corrupt_snapshot:
        summary = json.loads((output / "summary.json").read_text())
        assert [row["name"] for row in summary["methods"]] == list(m.METHODS)
        assert not summary["all_three_ID_accuracy_measured"]
        assert summary["requested_female_intervals"] == 2
        for value in seal["outputs"].values():
            assert m.female.sha(value["path"]) == value["sha256"]
