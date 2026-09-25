"""Finite saved-only packet orchestration; no actual inputs or fits."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "work/three_body"))
import prepare_body_point200_packets as m

from test_body_point_transport import fixture


def prepared_fixture():
    args = fixture(second_anchor=True)
    native = args["point_evidence"][0].vote.native
    calls = []
    reader = SimpleNamespace(
        prepare=lambda frame: calls.append(("prepare", frame.frame_idx)),
        provider=lambda method: (
            lambda n: calls.append((method, n)) or tuple(args["point_evidence"])
        ),
    )
    frame = m.inputs.FrameObservation(
        native, {}, {}, dict(zip(m.VIEWS, args["masks_by_view"], strict=True)), {}
    )
    samples = dict(
        frame_idx=np.asarray(native, np.int64),
        fit_pixels=args["fit_pixels"],
        fit_target=args["fit_foreground"],
        fit_weights=args["fit_weights"],
        fit_foreground=np.zeros(args["fit_foreground"].shape, bool),
    )
    return args, reader, calls, frame, samples


def test_pair_uses_old_f32_target_and_computed_fit_rays_not_bool_companion():
    args, reader, calls, frame, samples = prepared_fixture()
    packets = m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])
    assert calls == [("prepare", 101), ("lk", 101), ("ct_first", 101)]
    assert tuple(packets) == m.EVIDENCE
    for packet in packets.values():
        assert packet.fit_foreground.dtype == np.float32
        assert packet.fit_foreground.all() and packet.old_mass.any()
        assert packet.fit_ray_valid.all()
        assert packet.anchor_ids.tolist() == [0, 1]
        m.body._same(packet.fit_foreground, samples["fit_target"], "actual y")


@pytest.mark.parametrize("fault", ["native", "target_dtype", "weight_dtype"])
def test_wrong_native_or_nonoriginal_sample_dtype_is_rejected(fault):
    args, reader, _, frame, samples = prepared_fixture()
    if fault == "native":
        samples["frame_idx"] += 1
    elif fault == "target_dtype":
        samples["fit_target"] = samples["fit_target"].astype(bool)
    else:
        samples["fit_weights"] = samples["fit_weights"].astype(np.float64)
    with pytest.raises(ValueError):
        m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])


def test_mask_reader_only_consumes_three_observations_and_validates_them(monkeypatch):
    native, reads, validations = m.FIRST, [], []
    members = {
        f"{view}.npz": {"masks": np.zeros((i, 4, 8), bool), "mask_available": np.array(i != 1)}
        for i, view in enumerate(m.VIEWS)
    }

    def read(relative, digest):
        assert relative.endswith(".npz") and digest == "a" * 64
        reads.append(relative)
        return members[relative]

    reader = SimpleNamespace(
        rows={
            (native, v): dict(observation_path=f"{v}.npz", observation_sha256="a" * 64)
            for v in m.VIEWS
        },
        read=read,
        at=lambda *args: pytest.fail("full RGB reader must not be called"),
    )
    monkeypatch.setattr(m.prepared, "arrays_from_bytes", lambda value: value)
    monkeypatch.setattr(
        m.inputs, "validate_observation", lambda arrays, row: validations.append(row)
    )
    frame = m.mask_frame(reader, native)
    assert reads == [f"{v}.npz" for v in m.VIEWS] and len(validations) == 3
    assert frame.masks[m.VIEWS[0]].shape == (0, 4, 8)
    assert frame.masks[m.VIEWS[1]] is None
    assert frame.masks[m.VIEWS[2]].shape == (2, 4, 8)
    assert frame.images_bgr == frame.detections == frame.backgrounds_bgr == {}


def test_real_existing_codec_readback_preserves_all_packet_arrays(tmp_path):
    args, reader, _, frame, samples = prepared_fixture()
    packet = m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])["ct_first"]
    output = m.base.Output(tmp_path / "new_packet")
    restored = m.save_checked_packet(output, "packets/ct_first/native_101", packet, args["rig"])
    m.body._same(packet, restored, "independent disk readback")
    assert len(output.records) == 2
    output.final_guard()


def test_saved_packet_corruption_is_not_promoted_to_a_successful_readback(monkeypatch, tmp_path):
    args, reader, _, frame, samples = prepared_fixture()
    packet = m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])["lk"]
    original_loader = m.old_evaluation.load_tree

    def changed(*args):
        result = original_loader(*args)
        result["single_count"].flat[0] += 1
        return result

    monkeypatch.setattr(m.old_evaluation, "load_tree", changed)
    output = m.base.Output(tmp_path / "bad_packet")
    with pytest.raises(ValueError):
        m.save_checked_packet(output, "packet", packet, args["rig"])
    assert len(output.records) == 2  # Written evidence is retained after failure.


def test_all_denominators_include_t0_invalid_and_each_anchor_without_renormalizing(monkeypatch):
    args, reader, _, frame, samples = prepared_fixture()
    packets = m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])
    monkeypatch.setattr(m, "FIRST", 100)
    monkeypatch.setattr(m, "STOP", 102)
    rows = []
    for native in (100, 101):
        for method, packet in packets.items():
            record = m.packet_statistics(method, packet)
            record["native"] = native
            rows.append(record)
    summary = m.summarize(rows, args["anchors"])
    for result in summary.values():
        assert result["all_query_times"] == 36
        assert result["tracking_query_times"] == 18
        assert result["group_times_per_anchor"] == 18
        assert result["body_view_times_per_anchor"] == 18
        assert result["admitted_group_times_by_anchor"] == [12, 12]
        assert result["reason_counts_by_anchor"][0]["no_strict_initial_link"] == 6
    missing = [r for r in rows if not (r["evidence"] == "lk" and r["native"] == 101)]
    with pytest.raises(ValueError, match="native200"):
        m.summarize(missing, args["anchors"])


def test_input_failure_persists_failed_seal_without_attempting_packet_work(monkeypatch, tmp_path):
    def fail(evidence):
        raise ValueError("synthetic source/input mismatch")

    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "load_inputs", fail)
    destination = tmp_path / "failed_prep"
    with pytest.raises(ValueError, match="synthetic source"):
        m.run(destination)
    seal = json.loads((destination / "preparation.provenance.json").read_text())
    assert seal["status"] == "failed" and seal["completed_packets"] == 0
    assert seal["outputs"] == {}


def test_summary_rejects_saved_admission_flag_disagreeing_with_reasons(monkeypatch):
    args, reader, _, frame, samples = prepared_fixture()
    packets = m.packet_pair(frame, samples, reader, args["anchors"], args["rig"])
    monkeypatch.setattr(m, "FIRST", 101)
    monkeypatch.setattr(m, "STOP", 102)
    rows = [m.packet_statistics(method, packet) for method, packet in packets.items()]
    rows[0]["admitted_by_anchor_view_group"][0][0][0] = False
    with pytest.raises(ValueError, match="admission/reason"):
        m.summarize(rows, args["anchors"])
