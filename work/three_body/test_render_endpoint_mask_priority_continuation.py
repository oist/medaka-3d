"""Isolated small saved-array / native-PNG display contracts; no science inputs."""

import io
from pathlib import Path

import numpy as np
import pytest
import render_endpoint_mask_priority_continuation as m


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("real source/header/rig/video input forbidden")

    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "REPO", tmp_path)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    for name in ("source_code", "load_headers", "load_rig"):
        monkeypatch.setattr(m, name, forbidden)
    monkeypatch.setattr(m.drawing.prior.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(m.photos, "native_frame", forbidden)
    monkeypatch.setattr(m.photos.prior.raw, "iter_native", forbidden)
    original_load = np.load

    def load(path, *args, **kwargs):
        assert Path(path).resolve().is_relative_to(tmp_path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", load)

    def member(root, seal, name, evidence):
        path = (Path(root) / name).resolve()
        assert path.is_relative_to(tmp_path)
        record = seal["outputs"][name]
        assert record["path"] == str(path) and record["size_bytes"] == path.stat().st_size
        evidence.add(path, record["sha256"])
        return path

    monkeypatch.setattr(m.drawing.previous, "member", member)


def sample_fixture():
    samples = {}
    for ai, arm in enumerate(m.ARMS):
        pose = np.zeros((3, 3, 3, 3), np.float64)
        for t in range(3):
            for b in range(3):
                pose[t, b, :, :2] = [
                    [20 + ai + t, 25 + 35 * b],
                    [35, 25 + 35 * b],
                    [50, 25 + 35 * b],
                ]
        samples[arm + "_pose_xyz"] = pose
        samples[arm + "_frame_idx"] = np.array(m.FRAMES, np.int64)
        samples[arm + "_observed"] = np.ones((3, 3, 3), bool)
        samples[arm + "_sources"] = np.zeros((3, 3, 3, 3), np.int32)
        samples[arm + "_identity_review"] = np.ones((3, 3), bool)
    probes = [
        dict(
            frame_idx=frame,
            classification_by_view={
                v: ["same_instance", "split_instances", "uncovered_endpoint"] for v in m.VIEWS
            },
            split_by_raw_body=[False, True, False],
            saved_observed=samples["endpoint_priority_observed"][i].tolist(),
            saved_identity_review=samples["endpoint_priority_identity_review"][i].tolist(),
        )
        for i, frame in enumerate(m.FRAMES)
    ]
    return samples, probes


class Camera:
    def __init__(self, offset):
        self.offset = offset

    def project(self, pose):
        return pose[..., :2] + self.offset


class Rig:
    def camera(self, view):
        return Camera(m.VIEWS.index(view))


def saved_context(tmp_path, samples, probes):
    out = m.proof.Output(tmp_path / "synthetic_comparison")
    archive = dict(samples)
    # A whole-archive decode would reject these object placeholders. The renderer
    # must only decode the requested15, retaining27 membership scope explicitly.
    for arm in m.ARMS:
        for key in set(m.comparison.FIELDS) - set(m.FIELDS):
            archive[arm + "_" + key] = np.array([object()], dtype=object)
    stream = io.BytesIO()
    np.savez_compressed(stream, **archive)
    path = out.root / "review_samples.npz"
    path.write_bytes(stream.getvalue())
    out.records[path.name] = dict(
        path=str(path), sha256=m.proof.sha(path), size_bytes=path.stat().st_size
    )
    signs = {k: m.proof.signature(v) for k, v in samples.items()}
    signs.update({k: {"synthetic_undecoded": True} for k in set(archive) - set(samples)})
    return dict(
        result_root=out.root, result_seal={"outputs": out.records}, signatures=signs, probes=probes
    )


def test_exact_native_order_support_axis_and_review_shape():
    samples, _ = sample_fixture()
    m.validate_samples(samples)
    samples["nodewise_sources"][0, 0, 0, :] = -1  # One missing view still leaves two.
    m.validate_samples(samples)
    samples["nodewise_sources"][0, 0, 1, :] = -1
    with pytest.raises(ValueError, match="source count"):
        m.validate_samples(samples)
    samples, _ = sample_fixture()
    samples["nodewise_identity_review"] = np.ones((3, 3, 3), bool)
    with pytest.raises(ValueError, match="axes"):
        m.validate_samples(samples)
    samples, _ = sample_fixture()
    samples["nodewise_frame_idx"] = np.array(m.FRAMES[::-1])
    with pytest.raises(ValueError, match="order"):
        m.validate_samples(samples)


def test_selected15_arrays_preserve_raw_slots_and_bind_final_review(tmp_path):
    samples, probes = sample_fixture()
    context = saved_context(tmp_path, samples, probes)
    loaded = m.load_samples(context, m.proof.Evidence())
    for key in samples:
        np.testing.assert_array_equal(loaded[key], samples[key])
        assert not loaded[key].flags.writeable
    context["probes"][0]["saved_identity_review"][0] = False
    with pytest.raises(ValueError, match="plotted sample"):
        m.load_samples(context, m.proof.Evidence())


def test_real_drawing_same_crop_open_nodes_no_rawslot_remap(monkeypatch):
    samples, probes = sample_fixture()
    before = {k: v.tobytes() for k, v in samples.items()}
    image = np.full((120, 140, 3), 72, np.uint8)
    checked = []
    close = m.drawing.prior.plt.close

    def inspect(fig):
        if not hasattr(fig, "axes"):
            close(fig)
            return
        assert len(fig.axes) == 9
        for ai, arm in enumerate(m.ARMS):
            for vi, view in enumerate(m.VIEWS):
                ax = fig.axes[3 * ai + vi]
                assert len(ax.images) == 1
                nodes = [line for line in ax.lines if line.get_marker() in ("o", "s", "^")]
                assert len(nodes) == 9 and all(x.get_markerfacecolor() == "none" for x in nodes)
                expect = Rig().camera(view).project(samples[arm + "_pose_xyz"][0])
                # Frozen drawing emits each body's M/C/T in raw-slot order.
                actual = np.array([[p.get_xdata()[0], p.get_ydata()[0]] for p in nodes])
                np.testing.assert_array_equal(actual, expect.reshape(9, 2))
                assert ax.get_xlim() == fig.axes[vi].get_xlim()
                assert ax.get_ylim() == fig.axes[vi].get_ylim()
        assert "end at91300" in fig._suptitle.get_text()
        assert "SPLIT" in fig.axes[6].get_title()
        assert "uncovered" in fig.axes[8].get_title()
        checked.append(True)
        close(fig)

    monkeypatch.setattr(m.drawing.prior.plt, "close", inspect)
    data, projections, crops = m.draw_frame(
        samples, 0, Rig(), {v: image for v in m.VIEWS}, probes[0]
    )
    assert data.startswith(b"\x89PNG") and checked == [True]
    assert set(crops) == set(m.VIEWS) and set(projections) == set(m.ARMS)
    assert before == {k: v.tobytes() for k, v in samples.items()}


def test_fake_run_reads_only_small_samples_and_nine_saved_pngs(tmp_path, monkeypatch):
    samples, probes = sample_fixture()
    context = saved_context(tmp_path, samples, probes)
    photo_out = m.proof.Output(tmp_path / "synthetic_photos")
    image = np.zeros((1200, 1200, 3), np.uint8)
    image[10:32, 40:92] = [31, 72, 159]
    okay, encoded = m.drawing.prior.cv2.imencode(".png", image)
    assert okay
    photos = {}
    for frame in m.FRAMES:
        for view in m.VIEWS:
            ref = photo_out.write(f"native/frame_{frame}_{view}.png", encoded.tobytes())
            photos[frame, view] = dict(
                frame_idx=frame, view=view, image=ref, bgr_signature=m.proof.signature(image)
            )
    context.update(
        photo_root=photo_out.root,
        photo_seal={"outputs": photo_out.records},
        photos=photos,
        rig_refs={},
    )
    monkeypatch.setattr(m, "source_code", lambda evidence: {"synthetic_only": "fake"})
    monkeypatch.setattr(m, "load_headers", lambda *args: context)
    monkeypatch.setattr(m, "load_rig", lambda *args: Rig())
    seen = []

    def draw(loaded, index, rig, images, probe):
        assert index == len(seen)
        assert set(images) == set(m.VIEWS)
        assert all(np.array_equal(x, image) for x in images.values())
        assert probe["frame_idx"] == m.FRAMES[index]
        seen.append(index)
        return (
            encoded.tobytes(),
            {
                a: {v: rig.camera(v).project(loaded[a + "_pose_xyz"][index]) for v in m.VIEWS}
                for a in m.ARMS
            },
            {},
        )

    monkeypatch.setattr(m, "draw_frame", draw)
    output = tmp_path / "synthetic_output"
    seal = m.run(context["result_root"], "synthetic", photo_out.root, "synthetic", output)
    assert seal["status"] == "ok" and len(seal["outputs"]) == 6 and seen == [0, 1, 2]
    summary = m.proof.read_json(output / "summary.json")
    assert summary["native_PNG_decodes"] == 9 and summary["sample_arrays_decoded"] == 15
    assert summary["original_video_SLEAP_mask_checkpoint_or_full_candidate_reads"] == 0
    for record in seal["outputs"].values():
        assert Path(record["path"]).is_relative_to(tmp_path)
