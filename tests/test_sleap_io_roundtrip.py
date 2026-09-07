"""The guard on the sleap-io pin.

`PRIOR_FINDINGS.md` §4.5: sleap-io 0.8.0's interactive save silently rewired
camera views — `side1 <- side2`, `side2 <- top`. Every key stayed valid, so there
were no orphans and no exception; auto-resume then reloaded and re-saved the
corruption, propagating it. A round of annotation was lost before anyone noticed.

The retired repo's fix was to pin `sleap-io==0.7.0`, then it drifted to three
different answers across three files. We cannot even use 0.7.0: `sleap-nn==0.3.1`
requires `sleap-io>=0.9.2,<0.10.0`, so the version is forced to **0.9.2** — which
is exactly the version §4.5 flags as never re-validated.

So the pin is guarded by a test instead of a comment. This writes a multi-video
labels file whose point coordinates encode which video each instance belongs to,
reopens it, and asserts the mapping survived. A view scramble moves the
coordinates to the wrong video and fails here.

**If this test fails, the pinned version cannot be used for writing** — no matter
what else works. Run it after any change to either pin.
"""

from __future__ import annotations

import numpy as np
import pytest
import sleap_io as sio

#: Recording views, in the order the retired repo's bug rotated them.
VIEWS = ("top_cam/cam0_top.mp4", "side1_cam/cam1_side.mp4", "side2_cam/cam2_side.mp4")

#: Coordinates are a function of the view index, so a scramble is detectable by
#: value alone. Kept far apart so no rounding can blur two views together.
def expected_points(view_index: int) -> np.ndarray:
    base = 100.0 * (view_index + 1)
    return np.array([[base, base], [base + 10.0, base + 20.0]])


@pytest.fixture
def labels() -> sio.Labels:
    skeleton = sio.Skeleton(["mouth", "tail"])
    videos, frames = [], []
    for index, name in enumerate(VIEWS):
        video = sio.Video(filename=f"Z:/fake/{name}")
        videos.append(video)
        frames.append(
            sio.LabeledFrame(
                video=video,
                frame_idx=index,
                instances=[
                    sio.Instance.from_numpy(expected_points(index), skeleton=skeleton)
                ],
            )
        )
    return sio.Labels(labeled_frames=frames, videos=videos, skeletons=[skeleton])


def test_save_reopen_preserves_video_to_instance_mapping(labels, tmp_path):
    """The §4.5 regression: instances must not migrate between views on save."""
    path = tmp_path / "roundtrip.slp"
    sio.save_slp(labels, str(path))
    reopened = sio.load_slp(str(path))

    assert len(reopened.labeled_frames) == len(VIEWS)

    found: dict[str, np.ndarray] = {}
    for frame in reopened.labeled_frames:
        name = str(frame.video.filename).replace("\\", "/")
        assert len(frame.instances) == 1, f"{name}: expected 1 instance"
        found[name] = np.asarray(frame.instances[0].numpy(), dtype=float)[:, :2]

    for index, view in enumerate(VIEWS):
        key = next((k for k in found if k.endswith(view)), None)
        assert key is not None, f"view {view} vanished on reopen; got {sorted(found)}"
        np.testing.assert_allclose(
            found[key],
            expected_points(index),
            err_msg=(
                f"VIEW SCRAMBLE: {view} came back holding another view's "
                f"coordinates. This is the sleap-io 0.8.0 failure "
                f"(PRIOR_FINDINGS 4.5). Do not write labels with this version."
            ),
        )


def test_save_reopen_preserves_frame_indices(labels, tmp_path):
    """A frame_idx shift would silently misalign predictions against video."""
    path = tmp_path / "roundtrip.slp"
    sio.save_slp(labels, str(path))
    reopened = sio.load_slp(str(path))

    by_view = {
        str(f.video.filename).replace("\\", "/").rsplit("/", 2)[-2]: f.frame_idx
        for f in reopened.labeled_frames
    }
    for index, view in enumerate(VIEWS):
        folder = view.split("/")[0]
        assert by_view.get(folder) == index, (
            f"{folder}: frame_idx came back as {by_view.get(folder)}, expected {index}"
        )


def test_pinned_versions_are_the_validated_ones():
    """This test file validates specific versions. Fail loudly if they move."""
    import sleap_nn

    assert sio.__version__ == "0.9.2", (
        f"sleap-io is {sio.__version__}, not the validated 0.9.2. Re-run this "
        f"module's round-trip tests before trusting writes, then update the pin."
    )
    assert sleap_nn.__version__ == "0.3.1", (
        f"sleap-nn is {sleap_nn.__version__}, not the validated 0.3.1."
    )
