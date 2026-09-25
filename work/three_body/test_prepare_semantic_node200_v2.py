"""FourCC retry delta only; no repeat of the qualified27-node core tests."""

import ast
from pathlib import Path

import cv2
import numpy as np
import prepare_semantic_node200_v2 as m
import pytest
from test_prepare_semantic_node200 import Capture


@pytest.mark.parametrize("alias", ["FFV1", "ffv1"])
def test_exact_two_decoder_aliases_preserve_pixels_and_native(monkeypatch, alias):
    monkeypatch.setattr(m, "FRAMES", (m.START, m.START + 1))
    images = [np.zeros((2, 3, 3), np.uint8), np.full((2, 3, 3), 11, np.uint8)]
    cap = Capture(images)
    cap.settings[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*alias)
    hashes = [m.signature(x)["sha256"] for x in images]
    rows = list(m.read200(cap, hashes, shape=(2, 3)))
    assert [n for n, _ in rows] == list(m.FRAMES)
    assert all(x is y for (_, x), y in zip(rows, images, strict=True))
    assert cap.reads == 2


@pytest.mark.parametrize("alias", ["H264", "FfV1", "FFV2"])
def test_alias_does_not_mean_arbitrary_or_general_casefold(monkeypatch, alias):
    monkeypatch.setattr(m, "FRAMES", (m.START,))
    image = np.zeros((2, 3, 3), np.uint8)
    cap = Capture([image])
    cap.settings[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*alias)
    with pytest.raises(ValueError, match="codec"):
        list(m.read200(cap, [m.signature(image)["sha256"]], shape=(2, 3)))
    assert cap.reads == 0


def test_lowercase_does_not_weaken_pixel_guard(monkeypatch):
    monkeypatch.setattr(m, "FRAMES", (m.START,))
    cap = Capture([np.zeros((2, 3, 3), np.uint8)])
    cap.settings[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"ffv1")
    with pytest.raises(ValueError, match="BGR"):
        list(m.read200(cap, ["0" * 64], shape=(2, 3)))


def test_reported_codec_record_is_observation_not_header_claim():
    cap = Capture([np.zeros((2, 3, 3), np.uint8)])
    cap.settings[cv2.CAP_PROP_FOURCC] = cv2.VideoWriter_fourcc(*"ffv1")
    cap.getBackendName = lambda: "FFMPEG"
    report = m.capture_metadata(cap, "top_cam")
    assert report["reported_fourcc_integer"] == 829843046
    assert report["reported_fourcc_bytes"] == list(b"ffv1")
    assert report["RIFF_header_reparsed_by_this_preparation"] is False
    assert cap.reads == 0


def test_all_other_v1_functions_are_AST_exact():
    def funcs(path):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(Path(path).read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef)
        }

    old = funcs(m.FAILED_SOURCE)
    new = funcs(m.__file__)
    changed = {k for k in old if old[k] != new[k]}
    assert changed == {"read200", "source_code", "run"}
    assert set(new) - set(old) == {"retry_metadata", "capture_metadata"}


def test_retry_binds_failure_and_is_planned_without_reading_v1_payloads(monkeypatch):
    failure = dict(
        stage="semantic_node200_preparation_v1",
        status="failed",
        finished_at="fixed",
        error=dict(type="ValueError", message="lossless FFV1 codec required"),
        inputs={str(m.FAILED_SOURCE): m.FAILED_SOURCE_SHA},
    )
    calls = []
    monkeypatch.setattr(m, "bind", lambda p, s, e: calls.append((p, s)) or p)
    monkeypatch.setattr(m, "read_json", lambda p: failure)
    value = m.retry_metadata({})
    assert calls == [
        (m.FAILED / "preparation.provenance.json", m.FAILED_SHA),
        (m.FAILED_SOURCE, m.FAILED_SOURCE_SHA),
    ]
    assert value["all600_new_preparation_planned"]
    assert not value["v1_payloads_adopted"]
    failure["status"] = "ok"
    with pytest.raises(ValueError, match="failure"):
        m.retry_metadata({})
