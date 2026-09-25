from copy import deepcopy

import numpy as np
import pytest
import score_long_female_160s as m

from medaka3d.eval.generative import MouthAnchor


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(m, "FIRST", 10)
    monkeypatch.setattr(m, "STOP", 13)
    monkeypatch.setattr(m, "CHUNK", 3)
    source = np.broadcast_to(np.arange(3)[None, :, None, None], (3, 3, 3, 3)).copy()
    block = dict(
        frame_idx=np.arange(10, 13, dtype=np.int64),
        sources=source,
        observed=np.ones((3, 3, 3), bool),
        original_label_available=np.ones((3, 3), bool),
        original_identity_review=np.zeros((3, 3), bool),
    )
    anchors = [MouthAnchor(f, {"side1_cam": 0}, {}, {}) for f in (10, 11, 12)]
    return block, anchors


def test_reviewed_middle_anchor_is_not_bridged(small):
    block, anchors = small
    block["original_identity_review"][1, 0] = True
    block["original_label_available"][1, 0] = False
    before = deepcopy(block)
    scored = m.score_blocks([block], anchors, method="new")
    assert scored["strict_raw"]["interval_counts"] == dict(preserved=2, swap=0, unscored=0)
    assert scored["strict_with_original_acceptance"]["interval_counts"] == dict(
        preserved=0, swap=0, unscored=2
    )
    assert all(np.array_equal(block[k], v) for k, v in before.items())


def test_baseline_original_label_unavailable_preserves_raw_support(small):
    block, anchors = small
    block["original_label_available"][0, 0] = False
    scored = m.score_blocks([block], anchors, method="c34_lambda_0")
    assert scored["strict_raw"]["available_anchors"] == 3
    assert scored["strict_with_original_acceptance"]["available_anchors"] == 2
    assert scored["strict_with_original_acceptance"]["interval_counts"]["unscored"] == 1


def test_source_switch_is_endpoint_change(small):
    block, anchors = small
    block["sources"][1:, [0, 1]] = block["sources"][1:, [1, 0]]
    scored = m.score_blocks([block], anchors, method="new")
    assert scored["strict_raw"]["interval_counts"] == dict(preserved=1, swap=1, unscored=0)


def test_unmatched_annotation_stays_in_denominator(small):
    block, anchors = small
    anchors[1] = MouthAnchor(11, {}, {}, {})
    score = m.score_blocks([block], anchors, method="new")["strict_raw"]
    assert score["requested_anchors"] == 3 and score["available_anchors"] == 2
    assert score["interval_counts"]["unscored"] == 2


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda b: b["sources"].__setitem__((0, 1, 0, 0), 0), "shared"),
        (lambda b: b["observed"].__setitem__((0, 0, 0), False), "support flags"),
        (lambda b: b["frame_idx"].__setitem__(1, 10), "native block"),
        (lambda b: b["original_label_available"].__setitem__((0, 0), False), "review semantics"),
    ],
)
def test_inconsistent_geometry_or_original_flags_refused(small, change, match):
    block, anchors = small
    change(block)
    with pytest.raises(ValueError, match=match):
        m.score_blocks([block], anchors, method="new")


def test_missing_baseline_or_wrong_block_coverage_refused():
    methods = [
        dict(
            name=name,
            blocks=[
                dict(frame_range=[m.FIRST + i * m.CHUNK, m.FIRST + (i + 1) * m.CHUNK])
                for i in range(16)
            ],
        )
        for name in m.METHODS
    ]
    good = dict(schema=m.COMPARE_STAGE + "_v1", methods=methods)
    m.validate_methods(good)
    missing = deepcopy(good)
    missing["methods"].pop()
    with pytest.raises(ValueError, match="all seven"):
        m.validate_methods(missing)
    bad = deepcopy(good)
    bad["methods"][2]["blocks"][4]["frame_range"][0] += 1
    with pytest.raises(ValueError, match="native block"):
        m.validate_methods(bad)


def test_failed_comparison_creates_no_evaluation(tmp_path, monkeypatch):
    import json

    monkeypatch.setattr(m, "REPO", tmp_path)
    compare = tmp_path / "comparison"
    compare.mkdir()
    (compare / "comparison.provenance.json").write_text(
        json.dumps(dict(status="failed", stage=m.COMPARE_STAGE)), encoding="utf-8"
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="successful geometry"):
        m.evaluate(compare, tmp_path, output)
    assert not output.exists()
