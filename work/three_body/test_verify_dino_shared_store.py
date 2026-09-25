"""Small metadata-only conversion contract fixtures; no old real payload."""

import copy

import numpy as np
import pytest
import verify_dino_shared_store as m


def fixture():
    crops = [dict(crop_id=f"c{i}") for i in range(71)]
    pairs = [dict(pair_index=i, earlier_crop_id="c0", later_crop_id="c1") for i in range(700)]
    plan = dict(
        schema=m.original.STAGE,
        conditions=["raw", "masked"],
        crops=crops,
        pairs=pairs,
        maximum_forward_calls=142,
        pair_condition_records=1400,
    )
    inference = [dict(crop_id=c["crop_id"], condition=s) for c in crops for s in ("raw", "masked")]
    records = [dict(**p, condition=s) for p in pairs for s in ("raw", "masked")]
    return plan, inference, records


def test_all142_1400_stable_metadata_order():
    values = fixture()
    before = copy.deepcopy(values)
    m.validate_membership(*values)
    assert before == values


@pytest.mark.parametrize(
    "fault", ["missing_token", "duplicate_token", "swapped_condition", "pair_id", "missing_pair"]
)
def test_input_membership_drift_rejected(fault):
    plan, inference, records = fixture()
    if fault == "missing_token":
        inference.pop()
    elif fault == "duplicate_token":
        inference[-1] = inference[-2]
    elif fault == "swapped_condition":
        records[0]["condition"] = "masked"
    elif fault == "pair_id":
        records[500]["pair_index"] = 777
    else:
        records.pop()
    with pytest.raises(ValueError):
        m.validate_membership(plan, inference, records)


def test_fixed256_token_shape_rejected_before_writer():
    row = dict(crop_id="test", condition="raw", token_signature={})
    arrays = {k: np.empty(0) for k in m.original.GEOMETRY_KEYS}
    arrays["final_layernorm_patch_tokens"] = np.ones((255, 384), np.float32)
    with pytest.raises(ValueError, match="full256"):
        m.convert_tokens([row], lambda _: arrays, None)


def test_existing_output_rejected_before_actualinput(monkeypatch, tmp_path):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m.original, "verified", lambda *a: pytest.fail("actual input opened"))
    with pytest.raises(ValueError):
        m.run(tmp_path)
