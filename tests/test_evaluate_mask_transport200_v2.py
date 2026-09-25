"""The evaluation must distinguish reused Original from the two new arms."""

import pytest

from test_evaluate_mask_transport200 import m, run_metadata


def reuse_evidence():
    plan, summary, control = run_metadata()
    plan.update(
        new_arms_this_process=["lk", "ct_first"],
        reused_original_source_seal_sha256=m.producer.FAILED_SHA,
    )
    summary.update(
        this_process_initialize_calls=2,
        this_process_restore_calls=0,
        new_advance_calls=400,
        reused_original_frames=200,
        reused_original_source_seal_sha256=m.producer.FAILED_SHA,
    )
    for index, row in enumerate(summary["arms"]):
        row["computed_in_this_process"] = index != 0
    summary["arms"][0]["reused_from_seal_sha256"] = m.producer.FAILED_SHA
    proof = dict(
        schema="mask_transport_original_reuse_qualification_v1",
        status="ok",
        passed=True,
        original_source_seal_sha256=m.producer.FAILED_SHA,
        original_source_sha256=m.producer.FAILED_SOURCE_SHA,
        frames=200,
        actual_fit4_byte_exact_frames=200,
        saved_frame_fit_and_records_validated=200,
        original_files_copied_byte_exact=408,
        initial_preparation_SL_masks_byte_exact_views=3,
        candidate_21_byte_exact=True,
        history_19_plus_offsets_byte_exact=True,
        core_39_byte_exact=True,
        core_metadata_unchanged=True,
        semantic_state_decode_passed=True,
        sidecar_validation_passed=True,
        original_context_and_sidecar_bytes_preserved=True,
        validation_initialize_calls=0,
        validation_restore_calls=0,
        validation_advance_calls=0,
        validation_fit_calls=0,
        GPU_used=False,
    )
    return plan, summary, control, proof


def test_two_new_arms_and_complete_reused_original_are_accepted():
    plan, summary, control, proof = reuse_evidence()
    m.validate_run(plan, summary, control)
    m.validate_reuse(plan, summary, proof)
    assert summary["advance_calls"] == 600 and summary["new_advance_calls"] == 400


@pytest.mark.parametrize("fault", ["repeat", "wrong_original", "incomplete", "restore", "state"])
def test_invalid_reuse_is_not_presented_as_three_new_arms(fault):
    plan, summary, _, proof = reuse_evidence()
    if fault == "repeat":
        summary["new_advance_calls"] = 600
    elif fault == "wrong_original":
        summary["arms"][0]["reused_from_seal_sha256"] = "0" * 64
    elif fault == "incomplete":
        proof["saved_frame_fit_and_records_validated"] = 199
    elif fault == "restore":
        proof["validation_restore_calls"] = 1
    else:
        proof["core_39_byte_exact"] = False
    with pytest.raises(ValueError):
        m.validate_reuse(plan, summary, proof)
