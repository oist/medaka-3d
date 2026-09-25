"""Cold validation, exact-byte reuse and fail-closed receipt invalidation."""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "work/three_body"), str(REPO / "labs/20260909_three_body")]
import mask_validation_receipts as receipts  # noqa: E402


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def case(tmp_path):
    encoder = receipts.prepare_sequence.PackedMasks((7, 9))
    for frame in range(510, 513):
        count = 1 if frame == 510 else 0
        masks = np.zeros((count, 7, 9), bool)
        if count:
            masks[0, 3, 2] = True
        encoder.append(
            frame, masks, np.full(count, .75, np.float32),
            np.tile(np.array([2, 3, 3, 4], np.float32), (count, 1)),
            np.zeros(count, np.int64),
        )
    data = encoder.arrays()
    data["view_available"][-1] = False
    npz = tmp_path / "masks.npz"
    np.savez_compressed(npz, **data)
    manifest_path, seal_path = tmp_path / "manifest.json", tmp_path / "staging.provenance.json"
    binding = {"protocol_sha256": "f" * 64, "detector": "fixture"}
    manifest = dict(
        binding=binding, source_interval_half_open=[510, 513], image_shape_hw=[7, 9],
        masks={"top_cam": {"path": npz.name}},
    )
    seal = dict(
        status="ok", stage="prepare_independent_mask_chunk", params={"binding": binding},
        outputs={"top_cam_masks": {"path": str(npz), "sha256": receipts.sha(npz)}},
    )
    write(manifest_path, manifest)
    write(seal_path, seal)
    scope = dict(
        view="top_cam", frame_range=[510, 513], image_shape_hw=[7, 9],
        manifest_path=str(manifest_path), manifest_sha256=receipts.sha(manifest_path),
        seal_path=str(seal_path), seal_sha256=receipts.sha(seal_path),
        detector_binding=binding, expected_npz_sha256=receipts.sha(npz),
    )
    return dict(
        root=tmp_path / "receipts", npz=npz, scope=scope, data=data,
        manifest=manifest, seal=seal,
    )


def validate(case, store=None):
    store = store or receipts.ReceiptStore(case["root"])
    result = store.validate(case["npz"], scope=case["scope"])
    store.final_guard()
    return store, result


def rewrite_data(case):
    np.savez_compressed(case["npz"], **case["data"])
    scope = case["scope"]
    scope["expected_npz_sha256"] = receipts.sha(case["npz"])
    case["seal"]["outputs"]["top_cam_masks"]["sha256"] = scope["expected_npz_sha256"]
    write(Path(scope["seal_path"]), case["seal"])
    scope["seal_sha256"] = receipts.sha(scope["seal_path"])


def test_cold_calls_original_and_fresh_context_hit_skips_decode(case, monkeypatch):
    original = receipts.chunk_masks.validate_packed_masks
    calls = []

    def counted(data):
        calls.append(data["view_available"].tolist())
        original(data)

    monkeypatch.setattr(receipts.chunk_masks, "validate_packed_masks", counted)
    store, cold = validate(case)
    _, warm = validate(case)
    assert calls == [[True, True, False]]
    assert cold["cache_hit"] is False and warm["cache_hit"] is True
    assert cold["receipt_sha256"] == warm["receipt_sha256"]
    warm["scope"]["frame_range"][0] = 0
    assert store.uses[0]["scope"]["frame_range"] == [510, 513]


@pytest.mark.parametrize("field", ["mask_bits", "source_track_id", "frame_mask_sha256", "scores"])
def test_new_bytes_with_updated_seal_require_full_semantic_validation(case, field):
    validate(case)
    if field == "mask_bits":
        case["data"][field][-1] |= 128  # Padding; decoded mask SHA alone would miss it.
    elif field == "frame_mask_sha256":
        case["data"][field][0] = b"0" * 64
    elif field == "scores":
        case["data"][field][0] = np.nan
    else:
        case["data"][field][0] = 0
    rewrite_data(case)
    with pytest.raises(ValueError):
        validate(case)
    assert len(list(case["root"].glob("*/*.success.json"))) == 1


def test_changed_valid_bytes_get_new_receipt(case):
    _, first = validate(case)
    case["data"]["scores"][0] = .8
    rewrite_data(case)
    _, second = validate(case)
    assert second["cache_hit"] is False
    assert first["key"] != second["key"]


@pytest.mark.parametrize("which", ["npz", "manifest_path", "seal_path"])
def test_current_file_sha_is_checked_on_hit(case, which):
    validate(case)
    path = case["npz"] if which == "npz" else Path(case["scope"][which])
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="evidence changed"):
        validate(case)


@pytest.mark.parametrize(
    "damage", ["body", "commit", "failed_status", "false_result", "wrong_scope"],
)
def test_corrupt_or_false_receipts_revalidate_and_preserve_prior_entry(case, damage):
    _, first = validate(case)
    body_path, commit_path = Path(first["receipt_path"]), Path(first["commit_path"])
    if damage in ("body", "commit"):
        (body_path if damage == "body" else commit_path).write_text("{", encoding="utf-8")
    else:
        commit = receipts.read(commit_path)
        body = receipts.read(body_path)
        if damage == "failed_status":
            commit["status"] = "failed"
        elif damage == "false_result":
            body["result"]["every_decoded_frame_sha_verified"] = False
        else:
            body["request"]["scope"]["view"] = "side1_cam"
        write(body_path, body)
        commit["sha256"] = receipts.sha(body_path)
        write(commit_path, commit)
    _, second = validate(case)
    assert second["cache_hit"] is False
    assert len(second["rejected_entries"]) == 1
    assert body_path.exists() and commit_path.exists()
    assert len(list(case["root"].glob("*/*.success.json"))) == 2


def test_uncommitted_receipt_is_never_hit(case):
    _, first = validate(case)
    commit = Path(first["commit_path"])
    commit.rename(commit.with_suffix(".incomplete"))
    _, second = validate(case)
    assert second["cache_hit"] is False


def test_validator_binding_difference_revalidates(case):
    validate(case)
    store = receipts.ReceiptStore(case["root"])
    store.validator_binding["numpy"] = "different-runtime-fixture"
    _, result = validate(case, store)
    assert result["cache_hit"] is False


@pytest.mark.parametrize("which", ["receipt_path", "commit_path", "npz_path"])
def test_final_guard_catches_mutation_after_use(case, which):
    store, result = validate(case)
    with Path(result[which]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="evidence changed"):
        store.final_guard()


def test_detector_failure_never_creates_receipt(case):
    case["seal"]["status"] = "failed"
    write(Path(case["scope"]["seal_path"]), case["seal"])
    case["scope"]["seal_sha256"] = receipts.sha(case["scope"]["seal_path"])
    with pytest.raises(ValueError, match="successful detector seal"):
        validate(case)
    assert not list(case["root"].glob("*/*.success.json"))


def test_scope_binding_cannot_relabel_existing_evidence(case):
    scope = copy.deepcopy(case["scope"])
    scope["detector_binding"]["detector"] = "foreign"
    with pytest.raises(ValueError, match="detector binding differs"):
        receipts.ReceiptStore(case["root"]).validate(case["npz"], scope=scope)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"status":"failed","status":"ok"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        receipts.read(path)


def test_request_uses_its_snapshot_if_callers_scope_changes(case):
    class ChangingCaller(receipts.ReceiptStore):
        def _request(self, path, scope):
            result = super()._request(path, scope)
            scope["frame_range"][0] = 0
            return result

    store = ChangingCaller(case["root"])
    result = store.validate(case["npz"], scope=case["scope"])
    assert result["scope"]["frame_range"] == [510, 513]
    assert receipts.read(result["receipt_path"])["request"]["scope"] == result["scope"]
    store.final_guard()
