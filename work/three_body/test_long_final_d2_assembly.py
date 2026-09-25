"""Synthetic D2 evidence and lossless streaming checks; no producer/model/video."""

import copy
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import long_final_d2_assembly as audit
from chunk_masks import concatenate_packed_masks
from prepare_sequence import PackedMasks


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return path


def info(path):
    return dict(path=str(path.resolve()), sha256=digest(path), size=path.stat().st_size)


class Evidence:
    """Small fixture inode memo with the real injectable add/final_guard contract."""

    def __init__(self):
        self.paths, self.identities, self.hash_calls = {}, {}, 0

    def add(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        stat = path.stat()
        identity = stat.st_dev, stat.st_ino
        if identity not in self.identities:
            self.identities[identity] = digest(path)
            self.hash_calls += 1
        actual = self.identities[identity]
        if expected is not None and actual != expected:
            raise ValueError("fixture SHA mismatch")
        self.paths[str(path)] = actual
        return actual

    def final_guard(self):
        for path, expected in self.paths.items():
            if digest(path) != expected:
                raise ValueError("fixture final SHA mismatch")


def encode(start, counts, shape=(7, 9), unavailable=()):
    encoder = PackedMasks(shape)
    for offset, count in enumerate(counts):
        masks = np.zeros((count, *shape), bool)
        for i in range(count):
            if i == 0 and count > 1:
                continue  # Empty detection: no bits, still an instance.
            masks[i, 1:4, 2:5] = True
        encoder.append(
            start + offset,
            masks,
            np.linspace(0.5, 1, count, dtype=np.float32),
            np.tile(np.array([-0.0, 0.125, 7.25, 5.875], np.float32), (count, 1)),
            np.arange(count, dtype=np.int64),
        )
    result = encoder.arrays()
    result["view_available"][list(unavailable)] = False
    return result


@pytest.fixture
def packed():
    return [encode(510, [2, 0], unavailable=(1,)), encode(512, [1]), encode(513, [0, 3])]


def saved_join(tmp_path, chunks):
    path = tmp_path / "assembled.npz"
    np.savez_compressed(path, **concatenate_packed_masks(chunks))
    return path


def test_stream_matches_frozen_concatenation_all_fields_and_does_not_mutate(
    tmp_path, packed, monkeypatch
):
    before = [{key: value.tobytes() for key, value in row.items()} for row in packed]
    path = saved_join(tmp_path, packed)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("global np.load forbidden"))
    rows = []
    result = audit.verify_packed_stream(path, iter(packed), on_chunk=rows.append)
    assert result == dict(
        chunks=3,
        frames=5,
        instances=6,
        mask_bytes=8,
        exact_fields=13,
        decoded_masks_rechecked=False,
    )
    assert [(r["start"], r["stop"]) for r in rows] == [(510, 512), (512, 513), (513, 515)]
    assert before == [{key: value.tobytes() for key, value in row.items()} for row in packed]


def test_empty_available_unavailable_and_zero_byte_detections_are_retained(tmp_path):
    chunks = [encode(3, [0], unavailable=(0,)), encode(4, [0])]
    path = saved_join(tmp_path, chunks)
    result = audit.verify_packed_stream(path, chunks)
    assert result["instances"] == result["mask_bytes"] == 0
    chunks[0]["view_available"][0] = True
    with pytest.raises(ValueError, match="view_available"):
        audit.verify_packed_stream(path, chunks)


@pytest.mark.parametrize("key", list(audit.SCHEMA_DTYPES))
def test_every_global_field_corruption_is_rejected(tmp_path, packed, key):
    arrays = concatenate_packed_masks(packed)
    value = arrays[key]
    if key == "frame_mask_sha256":
        value[0] = b"a" * 64
    elif key == "view_available":
        value[0] = ~value[0]
    elif key == "mask_bits":
        value[0] ^= np.uint8(1)
    else:
        value.reshape(-1)[0] += 1
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=key):
        audit.verify_packed_stream(path, packed)


def test_signed_zero_is_a_byte_difference(tmp_path, packed):
    arrays = concatenate_packed_masks(packed)
    assert np.signbit(arrays["pred_boxes"][0, 0])
    arrays["pred_boxes"][0, 0] = 0.0
    path = tmp_path / "zero.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="pred_boxes"):
        audit.verify_packed_stream(path, packed)


@pytest.mark.parametrize("failure", ["dtype", "tail", "missing", "extra", "fortran", "duplicate"])
def test_global_header_schema_and_terminal_payload_rejected(tmp_path, packed, failure):
    arrays = concatenate_packed_masks(packed)
    if failure == "dtype":
        arrays["scores"] = arrays["scores"].astype(np.float64)
    elif failure == "tail":
        arrays["scores"] = np.r_[arrays["scores"], np.float32(0.7)]
    elif failure == "missing":
        arrays.pop("scores")
    elif failure == "extra":
        arrays["unknown"] = np.zeros(1)
    elif failure == "fortran":
        arrays["pred_boxes"] = np.asfortranarray(arrays["pred_boxes"])
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, **arrays)
    if failure == "duplicate":
        with ZipFile(path, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("scores.npy", b"duplicate")
    with pytest.raises(ValueError):
        audit.verify_packed_stream(path, packed)


def test_trailing_bytes_inside_npy_cannot_hide_after_declared_shape(tmp_path, packed):
    arrays = concatenate_packed_masks(packed)
    path = tmp_path / "trailing.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            archive.writestr(key + ".npy", stream.getvalue() + (b"x" if key == "scores" else b""))
    with pytest.raises(ValueError, match=r"terminal.*scores"):
        audit.verify_packed_stream(path, packed)


@pytest.mark.parametrize("field", ["frame_offsets", "mask_byte_offsets"])
def test_middle_chunk_offset_must_include_prior_instance_or_byte_count(tmp_path, packed, field):
    arrays = concatenate_packed_masks(packed)
    position = (
        len(packed[0]["frame_idx"]) if field == "frame_offsets" else len(packed[0]["mask_boxes"])
    )
    arrays[field][position + 1] = packed[1][field][1]
    path = tmp_path / "bad-rebase.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match=field):
        audit.verify_packed_stream(path, packed)


def test_short_npy_member_payload_rejected_at_eof(tmp_path, packed):
    arrays = concatenate_packed_masks(packed)
    path = tmp_path / "short.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for key, value in arrays.items():
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=False)
            raw = stream.getvalue()
            archive.writestr(key + ".npy", raw[:-1] if key == "scores" else raw)
    with pytest.raises(ValueError, match="scores"):
        audit.verify_packed_stream(path, packed)


@pytest.mark.parametrize("value", [None, "", "F" * 64, "f" * 63])
def test_explicit_invocation_sha_required_before_any_io(value):
    with pytest.raises(ValueError, match="SHA256"):
        audit.load_completed_assembly("unopened", value, {}, [], Evidence())


def test_gap_empty_sequence_and_callback_failure_stop_before_success(tmp_path, packed):
    path = saved_join(tmp_path, packed)
    with pytest.raises(ValueError, match="at least one"):
        audit.verify_packed_stream(path, [])
    with pytest.raises(ValueError, match="consecutive"):
        audit.verify_packed_stream(path, [packed[0], packed[2]])
    seen = []

    def stop(row):
        seen.append(row)
        raise RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        audit.verify_packed_stream(path, packed, on_chunk=stop)
    assert len(seen) == 1
    # Streams were closed even after callback failure.
    assert audit.verify_packed_stream(path, packed)["chunks"] == 3


def scope():
    entries = [
        dict(
            start=audit.FIRST + i * audit.CHUNK,
            stop=audit.FIRST + (i + 1) * audit.CHUNK,
            attempt_path=f"chunks/c{i}/attempt_0001",
            manifest_sha256="a" * 64,
            provenance_sha256="b" * 64,
        )
        for i in range(audit.COUNT)
    ]
    summary = dict(
        status="complete",
        tracking_performed=False,
        completed_native_frames=298800,
        video_payload_bytes_copied=0,
        protocol_sha256="c" * 64,
        chunks=[dict(row, reused=i > 0, rejected_attempts=[]) for i, row in enumerate(entries)],
        rejected_assemblies=[],
    )
    index = dict(
        schema="medaka_mask_chunk_index_v1",
        source_frame_range=[49800, 348600],
        chunk_frames=1660,
        protocol_sha256="c" * 64,
        chunks=entries,
    )
    consumed = [dict(row, index=i) for i, row in enumerate(entries)]
    return summary, index, consumed


def test_exact_real_180_scope_retains_rejected_attempts_without_promoting_them():
    summary, index, consumed = scope()
    summary["chunks"][0]["rejected_attempts"] = [dict(path="old", reason="failed")]
    summary["rejected_assemblies"] = [dict(path="bad", reason="incomplete")]
    before = copy.deepcopy((summary, index, consumed))
    assert audit.validate_index(summary, index, consumed, "c" * 64) is None
    assert (summary, index, consumed) == before


@pytest.mark.parametrize(
    "failure",
    [
        "incomplete",
        "tracking",
        "copied",
        "frames",
        "range",
        "count",
        "order",
        "attempt",
        "manifest",
        "seal",
        "protocol",
        "bool_index",
        "float_start",
        "missing_history",
    ],
)
def test_scope_identity_and_producer_flags_reject_drift(failure):
    summary, index, consumed = scope()
    if failure in {"incomplete", "tracking", "copied", "frames"}:
        key, value = {
            "incomplete": ("status", "incomplete"),
            "tracking": ("tracking_performed", 0),
            "copied": ("video_payload_bytes_copied", False),
            "frames": ("completed_native_frames", 298799),
        }[failure]
        summary[key] = value
    elif failure == "range":
        index["source_frame_range"][0] += 1
    elif failure == "count":
        index["chunks"].pop()
    elif failure == "order":
        index["chunks"].reverse()
    elif failure in {"attempt", "manifest", "seal"}:
        key = {
            "attempt": "attempt_path",
            "manifest": "manifest_sha256",
            "seal": "provenance_sha256",
        }[failure]
        consumed[5][key] = "changed"
    elif failure == "protocol":
        summary["protocol_sha256"] = "f" * 64
    elif failure == "bool_index":
        consumed[0]["index"] = False
    elif failure == "float_start":
        index["chunks"][0]["start"] = float(audit.FIRST)
    else:
        summary["chunks"][0].pop("reused")
    with pytest.raises(ValueError):
        audit.validate_index(summary, index, consumed, "c" * 64)


def make_seal(root, filename, stage, binding, inputs, outputs):
    return write_json(
        root / filename,
        dict(
            status="ok",
            stage=stage,
            finished_at="2026-09-10T00:00:00Z",
            params=dict(binding=binding),
            inputs=inputs,
            outputs=outputs,
        ),
    )


@pytest.fixture
def completed(tmp_path, monkeypatch):
    """Two tiny sealed chunks, all three views; no producer code is executed."""
    monkeypatch.setattr(audit, "STOP", 49804)
    monkeypatch.setattr(audit, "CHUNK", 2)
    monkeypatch.setattr(audit, "COUNT", 2)
    root = tmp_path / "d2"
    root.mkdir()
    inputs = {}
    for key in (
        "source_cache_manifest",
        "source_cache_provenance",
        "asset_manifest",
        "asset:top_weights",
        "asset:top_config",
        "asset:side_weights",
        "asset:side_config",
        *("video:" + view for view in audit.VIEWS),
    ):
        path = tmp_path / key.replace(":", "_")
        path.write_bytes(key.encode())
        inputs[key] = info(path)
    protocol = json.loads(
        (audit.REPO / "labs/20260909_three_body/long_input_protocol.json").read_text()
    )
    protocol.update(
        source_frame_range=[49800, 49804],
        chunk_frames=2,
        source_cache_manifest_sha256=inputs["source_cache_manifest"]["sha256"],
        asset_manifest_sha256=inputs["asset_manifest"]["sha256"],
        source_videos_sha256={v: inputs["video:" + v]["sha256"] for v in audit.VIEWS},
        model_assets_sha256={
            k: inputs["asset:" + k]["sha256"]
            for k in audit.frozen.validate_protocol.__globals__["ASSET_KEYS"]
        },
    )
    protocol_path = write_json(tmp_path / "protocol.json", protocol)
    inputs["protocol"] = info(protocol_path)
    source = "labs/20260909_three_body/chunk_masks.py"
    binding = dict(
        protocol_sha256=digest(protocol_path),
        input_sha256={k: v["sha256"] for k, v in inputs.items()},
        code_sha256={source: digest(audit.REPO / source)},
        runtime=dict(synthetic=True),
    )
    backgrounds = {}
    for view in audit.VIEWS:
        path = tmp_path / f"background_{view}.png"
        path.write_bytes(("background:" + view).encode())
        backgrounds[view] = path

    def payload(folder, start, stop, arrays, kind):
        folder.mkdir(parents=True)
        outputs = {}
        snap = folder / "code_snapshot" / source
        snap.parent.mkdir(parents=True)
        snap.write_bytes((audit.REPO / source).read_bytes())
        outputs["code:" + source] = info(snap)
        (folder / "protocol.json").write_bytes(protocol_path.read_bytes())
        outputs["protocol"] = info(folder / "protocol.json")
        raw, masks, bg, models = {}, {}, {}, {}
        for view in audit.VIEWS:
            link = folder / (view + ".mp4")
            os.link(inputs["video:" + view]["path"], link)
            raw[view] = dict(
                path=link.name,
                sha256=digest(link),
                storage_link="hardlink",
                source_offset_frames=start,
                source_total_frames=348600,
                frame_bgr_sha256=["d" * 64] * (stop - start),
            )
            path = folder / (view + ".npz")
            np.savez_compressed(path, **arrays)
            masks[view] = dict(
                path=path.name,
                sha256=digest(path),
                instances_per_frame=np.diff(arrays["frame_offsets"]).tolist(),
            )
            back = folder / backgrounds[view].name
            back.write_bytes(backgrounds[view].read_bytes())
            bg[view] = dict(
                path=back.name, sha256=digest(back), frames=protocol["background_frames"]
            )
            outputs.update(
                {
                    view + "_raw": info(link),
                    view + "_masks": info(path),
                    view + "_background": info(back),
                }
            )
        for model in ("top", "side"):
            cfg = folder / f"{model}_source_config.yaml"
            cfg.write_bytes(Path(inputs[f"asset:{model}_config"]["path"]).read_bytes())
            effective = folder / f"{model}_effective_config.yaml"
            effective.write_bytes(b"fixed-effective-config")
            models[model] = dict(
                weights=inputs[f"asset:{model}_weights"],
                config=inputs[f"asset:{model}_config"],
                effective_config=effective.name,
                effective_config_sha256=digest(effective),
            )
            outputs[model + "_source_config"] = info(cfg)
            outputs[model + "_effective_config"] = info(effective)
        manifest = {
            k: copy.deepcopy(protocol[k])
            for k in ("experiment", "fps", "image_shape_hw", "background_frames", "background_rule")
        }
        manifest.update(
            kind=kind,
            binding=binding,
            runtime=binding["runtime"],
            frames=list(range(start, stop)),
            source_interval_half_open=[start, stop],
            protocol_sha256=binding["protocol_sha256"],
            tracking_performed=False,
            human_truth_created=False,
            video_payload_bytes_copied=0,
            mask_schema="bbox_packbits_instances_v1",
            raw_storage="original_source_video",
            raw=raw,
            masks=masks,
            backgrounds=bg,
            models=models,
        )
        return manifest, outputs

    entries, consumed, arrays_list, manifests = [], [], [], []
    for i in range(2):
        start, stop = 49800 + 2 * i, 49802 + 2 * i
        arrays = encode(start, [0, 0], shape=(1200, 1200))
        arrays_list.append(arrays)
        folder = root / "chunks" / f"c{i}" / "attempt_0001"
        manifest, outputs = payload(folder, start, stop, arrays, "independent_detector_chunk")
        outputs["manifest"] = info(write_json(folder / "manifest.json", manifest))
        seal = make_seal(
            folder,
            "staging.provenance.json",
            "prepare_independent_mask_chunk",
            binding,
            inputs,
            outputs,
        )
        row = dict(
            start=start,
            stop=stop,
            attempt_path=folder.relative_to(root).as_posix(),
            manifest_sha256=digest(folder / "manifest.json"),
            provenance_sha256=digest(seal),
        )
        entries.append(row)
        consumed.append(dict(row, index=i, output_hashes=audit.normalized(outputs)))
        manifests.append((folder, manifest))
    assembled = root / "assemblies" / "attempt_0001"
    manifest, outputs = payload(
        assembled,
        49800,
        49804,
        concatenate_packed_masks(arrays_list),
        "assembled_independent_detector_chunks",
    )
    index = dict(
        schema="medaka_mask_chunk_index_v1",
        source_frame_range=[49800, 49804],
        chunk_frames=2,
        protocol_sha256=binding["protocol_sha256"],
        chunks=entries,
        attempt_paths_relative_to=str(root),
    )
    outputs["chunk_index"] = info(write_json(assembled / "chunks.json", index))
    manifest.update(
        chunks=2, chunk_index="chunks.json", chunk_index_sha256=digest(assembled / "chunks.json")
    )
    outputs["manifest"] = info(write_json(assembled / "manifest.json", manifest))
    assembly_inputs = copy.deepcopy(inputs)
    for i, (folder, chunk) in enumerate(manifests):
        files = [folder / "manifest.json", folder / "staging.provenance.json"]
        files.extend(
            folder / chunk[category][view]["path"]
            for category in ("masks", "backgrounds")
            for view in audit.VIEWS
        )
        assembly_inputs.update(
            {f"chunk:{i}:{p.relative_to(folder).as_posix()}": info(p) for p in files}
        )
    assembly_seal = make_seal(
        assembled,
        "staging.provenance.json",
        "assemble_independent_mask_chunks",
        binding,
        assembly_inputs,
        outputs,
    )
    run = root / "runs" / "run_0002"
    summary = dict(
        status="complete",
        tracking_performed=False,
        completed_native_frames=4,
        video_payload_bytes_copied=0,
        protocol_sha256=binding["protocol_sha256"],
        final_sequence_path=assembled.relative_to(root).as_posix(),
        final_manifest_sha256=digest(assembled / "manifest.json"),
        final_provenance_sha256=digest(assembly_seal),
        chunks=[dict(row, reused=bool(i), rejected_attempts=[]) for i, row in enumerate(entries)],
        rejected_assemblies=[],
    )
    summary_path = write_json(run / "summary.json", summary)
    seal = make_seal(
        run,
        "run.provenance.json",
        "restartable_independent_mask_preparation",
        binding,
        {},
        {"summary": info(summary_path)},
    )
    final_run = tmp_path / "tracking"
    consumed_path = write_json(final_run / "consumed_chunks.json", consumed)
    fixed = dict(
        input_stage_root=str(root),
        detector_binding=binding,
        detector_inputs=audit.normalized(inputs),
        protocol_sha256=binding["protocol_sha256"],
        background_sha256={v: digest(p) for v, p in backgrounds.items()},
    )
    chain = dict(
        chain_version="old_prefix_then_receipted_child_v1",
        tracking_processes=33,
        checkpoint_restores=32,
        final_audit_complete=False,
        binding=dict(inputs=fixed),
        final_run=final_run,
        final_record=dict(
            params=dict(through_frame=49804), outputs=dict(consumed_chunks=info(consumed_path))
        ),
    )
    return dict(seal=seal, chain=chain, consumed=consumed, root=root, assembled=assembled)


def load_fixture(fixture):
    return audit.load_completed_assembly(
        fixture["seal"], digest(fixture["seal"]), fixture["chain"], fixture["consumed"], Evidence()
    )


def test_completed_seals_to_all_three_streams_and_final_guard(completed):
    context = load_fixture(completed)
    rows = []
    result = audit.verify_packed_assembly(context, on_chunk=rows.append)
    assert result["packed_arrays_verified"] is True and result["final_audit_complete"] is False
    assert result["chunk_view_comparisons"] == 6 and result["logical_field_comparisons"] == 78
    assert [(row["view"], row["index"]) for row in rows] == [
        (v, i) for v in audit.VIEWS for i in range(2)
    ]
    assert context["summary"]["packed_arrays_verified"] is False
    context["evidence"].final_guard()
    assert context["evidence"].hash_calls < len(context["evidence"].paths)


@pytest.mark.parametrize(
    "target", ["summary", "chunk_payload", "assembly_payload", "snapshot", "registered_source"]
)
def test_file_tampering_is_rejected_before_array_open(completed, target, monkeypatch):
    root, assembled = completed["root"], completed["assembled"]
    path = {
        "summary": completed["seal"].parent / "summary.json",
        "chunk_payload": root / "chunks/c0/attempt_0001/top_cam.npz",
        "assembly_payload": assembled / "top_cam.npz",
        "snapshot": assembled / "code_snapshot/labs/20260909_three_body/chunk_masks.py",
        "registered_source": Path(
            completed["chain"]["binding"]["inputs"]["detector_inputs"]["asset:top_weights"]["path"]
        ),
    }[target]
    with path.open("ab") as stream:
        stream.write(b"tampered")
    monkeypatch.setattr(
        np, "load", lambda *a, **k: pytest.fail("metadata failure must precede arrays")
    )
    with pytest.raises(ValueError):
        load_fixture(completed)


@pytest.mark.parametrize(
    "failure", ["status", "stage", "incomplete_flag", "wrong_sha", "unfinished_chain", "consumed"]
)
def test_completion_gate_rejects_before_payload_decode(completed, failure):
    seal = completed["seal"]
    record = json.loads(seal.read_text())
    if failure == "status":
        record["status"] = "failed"
    elif failure == "stage":
        record["stage"] = "prepare_independent_mask_chunk"
    elif failure == "incomplete_flag":
        record["params"]["global_preparation_complete"] = False
    elif failure == "unfinished_chain":
        completed["chain"]["final_record"]["params"]["through_frame"] -= 1
    elif failure == "consumed":
        completed["consumed"][0]["attempt_path"] = "other"
    write_json(seal, record)
    with pytest.raises(ValueError):
        audit.load_completed_assembly(
            seal,
            "0" * 64 if failure == "wrong_sha" else digest(seal),
            completed["chain"],
            completed["consumed"],
            Evidence(),
        )


@pytest.mark.parametrize(
    "failure", ["runtime", "model", "background", "tracking", "raw", "snapshot"]
)
def test_resealed_manifest_contract_cannot_drift_from_tracking_binding(completed, failure):
    folder = completed["assembled"]
    manifest = audit.read_json(folder / "manifest.json")
    record = audit.read_json(folder / "staging.provenance.json")
    protocol = audit.read_json(folder / "protocol.json")
    fixed = completed["chain"]["binding"]["inputs"]
    if failure == "runtime":
        manifest["runtime"]["synthetic"] = False
    elif failure == "model":
        manifest["models"]["top"]["weights"]["sha256"] = "0" * 64
    elif failure == "background":
        manifest["backgrounds"]["top_cam"]["frames"][0] += 1
    elif failure == "tracking":
        manifest["tracking_performed"] = 0
    elif failure == "raw":
        manifest["raw"]["top_cam"]["storage_link"] = "copy"
    else:
        record["outputs"]["code:labs/20260909_three_body/chunk_masks.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        audit.verify_manifest(
            folder,
            manifest,
            record,
            fixed,
            protocol,
            49800,
            49804,
            "assembled_independent_detector_chunks",
            Evidence(),
        )
