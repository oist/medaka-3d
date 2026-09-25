"""Small file-backed equivalence tests; no real source videos, detector or GPU."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parent
REPO = WORK.parents[1]
for directory in (WORK, REPO / "labs/20260909_three_body"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cached_long_tracking_inputs as cached  # noqa: E402
import long_tracking_inputs as original  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def validate_fake_packed(packed):
    # The receipt implementation owns the real 13-array validator tests.
    if set(packed) != {"valid"} or packed["valid"].tolist() != [1]:
        raise ValueError("invalid fake packed content")


class FakeReceipts:
    def __init__(self):
        self.uses = []
        self.guards = 0

    def validate(self, path, *, scope):
        assert digest(path) == scope["expected_npz_sha256"]
        assert digest(scope["manifest_path"]) == scope["manifest_sha256"]
        assert digest(scope["seal_path"]) == scope["seal_sha256"]
        with np.load(path, allow_pickle=False) as arrays:
            validate_fake_packed({name: arrays[name] for name in arrays.files})
        record = {"path": str(path), "scope": copy.deepcopy(scope)}
        self.uses.append(record)
        return record

    def final_guard(self):
        self.guards += 1
        for record in self.uses:
            assert digest(record["path"]) == record["scope"]["expected_npz_sha256"]


def clone(function, namespace):
    """Isolate test globals; never monkeypatch the live frozen module."""
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


@pytest.fixture
def fixture(tmp_path):
    stage, seed, inputs = (tmp_path / name for name in ("stage", "seed", "inputs"))
    for path in (stage, seed, inputs):
        path.mkdir()
    seed_file = seed / "fixture.txt"
    seed_file.write_bytes(b"immutable seed fixture")
    code = inputs / "fixture_code.py"
    code.write_bytes(b"# file checksum fixture\n")
    sources = {}
    for view in original.VIEWS:
        path = inputs / f"{view}.mp4"
        path.write_bytes(view.encode())
        sources[view] = path
    protocol_path = inputs / "protocol.json"
    protocol = {"source_frame_range": [49800, 49806], "chunk_frames": 2}
    write_json(protocol_path, protocol)
    context = SimpleNamespace(
        protocol=protocol,
        runtime=SimpleNamespace(metadata={"fixture_runtime": 1}),
        binding={"protocol_sha256": digest(protocol_path), "fixture": True},
        input_records={view: {"path": str(path), "sha256": digest(path)}
                       for view, path in sources.items()},
        code_records={"fixture_code": {"path": str(code), "sha256": digest(code)}},
        sources=sources,
    )
    attempts = []
    for start in range(49800, 49806, 2):
        path = stage / "chunks" / f"f{start:06d}-{start + 2:06d}" / "attempt_0001"
        path.mkdir(parents=True)
        manifest = {
            "experiment": original.EXPERIMENT, "fps": 166,
            "image_shape_hw": [1200, 1200],
            "protocol_sha256": context.binding["protocol_sha256"],
            "tracking_performed": False, "human_truth_created": False,
            "raw": {}, "masks": {}, "backgrounds": {},
        }
        outputs = {}
        for view in original.VIEWS:
            raw, mask, background = (path / f"{view}{ext}" for ext in (".mp4", ".npz", ".png"))
            os.link(sources[view], raw)
            np.savez(mask, valid=np.array([1], np.int64))
            background.write_bytes((view + " background").encode())
            manifest["raw"][view] = {"frame_bgr_sha256": ["0" * 64] * 2}
            manifest["masks"][view] = {"path": mask.name}
            manifest["backgrounds"][view] = {"sha256": digest(background)}
            for role, target in ((view + "_raw", raw), (view + "_masks", mask),
                                 (view + "_background", background)):
                outputs[role] = {"path": str(target), "sha256": digest(target)}
        write_json(path / "manifest.json", manifest)
        outputs["manifest"] = {"path": str(path / "manifest.json"),
                               "sha256": digest(path / "manifest.json")}
        seal = {"status": "ok", "stage": "prepare_independent_mask_chunk",
                "inputs": context.input_records, "outputs": outputs}
        write_json(path / "staging.provenance.json", seal)
        attempts.append(path)
    pose = np.arange(27, dtype=np.float64).reshape(3, 3, 3)
    rig = object()
    seed_binding = {"fixture_seed": digest(seed_file)}

    def engine(module):
        visits = []

        def validate_attempt(ctx, path, start, stop):
            visits.append((str(path), start, stop))
            assert ctx is context
            return original.read_json(path / "manifest.json")

        namespace = dict(vars(module))
        namespace.update(
            FULL_RANGE=(49800, 49806), PROTOCOL_PATH=protocol_path,
            PROTOCOL_SHA256=digest(protocol_path),
            STAGER_CODE={"fixture_code": digest(code)}, RECORDED_RUNTIME=context.runtime.metadata,
            _make_context=lambda: context,
            _verify_seed=lambda *_: (pose.copy(), rig, copy.deepcopy(seed_binding),
                                    {str(seed_file): digest(seed_file)}),
            validate_packed_masks=validate_fake_packed,
            staging=SimpleNamespace(
                validate_protocol=lambda value: value,
                chunk_ranges=lambda _: [(49800, 49802), (49802, 49804), (49804, 49806)],
                exact_contract=original.staging.exact_contract,
                validate_attempt=validate_attempt,
            ),
        )
        namespace["_select_chunk"] = clone(module._select_chunk, namespace)
        function = module.load_long_inputs if module is original else module.load_cached_long_inputs
        return clone(function, namespace), namespace, visits

    return SimpleNamespace(stage=stage, seed=seed, inputs=inputs, attempts=attempts,
                           context=context, engine=engine, seed_file=seed_file)


def assert_same(left, right):
    assert type(left) is type(right) is original.LongInputs
    for name in left.__dataclass_fields__:
        a, b = getattr(left, name), getattr(right, name)
        if isinstance(a, np.ndarray):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes()), name
        elif name == "rig":
            assert a is b
        else:
            assert a == b, name


@pytest.mark.parametrize("stop", [49802, 49804, 49806])
def test_exact_all_longinputs_fields_and_same_attempt_validation(fixture, stop):
    old, _, old_visits = fixture.engine(original)
    new, _, new_visits = fixture.engine(cached)
    store = FakeReceipts()
    args = (fixture.seed, fixture.stage, fixture.inputs)
    expected = old(*args, through_frame=stop)
    actual = new(*args, through_frame=stop, receipt_store=store)
    assert_same(expected, actual)
    assert old_visits == new_visits
    assert len(store.uses) == 3 * ((stop - 49800) // 2)
    assert store.guards == 1
    for index, item in enumerate(store.uses):
        start = 49800 + 2 * (index // 3)
        scope = item["scope"]
        assert scope["view"] == original.VIEWS[index % 3]
        assert scope["frame_range"] == [start, start + 2]
        assert scope["image_shape_hw"] == [1200, 1200]
        assert scope["detector_binding"] == fixture.context.binding
        assert set(scope) == {"view", "frame_range", "image_shape_hw", "manifest_path",
                              "manifest_sha256", "seal_path", "seal_sha256",
                              "detector_binding", "expected_npz_sha256"}


@pytest.mark.parametrize("stop", [True, np.bool_(True), 49802.0, 49801, 49808])
def test_bad_endpoint_matches_original_rejection(fixture, stop):
    for module in (original, cached):
        function, _, _ = fixture.engine(module)
        kwargs = {} if module is original else {"receipt_store": FakeReceipts()}
        with pytest.raises(ValueError):
            function(fixture.seed, fixture.stage, fixture.inputs, through_frame=stop, **kwargs)


@pytest.mark.parametrize("change", ["failed", "role", "fps", "pixel_hash", "bad_npz"])
def test_invalid_first_attempt_falls_back_to_same_lowest_valid(fixture, change):
    import shutil

    first = fixture.attempts[0]
    second = first.with_name("attempt_0002")
    shutil.copytree(first, second, copy_function=os.link)
    # Rewrite metadata through replace so the new attempt's hardlinks stay intact.
    seal_path = first / "staging.provenance.json"
    manifest_path = first / "manifest.json"
    if change in ("failed", "role"):
        value = original.read_json(seal_path)
        if change == "failed":
            value["status"] = "failed"
        else:
            value["inputs"] = {}
        replacement = first / "changed.json"
        write_json(replacement, value)
        os.replace(replacement, seal_path)
    elif change in ("fps", "pixel_hash"):
        value = original.read_json(manifest_path)
        if change == "fps":
            value["fps"] = 20
        else:
            value["raw"][original.VIEWS[0]]["frame_bgr_sha256"] = ["not a digest"]
        replacement = first / "changed.json"
        write_json(replacement, value)
        os.replace(replacement, manifest_path)
    else:
        target = first / f"{original.VIEWS[0]}.npz"
        replacement = first / "changed.npz"
        np.savez(replacement, valid=np.array([0]))
        os.replace(replacement, target)
        value = original.read_json(seal_path)
        value["outputs"][original.VIEWS[0] + "_masks"]["sha256"] = digest(target)
        replacement = first / "changed.json"
        write_json(replacement, value)
        os.replace(replacement, seal_path)
    selected = []
    for module in (original, cached):
        _, namespace, _ = fixture.engine(module)
        kwargs = {} if module is original else {"receipt_store": FakeReceipts()}
        selected.append(namespace["_select_chunk"](
            fixture.context, fixture.stage, 49800, 49802, **kwargs)[0])
    assert selected == [second.resolve(), second.resolve()]


def test_final_original_input_guard_remains(fixture):
    new, namespace, _ = fixture.engine(cached)
    selector = namespace["_select_chunk"]

    def mutate_after_selection(*args):
        result = selector(*args)
        fixture.seed_file.write_bytes(b"changed after seed verification")
        return result

    namespace["_select_chunk"] = mutate_after_selection
    store = FakeReceipts()
    with pytest.raises(ValueError, match="long input changed during validation"):
        new(fixture.seed, fixture.stage, fixture.inputs, through_frame=49802, receipt_store=store)
    assert store.guards == 0


def test_receipt_final_guard_failure_cannot_return_context(fixture):
    class ChangedReceipts(FakeReceipts):
        def final_guard(self):
            raise ValueError("changed receipt")

    new, _, _ = fixture.engine(cached)
    with pytest.raises(ValueError, match="changed receipt"):
        new(fixture.seed, fixture.stage, fixture.inputs, through_frame=49802,
            receipt_store=ChangedReceipts())
