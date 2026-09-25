"""Isolated old/new phase orchestration parity; no actual checkpoints or fits."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import FunctionType, SimpleNamespace

import numpy as np
import pytest

WORK = Path(__file__).resolve().parent
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

import long_phase_audit as original  # noqa: E402
import receipted_long_phase_audit as cached  # noqa: E402


def clone(function, namespace):
    result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


class Evidence:
    def __init__(self, events):
        self.paths = {}
        self.events = events

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        value = expected or "c" * 64
        assert self.paths.get(path, value) == value
        self.paths[path] = value
        self.events.append(("evidence", path))
        return value


class Store:
    def __init__(self, root, events):
        self.evidence_hashes = {str(root / "receipt.json"): "d" * 64}
        self.events = events
        self.guards = 0

    def final_guard(self):
        self.guards += 1
        self.events.append(("receipt_guard",))


def fixture(tmp_path, module, start=66400, stop=76360):
    events = []
    evidence, store = Evidence(events), Store(tmp_path, events)
    parent, output = (tmp_path / name for name in ("parent", "output"))
    chunks = [dict(index=i, start=f, stop=f + original.CHUNK)
              for i, f in enumerate(range(original.FIRST, stop, original.CHUNK))]
    input_binding = {"seed": {"registration_path": str(tmp_path / "seed/registration.json"),
                              "input_root": str(tmp_path / "inputs")},
                     "input_stage_root": str(tmp_path / "stage")}
    protocol = {"source_frame_range": [original.FIRST, original.FULL_STOP],
                "frozen_tracker_sources_sha256": {"frozen.py": "f" * 64},
                "input_protocol_sha256": "a" * 64,
                "package_versions": {"fixture_runtime": 1}, "torch_cpu_threads": 2}
    binding = {"tracking_protocol_sha256": "e" * 64,
               "code_sha256": {str(tmp_path / "code.py"): "b" * 64},
               "package_versions": protocol["package_versions"], "torch_cpu_threads": 2,
               "inputs": input_binding}
    inputs = SimpleNamespace(fixed_bindings=input_binding,
                             full_frame_range=(original.FIRST, original.FULL_STOP),
                             frame_idx=np.arange(original.FIRST, stop), chunk_records=chunks,
                             verified_hashes={str(tmp_path / "original_input.npz"): "a" * 64})
    fixed_context = {"registered": "context"}
    before = {"arrays": {"state_pose": np.zeros((6, 3, 3, 3)),
                          "history_offsets": np.array([0, 99123])},
              "metadata": {"restore_count": 8, "context": fixed_context}}
    counts = {"direct_pectoral_fish_frames": 15, "accepted_pectoral_fish_frames": 12,
              "review_fish_frames": 4, "direct_pectoral_per_slot": [5, 5, 5],
              "accepted_pectoral_per_slot": [3, 4, 5]}
    after = {"metadata": {"restore_count": 9, "context": fixed_context},
             "candidate": {"fixture": "unchanged candidate"},
             "comparisons": {"all_21_fixture": "exact"},
             "summary": {"direct_pectoral_count": 15, "accepted_pectoral_count": 12,
                         "identity_review_count": 4, "per_slot_direct_pectoral": [5, 5, 5],
                         "per_slot_accepted_pectoral": [3, 4, 5]}}
    history = {"active_beam_count": 6, "retained_history_rows": 150000,
               "all_suffix_parent_recurrences_exact": True}

    def load_inputs(*args, **kwargs):
        events.append(("load_inputs", args, dict(kwargs)))
        if module is cached:
            assert kwargs["receipt_store"] is store
        return inputs

    def sealed(root, digests):
        root = Path(root).resolve()
        digests.add(root / "continuation.provenance.json")
        params = {"through_frame": start if root == parent else stop, "bindings": binding,
                  "resume_phase": str(parent)}
        return root, {"params": params}

    def load_phase(root, seal, actual_inputs, digests, actual_protocol):
        assert actual_inputs is inputs and actual_protocol is protocol
        events.append(("load_phase", str(root)))
        digests.add(root / "candidate.npz")
        return before if root == parent else after

    def suffix_history(previous, current):
        assert previous is before and current is after
        events.append(("suffix_history",))
        return history

    namespace = dict(vars(module))
    namespace.update(
        REPO=tmp_path,
        COMMON=SimpleNamespace(read_json=lambda _: protocol),
        sealed=sealed, load_phase=load_phase, suffix_history=suffix_history,
        old=SimpleNamespace(prior=SimpleNamespace(
            first=SimpleNamespace(verified=SimpleNamespace(load_long_inputs=load_inputs)),
            support=SimpleNamespace(direct_counts=lambda _: counts))),
        load_cached_long_inputs=load_inputs,
    )
    for name in ("verified_inputs", "prepare_phase", "_verify_phase", "verify_phase"):
        namespace[name] = clone(getattr(module, name), namespace)
    if module is cached:
        namespace["_bind_receipt_evidence"] = clone(module._bind_receipt_evidence, namespace)
    kwargs = {} if module is original else {"receipt_store": store}
    return SimpleNamespace(namespace=namespace, evidence=evidence, store=store, kwargs=kwargs,
                           events=events, parent=parent, output=output, start=start, stop=stop,
                           inputs=inputs, protocol=protocol, binding=binding, before=before,
                           after=after)


def prepare(f):
    return f.namespace["prepare_phase"](f.parent, f.stop, f.output, f.evidence, **f.kwargs)


def verify(f, plan):
    return f.namespace["verify_phase"](f.output, plan, f.evidence, **f.kwargs)


@pytest.mark.parametrize("start,stop", [(66400, 76360), (345280, 348600)])
def test_plan_context_and_verify_result_exact_except_separate_evidence(tmp_path, start, stop):
    old = fixture(tmp_path, original, start, stop)
    new = fixture(tmp_path, cached, start, stop)
    old_plan, old_context = prepare(old)
    new_plan, new_context = prepare(new)
    assert original.COMMON.normalize(old_plan) == original.COMMON.normalize(new_plan)
    assert set(new_context) == set(old_context) == {"inputs", "parent", "protocol"}
    assert new_context["inputs"] is new.inputs
    assert new_context["parent"] is new.before
    assert new_context["protocol"] is new.protocol
    for name in vars(old.inputs):
        a, b = getattr(old.inputs, name), getattr(new.inputs, name)
        if isinstance(a, np.ndarray):
            assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())
        else:
            assert a == b
    before_hashes = copy.deepcopy(new.inputs.verified_hashes)
    before_chunks = copy.deepcopy(new.inputs.chunk_records)
    assert verify(old, old_plan) == verify(new, new_plan)
    assert new.inputs.verified_hashes == before_hashes
    assert new.inputs.chunk_records == before_chunks
    assert new.inputs.fixed_bindings == old.inputs.fixed_bindings
    assert new.store.guards == 2
    assert {**old.evidence.paths, **new.store.evidence_hashes} == new.evidence.paths
    assert new.events[-2:] == [("receipt_guard",),
                              ("evidence", str(tmp_path / "receipt.json"))]
    assert new_plan["full_sequence_complete"] is (stop == original.FULL_STOP)
    assert not new.output.exists()


@pytest.mark.parametrize("bad", ["binding", "runtime", "threads", "frames", "chunks"])
def test_original_input_contract_rejections_are_preserved(tmp_path, bad):
    for module in (original, cached):
        f = fixture(tmp_path, module)
        if bad == "binding":
            f.inputs.fixed_bindings = {"changed": True}
        elif bad == "runtime":
            f.binding["package_versions"] = {"changed": True}
        elif bad == "threads":
            f.binding["torch_cpu_threads"] = 1
        elif bad == "frames":
            f.inputs.frame_idx = f.inputs.frame_idx[:-1]
        else:
            f.inputs.chunk_records = f.inputs.chunk_records[:-1]
        with pytest.raises((ValueError, AssertionError)):
            prepare(f)
        assert f.store.guards == 0


@pytest.mark.parametrize("bad", ["initializations", "reset", "chunks", "context", "restores",
                                 "rows", "summary", "output"])
def test_same_phase_failure_guards_run_before_receipt_evidence(tmp_path, bad):
    for module in (original, cached):
        f = fixture(tmp_path, module)
        plan, _ = prepare(f)
        guards_before = f.store.guards
        if bad == "initializations":
            plan["expected_initializations"] = 2
        elif bad == "reset":
            plan["expected_external_resets"] = 1
        elif bad == "chunks":
            plan["full_selected_chunk_records"] = []
        elif bad == "context":
            plan["parent_checkpoint_context"] = {"changed": True}
        elif bad == "restores":
            plan["expected_restores"] = 8
        elif bad == "rows":
            plan["parent_history_rows"] += 1
        elif bad == "summary":
            f.after["summary"]["accepted_pectoral_count"] += 1
        else:
            plan["expected_output"] = str(tmp_path / "another_output")
        with pytest.raises((ValueError, AssertionError)):
            verify(f, plan)
        assert f.store.guards == guards_before


@pytest.mark.parametrize("method", ["prepare", "verify"])
def test_changed_receipt_fails_instead_of_returning_success(tmp_path, method):
    f = fixture(tmp_path, cached)
    plan = prepare(f)[0] if method == "verify" else None

    def fail():
        raise ValueError("receipt evidence changed")

    f.store.final_guard = fail
    with pytest.raises(ValueError, match="receipt evidence changed"):
        prepare(f) if method == "prepare" else verify(f, plan)
