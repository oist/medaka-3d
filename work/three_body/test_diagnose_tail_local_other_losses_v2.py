"""Synthetic evidence-boundary regression only; no real data or fitting."""

import ast
from pathlib import Path

import diagnose_tail_local_other_losses_v2 as d
import pytest


def test_packet_binds_raw_sl_while_fit_binds_calibration_and_packet(monkeypatch):
    shared = {str(Path(f"v{i}.npz").resolve()): f"s{i}" for i in range(3)}
    shared.update({str(Path(f"c{i}.json").resolve()): f"c{i}" for i in range(4)})
    packet, fit = dict(shared), {p: h for p, h in shared.items() if p.endswith(".json")}
    calls = []

    def bind(seal, path, hashes):
        calls.append((seal is packet, path))
        if path not in seal:
            raise ValueError("missing direct input")
        hashes[path] = seal[path]

    monkeypatch.setattr(d.fit, "bind_input", bind)
    hashes = {}
    d.bind_shared_inputs(shared, packet, fit, hashes)
    assert hashes == shared and len(calls) == 11
    assert all(is_packet for is_packet, p in calls if p.endswith(".npz"))
    bad = dict(packet)
    bad[next(p for p in shared if p.endswith(".npz"))] = "changed"
    with pytest.raises(ValueError, match="same registered inputs"):
        d.bind_shared_inputs(shared, bad, fit, {})


def test_numeric_diagnostics_are_unchanged_from_preserved_failed_revision():
    def functions(path):
        return {
            f.name: ast.dump(f, include_attributes=False)
            for f in ast.parse(path.read_text(encoding="utf-8")).body
            if isinstance(f, ast.FunctionDef)
        }

    old = functions(Path(d.__file__).with_name("diagnose_tail_local_other_losses.py"))
    new = functions(Path(d.__file__))
    for name in ("geometry", "case_details"):
        assert old[name] == new[name]
