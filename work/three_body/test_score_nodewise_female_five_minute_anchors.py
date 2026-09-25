"""Synthetic adapter integration; completed comparison/source trust is substituted.

The frozen upstream comparison loader is already tested by the Nodewise ID suite.
Here real new ID/label headers, archive selection and frozen evaluation are used.
"""

from pathlib import Path

import numpy as np
import pytest
import score_nodewise_female_five_minute_anchors as m

NP_LOAD = np.load


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    modules = (
        m,
        m.identity,
        m.identity.comparison,
        m.common,
        m.prior,
        m.prior.control,
        m.metrics,
        m.identity.frozen,
    )
    for module in modules:
        monkeypatch.setattr(module, "WORK", tmp_path)
        if hasattr(module, "REPO"):
            monkeypatch.setattr(module, "REPO", tmp_path)
        for key, value in dict(FIRST=100, START=104, STOP=164).items():
            if hasattr(module, key):
                monkeypatch.setattr(module, key, value)
    monkeypatch.setattr(m.proof, "REPO", tmp_path)
    monkeypatch.setattr(m.prior, "FRAMES", tuple(range(106, 163)))
    monkeypatch.setattr(m.prior.control, "COMPARISON_ROOT", tmp_path / "old_completed_comparison")

    def forbidden(*args, **kwargs):
        pytest.fail("real data, solver or unrelated workflow entry reached")

    for module, names in (
        (m.identity, ("comparison_headers", "load_candidate", "source_code", "run")),
        (m.identity.frozen, ("solve_once", "qualification")),
        (m.prior, ("load_mouths", "load_beam", "selected_records", "match_records", "run")),
        (m.metrics, ("source_code", "identity_headers", "load_saved_anchors", "run")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(m, "source_code", lambda evidence: {"synthetic_source_trust": "only"})
    real_open = Path.open
    real_bytes, real_text = Path.read_bytes, Path.read_text

    def guard(path):
        assert Path(path).resolve().is_relative_to(tmp_path), "real filesystem reached"

    def safe_open(path, *args, **kwargs):
        guard(path)
        return real_open(path, *args, **kwargs)

    def safe_bytes(path):
        guard(path)
        return real_bytes(path)

    def safe_text(path, *args, **kwargs):
        guard(path)
        return real_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", safe_open)
    monkeypatch.setattr(Path, "read_bytes", safe_bytes)
    monkeypatch.setattr(Path, "read_text", safe_text)
    reads = []

    class Archive:
        def __init__(self, path, *args, **kwargs):
            guard(path)
            self.path = Path(path)
            reads.append((self.path.name, "OPEN"))
            self.inner = NP_LOAD(path, *args, **kwargs)
            self.files = self.inner.files

        def __getitem__(self, name):
            reads.append((self.path.name, name))
            return self.inner[name]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.inner.close()

    monkeypatch.setattr(np, "load", Archive)
    return reads


def bundle(tmp_path, monkeypatch, damage=None):
    def seal(root, name, stage, out, inputs):
        value = dict(stage=stage, status="ok", outputs=out.records.copy(), inputs=inputs)
        path = root / name
        path.write_bytes(m.proof.json_bytes(value))
        return value, m.proof.sha(path)

    # No registered payloads are created: only explicit synthetic path/SHA edges.
    names = [f"20251225_K_1_2_FCMC__{v}.npz" for v in m.prior.generative.VIEWS]
    names += ["calibration_manifest.json"] + [f"cam{i}_bundle_refractive.json" for i in range(3)]
    shared = {str(tmp_path / "registered" / name): "a" * 64 for name in names}
    old_root = m.prior.control.COMPARISON_ROOT
    old_out = m.proof.Output(old_root)
    old_out.write("plan.json", dict(shared_SL_calibration_sha256=shared))
    _, old_sha = seal(
        old_root, "comparison.provenance.json", m.prior.control.comparison.STAGE, old_out, shared
    )
    monkeypatch.setattr(m.prior.control, "COMPARISON_SEAL_SHA", old_sha)

    label_root = tmp_path / m.prior.STAGE
    label_out = m.proof.Output(label_root)
    label_out.write(
        "summary.json",
        dict(
            requested_anchors=57,
            requested_adjacent_intervals=56,
            annotation_CSV_sha256=m.prior.GT_SHA,
            female_endpoint_evaluation_only=True,
            same_horizon_first11_control_exact=True,
        ),
    )
    fixed = dict(
        frames=list(m.prior.FRAMES),
        CSV_sha256=m.prior.GT_SHA,
        source_interval=[m.START, m.STOP],
        labels_created=False,
    )
    if damage == "anchor_order":
        fixed["frames"][0], fixed["frames"][1] = fixed["frames"][1], fixed["frames"][0]
    label_out.write("fixed_female_anchor_set.json", fixed)
    matches = [
        dict(
            frame=f,
            detection_rows={v: 1 for v in m.prior.SIDES},
            pixels={v: [10.0, 20.0] for v in m.prior.SIDES},
            match_errors_px={v: 0.0 for v in m.prior.SIDES},
        )
        for f in m.prior.FRAMES
    ]
    label_out.write(
        "anchor_observations.json",
        dict(matches=matches, diagnostics=[dict(frame=f) for f in m.prior.FRAMES]),
    )
    _, label_sha = seal(
        label_root,
        "evaluation.provenance.json",
        m.prior.STAGE,
        label_out,
        {
            **shared,
            str(old_root / "comparison.provenance.json"): old_sha,
            str(Path(m.prior.__file__).resolve()): m.metrics.PRIOR_SHA,
        },
    )
    monkeypatch.setattr(m.metrics, "PRIOR_SEAL_SHA", label_sha)

    producer_root = tmp_path / m.identity.comparison.NODEWISE_STAGE
    out = m.proof.Output(producer_root)
    n = m.STOP - m.identity.FIRST
    raw = dict(
        frame_idx=np.arange(m.identity.FIRST, m.STOP, dtype=np.int64),
        pose_xyz=np.zeros((n, 3, 3, 3)),
        sources=np.broadcast_to(
            np.arange(3, dtype=np.int32)[None, :, None, None], (n, 3, 3, 3)
        ).copy(),
        observed=np.ones((n, 3, 3), bool),
        identity_review=np.zeros((n, 3), bool),
    )
    raw["observed"][8, 1, 0] = False  # anchor108: keep latent but not direct
    raw["identity_review"][9, 1] = True  # anchor109: distinct original-review stratum
    for key in m.identity.comparison.ALL21 - set(raw):
        raw[key] = np.zeros(1)  # sixteen undecoded fields remain producer declarations
    signatures = m.common.signature(raw)
    if damage == "candidate_signature":
        raw["sources"][10, 1, 1, 0] = 2
    candidate = out.write("candidate.npz", raw)
    new_shared = shared.copy()
    if damage == "top_lineage":
        key = next(p for p in new_shared if p.endswith("__top_cam.npz"))
        new_shared[key] = "b" * 64
    idx = out.write(
        "input_index.json",
        dict(
            prefix_chunks=[{}] * 40,
            original_binding=dict(inputs=dict(seed=dict(SL_and_calibration_sha256=new_shared))),
        ),
    )
    producer, producer_sha = seal(
        producer_root, "tracking.provenance.json", producer_root.name, out, shared
    )
    ref = dict(
        root=str(producer_root),
        tracking_seal_sha256=producer_sha,
        candidate=candidate,
        all21_declared_signatures=signatures,
        decoded_field_signatures={k: signatures[k] for k in m.identity.comparison.FIELDS},
    )
    headers = dict(
        comparison_sha256="c" * 64,
        reference=ref,
        producer_root=producer_root,
        producer_seal=producer,
        comparison_seal=dict(inputs={idx["path"]: idx["sha256"]}),
    )

    def comparison_headers(digest, evidence):
        assert digest == headers["comparison_sha256"]
        return headers  # explicit successful-full5/source trust boundary, no real upstream read

    monkeypatch.setattr(m.identity, "comparison_headers", comparison_headers)
    id_root = tmp_path / m.identity.STAGE
    id_out = m.proof.Output(id_root)
    count = m.STOP - m.START
    state = np.full(count, 3, dtype=np.int64)  # P120, not its inverse P201
    state[0] = 0
    p = m.prior.PERMS[state].copy()
    gap = np.ones((count, 3))
    gap[0] = np.inf
    status = np.full((count, 3), "finite_gap", dtype="U32")
    status[0] = "fixed_initial_constraint"
    gap[5, 1], status[5, 1] = np.inf, "not_retained_unknown"
    mapping = dict(
        frame_idx=np.arange(m.START, m.STOP, dtype=np.int64),
        path_state=state,
        permutation_by_id=p,
        conditional_identity_alternative_gap=gap,
        conditional_identity_alternative_status=status,
    )
    for key in set(m.identity.frozen.MAPPING_FIELDS) - set(mapping):
        mapping[key] = np.zeros(1)  # remaining9 members are not decoded/qualified
    if damage == "mapping_native":
        mapping["frame_idx"][3] += 1
    elif damage == "mapping_initial":
        mapping["permutation_by_id"][0] = [1, 2, 0]
    elif damage == "mapping_members":
        mapping["extra"] = np.zeros(1)
    id_out.write("identity_mapping_and_ambiguity.npz", mapping)
    scope = dict(
        schema=m.identity.STAGE,
        native_interval=[m.START, m.STOP],
        native_frames=count,
        retained_raw_interval=[m.identity.FIRST, m.STOP],
        terminal_horizon=m.STOP,
        comparison_seal_sha256=headers["comparison_sha256"],
        comparison_source_sha256=m.identity.COMPARISON_SHA,
        tracking_seal_sha256=producer_sha,
        tracking_source_sha256=m.identity.comparison.NODEWISE_RUNNER_SHA,
        raw_candidate_reference=ref,
        decoded_field_signatures={k: signatures[k] for k in m.identity.FIELDS},
        decoded_candidate_fields=list(m.identity.FIELDS),
        undecoded18_numerically_verified=False,
        beam_source_sha256=m.identity.frozen.BEAM_SHA,
        beam_width=256,
        node_indices=[1, 0],
        dp_config=m.identity.frozen.CONFIG,
        initial_P=[0, 1, 2],
        initialization_frame=m.START,
        intermediate_ID_resets=0,
        global_optimality_certified=False,
        gaps_conditional_on_retained_graph=True,
        gaps_are_probabilities=False,
        physical_ID_truth=False,
        biological_naming_applied=False,
        review_flags_cleared=False,
        raw_candidate_archive_modified=False,
        source_sha256={str(Path(m.identity.__file__).resolve()): m.IDENTITY_SHA},
    )
    if damage == "identity_horizon":
        scope["terminal_horizon"] -= 1
    id_out.write("plan.json", scope)
    id_out.write(
        "summary.json",
        dict(
            **scope,
            status="ok",
            solve_calls=1,
            decoded_fields_unchanged=True,
            support_and_geometry_reestimated=False,
            exact_DP_rerun=False,
            numerical_tie_tolerance=1e-12,
        ),
    )
    id_out.write("beam_scalars.json", dict(numerical_tie_tolerance=1e-12))
    _, id_sha = seal(
        id_root,
        "identity.provenance.json",
        "wrong_stage" if damage == "identity_stage" else m.identity.STAGE,
        id_out,
        {
            str(Path(m.identity.__file__).resolve()): m.IDENTITY_SHA,
            str(Path(m.identity.frozen.__file__).resolve()): m.identity.FROZEN_SHA,
            str(Path(m.identity.frozen.beam.__file__).resolve()): m.identity.frozen.BEAM_SHA,
            str(tmp_path / m.identity.comparison.STAGE / "comparison.provenance.json"): headers[
                "comparison_sha256"
            ],
            str(producer_root / "tracking.provenance.json"): producer_sha,
            candidate["path"]: candidate["sha256"],
        },
    )
    return headers["comparison_sha256"], id_sha


def test_adapter_connection(tmp_path, monkeypatch, isolate):
    comp, ident = bundle(tmp_path, monkeypatch)
    calls = []
    evaluate = m.metrics.evaluate

    def counted(*args):
        calls.append("evaluate")
        return evaluate(*args)

    monkeypatch.setattr(m.metrics, "evaluate", counted)
    assert m.run(comp, ident, tmp_path / "result")["status"] == "ok"
    assert calls == ["evaluate"]
    assert isolate == [("candidate.npz", k) for k in ("OPEN", *m.prior.FIELDS)] + [
        ("identity_mapping_and_ambiguity.npz", k) for k in ("OPEN", *m.MAPPING_FIELDS)
    ]
    result = m.proof.read_json(tmp_path / "result/summary.json")
    raw, ids = (result["results"][k] for k in ("raw_slots", "automatic_IDs"))
    assert raw["raw"]["anchors"][0]["identity"] == 1
    assert ids["raw"]["anchors"][0]["identity"] == 0
    assert ids["endpoint_details"][0]["raw_slot_by_ID"] == [1, 2, 0]
    assert raw["raw"]["anchors"][2]["identity"] is None
    assert raw["raw"]["anchors"][3]["identity"] == 1
    assert raw["with_original_review"]["anchors"][3]["identity"] is None
    assert len(raw["raw"]["anchors"]) == 57 and len(ids["raw"]["anchors"]) == 57
    assert (tmp_path / "result/scores/automatic_IDs_with_original_review.json").exists()
    assert result["identity_solves"] == 0 and not result["other_methods_reevaluated"]


@pytest.mark.parametrize(
    "damage",
    [
        "identity_stage",
        "identity_horizon",
        "top_lineage",
        "anchor_order",
        "candidate_signature",
        "mapping_native",
        "mapping_initial",
        "mapping_members",
    ],
)
def test_rejection(tmp_path, monkeypatch, isolate, damage):
    comp, ident = bundle(tmp_path, monkeypatch, damage)
    monkeypatch.setattr(
        m.metrics, "evaluate", lambda *args: pytest.fail("invalid input was scored")
    )
    with pytest.raises(ValueError):
        m.run(comp, ident, tmp_path / "failed")
    seal = m.proof.read_json(tmp_path / "failed/evaluation.provenance.json")
    assert seal["status"] == "failed" and not (tmp_path / "failed/summary.json").exists()
    if damage in {"identity_stage", "identity_horizon", "top_lineage", "anchor_order"}:
        assert isolate == []
