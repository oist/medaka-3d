"""Synthetic join/provenance cases only; no completed geometry/image join."""

import copy
import json

import join_dino_minute_geometry as m
import numpy as np
import pytest


def endpoint_fixture():
    crops = [
        dict(
            mask_row=i,
            crop_id=f"test{i}",
            mask_pixels=30,
            core_pixels=12,
            mask_sha256="a" * 64,
            core_sha256="b" * 64,
            path=f"crops/{i}.npz",
            sha256="c" * 64,
        )
        for i in range(4)
    ]
    masks = [
        dict(
            mask_row=i,
            crop_id=c["crop_id"],
            mask_pixels=c["mask_pixels"],
            core_pixels=c["core_pixels"],
            core_valid=True,
            mask_sha256=c["mask_sha256"],
            core_sha256=c["core_sha256"],
            crop_path=c["path"],
            crop_sha256=c["sha256"],
        )
        for i, c in enumerate(crops)
    ]
    rows = [
        dict(
            slot=i,
            assigned_mask_row=i,
            assignment_reason="usable",
            observed_mouth_chest=[True, True],
            view_sources_mouth_chest=[i, i],
            direct_mouth_chest=True,
            geometry_eligible=True,
            reasons=[],
            body_coherence=dict(
                requested_points=33, valid_points=33, best_mask_rows=[i], C_single=0.0
            ),
            core_valid=True,
            core_pixels=12,
            crop_id=crops[i]["crop_id"],
            identity_review=True,
            endpoint_eligible=True,
            endpoint_reasons=[],
        )
        for i in range(3)
    ]
    reference = dict(
        slots=rows,
        all_masks=masks,
        mask_count=4,
        mask_available=True,
        assigned_mask_rows=[0, 1, 2],
        slot_geometry_eligible=[True] * 3,
        slot_endpoint_eligible=[True] * 3,
        all3_distinct=True,
        all3_geometry_eligible=True,
        all3_endpoint_eligible=True,
        review_excluded_from_eligibility=False,
        token_validity_evaluated=False,
    )
    return reference, crops


def test_endpoint_matches_frozen_fixed6_logic_and_keeps_review():
    reference, crops = endpoint_fixture()
    before = copy.deepcopy((reference, crops))
    actual = m.endpoint(reference, crops)
    masks = [dict(descriptor_valid=r["core_valid"]) for r in reference["all_masks"]]
    context = dict(
        slot_reasons=[[]] * 3, all3_distinct=True, scorable=True, slot_descriptors=[[1.0] * 6] * 3
    )
    expected = m.fixed.endpoint(reference, context, masks)
    assert {k: actual[k] for k in expected} == expected
    assert actual["review"] == [True] * 3 and actual["ready"]
    assert (reference, crops) == before


@pytest.mark.parametrize(
    "damage",
    [
        "core",
        "tie",
        "body",
        "direct",
        "view_source",
        "review_policy",
        "row",
        "core_sha",
        "ready",
        "reason",
    ],
)
def test_sealed_gate_or_crop_binding_drift_rejected(damage):
    reference, crops = endpoint_fixture()
    row = reference["slots"][0]
    if damage == "core":
        crops[0]["core_pixels"] = 11
    elif damage == "tie":
        row["body_coherence"]["best_mask_rows"] = [0, 3]
    elif damage == "body":
        row["body_coherence"]["valid_points"] = 32
    elif damage == "direct":
        row["observed_mouth_chest"][0] = False
    elif damage == "view_source":
        row["view_sources_mouth_chest"][1] = -1
    elif damage == "review_policy":
        reference["review_excluded_from_eligibility"] = True
    elif damage == "row":
        row["assigned_mask_row"] = 4
    elif damage == "core_sha":
        crops[0]["core_sha256"] = "d" * 64
    elif damage == "ready":
        reference["all3_endpoint_eligible"] = False
    else:
        row["endpoint_reasons"] = ["invented"]
    with pytest.raises(ValueError):
        m.endpoint(reference, crops)


def test_core_failure_leaves_other_slot_entries_and_does_not_use_review():
    reference, crops = endpoint_fixture()
    crops[2]["core_pixels"] = reference["all_masks"][2]["core_pixels"] = 11
    reference["all_masks"][2]["core_valid"] = False
    reference["slots"][2].update(
        core_pixels=11,
        core_valid=False,
        endpoint_eligible=False,
        endpoint_reasons=["assigned_core_invalid"],
    )
    reference["slot_endpoint_eligible"][2] = reference["all3_endpoint_eligible"] = False
    result = m.endpoint(reference, crops)
    assert result["eligible"] == [True, True, False]
    assert not result["ready"] and result["review"] == [True] * 3


def test_all354_fixed_requests_never_bridge_an_unscorable_time():
    rows = m.requests()
    assert len(rows) == 354 and rows[0]["earlier_frame"] == 66483
    assert rows[-1]["later_frame"] == 76277
    assert all(r["later_frame"] - r["earlier_frame"] == 166 for r in rows)
    assert [r["request_index"] for r in rows] == list(range(354))
    assert [r["method"] for r in rows[:6]] == ["original", "tail"] * 3


def partial_case():
    left = dict(
        assigned=[2, 0, -1],
        eligible=[True, True, False],
        reasons=[[], [], ["no_joint_instance"]],
        ready=False,
        review=[True, False, True],
    )
    right = dict(
        assigned=[1, 3, -1],
        eligible=[True, True, False],
        reasons=[[], [], ["no_direct_mouth_pect"]],
        ready=False,
        review=[True, False, True],
    )
    refs = {("original", 1, "top_cam"): left, ("original", 2, "top_cam"): right}
    request = dict(
        request_index=0, method="original", earlier_frame=1, later_frame=2, view="top_cam"
    )
    table = {}
    for condition, values in zip(
        m.CONDITIONS, ([[0.25, 0.5], [0.1, 0.2]], [[0.6, 0.2], [0.3, 0.4]]), strict=True
    ):
        for i in range(2):
            for j in range(2):
                table[1, 2, "top_cam", left["assigned"][i], right["assigned"][j], condition] = dict(
                    pair_index=2 * i + j,
                    condition=condition,
                    status="comparable",
                    comparable=True,
                    median_mutual_cosine_distance=values[i][j],
                    mutual_edge_count=1,
                )
    return request, refs, table


def test_partial_matrix_equations_match_fixed6_and_keep_all18_entries():
    request, refs, table = partial_case()
    before = copy.deepcopy((request, refs, table))
    value = m.conditional_matrix(request, refs, table)
    old_request = dict(
        request,
        distances=[[0.0, 0.0, "NaN"], [0.0, 0.0, "NaN"], ["NaN"] * 3],
        earlier_ready=False,
        later_ready=False,
        scorable=False,
        earlier_slot_reasons=refs["original", 1, "top_cam"]["reasons"],
        later_slot_reasons=refs["original", 2, "top_cam"]["reasons"],
    )
    old_request["distance_entry_reasons"] = [
        [
            []
            if i < 2 and j < 2
            else dict(
                earlier=old_request["earlier_slot_reasons"][i],
                later=old_request["later_slot_reasons"][j],
            )
            for j in range(3)
        ]
        for i in range(3)
    ]
    expected = m.fixed.conditional_matrix(old_request, refs, table)
    for key in ("distance", "mask_pair_index"):
        assert value[key].tobytes() == expected[key].tobytes()
    assert len(value["entries"]) == 18 and not value["all3_gate"]
    assert (request, refs, table) == before


def test_noeligible_tokens_keeps_raw_score_and_all_lost_reasons():
    request, refs, table = partial_case()
    table[1, 2, "top_cam", 2, 1, "masked"].update(
        median_mutual_cosine_distance=np.nan, status="no_eligible_tokens", comparable=False
    )
    arrays, rows, cases, paired = m.join_requests([request], refs, table)
    assert np.isfinite(arrays["distance"]).sum() == 7
    assert np.isnan(arrays["permutation_cost"]).all()
    assert len(rows) == 6 and len(cases[0]["entries"]) == 18 and len(paired) == 3
    assert paired[0]["both_same_and_cross_comparable"] is False
    assert paired[0]["best_target_set_equal"] is None
    assert cases[0]["entries"][1]["image_status"] == "no_eligible_tokens"
    assert cases[0]["entries"][-1]["endpoint_reasons"]["earlier"] == ["no_joint_instance"]


@pytest.mark.parametrize("fault", ["missing", "pair_drift"])
def test_missing_registered_pair_and_background_drift_rejected(fault):
    request, refs, table = partial_case()
    if fault == "missing":
        del table[1, 2, "top_cam", 2, 1, "masked"]
    else:
        table[1, 2, "top_cam", 2, 1, "masked"]["pair_index"] = 100
    with pytest.raises(ValueError):
        m.conditional_matrix(request, refs, table)


def image_fixture():
    pair = dict(
        pair_index=0,
        request_index=0,
        earlier_frame=1,
        later_frame=2,
        view="top_cam",
        earlier_mask_row=0,
        later_mask_row=1,
        earlier_crop_id="a",
        later_crop_id="b",
    )
    rows, stored = [], {}
    for condition in m.CONDITIONS:
        distance = np.full((256, 256), np.nan)
        mutual = np.zeros((256, 256), bool)
        if condition == "raw":
            distance[0, 3] = 0.25
            mutual[0, 3] = True
        present = condition == "raw"
        side = dict(
            eligible_count=1,
            valid_count=1,
            matched_fraction=1.0 if present else 0.0,
            tie_fraction=0.0,
        )
        metadata = dict(
            status="comparable" if present else "no_eligible_tokens",
            comparable=present,
            mutual_distance_distribution=dict(
                count=int(present), median=0.25 if present else {"nonfinite": "NaN"}
            ),
            left=side,
            right=side,
        )
        key = "p0000_" + condition
        rows.append(
            dict(
                pair,
                condition=condition,
                shared_pair_id=key,
                metadata=metadata,
                status=metadata["status"],
                comparable=present,
            )
        )
        stored[key] = (dict(distances=distance, mutual_nn=mutual), metadata)
    return dict(pairs=[pair]), rows, stored


def test_all_image_conditions_metric_exact_one_edge_no_threshold():
    plan, rows, stored = image_fixture()
    calls = []

    def read(key):
        calls.append(key)
        return stored[key]

    lookup, records = m.image_metrics(plan, rows, read)
    assert calls == ["p0000_raw", "p0000_masked"]
    assert len(lookup) == len(records) == 2 and records[0]["mutual_edge_count"] == 1
    assert records[0]["median_mutual_cosine_distance"] == 0.25
    assert np.isnan(records[1]["median_mutual_cosine_distance"])


@pytest.mark.parametrize("fault", ["order", "id", "median", "metadata", "count", "shape"])
def test_image_shared_schema_or_metric_drift_rejected(fault):
    plan, rows, stored = image_fixture()
    if fault == "order":
        rows.reverse()
    elif fault == "id":
        rows[0]["shared_pair_id"] = "wrong"
    elif fault == "median":
        stored["p0000_raw"][0]["distances"][0, 3] = 0.5
    elif fault == "metadata":
        rows[0]["metadata"] = copy.deepcopy(rows[0]["metadata"])
        rows[0]["metadata"]["left"]["valid_count"] = 2
    elif fault == "count":
        rows.pop()
    else:
        stored["p0000_raw"][0]["mutual_nn"] = np.zeros((1, 1), bool)
    with pytest.raises(ValueError):
        m.image_metrics(plan, rows, stored.__getitem__)


def test_singleton_and_exact_ties_never_become_id_success():
    single = m.fixed.row_ranking([0.2, np.nan, np.nan], same_slot=0)
    tied = m.fixed.row_ranking([0.2, 0.2, 0.3], same_slot=0)
    crossed = m.fixed.row_ranking([np.nan, 0.1, np.nan], same_slot=0)
    stats = m.group_stats([single, tied, crossed])
    assert stats["same_only"] == stats["cross_only"] == stats["same_and_cross"] == 1
    assert stats["conditional_same_slot_minimum"] == 1 and stats["exact_zero_margin"] == 1
    assert single["conditional_same_slot_is_minimum"] is None
    assert np.isnan(m.fixed.six_costs(np.zeros((3, 3)), False)).all()


def method_targets(method):
    base = 0 if method == "original" else 3
    return [
        dict(
            later_slot=i,
            earlier_mask_row=base,
            later_mask_row=i,
            mask_pair_index=base * 3 + i,
            metric=dict(shared_pair_id=f"p{base * 3 + i:04d}_raw"),
        )
        for i in range(3)
    ]


def test_method_availability_intersection_never_fills_missing_targets():
    rows = []
    for method, values in (("original", [0.1, 0.2, np.nan]), ("tail", [0.3, np.nan, 0.1])):
        rows.append(
            dict(
                method=method,
                earlier_frame=1,
                later_frame=2,
                view="top_cam",
                condition="raw",
                source_slot=0,
                distances=values,
                targets=method_targets(method),
                **m.fixed.row_ranking(values, same_slot=0),
            )
        )
    result = m.method_comparison(rows)[0]
    assert result["common_finite_numeric_target_slots"] == [0]
    assert result["original_only_targets"] == [1] and result["tail_only_targets"] == [2]
    assert result["own_target_sets_both_same_and_cross"] is True
    assert result["common_targets_both_same_and_cross"] is False
    assert np.isnan(result["common_targets_margin_delta"])
    assert result["original_common_targets"]["reason"] == "same_only_no_cross_comparison"
    assert result["tail_common_targets"]["conditional_same_slot_is_minimum"] is None
    assert result["same_numeric_slot_not_biological_alignment"] is True


def test_common_targets_are_reranked_using_identical_columns():
    rows = []
    for method, values in (("original", [0.2, 0.3, 0.1]), ("tail", [0.4, 0.6, np.nan])):
        rows.append(
            dict(
                method=method,
                earlier_frame=1,
                later_frame=2,
                view="top_cam",
                condition="raw",
                source_slot=0,
                distances=values,
                targets=method_targets(method),
                **m.fixed.row_ranking(values, same_slot=0),
            )
        )
    result = m.method_comparison(rows)[0]
    assert result["common_targets_both_same_and_cross"]
    assert result["original_own_targets_margin"] == -0.1
    assert result["original_common_targets"]["same_slot_rank_min"] == 1
    assert result["original_common_targets"]["best_targets"] == [0]
    assert np.isnan(result["original_common_targets"]["distances"][2])
    assert result["common_targets_margin_delta"] == (0.6 - 0.4) - (0.3 - 0.2)


def test_store_ref_identity_is_independent_of_identical_numeric_arrays():
    plan = dict(
        crops=[dict(crop_id="a"), dict(crop_id="b")],
        pairs=[dict(pair_index=0, earlier_crop_id="a", later_crop_id="b")],
    )
    index = dict(
        crops={c: {} for c in ("a", "b")},
        features={c + "__" + s: dict(crop_id=c) for c in ("a", "b") for s in m.CONDITIONS},
        pairs={"p0000_" + s: dict(left_id="a__" + s, right_id="b__" + s) for s in m.CONDITIONS},
    )
    m.verify_store_references(index, plan)
    index["pairs"]["p0000_raw"]["left_id"] = "b__raw"
    with pytest.raises(ValueError, match="endpoint"):
        m.verify_store_references(index, plan)


def test_identical_anonymous_pairs_keep_exact_distance_bytes_separate_from_numeric_slots():
    values = [0.2, 0.3, 0.4]
    rows = [
        dict(
            method=method,
            earlier_frame=1,
            later_frame=2,
            view="top_cam",
            condition="raw",
            source_slot=0,
            distances=values.copy(),
            targets=method_targets("original"),
            **m.fixed.row_ranking(values, same_slot=0),
        )
        for method in m.METHODS
    ]
    rows[1]["targets"][2] = method_targets("tail")[2]
    result = m.method_comparison(rows)[0]
    assert result["identical_anonymous_pair_target_slots"] == [0, 1]
    assert result["different_anonymous_pair_target_slots"] == [2]
    assert result["identical_pair_subset_same_and_cross"]
    rows[1]["distances"][0] = np.nextafter(0.2, 1)
    with pytest.raises(ValueError, match="distance bytes"):
        m.method_comparison(rows)


def metadata_fixture():
    plan = dict(
        frames=list(m.FRAMES),
        views=list(m.VIEWS),
        crops=[],
        requests=[],
        pairs=[],
        conditions=list(m.CONDITIONS),
        maximum_forward_calls=1186,
        candidate_references={
            k: dict(path=str(p), sha256=d) for k, (p, d) in m.crop.CANDIDATES.items()
        },
        shared_SL_calibration_sha256={
            str(m.WORK / f"synthetic_{i}.npz"): str(i) * 64 for i in range(7)
        },
    )
    common = {
        str(m.images.PREPARATION / "preparation.provenance.json"): m.images.PREPARATION_SHA,
        str(m.crop.PLAN): m.crop.PLAN_SHA,
    }
    preparation = dict(
        inputs={
            **plan["shared_SL_calibration_sha256"],
            **{str(p): d for p, d in m.crop.CANDIDATES.values()},
        },
        outputs={"plan.json": dict(path=str(m.images.PREPARATION / "plan.json"), sha256="a" * 64)},
    )
    geometry = dict(inputs={**common, **preparation["inputs"]})
    # Deliberately NO direct SL/calibration inputs in the independent image stage.
    image = dict(
        inputs={**common, **{r["path"]: r["sha256"] for r in preparation["outputs"].values()}}
    )
    bindings = dict(
        preparation_sha256=m.images.PREPARATION_SHA,
        compatibility_sha256=m.COMPATIBILITY_SHA,
        source_plan_sha256=m.crop.PLAN_SHA,
        codec_source_sha256=m.PINS[m.Path(m.store.__file__).resolve()],
        original_model_runner_sha256=m.images.PINS[m.Path(m.images.old.__file__).resolve()],
        device="cpu",
        dtype="float32",
        torch_cpu_threads=2,
    )
    image_plan = {**plan, "bindings": bindings}
    geom_summary = dict(
        status="ok",
        frames=list(m.FRAMES),
        views=list(m.VIEWS),
        methods=list(m.METHODS),
        horizon_stop=76360,
        geometry_precedes_feature_join=True,
        crop_preparation_sha256=m.images.PREPARATION_SHA,
        comparison_sha256=m.crop.COMPARISON_SHA,
        candidate_sha256={k: d for k, (_, d) in m.crop.CANDIDATES.items()},
        shared_SL_calibration_sha256=plan["shared_SL_calibration_sha256"],
    )
    image_summary = dict(
        status="ok",
        geometry_join_performed=False,
        mask_pairs=1934,
        pair_condition_records=3868,
        shared_store_manifest_sha256=m.MANIFEST_SHA,
    )
    return [geometry, image, preparation, plan, image_plan, geom_summary, image_summary]


def test_metadata_transitive_image_geometry_binding_never_claims_direct_SL_consumption():
    fixture = metadata_fixture()
    result = m.metadata_contract(*fixture)
    assert result["preparation_sha256"] == m.images.PREPARATION_SHA
    assert not set(fixture[3]["shared_SL_calibration_sha256"]) & set(fixture[1]["inputs"])


@pytest.mark.parametrize(
    "damage", ["prep_output", "SL", "candidate_path", "candidate_sha", "horizon"]
)
def test_metadata_changed_origin_or_failed_common_context_rejected(damage):
    fixture = metadata_fixture()
    geometry, image, _preparation, plan, _image_plan, geom_summary, _image_summary = fixture
    if damage == "prep_output":
        image["inputs"][str(m.images.PREPARATION / "plan.json")] = "z" * 64
    elif damage == "SL":
        del geometry["inputs"][next(iter(plan["shared_SL_calibration_sha256"]))]
    elif damage == "candidate_path":
        plan["candidate_references"]["original"]["path"] = str(m.WORK / "wrong.npz")
    elif damage == "candidate_sha":
        geom_summary["candidate_sha256"]["tail"] = "z" * 64
    else:
        geom_summary["horizon_stop"] = 68060
    with pytest.raises(ValueError):
        m.metadata_contract(*fixture)


def test_cross_view_exact_rank_preserves_ties_without_mixing_costs():
    planned = m.requests()
    arrays = {"permutation_cost": np.full((354, 2, 6), np.nan)}
    # First pair original, top and side1. Absolute scales deliberately differ.
    arrays["permutation_cost"][0, 0] = [1, 1, 4, 5, 6, 7]
    arrays["permutation_cost"][2, 0] = [30, 10, 30, 40, 50, 60]
    result = m.cross_view_ranks(planned, arrays)
    assert len(result) == 708
    row = result[0]
    assert row["comparison_available"]
    assert row["first_minima_ranks_in_second"] == [
        dict(permutation_index=0, rank_min=2, rank_max=3),
        dict(permutation_index=1, rank_min=1, rank_max=1),
    ]
    assert sum(r["comparison_available"] for r in result) == 1


def test_named_input_and_success_output_contract(tmp_path):
    payload = tmp_path / "summary.json"
    payload.write_text('{"status":"ok"}')
    digest = m.assets.sha(payload)
    record = dict(
        stage="synthetic",
        status="ok",
        finished_at="now",
        inputs={str(payload): digest},
        outputs={
            "summary.json": dict(
                path=str(payload), size_bytes=payload.stat().st_size, sha256=digest
            )
        },
    )
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps(record))
    evidence = {}
    assert m.sealed(tmp_path, "seal.json", m.assets.sha(seal), "synthetic", evidence) == record
    m.input_member(record, payload, digest)
    assert m.json_output(tmp_path, record, "summary.json", evidence) == dict(status="ok")
    with pytest.raises(ValueError):
        m.input_member(record, payload, "a" * 64)
    (tmp_path / "stray").write_text("unregistered")
    with pytest.raises(ValueError):
        m.sealed(tmp_path, "seal.json", m.assets.sha(seal), "synthetic", {})


def test_failed_startup_keeps_failed_new_seal_without_numeric_work(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WORK", tmp_path)
    monkeypatch.setattr(m, "workspace_code", lambda: {})
    monkeypatch.setattr(m, "PINS", {})
    monkeypatch.setattr(
        m, "load_inputs", lambda _: (_ for _ in ()).throw(ValueError("fixture binding"))
    )
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="fixture binding"):
        m.run(output)
    result = json.loads((output / "join.provenance.json").read_text())
    assert result["status"] == "failed" and result["stage"] == m.STAGE
    assert not (output / "joined_arrays.npz").exists()
