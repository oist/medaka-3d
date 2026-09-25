"""Risk-bounded propagation of actual human endpoint correspondences.

Global G0/G1/G2 denote the initial raw slots, conditionally; they are not sex
labels or verified biological identities. This pure export never fits geometry,
adds observed nodes, infers a switch time, or changes a supplied input array.
"""

from __future__ import annotations

import copy
from collections import defaultdict

import numpy as np
from review_constraints import _array_digest, combine_constraints
from tracklet_ledger import build_tracklet_ledger

ORIGINS = {0: 'unresolved', 1: 'initial_automatic', 2: 'human_endpoint',
           3: 'conditional_machine_after_human'}


def _geometry(geometry):
    frames = np.asarray(geometry['frame_idx'])
    if (frames.ndim != 1 or not len(frames) or frames.dtype.kind not in 'iu'
            or np.any(frames < 0) or np.any(frames > np.iinfo(np.int64).max)):
        raise ValueError('nonnegative int64-compatible native frames required')
    frames = frames.astype(np.int64, copy=True)
    if np.any(np.diff(frames) <= 0):
        raise ValueError('native frames must be strictly increasing')
    n = len(frames)
    shapes = dict(pose_xyz=(n, 3, 3, 3), sources=(n, 3, 3, 3),
                  observed=(n, 3, 3), identity_review=(n, 3))
    for key, shape in shapes.items():
        if np.asarray(geometry[key]).shape != shape:
            raise ValueError('invalid geometry shape: ' + key)
    pose, source = np.asarray(geometry['pose_xyz']), np.asarray(geometry['sources'])
    observed = np.asarray(geometry['observed'])
    review = np.asarray(geometry['identity_review'])
    if (not np.isfinite(pose).all() or source.dtype.kind not in 'iu'
            or np.any(source < -1) or np.any(source > np.iinfo(np.int64).max)
            or observed.dtype != bool or review.dtype != bool):
        raise ValueError('invalid pose, source, observation or review values')
    if np.any(observed & ((source >= 0).sum(axis=2) < 2)):
        raise ValueError('observed nodes require two original source views')
    for a, b in ((0, 1), (0, 2), (1, 2)):
        if np.any((source[:, a] >= 0) & (source[:, a] == source[:, b])):
            raise ValueError('source rows must be exclusive per view and anatomical node')
    direct = np.where(observed[:, :, 1, None], pose[:, :, 1], np.nan)
    accepted = np.where(~review[..., None], direct, np.nan)
    for key, expected in (('xyz', direct), ('accepted_xyz', accepted)):
        if key in geometry and not np.array_equal(geometry[key], expected, equal_nan=True):
            raise ValueError('original coordinate acceptance differs: ' + key)
    return frames


def build_review_correction(geometry, bundles, *, tracker_config, candidate_sha256):
    """Compile endpoint claims, conditional intervals and provenance into arrays.

An actual extracted bundle is required for every human link. Empty bundles
produce an explicit zero-review control, bounded by the risk ledger; they do
not convert the proposed review template into evidence. Its acceptance can be
lower than the old tracker because ledger risk boundaries start earlier.
"""
    frames = _geometry(geometry)
    if (not isinstance(candidate_sha256, str) or len(candidate_sha256) != 64
            or any(c not in '0123456789abcdef' for c in candidate_sha256)):
        raise ValueError('candidate SHA256 binding required')
    before = {k: _array_digest(v) for k, v in geometry.items() if isinstance(v, np.ndarray)}
    ledger = build_tracklet_ledger(geometry, tracker_config=tracker_config)
    bundles = sorted(copy.deepcopy(list(bundles)), key=lambda b: b['content_sha256'])
    for bundle in bundles:
        if bundle['origin']['candidate_sha256'] != candidate_sha256:
            raise ValueError('human constraints belong to a different completed candidate')
        for key, digest in bundle['provenance']['candidate_arrays_sha256'].items():
            if key not in geometry or _array_digest(geometry[key]) != digest:
                raise ValueError('constraint array binding differs: ' + key)
    combined = combine_constraints(bundles) if bundles else None
    tracklets = ledger['tracklets']
    tracklet_at = np.full((len(frames), 3), -1, dtype=np.int32)
    for index, item in enumerate(tracklets):
        active = (frames >= item['start_frame']) & (frames < item['stop_frame_exclusive'])
        tracklet_at[active, item['slot']] = index
    frame_index = {int(f): i for i, f in enumerate(frames)}
    claims, link_records, operations, blank_intervals = [], [], [], []
    endpoint_claims = defaultdict(list)

    def at_tracklet(frame, slot):
        t = frame_index.get(int(frame))
        return -1 if t is None else int(tracklet_at[t, slot])

    def blanked(identity, frame):
        return any(r['identity'] == identity and r['start_frame'] <= frame < r['stop_frame']
                   for r in blank_intervals)

    def supports(claim, frame, slot):
        if not claim['valid'] or claim['slot'] != slot:
            return False
        if claim['kind'] == 'human' and frame == claim['anchor_frame']:
            return True  # identity at a confirmed endpoint, even if its risk blocks emission
        return (claim['start_frame'] <= frame < claim['stop_frame']
                and not blanked(claim['identity'], frame))

    def resolve(frame, slot, before_index=None):
        population = claims if before_index is None else claims[:before_index]
        hits = [c['claim_index'] for c in population if supports(c, frame, slot)]
        return sorted({claims[i]['identity'] for i in hits}), hits

    def refresh():
        # Claims are created in increasing right-endpoint time, so dependencies
        # are acyclic. Re-evaluate after any retrospective machine cut.
        for claim in claims:
            if claim['kind'] == 'initial':
                continue
            ids, parents = resolve(claim['left_frame'], claim['left_slot'], claim['claim_index'])
            claim['valid'] = (not claim['hard_conflict'] and ids == [claim['identity']])
            claim['parent_claim_indices'] = [i for i in parents
                                             if claims[i]['identity'] == claim['identity']]
            claim['dependency_unresolved'] = not claim['valid'] and not claim['hard_conflict']

    def trim(identity, boundary, reason, link_ids):
        changed = []
        for claim in claims:
            if claim['identity'] != identity:
                continue
            # An independently supported human endpoint survives a later cut
            # of its subsequent machine propagation. Its lineage is rechecked.
            cutoff = max(boundary, claim['anchor_frame'] + 1
                         if claim['kind'] == 'human' else boundary)
            if claim['stop_frame'] > cutoff:
                claim['stop_frame'] = cutoff
                changed.append(claim['claim_index'])
        operations.append(dict(kind=reason, identity=int(identity), boundary_frame=int(boundary),
                               affected_claim_indices=changed, link_ids=list(link_ids)))

    for identity in range(3):
        index = at_tracklet(frames[0], identity)
        if index < 0:
            continue
        item = tracklets[index]
        if item['unresolved_risk_ids'] or bool(geometry['identity_review'][0, identity]):
            continue
        claims.append(dict(claim_index=len(claims), identity=identity, slot=identity,
                           kind='initial', anchor_frame=int(frames[0]),
                           start_frame=int(frames[0]), stop_frame=item['stop_frame_exclusive'],
                           original_stop_frame=item['stop_frame_exclusive'], tracklet_index=index,
                           valid=True, hard_conflict=False, dependency_unresolved=False,
                           parent_claim_indices=[], link_ids=[], review_sha256=[],
                           crossed_risk_ids=[], endpoint_emission_allowed=True))

    links = [] if combined is None else sorted(
        combined['links'], key=lambda x: (x['stop_frame'], x['start_frame'], x['link_id']))
    documents = {} if combined is None else combined['document_provenance']
    scope_review = {v['clip_scope_id']: v['provenance']['review_sha256']
                    for v in documents.values()}
    by_right = defaultdict(list)
    for link in links:
        by_right[link['stop_frame']].append(link)
    for q, group in sorted(by_right.items()):
        refresh()
        pending = []
        for link in group:
            item = copy.deepcopy(link)
            item.update(global_identity=None, correction_status='local_or_unconfirmed',
                        applied_claim_index=None)
            link_records.append(item)
            if link['status'] not in ('confirmed_endpoint_link', 'conflict'):
                item['correction_status'] = link['status']
                continue
            p, left, right = link['start_frame'], link['left_raw_slot'], link['right_raw_slot']
            if left is None or right is None or not p < q:
                raise ValueError('invalid confirmed endpoint constraints')
            ids, parents = resolve(p, left)
            if link['status'] == 'conflict':
                item['correction_status'] = 'human_constraint_conflict'
                for identity in ids:
                    trim(identity, p + 1, 'conflicting_human_component', [link['link_id']])
                continue
            if len(ids) != 1:
                item['correction_status'] = 'left_global_identity_unresolved'
                continue
            identity = ids[0]
            item['global_identity'] = identity
            pending.append((link, item, identity, parents))

        # Same-time human constraints must be resolved as a set, never by JSON
        # input order. Both one-ID/two-slots and one-slot/two-IDs are conflicts.
        identities, slots = defaultdict(set), defaultdict(set)
        for link, _, identity, _ in pending:
            identities[identity].add(link['right_raw_slot'])
            slots[link['right_raw_slot']].add(identity)
        bad = {i for i, rows in identities.items() if len(rows) > 1}
        bad |= {i for ids in slots.values() if len(ids) > 1 for i in ids}
        for link, item, identity, _ in pending:
            if identity in bad:
                item['correction_status'] = 'human_global_identity_conflict'
                trim(identity, link['start_frame'] + 1, 'human_global_identity_conflict',
                     [link['link_id']])

        good = [p for p in pending if p[2] not in bad]
        # Inspect collisions before cutting any old claim. A new human endpoint
        # outweighs an unconfirmed machine continuation, but never names the
        # displaced identity at another slot.
        displaced = []
        for link, _, identity, _ in good:
            right, p = link['right_raw_slot'], link['start_frame']
            ids, hits = resolve(q, right)
            for other in ids:
                if other == identity:
                    continue
                human_here = any(claims[i]['kind'] == 'human'
                                 and claims[i]['anchor_frame'] == q
                                 and claims[i]['identity'] == other for i in hits)
                if human_here:
                    bad.update((identity, other))
                else:
                    displaced.append((other, p + 1, link['link_id']))
        for other, boundary, link_id in displaced:
            trim(other, boundary, 'machine_identity_displaced_by_human_endpoint', [link_id])
        for link, item, identity, _ in good:
            if identity in bad:
                item['correction_status'] = 'human_global_identity_conflict'
                trim(identity, link['start_frame'] + 1, 'human_global_identity_conflict',
                     [link['link_id']])
                continue
            p, right = link['start_frame'], link['right_raw_slot']
            trim(identity, p + 1, 'human_link_unknown_interior_and_old_continuation',
                 [link['link_id']])
            blank_intervals.append(dict(identity=identity, start_frame=p + 1, stop_frame=q,
                                        link_id=link['link_id'], meaning='switch_time_unannotated'))

        refresh()
        for link, item, identity, _ in good:
            if identity in bad:
                continue
            p, left, right = link['start_frame'], link['left_raw_slot'], link['right_raw_slot']
            ids, parents = resolve(p, left)
            if ids != [identity]:
                item['correction_status'] = 'parent_invalidated_by_conflict_or_retrospective_cut'
                continue
            tracklet = at_tracklet(q, right)
            allowed = tracklet >= 0
            stop = tracklets[tracklet]['stop_frame_exclusive'] if allowed else q
            crossed = [risk['risk_id'] for risk in ledger['risk_events']
                       if (left in risk['affected_slots'] or right in risk['affected_slots'])
                       and ((risk['representation'] == 'unknown_extent_boundary'
                             and p < risk['start_frame'] <= q)
                            or (risk['representation'] == 'available_interval'
                                and p < risk['stop_frame_exclusive'] <= q))]
            claim = dict(claim_index=len(claims), identity=identity, slot=right, kind='human',
                         anchor_frame=q, start_frame=q, stop_frame=stop,
                         original_stop_frame=stop, tracklet_index=tracklet,
                         valid=True, hard_conflict=False, dependency_unresolved=False,
                         left_frame=p, left_slot=left, parent_claim_indices=parents,
                         link_ids=[link['link_id']], component_id=link.get('component_id'),
                         review_sha256=[scope_review[link['clip_scope_id']]],
                         crossed_risk_ids=crossed, endpoint_emission_allowed=allowed)
            claims.append(claim)
            endpoint_claims[link['right_endpoint_id']].append(claim['claim_index'])
            item.update(applied_claim_index=claim['claim_index'],
                        correction_status=('human_endpoint_and_bounded_propagation' if allowed
                                           else 'human_endpoint_identity_only_active_risk'))
    refresh()

    n = len(frames)
    mapping = np.full((n, 3), -1, dtype=np.int32)
    origin = np.zeros((n, 3), dtype=np.uint8)
    claim_index = np.full((n, 3), -1, dtype=np.int32)
    chosen_tracklet = np.full((n, 3), -1, dtype=np.int32)
    conflicts = np.zeros((n, 3), dtype=bool)
    # Three IDs and a small number of human claims: retain every interval and
    # source reference. No latent coordinate supplies a missing identity.
    for t, frame in enumerate(frames):
        choices = [[] for _ in range(3)]
        for claim in claims:
            if (supports(claim, int(frame), claim['slot'])
                    and at_tracklet(frame, claim['slot']) >= 0
                    and (claim['kind'] != 'human' or frame != claim['anchor_frame']
                         or claim['endpoint_emission_allowed'])):
                choices[claim['identity']].append(claim)
        for identity, candidates in enumerate(choices):
            rows = {c['slot'] for c in candidates}
            if len(rows) > 1:
                conflicts[t, identity] = True
                continue
            if not candidates:
                continue
            chosen = max(candidates, key=lambda c: (c['kind'] == 'human'
                         and c['anchor_frame'] == frame, c['anchor_frame'], -c['claim_index']))
            mapping[t, identity] = chosen['slot']
            origin[t, identity] = (1 if chosen['kind'] == 'initial' else
                                   2 if frame == chosen['anchor_frame'] else 3)
            claim_index[t, identity] = chosen['claim_index']
            chosen_tracklet[t, identity] = at_tracklet(frame, chosen['slot'])
        for slot in range(3):
            owners = np.flatnonzero(mapping[t] == slot)
            if len(owners) > 1:
                conflicts[t, owners] = True
                mapping[t, owners] = -1
                origin[t, owners] = 0
                claim_index[t, owners] = -1
                chosen_tracklet[t, owners] = -1
    for operation in operations:
        if operation['kind'] in ('conflicting_human_component', 'human_global_identity_conflict'):
            identity = operation['identity']
            active = (frames >= operation['boundary_frame']) & (mapping[:, identity] < 0)
            conflicts[active, identity] = True
    latent = np.full((n, 3, 3, 3), np.nan, dtype=np.float64)
    sources = np.full((n, 3, 3, 3), -1, dtype=np.int64)
    observed = np.zeros((n, 3, 3), dtype=bool)
    prior_review = np.ones((n, 3), dtype=bool)
    for t, identity in np.argwhere(mapping >= 0):
        slot = mapping[t, identity]
        latent[t, identity] = geometry['pose_xyz'][t, slot]
        sources[t, identity] = geometry['sources'][t, slot]
        observed[t, identity] = geometry['observed'][t, slot]
        prior_review[t, identity] = geometry['identity_review'][t, slot]
    nodes = np.where(observed[..., None], latent, np.nan)
    arrays = dict(frame_idx=frames, corrected_identity_to_raw_slot=mapping,
                  corrected_pose_xyz=latent, corrected_sources=sources,
                  corrected_observed=observed, corrected_node_xyz=nodes,
                  corrected_xyz=nodes[:, :, 1].copy(), identity_origin=origin,
                  claim_index=claim_index, tracklet_index=chosen_tracklet,
                  identity_conflict=conflicts, prior_raw_review_for_identity=prior_review)
    for item in link_records:
        index = item['applied_claim_index']
        if index is not None and not claims[index]['valid']:
            item['correction_status'] = 'dependency_invalidated'
    metadata = dict(schema='risk_bounded_human_correction_v1',
                    candidate_sha256=candidate_sha256,
                    identity_names=['G0', 'G1', 'G2'],
                    naming='initial raw-slot anonymous identities; conditionally propagated',
                    biological_sex_names_assigned=False, identity_origin_codes=ORIGINS,
                    original_risk_ledger=ledger, combined_endpoint_constraints=combined,
                    claims=claims, link_applications=link_records, operations=operations,
                    unannotated_interiors=blank_intervals,
                    counts=dict(native_frames=n, input_review_bundles=len(bundles),
                                distinct_review_documents=len(documents),
                                matched_global_slot_frames=int((mapping >= 0).sum()),
                                accepted_direct_pectoral_frames=int(observed[:, :, 1].sum()),
                                human_endpoint_slot_frames=int((origin == 2).sum()),
                                conditional_after_human_slot_frames=int((origin == 3).sum()),
                                conflict_slot_frames=int(conflicts.sum())),
                    original_arrays_unchanged=True, new_observations_created=False,
                    dense_interiors_human_certified=False, biological_identity_verified=False,
                    limitations=[
                        'The old risk detector can miss swaps; propagated identity is conditional.',
                        'Coordinates reuse original direct support, not anatomical truth.',
                        'Initial anonymous names do not label the female or either male.',
                        'Clicks used here cannot also establish independent correction accuracy.',
                        'Unknown interiors stay empty except additional supported human endpoints.',
                        'A zero-review run is bookkeeping, not an accuracy improvement.',
                    ])
    if any(_array_digest(geometry[k]) != digest for k, digest in before.items()):
        raise AssertionError('correction mutated an original array')
    return dict(arrays=arrays, metadata=metadata)
