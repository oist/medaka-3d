"""Registered CPU-only rescore of the completed, fixed A body candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / 'labs/20260909_three_body'
sys.path.insert(0, str(LAB))
from diagnose_body_pool_79372 import assert_hashes, read_npz, sha, verify_complete  # noqa: E402
from instance_assignment import assign, half_ray_costs  # noqa: E402
from node_geometry import RayObservations, supported_nodes  # noqa: E402

from medaka3d.calib import load_rig  # noqa: E402
from medaka3d.provenance import provenance  # noqa: E402
from medaka3d.types import VIEWS  # noqa: E402

FRAMES = (79316, 79318, 79322, 79355, 79370, 79372, 79381, 79394,
          79396, 79405, 79547, 82719, 82720)


def plan_cost(costs, plan, weights=(1., 1., .35), missing=9.):
    value = np.full((3, 3, 3), missing, dtype=np.float64)
    for fish, view, node in np.argwhere(plan >= 0):
        value[fish, view, node] = costs[view][fish, plan[fish, view, node], node]
    return float((value * np.asarray(weights)).sum() / (9. * sum(weights)))


def conflicts(plan):
    chimeric = shared = 0
    for view in range(3):
        used = [set(plan[fish, view]) - {-1} for fish in range(3)]
        chimeric += sum(len(row) > 1 for row in used)
        shared += sum(bool(used[a] & used[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    return chimeric, shared


def rescore(poses, original_plan, original_parts, original_scores, rays, rig):
    result = {'original_plan': original_plan.copy(), 'original_scores': original_scores.copy()}
    for name in ('nodewise', 'coherent'):
        result[name + '_node_cost'] = np.empty(len(poses))
        result[name + '_plan'] = np.empty_like(original_plan)
        result[name + '_sources'] = np.empty_like(original_plan)
        result[name + '_direct_nodes'] = np.empty((len(poses), 3, 3), dtype=bool)
        result[name + '_optimal_view_counts'] = np.empty((len(poses), 3), dtype=np.int64)
    maximum_replay_difference = 0.
    for index, pose in enumerate(poses):
        costs = half_ray_costs(pose, rays)
        replay = plan_cost(costs, original_plan[index])
        np.testing.assert_allclose(replay, original_parts[index, 0], rtol=1e-12, atol=1e-12)
        maximum_replay_difference = max(maximum_replay_difference,
                                        abs(replay - original_parts[index, 0]))
        for name, coherent in (('nodewise', False), ('coherent', True)):
            answer = assign(costs, coherent=coherent)
            np.testing.assert_allclose(plan_cost(costs, answer.plan), answer.cost,
                                       rtol=1e-12, atol=1e-12)
            source, direct, _ = supported_nodes(pose, answer.plan, rays, rig)
            result[name + '_node_cost'][index] = answer.cost
            result[name + '_plan'][index] = answer.plan
            result[name + '_sources'][index] = source
            result[name + '_direct_nodes'][index] = direct
            result[name + '_optimal_view_counts'][index] = answer.optimal_view_counts
        if result['coherent_node_cost'][index] < result['nodewise_node_cost'][index] - 1e-12:
            raise AssertionError('coherent feasible subset beat relaxed exact minimum')
        if result['nodewise_node_cost'][index] > replay + 1e-12:
            raise AssertionError('fresh relaxed minimum exceeded old feasible plan')
    for name in ('nodewise', 'coherent'):
        result[name + '_scores'] = (original_scores - original_parts[:, 0]
                                    + result[name + '_node_cost'])
    return result, maximum_replay_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path,
                        default=REPO / 'work/three_body/body_volume_probe_13frames_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(REPO) or output.exists():
        raise ValueError('a new workspace output directory is required')
    output.mkdir(parents=True)
    with provenance('three_body_fixed_pose_instance_association', output, name='rescore') as rec:
        seal, original_summary, calibration, hashes = verify_complete(args.input)
        if tuple(original_summary['native_frame_probes']) != FRAMES:
            raise ValueError('registered frame set differs')
        cfg = original_summary['tracker_config']
        if cfg['ray_sigma_mm'] != .7 or cfg['missing_cost'] != 9.:
            raise ValueError('registered observation parameters differ')
        sources = [Path(__file__), REPO / 'work/three_body/diagnose_body_pool_79372.py',
                   LAB / 'instance_assignment.py', LAB / 'INSTANCE_ASSOCIATION_PROTOCOL.md',
                   LAB / 'node_geometry.py', REPO / 'src/medaka3d/calib.py',
                   REPO / 'src/medaka3d/types.py', REPO / 'src/medaka3d/provenance.py']
        for source in sources:
            digest = sha(source)
            previous = args.input / 'code_snapshot' / source.relative_to(REPO)
            if source in (LAB / 'node_geometry.py', REPO / 'src/medaka3d/calib.py',
                          REPO / 'src/medaka3d/types.py') and sha(previous) != digest:
                raise ValueError('geometry/calibration code differs from the frozen A source')
            hashes[str(source)] = digest
            destination = output / 'code_snapshot' / source.relative_to(REPO)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if sha(destination) != digest:
                raise ValueError('source changed during snapshot')
            rec.output('code:' + str(source.relative_to(REPO)), destination, sha256=digest)
        for path, digest in hashes.items():
            rec.input(str(Path(path).relative_to(REPO)), path, sha256=digest)
        rig = load_rig(calibration)
        started = time.perf_counter()
        rows, details, maximum_replay = [], [], 0.
        total_poses = 0
        for frame in FRAMES:
            root = args.input / f'frame_{frame}'
            for role in ('candidates', 'observations'):
                path = (root / f'{role}.npz').resolve()
                if (Path(seal['outputs'][f'{role}/{frame}']['path']).resolve() != path
                        or str(path) not in hashes):
                    raise ValueError('read path is not bound to its registered producer role')
            candidates = read_npz(root / 'candidates.npz')
            observations = read_npz(root / 'observations.npz')
            rays = RayObservations.from_points(rig, {v: observations['sleap_' + v] for v in VIEWS})
            scored, replay_difference = rescore(
                candidates['fit_pose_xyz'], candidates['fit_plans'], candidates['fit_loss_parts'],
                candidates['fit_scores'], rays, rig)
            maximum_replay = max(maximum_replay, replay_difference)
            total_poses += len(candidates['fit_pose_xyz'])
            saved = output / f'frame_{frame}.npz'
            np.savez_compressed(saved, **scored)
            rec.output(f'candidates/{frame}', saved, sha256=sha(saved))
            detail = {'frame': frame, 'poses': len(candidates['fit_pose_xyz']), 'arms': {}}
            for arm in ('refit_original', 'mask', 'random'):
                if arm == 'refit_original':
                    indexes = np.array([0])
                else:
                    mapping = candidates[arm + '_fit_index']
                    indexes = mapping[mapping >= 0]
                detail['arms'][arm] = {}
                for mode in ('original', 'nodewise', 'coherent'):
                    scores = scored[mode + '_scores']
                    index = int(indexes[np.argmin(scores[indexes])])
                    plan = scored[mode + '_plan'][index]
                    chimera, shared = conflicts(plan)
                    node_cost = (candidates['fit_loss_parts'][index, 0] if mode == 'original'
                                 else scored[mode + '_node_cost'][index])
                    direct = (candidates['fit_observed'][index] if mode == 'original'
                              else scored[mode + '_direct_nodes'][index])
                    row = dict(frame=frame, arm=arm, assignment=mode, fit_index=index,
                               score=float(scores[index]), node_cost=float(node_cost),
                               spatial_error=float(candidates['evaluation_mean_view_loss'][index+1]),
                               chimeric_fish_views=chimera, shared_row_fish_pairs=shared,
                               direct_nodes=int(direct.sum()),
                               direct_pectoral=int(direct[:, 1].sum()))
                    parts = candidates['fit_loss_parts'][index]
                    pose = candidates['fit_pose_xyz'][index]
                    predicted = candidates['predicted_pose_xyz']
                    relative = pose - pose[:, 1:2]
                    previous_relative = predicted - predicted[:, 1:2]
                    orientation = np.sum((relative - previous_relative)**2, axis=-1).mean() / 25
                    row.update(weighted_shape=float(cfg['shape_weight'] * parts[1]),
                               weighted_pectoral_motion=float(cfg['motion_weight'] * parts[2]),
                               weighted_orientation=float(cfg['motion_weight'] * .05 * orientation),
                               weighted_repulsion=float(cfg['repulsion_weight'] * parts[3]),
                               weighted_silhouette=float(cfg['mask_weight'] * parts[4]))
                    np.testing.assert_allclose(
                        sum(value for key, value in row.items() if key.startswith('weighted_'))
                        + row['node_cost'], row['score'], rtol=1e-12, atol=1e-12)
                    rows.append(row)
                    detail['arms'][arm][mode] = row
            details.append(detail)
            print(json.dumps({'frame': frame, 'poses': len(candidates['fit_pose_xyz']),
                              'mask_winners': {m: detail['arms']['mask'][m]['fit_index']
                                               for m in ('original', 'nodewise', 'coherent')} }),
                  flush=True)
        if total_poses != 1384:
            raise ValueError('registered A pool count differs')
        summary = dict(frames=details, total_poses=total_poses,
                       elapsed_seconds=time.perf_counter()-started,
                       maximum_original_node_replay_difference=maximum_replay,
                       new_fit=False, old_coordinates_changed=False, old_review_cleared=False,
                       independent_identity_truth=False, device='cpu',
                       limitations=('Opened development frames; fixed post-fit finite pool. '
                                    'No temporal or ID claim.'))
        path = output / 'summary.json'
        path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        rec.output('summary', path, sha256=sha(path))
        path = output / 'selected.csv'
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        rec.output('selected', path, sha256=sha(path))
        assert_hashes({**hashes, **{v['path']: v['sha256'] for v in rec.outputs.values()}})
        rec.param('all_input_source_output_sha_unchanged', True)
        rec.param('candidate_count', total_poses)
        rec.param('no_fit_or_trajectory_mutation', True)


if __name__ == '__main__':
    main()
