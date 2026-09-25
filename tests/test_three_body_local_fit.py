"""Local comparisons use original motion history and the actual frozen score."""

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

LAB = Path(__file__).resolve().parents[1] / 'labs/20260909_three_body'
sys.path.insert(0, str(LAB))
import body_local_fit as local  # noqa: E402
from node_geometry import RayObservations, assignment_plans  # noqa: E402
from observations import PixelTargets  # noqa: E402
from tracker import Proposal, State, ThreeBodyTracker, TrackerConfig  # noqa: E402


def scene():
    return np.array([[40, 50, 60], [80, 90, 100], [130, 130, 140]])[:, None] + np.array(
        [[6., 0., 0.], [0., 0., 0.], [-16., 2., 1.]])[None]


def history():
    body = np.array([[True, True, False], [True, False, True],
                     [True, True, False], [True, True, True]])
    poses = scene()[None] + np.array([1, 4, 14, 15])[:, None, None, None] * [1, 0, 0]
    missing = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    return dict(frame_idx=np.arange(100, 104), pose_xyz=poses, body_supported=body, missing=missing)


def test_parent_history_keeps_supported_motion_clips_speed_and_decays_missing_velocity():
    data = history()
    before = {k: v.copy() for k, v in data.items()}
    result = local.replay_chosen_parents(data, scene())
    expected_vx = np.array([[0, 0, 0], [.5, .5, 0], [1.75, .4, 1.5], [2, 2, 1.2]])
    np.testing.assert_allclose(result['parent_velocity'][:, :, 0], expected_vx, atol=1e-15)
    np.testing.assert_array_equal(result['parent_velocity'][:, :, 1:], 0.)
    np.testing.assert_array_equal(result['parent_missing'],
                                  np.vstack(([0, 0, 0], data['missing'][:-1])))
    np.testing.assert_array_equal(result['predicted_pose_xyz'][0], scene())
    for t in range(1, 4):
        np.testing.assert_allclose(result['predicted_pose_xyz'][t],
                                   data['pose_xyz'][t-1] + result['parent_velocity'][t, :, None])
    for k, v in data.items():
        np.testing.assert_array_equal(v, before[k])


@pytest.mark.parametrize('key', ['frame_idx', 'missing', 'body_supported', 'pose_xyz'])
def test_inconsistent_parent_history_rejected(key):
    data = history()
    if key == 'frame_idx':
        data[key][2] += 1
    elif key == 'missing':
        data[key][1, 0] = 4
    elif key == 'body_supported':
        data[key] = data[key].astype(int)
    else:
        data[key][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        local.replay_chosen_parents(data, scene())


def test_unsigned_frame_wrap_is_rejected_before_continuity_comparison():
    data = {k: v[:2] for k, v in history().items()}
    data['frame_idx'] = np.array([2**64-1, 0], dtype=np.uint64)
    with pytest.raises(ValueError, match='int64-compatible'):
        local.replay_chosen_parents(data, scene())


def make_case(rig):
    seed = scene()
    points = {v: rig.camera(v).project(seed + np.array([.12, -.03, .07])) for v in rig.views}
    points[rig.views[2]][2, 2] = np.nan
    rays = RayObservations.from_points(rig, points)
    # Candidate-independent synthetic pixels, including explicit invalid padding.
    pixels = {v: (rig.camera(v).project(seed).reshape(-1, 2)[:, None] +
                  np.array([[0., 0.], [2., -1.]])[None]).reshape(-1, 2) for v in rig.views}
    pixels[rig.views[1]][-1] = np.nan
    target = np.tile([1., 0.], (3, 9))
    targets = PixelTargets(pixels, target, np.full((3, 18), 1/18), target.astype(bool))
    return seed, rays, targets


def original_objective(tracker, proposals, states, rays, targets):
    """Execute source through its nested objective, without copying that formula."""
    tree = ast.parse((LAB / 'tracker.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ThreeBodyTracker')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_fit')
    end = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.FunctionDef)
               and n.name == 'objective')
    fit.body = [*fit.body[:end + 1], ast.Return(ast.Name('objective', ast.Load()))]
    fit.name = 'original_prefix'
    namespace = dict(ThreeBodyTracker._fit.__globals__)
    exec(compile(ast.fix_missing_locations(ast.Module([fit], [])), str(LAB / 'tracker.py'), 'exec'),
         namespace)
    return namespace['original_prefix'](tracker, proposals, states, rays, targets)


def test_scoring_matches_actual_frozen_source_with_missing_rays_and_invalid_pixels(rig):
    seed, rays, targets = make_case(rig)
    fitter = local.LocalBodyFitter(rig, seed, TrackerConfig(device='cpu', opt_steps=2))
    poses = np.stack((seed, seed + np.array([.3, -.1, .04])))
    predicted = seed - [.1, 0, .04]
    gaps = np.array([0, 5, 27])
    plans = np.stack([assignment_plans(p, rays, keep=1)[0][1] for p in poses])
    proposals = [Proposal(p, predicted, plan, 0, 'test')
                 for p, plan in zip(poses, plans, strict=True)]
    state = State(seed, np.zeros((3, 3)), gaps, np.zeros(3, bool), 0.)
    oracle = original_objective(fitter.tracker, proposals, [state], rays, targets)
    expected = oracle(torch.as_tensor(poses))
    actual = fitter._scores(poses, plans, predicted, gaps, rays, targets)
    for a, e in zip(actual, expected[:2], strict=True):
        np.testing.assert_array_equal(a, e.detach().numpy())


def test_complete_cpu_local_fit_preserves_trials_and_batch_invariance_without_tracking(rig):
    seed, rays, targets = make_case(rig)
    cfg = TrackerConfig(device='cpu', opt_steps=2)
    starts = np.stack((seed, seed + np.array([.1, -.03, 0]), seed + np.array([.4, .1, -.1])))
    original = starts.copy()
    first = local.LocalBodyFitter(rig, seed, cfg, batch_size=1)
    second = local.LocalBodyFitter(rig, seed, cfg, batch_size=3)
    one = first.run(starts, seed, np.array([0, 1, 5]), rays, targets)
    many = second.run(starts, seed, np.array([0, 1, 5]), rays, targets)
    for key in one:
        if isinstance(one[key], np.ndarray):
            np.testing.assert_array_equal(one[key], many[key])
    assert np.all(one['scores'] <= one['initial_scores'] + 1e-12)
    assert one['observed'].shape == (3, 3, 3)
    assert one['sources'].shape == (3, 3, 3, 3)
    assert 'identity_review' not in one and 'accepted_xyz' not in one
    assert first.tracker.initialize_count == second.tracker.initialize_count == 0
    np.testing.assert_array_equal(starts, original)


def test_outside_trial_and_noninteger_gaps_rejected_before_fit(rig):
    seed, rays, targets = make_case(rig)
    fitter = local.LocalBodyFitter(rig, seed, TrackerConfig(device='cpu', opt_steps=1))
    outside = seed[None].copy()
    outside[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match='outside'):
        fitter.run(outside, seed, np.zeros(3, int), rays, targets)
    with pytest.raises(ValueError, match='integer'):
        fitter.run(seed[None], seed, np.zeros(3), rays, targets)
    with pytest.raises(ValueError):
        local.LocalBodyFitter(rig, seed, fitter.tracker.config, batch_size=25)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA capture requires a GPU')
def test_new_local_fitter_cpu_gpu_equivalence(rig):
    seed, rays, targets = make_case(rig)
    starts = np.stack((seed, seed + np.array([.25, -.05, .1])))
    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        outputs = []
        for device in ('cpu', 'cuda'):
            fitter = local.LocalBodyFitter(rig, seed, TrackerConfig(device=device, opt_steps=3))
            outputs.append(fitter.run(starts, seed, np.array([0, 2, 11]), rays, targets))
        for key in outputs[0]:
            if isinstance(outputs[0][key], np.ndarray):
                if np.issubdtype(outputs[0][key].dtype, np.floating):
                    np.testing.assert_allclose(outputs[0][key], outputs[1][key],
                                               rtol=1e-12, atol=1e-12)
                else:
                    np.testing.assert_array_equal(outputs[0][key], outputs[1][key])
    finally:
        torch.set_num_threads(old_threads)
