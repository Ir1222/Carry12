"""CPU-only audit using the repository's actual MotionLib; no simulator required.

Run from any directory: python analysis/carrywith_amp_audit/audit_carrywith.py
Writes schema, per-frame measurements, distribution summaries, and a figure.
Only a minimal import shim is installed to avoid legged_gym's Isaac Gym imports.
"""
from pathlib import Path
import csv
import importlib.util
import json
import sys
import types

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
torch.set_num_threads(2)
torch.manual_seed(19)
np.random.seed(19)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


utils = load_module('audit_torch_utils', ROOT / 'legged_gym/legged_gym/utils/torch_utils.py')
shim = types.ModuleType('legged_gym.utils')
shim.torch_utils = utils
sys.modules['legged_gym'] = types.ModuleType('legged_gym')
sys.modules['legged_gym.utils'] = shim
motion_module = load_module('audit_motionlib', ROOT / 'legged_gym/legged_gym/envs/motionlib/motionlib_carrybox.py')
config = ROOT / 'legged_gym/resources/config/carrybox_no_relocation.yaml'
mapping = ROOT / 'legged_gym/resources/config/joint_id.txt'
names = [line.split()[1] for line in mapping.read_text().splitlines()]
lib = motion_module.MotionLib(str(config), str(mapping), names, 60, 'cpu', 10, [0.95, 1.05], 0.7)


def stats(value):
    v = torch.as_tensor(value).float().flatten()
    return dict(mean=v.mean().item(), std=v.std(unbiased=False).item(),
                **{k: x for k, x in zip(['min', 'p05', 'p50', 'p95', 'max'],
                                       torch.quantile(v, torch.tensor([0., .05, .5, .95, 1.])).tolist())})


def rmse(a, b):
    return ((a - b).square().mean().sqrt()).item()


report = {'fps_assumption_from_config': 60, 'policy_hz': 50,
          'amp_step_dimensions': 60, 'amp_window_dimensions': 600,
          'amp_velocity_slices_zero_based': {'linear': [48, 51], 'angular': [51, 54]},
          'files': {}, 'skill_weights': {}, 'sampling_seed': 19}
weights = yaml.safe_load(config.read_text())['motions']
total_weight = sum(e['weight'] for entries in weights.values() for e in entries)
for skill, entries in weights.items():
    report['skill_weights'][skill] = sum(e['weight'] for e in entries) / total_weight

rows = []
curves = []
for i, entry in enumerate(weights['carryWith']):
    path = (config.parent / entry['file']).resolve()
    data = torch.load(path, map_location='cpu', weights_only=True)
    start, end = int(lib.motion_start_ids['carryWith'][i]), int(lib.motion_end_ids['carryWith'][i])
    q = data['base_quat']
    n = len(q)
    world_v = lib.motion_global_lin_vel[start:end]
    body_v = lib.motion_base_lin_vel[start:end]
    yaw_v = utils.quat_rotate_inverse(utils.calc_heading_quat(q), world_v)
    euler_rate = lib.motion_global_ang_vel[start:end]
    body_w = lib.motion_base_ang_vel[start:end]
    q_unit = torch.nn.functional.normalize(q, dim=-1)
    dq = utils.quat_mul(q_unit[1:], utils.quat_conjugate(q_unit[:-1]))
    dq = torch.where(dq[:, 3:4] < 0, -dq, dq)
    vnorm = dq[:, :3].norm(dim=-1, keepdim=True)
    actual_world_w = dq[:, :3] / vnorm.clamp_min(1e-9) * (2 * torch.atan2(vnorm, dq[:, 3:4])) * 60
    actual_world_w = torch.cat([actual_world_w, actual_world_w[-1:]])
    actual_body_w = utils.quat_rotate_inverse(q, actual_world_w)
    raw_w = data['base_angular_velocity']
    raw_v = data['base_linear_velocity']
    raw_rpy = motion_module.euler_from_quaternion(q)
    direct_euler_diff = torch.diff(raw_rpy, dim=0) * 60
    forward_v = torch.diff(data['base_position'], dim=0) * 60
    backward_v = torch.cat([forward_v[:1], forward_v])
    joint_diff = torch.diff(data['joint_position'], dim=0) * 60
    item = {
        'frames': n, 'elapsed_seconds': (n - 1) / 60,
        'schema': {k: {'shape': list(v.shape), 'dtype': str(v.dtype),
                       'finite': bool(torch.isfinite(v).all())} for k, v in data.items()},
        'amp_body_vx': stats(body_v[:, 0]), 'amp_body_vy': stats(body_v[:, 1]),
        'amp_body_vz': stats(body_v[:, 2]), 'amp_body_wz': stats(body_w[:, 2]),
        'tracking_yaw_frame_vx': stats(yaw_v[:, 0]),
        'tracking_yaw_frame_vy': stats(yaw_v[:, 1]),
        'horizontal_speed': stats(world_v[:, :2].norm(dim=-1)),
        'unwrapped_yaw_rate': stats(euler_rate[:, 2]),
        'quaternion_world_wz': stats(actual_world_w[:, 2]),
        'raw_linear_velocity': [stats(raw_v[:, j]) for j in range(3)],
        'raw_angular_velocity': [stats(raw_w[:, j]) for j in range(3)],
        'joint_velocity': stats(data['joint_velocity']),
        'quaternion_norm': stats(q.norm(dim=-1)),
        'raw_v_vs_forward_difference_rmse': rmse(raw_v[:-1], forward_v),
        'raw_v_vs_backward_difference_rmse': rmse(raw_v[1:], backward_v[1:]),
        'raw_w_vs_unwrapped_forward_euler_rate_rmse': rmse(raw_w[:-1], euler_rate[:-1]),
        'raw_w_vs_wrapped_forward_euler_difference_rmse': rmse(raw_w[:-1], direct_euler_diff),
        'joint_v_vs_forward_difference_rmse': rmse(data['joint_velocity'][:-1], joint_diff),
        'amp_body_w_vs_quaternion_body_w_rmse': rmse(body_w[:-1], actual_body_w[:-1]),
        'amp_body_wz_vs_quaternion_body_wz_rmse': rmse(body_w[:-1, 2], actual_body_w[:-1, 2]),
        'roll': stats(raw_rpy[:, 0]), 'pitch': stats(raw_rpy[:, 1]),
        'raw_w_outliers_abs_gt_20': torch.nonzero(raw_w.abs() > 20).tolist(),
        'yaw_frame_vx_fraction_negative': float((yaw_v[:, 0] < 0).float().mean()),
        'amp_wz_fraction_abs_above_0_4': float((body_w[:, 2].abs() > .4).float().mean()),
        'raw_w_outlier_details': [dict(frame=t, axis=j, raw=float(raw_w[t, j]),
                                      loader_euler_rate=float(euler_rate[t, j]),
                                      quaternion_world_rate=float(actual_world_w[t, j]))
                                  for t, j in torch.nonzero(raw_w.abs() > 20).tolist()],
        'yaw_frame_vx_fraction_above_0_8': float((yaw_v[:, 0] > .8).float().mean()),
        'yaw_frame_vx_fraction_above_1_0': float((yaw_v[:, 0] > 1.).float().mean()),
        'yaw_frame_vx_fraction_below_0_2': float((yaw_v[:, 0] < .2).float().mean()),
        'yaw_rate_fraction_abs_below_0_1': float((euler_rate[:, 2].abs() < .1).float().mean()),
    }
    report['files'][path.name] = item
    curves.append((path.stem, np.arange(n) / 60, yaw_v[:, 0].numpy(), body_v[:, 0].numpy(), euler_rate[:, 2].numpy(), actual_world_w[:, 2].numpy()))
    for t in range(n):
        rows.append(dict(file=path.name, frame=t, seconds=t / 60,
                         amp_body_vx=float(body_v[t, 0]), amp_body_vy=float(body_v[t, 1]),
                         yaw_frame_vx=float(yaw_v[t, 0]), yaw_frame_vy=float(yaw_v[t, 1]),
                         amp_body_wz=float(body_w[t, 2]), unwrapped_yaw_rate=float(euler_rate[t, 2]),
                         quaternion_world_wz=float(actual_world_w[t, 2]), raw_wz=float(raw_w[t, 2])))

# Exercise the actual expert sampler with only carryWith clips enabled, retaining 5:2:5 weights.
saved_weights = lib.motion_weights_tot.clone()
lib.motion_weights_tot.zero_()
offset = sum(lib.num_motion[s] for s in lib.skills[:lib.skills.index('carryWith')])
lib.motion_weights_tot[offset:offset + 3] = saved_weights[offset:offset + 3]
sample = lib.get_expert_obs(20000).reshape(-1, 10, 60)
assert torch.isfinite(sample).all()
report['carry_only_actual_sampler'] = {
    'windows': len(sample), 'amp_body_vx': stats(sample[:, :, 48]),
    'amp_body_vy': stats(sample[:, :, 49]), 'amp_body_wz': stats(sample[:, :, 53]),
    'vx_fraction_above_0_8': float((sample[:, :, 48] > .8).float().mean()),
    'vx_fraction_above_1_0': float((sample[:, :, 48] > 1.).float().mean()),
    'vx_fraction_below_0_2': float((sample[:, :, 48] < .2).float().mean()),
    'wz_fraction_abs_below_0_1': float((sample[:, :, 53].abs() < .1).float().mean()),
    'wz_fraction_abs_above_0_4': float((sample[:, :, 53].abs() > .4).float().mean()),
    'vx_fraction_negative': float((sample[:, :, 48] < 0).float().mean()),
    'window_mean_vx': stats(sample[:, :, 48].mean(dim=1)),
    'window_mean_wz': stats(sample[:, :, 53].mean(dim=1)),
}
lib.motion_weights_tot.copy_(saved_weights)
full_sample = lib.get_expert_obs(1000)
assert full_sample.shape == (1000, 600) and torch.isfinite(full_sample).all()
report['full_sampler_smoke_check'] = {'shape': list(full_sample.shape), 'finite': True}

(OUT / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
with (OUT / 'per_frame.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey='row')
for i, (name, time, vx, body_vx, yaw_rate, world_wz) in enumerate(curves):
    axes[0, i].axhspan(.1, 1.2, color='green', alpha=.09, label='Command range')
    axes[0, i].plot(time, vx, label='Yaw-frame vx', linewidth=1.6)
    axes[0, i].plot(time, body_vx, '--', label='AMP body vx', linewidth=1)
    axes[0, i].set_title(name)
    axes[1, i].axhspan(-.4, .4, color='green', alpha=.09, label='Command envelope')
    axes[1, i].plot(time, yaw_rate, label='Loader Euler yaw derivative', linewidth=1.6)
    axes[1, i].plot(time, world_wz, '--', label='Quaternion world omega-z', linewidth=1)
    axes[1, i].set_xlabel('Time (s)')
    for ax in axes[:, i]:
        ax.grid(alpha=.2)
axes[0, 0].set_ylabel('Forward velocity (m/s)')
axes[1, 0].set_ylabel('Angular rate (rad/s)')
axes[0, 0].legend(fontsize=8)
axes[1, 0].legend(fontsize=8)
fig.suptitle('carryWith reference velocities at configured 60 FPS\nGreen bands show current command ranges; no policy rollout used')
fig.tight_layout()
fig.savefig(OUT / 'reference_velocity.png', dpi=170)
plt.close(fig)
print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))
for name, item in report['files'].items():
    print(name, json.dumps({k: item[k] for k in ['frames', 'elapsed_seconds', 'tracking_yaw_frame_vx', 'unwrapped_yaw_rate', 'amp_body_w_vs_quaternion_body_w_rmse', 'raw_w_outliers_abs_gt_20']}))
