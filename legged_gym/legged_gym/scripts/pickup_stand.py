"""Explicit training, bounded smoke/evaluation and validated HOLD-bank collection."""
import argparse
import json
from pathlib import Path
import sys

# Isaac Gym must precede torch.
import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'evaluate', 'collect', 'smoke'], default='smoke')
    parser.add_argument('--stage', choices=['hold', 'pickup', 'full'], default='hold')
    parser.add_argument('--hold-bank')
    parser.add_argument('--require-hold-bank', action='store_true')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--output', type=Path, default=Path('pickup_stand_metrics.json'))
    parser.add_argument('--baseline-metrics', type=Path)
    parser.add_argument('--reset-suite', choices=['approach', 'pickup', 'hold', 'mixed'], default='approach')
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'carrybox_pickup_stand'
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    env_cfg.pickup_stand.curriculum_stage = options.stage
    env_cfg.pickup_stand.hold_bank_path = options.hold_bank
    env_cfg.pickup_stand.require_hold_bank = options.require_hold_bank or options.reset_suite == 'hold'
    train_cfg.runner.run_name = options.stage
    if options.mode != 'train':
        env_cfg.noise.add_noise = False
        if options.reset_suite != 'mixed':
            weights = {'approach': [0, 0, 0, 0, 1], 'pickup': [0, 1, 1, 1, 0], 'hold': [1, 0, 0, 0, 0]}[options.reset_suite]
            setattr(env_cfg.pickup_stand, options.stage + '_reset_weights', weights)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    if options.mode == 'train':
        runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
        return
    if not args.finetune_path and not args.resume:
        raise ValueError('Evaluation/smoke/collection requires --finetune_path or --resume --resume_path')
    policy = runner.get_inference_policy(env.device)
    observation = env.get_observations()
    episodes, snapshots, bank_rows = [], [], []
    recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(options.steps):
        with torch.inference_mode():
            observation, _, _, done, info, _, _, _ = env.step(policy(observation))
        assert observation.shape == (env.num_envs, 738) and torch.isfinite(observation).all()
        diagnostic = info['pickup_stand']  # pre-autoreset snapshot
        hold = diagnostic['phase'] == 2
        if hold.any():
            assert torch.all(diagnostic['command_speed'][hold] == 0), 'Nonzero HOLD command'
            snapshots.append({k: float(v[hold].float().sum()) for k, v in diagnostic.items()
                              if k in ('base_linear_speed', 'base_angular_speed', 'box_linear_speed', 'box_angular_speed',
                                       'bilateral_contact', 'both_feet_contact', 'foot_slip',
                                       'box_height_deviation', 'box_orientation_deviation')})
            snapshots[-1]['count'] = int(hold.sum())
        for index in done.nonzero().flatten().tolist():
            episodes.append({k: float(v[index]) for k, v in diagnostic.items()})
        if options.mode == 'collect':
            eligible = (env.stable_elapsed >= env.cfg.pickup_stand.stable_seconds) & ~recorded & ~done.bool()
            for index in eligible.nonzero().flatten().tolist():
                # Preserve all velocities; never zero a moving reference.
                root = env.root_states[index].clone()
                box = env.box_states[index].clone()
                root[:3] -= env.env_origins[index]
                box[:3] -= env.env_origins[index]
                bank_rows.append(dict(root=root.cpu(), box=box.cpu(), dof_pos=env.dof_pos[index].cpu().clone(),
                    dof_vel=env.dof_vel[index].cpu().clone(), size=env._box_size[index].cpu().clone(),
                    mass=env.box_masses[index].cpu().clone(), ee=env.end_effector_pos[index].cpu().clone(),
                    bilateral_contact=diagnostic['bilateral_contact'][index].cpu().clone(),
                    both_feet_contact=diagnostic['both_feet_contact'][index].cpu().clone(),
                    foot_slip=diagnostic['foot_slip'][index].cpu().clone(),
                    validated_seconds=env.stable_elapsed[index].cpu().clone()))
            recorded |= eligible
            recorded[done.bool()] = False
    # Include censored episodes separately, never count a partial run as failure.
    partial = {k: v.cpu().tolist() for k, v in env.diagnostics().items()}
    def average(rows, key):
        return sum(row[key] for row in rows) / len(rows) if rows else None
    pickup_episodes = [r for r in episodes if r['reset_category'] != 0]
    approach_episodes = [r for r in episodes if r['reset_category'] == 4]
    stable_episodes = [r for r in episodes if r['stable_delay'] >= 0 and r['pickup_completed']]
    report = dict(completed_episodes=len(episodes), reset_suite=options.reset_suite,
        seed=env.cfg.seed, steps=options.steps, num_envs=env.num_envs,
        approach_success_rate=average(approach_episodes, 'approach_completed'),
        pickup_success_rate=average(pickup_episodes, 'pickup_completed'),
        hold_success_rate=average(episodes, 'hold_success'),
        pickup_to_stable_hold_seconds=average(stable_episodes, 'stable_delay'),
        successful_hold_seconds=average(episodes, 'longest_hold'),
        approach_seconds=average([r for r in approach_episodes if r['approach_completed']], 'approach_seconds'),
        pickup_seconds=average([r for r in pickup_episodes if r['pickup_completed']], 'pickup_seconds'),
        hold_step_means={k: sum(r[k] for r in snapshots) / sum(r['count'] for r in snapshots)
                         for k in snapshots[0] if k != 'count'} if snapshots else {},
        episodes=episodes, partial_episodes=partial)
    report['by_reset_category'] = {str(category): dict(
        episodes=sum(r['reset_category'] == category for r in episodes),
        pickup_success_rate=average([r for r in episodes if r['reset_category'] == category], 'pickup_completed'),
        hold_success_rate=average([r for r in episodes if r['reset_category'] == category], 'hold_success'))
        for category in range(5)}
    if options.baseline_metrics:
        baseline = json.loads(options.baseline_metrics.read_text())
        if any(baseline[key] != report[key] for key in ('seed', 'num_envs', 'steps', 'reset_suite')):
            raise ValueError('Baseline evaluation seed/env count/horizon/reset suite must match')
        report['regression_delta_from_baseline'] = {
            key: report[key] - baseline[key] if report[key] is not None and baseline[key] is not None else None
            for key in ('approach_success_rate', 'pickup_success_rate', 'approach_seconds', 'pickup_seconds', 'hold_success_rate')}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2))
    if options.mode == 'collect':
        if not bank_rows:
            raise RuntimeError('No physically validated stable HOLD observed; metrics saved, bank NOT fabricated')
        bank = {k: torch.stack([row[k] for row in bank_rows]) for k in bank_rows[0]}
        bank.update(version=1, dof_names=env.dof_names, source_checkpoint=args.finetune_path or args.resume_path)
        torch.save(bank, options.output.with_suffix('.pt'))
    print(json.dumps({k: v for k, v in report.items() if k not in ('episodes', 'partial_episodes')}, indent=2))


if __name__ == '__main__':
    main()
