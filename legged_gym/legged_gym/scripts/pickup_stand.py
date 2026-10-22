"""Scratch AMP/PPO training, short smoke, contact probe and phase-policy eval."""
import argparse
import json
from pathlib import Path
import sys

import isaacgym  # must precede torch
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import class_to_dict
from legged_gym.utils.pickup_stand_experiment import (
    assert_scratch_config, scratch_manifest, state_digest, validate_evaluation_checkpoint,
)


def write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')


def configure(options, args, env_cfg, train_cfg):
    # All training/smoke starts random. --checkpoint belongs exclusively to eval
    # and collection and is never forwarded to TaskRegistry's legacy loaders.
    assert_scratch_config(train_cfg, args)
    if options.mode in ('train', 'train-smoke', 'smoke', 'contact-check') and options.checkpoint:
        raise ValueError('Scratch/probe modes reject --checkpoint; use evaluate for a trained phase policy')
    if options.mode in ('evaluate', 'collect') and not options.checkpoint:
        raise ValueError('Evaluation/collection requires --checkpoint from this scratch experiment')
    train_cfg.runner.resume = False
    train_cfg.runner.finetune_path = train_cfg.runner.resume_path = None
    if options.hold_bank:
        env_cfg.pickup_stand.hold_bank_path = options.hold_bank
    if options.reset_weights is not None:
        env_cfg.pickup_stand.reset_weights = options.reset_weights
    if options.approach_speed is not None:
        env_cfg.rewards.target_speed_loco = options.approach_speed
    if options.hold_seconds is not None:
        env_cfg.pickup_stand.success_hold_seconds = options.hold_seconds
    if env_cfg.rewards.target_speed_loco <= 0 or env_cfg.pickup_stand.success_hold_seconds <= 0:
        raise ValueError('Approach speed and hold duration must be positive')
    suite = options.reset_suite
    if suite is None:
        suite = 'mixed' if options.mode in ('train', 'train-smoke', 'smoke') else 'approach'
    if options.mode == 'contact-check':
        suite = 'approach'
        env_cfg.pickup_stand.approach_rsi_probability = 0.
        # Probe known default stance under PD without random actions/perturbation.
        for name in dir(env_cfg.domain_rand):
            value = getattr(env_cfg.domain_rand, name)
            if isinstance(value, bool):
                setattr(env_cfg.domain_rand, name, False)
    if suite != 'mixed':
        if options.reset_weights is not None:
            raise ValueError('Use either an explicit reset suite or reset weights')
        env_cfg.pickup_stand.reset_weights = {
            'approach': [0, 0, 0, 0, 1], 'pickup': [0, 1, 1, 1, 0], 'hold': [1, 0, 0, 0, 0]}[suite]
    if args.num_envs is None and options.mode != 'train':
        args.num_envs = 32
    if options.mode == 'train-smoke':
        if not 1 <= options.smoke_iterations <= 10 or args.num_envs > 128:
            raise ValueError('Training smoke is bounded to 1..10 iterations and <=128 environments')
        train_cfg.runner.num_steps_per_env = 32
        train_cfg.runner.max_iterations = options.smoke_iterations
        args.max_iterations = options.smoke_iterations  # registry must not override the bound
        train_cfg.algorithm.num_learning_epochs = 2
        train_cfg.algorithm.num_mini_batches = 2
    if options.mode not in ('train', 'train-smoke'):
        env_cfg.noise.add_noise = False
    return suite


def contact_probe(env, steps, output):
    # No policy or checkpoint. This validates an unloaded stance, never asserts
    # that a visually loaded stance is physically valid.
    env.root_states[:, 7:13] = 0.
    env._reset_env_tensors(torch.arange(env.num_envs, device=env.device))
    traces, supported = [], torch.zeros(env.num_envs, device=env.device)
    best = torch.zeros_like(supported)
    for step in range(steps):
        with torch.inference_mode():
            _, _, _, done, info, _, _, _ = env.step(torch.zeros(env.num_envs, 29, device=env.device))
        d = info['pickup_stand']
        standing = d['both_feet_contact'] & (d['base_linear_speed'] <= env.cfg.pickup_stand.stable_base_speed)
        supported = torch.where(standing & ~done.bool(), supported + env.dt, 0.)
        best = torch.maximum(best, supported)
        if step % 10 == 0:
            traces.append({k: v.cpu().tolist() for k, v in d.items()
                if k.startswith(('completion/', 'completion_value/')) or 'force_z' in k or 'foot_support' in k or k == 'both_feet_contact'})
    passed = best >= .2
    report = dict(kind='unloaded_default_PD_contact_probe', support_manifest=env.support_manifest,
        threshold_N=env.cfg.pickup_stand.foot_force, max_continuous_support_seconds=best.cpu().tolist(),
        per_env_pass=passed.cpu().tolist(), traces=traces, loaded_hold_validated=False)
    write_json(output, report)
    if not passed.all():
        raise RuntimeError(f'Contact probe failed in {(~passed).sum().item()} envs; inspect {output}; thresholds were not relaxed')
    print(f'Contact probe passed; {output}')


def evaluate(env, runner, options, suite):
    policy = runner.get_inference_policy(env.device)
    observation = env.get_observations()
    episodes, partial, bank_rows = [], {}, []
    totals, counts, predicate_failures = {}, {}, {}
    recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for _ in range(options.steps):
        with torch.inference_mode():
            observation, _, rewards, done, info, _, _, amp = env.step(policy(observation))
        assert observation.shape == (env.num_envs, 738) and torch.isfinite(observation).all()
        assert torch.isfinite(rewards).all() and torch.isfinite(amp).all()
        d = info['pickup_stand']  # copied before autoreset
        remaining = ~done.bool()
        partial = {'env_ids': remaining.nonzero().flatten().cpu().tolist()}
        partial.update({k: v[remaining].cpu().tolist() for k, v in d.items()})
        assert torch.all(d['command_speed'] == 0), 'Carry velocity command leaked into phase task'
        for phase, name in [(0, 'approach'), (1, 'pickup'), (2, 'hold')]:
            mask = d['phase'] == phase
            counts[name] = counts.get(name, 0) + int(mask.sum())
            if not mask.any():
                continue
            sums = totals.setdefault(name, {})
            for key, value in d.items():
                if key.startswith(('completion/', 'completion_value/')) or key in (
                    'base_linear_speed', 'base_angular_speed', 'box_linear_speed', 'box_angular_speed',
                    'bilateral_contact', 'both_feet_contact', 'foot_slip', 'box_height_deviation',
                    'box_orientation_deviation', 'left_foot_force_z', 'right_foot_force_z',
                    'left_pitch_force_z', 'right_pitch_force_z', 'footstep_events'):
                    sums[key] = sums.get(key, 0.) + float(value[mask].float().sum())
                    if phase == 1 and key.startswith('completion/'):
                        predicate_failures[key] = predicate_failures.get(key, 0) + int((~value[mask]).sum())
        for index in done.nonzero().flatten().tolist():
            episodes.append({k: float(v[index]) for k, v in d.items()})
        if options.mode == 'collect':
            eligible = (env.stable_elapsed >= env.cfg.pickup_stand.stable_seconds) & ~recorded & ~done.bool()
            for index in eligible.nonzero().flatten().tolist():
                root, box = env.root_states[index].clone(), env.box_states[index].clone()
                root[:3] -= env.env_origins[index]
                box[:3] -= env.env_origins[index]
                row = dict(root=root.cpu(), box=box.cpu())
                for name, tensor in [('dof_pos', env.dof_pos), ('dof_vel', env.dof_vel), ('size', env._box_size),
                    ('mass', env.box_masses), ('ee', env.end_effector_pos), ('validated_seconds', env.stable_elapsed)]:
                    row[name] = tensor[index].cpu().clone()
                for name in ('bilateral_contact', 'both_feet_contact', 'foot_slip'):
                    row[name] = d[name][index].cpu().clone()
                bank_rows.append(row)
            recorded |= eligible
            recorded[done.bool()] = False
    def average(rows, key):
        return sum(row[key] for row in rows) / len(rows) if rows else None
    pickup = [r for r in episodes if r['reset_category'] != 0]
    approach = [r for r in episodes if r['reset_category'] == 4]
    stable = [r for r in pickup if r['stable_delay'] >= 0 and r['pickup_completed']]
    report = dict(completed_episodes=len(episodes), seed=env.cfg.seed, steps=options.steps,
        num_envs=env.num_envs, reset_suite=suite, policy_metadata=env.policy_metadata,
        support_manifest=env.support_manifest, observation_semantics=env.cfg.observation_semantics,
        pickup_success_rate=average(pickup, 'pickup_completed'), approach_success_rate=average(approach, 'approach_completed'),
        hold_success_rate=average(episodes, 'hold_success'), successful_hold_seconds=average(episodes, 'longest_hold'),
        pickup_to_stable_hold_seconds=average(stable, 'stable_delay'),
        approach_seconds=average([r for r in approach if r['approach_completed']], 'approach_seconds'),
        pickup_seconds=average([r for r in pickup if r['pickup_completed']], 'pickup_seconds'),
        phase_step_means={p: {k: v / counts[p] for k, v in values.items()} for p, values in totals.items()},
        pickup_predicate_failures=predicate_failures, episodes=episodes, partial_episodes=partial)
    report['checkpoint_loaded'] = options.checkpoint is not None
    report['checkpoint'] = str(options.checkpoint) if options.checkpoint else None
    report['reset_weights'] = env.cfg.pickup_stand.reset_weights
    report['success_hold_seconds'] = env.cfg.pickup_stand.success_hold_seconds
    report['approach_speed'] = env.cfg.rewards.target_speed_loco
    report['by_reset_category'] = {str(category): dict(
        episodes=sum(r['reset_category'] == category for r in episodes),
        pickup_success_rate=average([r for r in episodes if r['reset_category'] == category], 'pickup_completed'),
        hold_success_rate=average([r for r in episodes if r['reset_category'] == category], 'hold_success'))
        for category in range(5)}
    if options.baseline_metrics:
        baseline = json.loads(options.baseline_metrics.read_text())
        for key in ('seed', 'steps', 'num_envs', 'reset_suite', 'observation_semantics',
                    'reset_weights', 'success_hold_seconds', 'approach_speed'):
            if baseline[key] != report[key]:
                raise ValueError(f'Baseline {key} does not match')
        report['regression_delta_from_baseline'] = {k: report[k] - baseline[k]
            if report[k] is not None and baseline[k] is not None else None
            for k in ('pickup_success_rate', 'approach_success_rate', 'hold_success_rate', 'approach_seconds', 'pickup_seconds')}
    write_json(options.output, report)
    if options.mode == 'collect':
        if not bank_rows:
            raise RuntimeError('No stable simulated HOLD states observed; no bank written')
        bank = {k: torch.stack([r[k] for r in bank_rows]) for k in bank_rows[0]}
        bank.update(version=2, dof_names=env.dof_names, source_checkpoint=str(options.checkpoint), support_manifest=env.support_manifest)
        torch.save(bank, options.output.with_suffix('.pt'))
    print(json.dumps({k: v for k, v in report.items() if k not in ('episodes', 'partial_episodes')}, indent=2))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--mode', choices=['train', 'train-smoke', 'smoke', 'evaluate', 'collect', 'contact-check'], default='smoke')
    parser.add_argument('--checkpoint', type=Path, help='Evaluation/collection only: checkpoint with phase-v2 metadata')
    parser.add_argument('--hold-bank')
    parser.add_argument('--reset-weights', type=float, nargs=5)
    parser.add_argument('--reset-suite', choices=['approach', 'pickup', 'hold', 'mixed'])
    parser.add_argument('--approach-speed', type=float)
    parser.add_argument('--hold-seconds', type=float)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--smoke-iterations', type=int, default=2)
    parser.add_argument('--output', type=Path, default=Path('outputs/pickup_stand/scratch_smoke.json'))
    parser.add_argument('--baseline-metrics', type=Path)
    options, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = 'carrybox_pickup_stand'
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    suite = configure(options, args, env_cfg, train_cfg)
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=env_cfg)
    if options.mode == 'contact-check':
        env.reset()
        contact_probe(env, options.steps, options.output)
        return
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=train_cfg)
    initial = scratch_manifest(runner)
    initial.update(support_manifest=env.support_manifest, reset_weights=env.cfg.pickup_stand.reset_weights,
        reward_scales=class_to_dict(env.cfg.rewards.scales))
    write_json(options.output.with_suffix('.initialization.json'), initial)
    if options.mode in ('train', 'train-smoke'):
        runner.learn(train_cfg.runner.max_iterations, init_at_random_ep_len=False)
        report = dict(initialization=initial, completed_iterations=runner.current_learning_iteration + 1,
            actor_changed=initial['actor_sha256'] != state_digest(runner.alg.actor_critic.actor),
            critic_changed=initial['critic_sha256'] != state_digest(runner.alg.actor_critic.critic),
            discriminator_changed=initial['discriminator_sha256'] != state_digest(runner.alg.amp),
            optimizer_state_entries=len(runner.alg.optimizer.state))
        for module in (runner.alg.actor_critic, runner.alg.amp):
            if not all(torch.isfinite(p).all() for p in module.parameters()):
                raise RuntimeError('Nonfinite parameters after training')
        write_json(options.output, report)
    else:
        if options.checkpoint:
            validate_evaluation_checkpoint(options.checkpoint, env.policy_metadata)
            runner.load(str(options.checkpoint), load_optimizer=False)
        evaluate(env, runner, options, suite)


if __name__ == '__main__':
    main()
