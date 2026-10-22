"""Offline motion and checkpoint audit; requires only CPU PyTorch and PyYAML."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
import yaml
import ast
import subprocess
import struct
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PHYSHSI_COMMIT = '07bf22bdaf9c23c1d40366334a744c1816b6d42b'


def inspect_original_physhsi():
    paths = ['legged_gym/legged_gym/envs/g1/carrybox.py',
             'legged_gym/legged_gym/envs/g1/carrybox_config.py',
             'rsl_rl/rsl_rl/runners/him_on_policy_runner.py']
    source = {p: subprocess.check_output(['git', 'show', PHYSHSI_COMMIT + ':' + p], cwd=ROOT, text=True) for p in paths}
    tree = ast.parse(source[paths[0]])
    methods = {n.name: ast.get_source_segment(source[paths[0]], n)
               for cls in tree.body if isinstance(cls, ast.ClassDef)
               for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert 'target_speed_loco' in methods['_reward_walk_task']
    assert 'torch.bernoulli' in methods['_reset_hybrid_state_init']
    cfg = ast.parse(source[paths[1]])
    target = next(n.value.value for n in ast.walk(cfg) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'target_speed_loco' for t in n.targets))
    return dict(repository='InternRobotics/PhysHSI', commit=PHYSHSI_COMMIT,
        target_speed_loco=target, hybrid_rsi=True, amp_features=600, amp_coefficient=.25,
        sources={p: subprocess.check_output(['git', 'rev-parse', PHYSHSI_COMMIT + ':' + p], cwd=ROOT, text=True).strip() for p in paths},
        adapted=['fixed-speed world-frame approach/facing', 'hand placement and box lifting', 'AMP + hybrid RSI'],
        excluded=['relocation', 'putDown', 'carryWith'],
        walk_task=methods['_reward_walk_task'], carryup_task=methods['_reward_carryup_task'])


def inspect_foot_geometry():
    urdf = ROOT / 'legged_gym/resources/robots/g1/urdf/g1_29dof.urdf'
    links = {l.get('name'): l for l in ET.parse(urdf).findall('link')}
    report = {}
    for side in ('left', 'right'):
        for part in ('pitch', 'roll'):
            name = f'{side}_ankle_{part}_link'
            meshes = []
            for mesh in links[name].findall('collision/geometry/mesh'):
                path = (urdf.parent / mesh.get('filename')).resolve()
                raw = path.read_bytes()
                count = struct.unpack_from('<I', raw, 80)[0]
                if len(raw) != 84 + count * 50:
                    raise ValueError(f'Expected binary STL for geometry audit: {path}')
                vertices = [struct.unpack_from('<9f', raw, 84 + i * 50 + 12) for i in range(count)]
                points = torch.tensor(vertices).reshape(-1, 3)
                meshes.append(dict(file=path.relative_to(ROOT).as_posix(), triangles=count,
                    bounds_min=points.amin(0).tolist(), bounds_max=points.amax(0).tolist()))
            report[name] = meshes
    return dict(collision_meshes=report, selected_support_links=['left_ankle_roll_link', 'right_ankle_roll_link'],
        previous_detection_links=['left_ankle_pitch_link', 'right_ankle_pitch_link'],
        preserve_amp_end_effector_links=True, isaac_gym_runtime_verified=False)


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def derivative(value, fps=60):
    return torch.cat(((value[1:] - value[:-1]) * fps,
                      (value[-1:] - value[-2:-1]) * fps))


def quaternion_speed(q, fps=60):
    dot = (q[1:] * q[:-1]).sum(-1).abs().clamp(0, 1)
    values = 2 * torch.acos(dot) * fps
    return torch.cat((values, values[-1:]))


def inspect_references():
    config_dir = ROOT / 'legged_gym/resources/config'
    source = yaml.safe_load((config_dir / 'carrybox.yaml').read_text())['motions']
    no_relocation = yaml.safe_load((config_dir / 'carrybox_no_relocation.yaml').read_text())['motions']
    selected = yaml.safe_load((config_dir / 'carrybox_pickup_stand.yaml').read_text())['motions']
    assert set(selected) == {'pickUp', 'loco'}
    assert selected['pickUp'] == source['pickUp'] == no_relocation['pickUp']
    report = {'source_skills': list(source), 'no_relocation_skills': list(no_relocation),
              'selected_skills': list(selected), 'pickup': [], 'available_expert_clips': 0}
    for skill, entries in selected.items():
        for entry in entries:
            path = (config_dir / entry['file']).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            d = torch.load(path, map_location='cpu', weights_only=False)
            assert len(d['base_height']) > 12 and d['joint_position'].shape[1] == 29
            report['available_expert_clips'] += 1
            if skill != 'pickUp':
                continue
            # Reproduce MotionLib's conventions, including its +5 cm reset bias.
            base = d['base_position'].double().clone()
            box = base + d['box_pos_local'].double()
            bias = base[:, 2] - d['base_height']
            corrected_base, corrected_box = base.clone(), box.clone()
            corrected_base[:, 2] -= bias - .05
            corrected_box[:, 2] -= bias - .05
            root_speed = derivative(base).norm(dim=-1)
            root_angular = quaternion_speed(d['base_quat'].double())
            box_speed = derivative(corrected_box).norm(dim=-1)
            joint_rms = d['joint_velocity'].square().mean(-1).sqrt()
            tilt = torch.acos((1 - 2 * d['base_quat'][:, :2].square().sum(-1)).clamp(-1, 1))
            foot_height_delta = (d['link_position'][:, 2, 2] - d['link_position'][:, 3, 2]).abs()
            candidate = ((root_speed < .12) & (root_angular < .2) & (box_speed < .12)
                         & (joint_rms < .3) & (d['base_height'] >= .70)
                         & (corrected_box[:, 2] >= .67) & (tilt <= .25)
                         & (foot_height_delta <= .025))
            tail = slice(-30, None)
            item = dict(file=path.relative_to(ROOT).as_posix(), frames=len(base), contact_index=int(d['contact_index']),
                        final_base_height=float(d['base_height'][-1]),
                        final_box_height=float(corrected_box[-1, 2]),
                        final_foot_height_difference=float(foot_height_delta[-1]),
                        final_upright_loaded_pose=bool((d['base_height'][-1] >= .70) & (corrected_box[-1, 2] >= .67) & (tilt[-1] <= .25)),
                        kinematic_hold_candidates_in_tail=int(candidate[tail].sum()),
                        physically_validated_hold_states=0)
            for key, value in [('root_speed', root_speed), ('root_angular_speed', root_angular),
                               ('box_speed', box_speed), ('joint_velocity_rms', joint_rms)]:
                item[key] = dict(final=float(value[-1]), tail_min=float(value[tail].min()), tail_max=float(value[tail].max()))
            report['pickup'].append(item)
    return report


def inspect_checkpoint(path):
    model = load_source('pickup_stand_actor', 'rsl_rl/rsl_rl/modules/actor_critic.py')
    amp_module = load_source('pickup_stand_amp', 'rsl_rl/rsl_rl/modules/amp.py')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    actor = model.ActorCritic(738, 126, 6, 29, actor_hidden_dims=[512, 256, 256], critic_hidden_dims=[512, 256, 256])
    amp = amp_module.AMP(600, .25, device='cpu')
    actor.load_state_dict(checkpoint['model_state_dict'], strict=True)
    amp.load_state_dict(checkpoint['amp_state_dict'], strict=True)
    with torch.no_grad():
        output = actor.act_inference(torch.zeros(2, 738))
    assert output.shape == (2, 29) and torch.isfinite(output).all()
    import xml.etree.ElementTree as ET
    robot = ET.parse(ROOT / 'legged_gym/resources/robots/g1/urdf/g1_29dof.urdf')
    order = [j.attrib['name'] for j in robot.findall('joint') if j.attrib['type'] != 'fixed']
    mapping = [line.split()[1] for line in (ROOT / 'legged_gym/resources/config/joint_id.txt').read_text().splitlines()]
    assert order == mapping
    return dict(path=str(path), actor_obs=738, critic_obs=126, actions=29, amp_obs=600,
                strict_actor_and_amp_load=True, urdf_matches_motion_joint_order=True,
                joint_order=order, historical_checkpoint_joint_metadata=checkpoint.get('infos'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiments/pickup_stand/audit.json')
    args = parser.parse_args()
    result = inspect_references()
    result['original_physhsi'] = inspect_original_physhsi()
    result['foot_geometry'] = inspect_foot_geometry()
    if args.checkpoint:
        result['checkpoint'] = inspect_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(result, indent=2, default=str))
