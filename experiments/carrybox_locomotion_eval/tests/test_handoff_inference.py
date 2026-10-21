"""CPU checks for evaluating the trained A5/B2 controller in the velocity suite."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "rsl_rl"))
from experiments.carrybox_locomotion_eval.evaluation.inference import handoff_velocity_policy
from rsl_rl.teacher_handoff import adapt_action, joint_groups

NAMES = [line.split()[1] for line in (ROOT / "legged_gym/resources/config/joint_id.txt").read_text().splitlines()]


def fixture():
    control = dict(control_type="P", action_scale=.25, decimation=4,
                   stiffness={"joint": 20.}, damping={"joint": .5})
    env = NS(num_obs=738, num_actions=29, num_envs=2, device="cpu", dt=.019999999552965164,
             dof_names=NAMES, actions=torch.zeros(2, 29), cfg=NS(control=NS(**control),
             normalization=NS(clip_actions=1.), domain_rand=NS(delay=False)))
    actor = torch.nn.Sequential(torch.nn.Linear(738, 29))
    with torch.no_grad():
        actor[0].weight.zero_()
        actor[0].bias.fill_(3.)
    contract = dict(actor_obs=738, dof_names=NAMES, action_scale=.25, dt=.02,
                    environment=dict(control=control, normalization=dict(clip_actions=1.)),
                    handoff=dict(adapter="rate_limited", target_rate_arms=2., target_rate_waist=1.5,
                                 target_rate_legs=12., joint_target_rates=None))
    return env, actor, contract, control


def test_saved_rate_adapter_and_executed_action_history():
    env, actor, contract, control = fixture()
    source = copy.deepcopy(contract)
    rng = torch.get_rng_state().clone()
    before = copy.deepcopy(actor.state_dict())
    policy = handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)
    assert torch.equal(torch.get_rng_state(), rng)
    obs = torch.zeros(2, 738)
    first = policy(obs)
    groups = joint_groups(NAMES)
    for group, bound in (("arms", .16), ("waist", .12), ("legs", .96)):
        assert torch.allclose(first[:, groups[group]], torch.full_like(first[:, groups[group]], bound))
    env.actions.copy_(first)
    second = policy(obs)
    assert second[0, groups["arms"][0]] == pytest.approx(.32)
    # An asynchronous reset of one env clears its limiter history through the
    # actual env.actions tensor, without resetting the other env's history.
    env.actions[0] = 0
    reset = policy(obs)
    assert torch.equal(reset[0], first[0])
    assert torch.equal(reset[1], second[1])
    assert not reset.requires_grad
    assert contract == source
    for key, value in before.items():
        assert torch.equal(value, actor.state_dict()[key])
    metadata = policy.evaluation_metadata
    assert metadata["teacher_pickup"] is False
    assert metadata["adapter"] == "rate_limited"


@pytest.mark.parametrize("key,value", [
    ("dt", .01), ("dt", float("nan")), ("actor_obs", 737),
    ("action_scale", .5), ("dof_names", list(reversed(NAMES))),
])
def test_contract_mismatches_fail_before_rollout(key, value):
    env, actor, contract, control = fixture()
    contract[key] = value
    with pytest.raises(ValueError, match=key):
        handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)


def test_invalid_rates_control_clip_and_hard_adapter():
    env, actor, contract, control = fixture()
    contract["handoff"]["joint_target_rates"] = [1.] * 28
    with pytest.raises(ValueError, match="target rates"):
        handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)
    contract["handoff"]["joint_target_rates"] = None
    with pytest.raises(ValueError, match="PD/control"):
        handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, dict(control, decimation=2))
    env.cfg.normalization.clip_actions = 2.
    with pytest.raises(ValueError, match="clipping"):
        handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)
    env.cfg.normalization.clip_actions = 1.
    contract["handoff"]["adapter"] = "hard"
    policy = handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)
    assert policy(torch.zeros(2, 738)).eq(1).all()


def test_per_joint_override_uses_the_original_adapter():
    env, actor, contract, control = fixture()
    rates = [1. + i*.1 for i in range(29)]
    contract["handoff"]["joint_target_rates"] = rates
    policy = handoff_velocity_policy(env, actor, dict(dims=[738, 29]), contract, control)
    obs = torch.zeros(2, 738)
    expected = adapt_action(actor(obs), env.actions, torch.tensor(rates), env.dt, .25, 1.,
                            torch.ones(2, dtype=torch.bool), "rate_limited")
    assert torch.equal(policy(obs), expected)
