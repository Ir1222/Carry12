import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import torch

from nforce_test_support import RolloutEnv, actor_critic_class, carrybox_configs
from evaluation.inference import assert_startup_compatibility, load_actor_only_for_inference


def _model():
    cfg, train = carrybox_configs()
    return actor_critic_class()(
        cfg.env.num_actor_obs, cfg.env.num_privileged_obs, cfg.env.num_actor_history,
        cfg.env.num_actions, actor_hidden_dims=train.policy.actor_hidden_dims,
        critic_hidden_dims=train.policy.critic_hidden_dims, activation=train.policy.activation,
    )


def _load(model, state):
    with tempfile.TemporaryDirectory() as directory:
        path = str(Path(directory) / "model.pt")
        torch.save(state, path)
        runner = SimpleNamespace(alg=SimpleNamespace(actor_critic=model))
        return load_actor_only_for_inference(runner, path, device="cpu")


def test_actor_only_checkpoint_matches_full_policy_actions():
    source, destination = _model(), _model()
    state = source.state_dict()
    critic_before = {key: value.clone() for key, value in destination.critic.state_dict().items()}
    _load(destination, {"model_state_dict": state, "optimizer_state_dict": {}, "iter": 19000})
    obs = torch.randn(3, 738)
    with torch.no_grad():
        expected = source.act_inference(obs)
        actual = destination.act_inference(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == (3, 29)
    for key, value in destination.critic.state_dict().items():
        torch.testing.assert_close(value, critic_before[key], rtol=0, atol=0)


def test_actor_only_checkpoint_accepts_no_critic_amp_or_optimizer():
    model = _model()
    state = {key: value for key, value in model.state_dict().items() if key.startswith("actor.")}
    _load(_model(), {"model_state_dict": state})


def test_actor_checkpoint_rejects_missing_weights_before_loading():
    model = _model()
    for missing in ("actor.0.weight", "actor.2.bias"):
        state = dict(model.state_dict())
        del state[missing]
        with TestCase().assertRaisesRegex(RuntimeError, "missing"):
            _load(model, {"model_state_dict": state})
    with TestCase().assertRaisesRegex(KeyError, "model_state_dict"):
        _load(model, {})


def test_actor_checkpoint_rejects_input_output_and_hidden_shape_mismatches():
    model = _model()
    original = {key: value.clone() for key, value in model.state_dict().items()}
    for key in ("actor.0.weight", "actor.2.weight", "actor.6.weight"):
        state = dict(original)
        if key == "actor.0.weight":
            state[key] = state[key][:, :-1]
            error = AssertionError
        else:
            state[key] = state[key][:-1]
            error = RuntimeError
        with TestCase().assertRaises(error):
            _load(model, {"model_state_dict": state})
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)


def test_startup_checks_actual_policy_command_and_action_shape():
    env = RolloutEnv(command=(0.4, 0.2, 0.1))
    obs, _ = env.reset()
    assert_startup_compatibility(env, obs, expected_command=(0.4, 0.0, 0.1))
    with TestCase().assertRaisesRegex(AssertionError, "command mismatch"):
        assert_startup_compatibility(env, obs, expected_command=env.command)
    env.num_actions = 28
    with TestCase().assertRaisesRegex(AssertionError, "29 actions"):
        assert_startup_compatibility(env, obs)
