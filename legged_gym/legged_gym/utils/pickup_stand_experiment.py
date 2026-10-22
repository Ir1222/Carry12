"""Scratch/evaluation provenance helpers; no Gym dependency."""
import hashlib
import torch


def assert_scratch_config(train_cfg, args):
    runner = train_cfg.runner
    for name in ('resume', 'resume_path', 'finetune_path', 'init_policy_path', 'checkpoint', 'load_run'):
        value = getattr(args, name, None)
        if value is not None and value is not False:
            raise ValueError(f'Scratch training rejects --{name}; use --mode evaluate --checkpoint PATH for evaluation')
    if runner.resume or runner.resume_path or runner.finetune_path:
        raise ValueError('Scratch runner config must not contain checkpoint initialization')


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scratch_manifest(runner):
    if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
        raise ValueError('Scratch startup requires iteration zero and an empty optimizer state')
    model = runner.alg.actor_critic
    return dict(initialization='scratch', checkpoint_loaded=False, iteration=0,
        optimizer_state_entries=0, actor_sha256=state_digest(model.actor),
        critic_sha256=state_digest(model.critic), discriminator_sha256=state_digest(runner.alg.amp),
        observation_semantics=runner.env.cfg.observation_semantics)


def validate_evaluation_checkpoint(path, metadata):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved = checkpoint.get('policy_metadata', {})
    for key in ('task', 'observation_semantics', 'actor_obs', 'critic_obs', 'actions', 'dof_names', 'approach_speed'):
        if saved.get(key) != metadata.get(key):
            raise ValueError(f'Checkpoint {key} mismatch: legacy command observations cannot be loaded as phase one-hot')
    return saved
