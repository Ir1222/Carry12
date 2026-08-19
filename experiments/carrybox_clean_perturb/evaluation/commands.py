"""Command helpers shared by controlled CarryBox evaluators."""


def set_fixed_evaluation_command(env, command):
    """Synchronize a fixed raw/policy command and rebuild actor observations."""
    command = tuple(float(value) for value in command)
    if len(command) != 3:
        raise ValueError(f"Expected a 3-D (vx, vy, yaw) command, got {command}")

    env.commands[:, 0] = command[0]
    env.commands[:, 1] = command[1]
    env.commands[:, 2] = command[2]
    env.carry_policy_commands[:, :3] = env.commands[:, :3]
    env.compute_observations()
    return env.get_observations()
