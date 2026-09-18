"""Offline physical-scale audit; never writes training targets or tolerances.

Run: python -m experiments.carry_preservation.analyze_constraints
Uses the existing MotionLib/FK analysis, all clips, and causal 50 Hz differences.
"""

import numpy as np

from . import analyze


def quantities(clip):
    sampled = clip["policy_sampled"]
    palms = np.stack([
        analyze.local(sampled["br"], sampled["poses"][side + "_palm_link"][0] - sampled["bp"])
        for side in ("left", "right")
    ], axis=1)
    velocity = np.diff(palms, axis=0) / analyze.POLICY_DT
    hand = clip["hand_box"]
    return {
        "arm_rad": clip["dofs"][:, 15:],
        "box_torso_m": clip["torso_link"]["position"],
        "box_torso_velocity_mps": clip["policy_velocity"],
        "box_torso_abs_velocity_mps": abs(clip["policy_velocity"]),
        "hand_box_m": hand,
        # Nominal scenario only: reference files have no measured dimensions.
        "nominal_signed_side_error_m": hand[..., 1] * [1, -1] - 0.175,
        "nominal_face_overflow_m": np.maximum(abs(hand[..., [0, 2]]) - [0.175, 0.150], 0),
        "hand_tangential_speed_mps": np.linalg.norm(velocity[..., [0, 2]], axis=-1),
    }


def main():
    data, _, _ = analyze.load_dataset()
    samples = {clip["name"]: quantities(clip) for clip in data}
    print("Reference box dimensions/contact are unmeasured; nominal side errors are illustrative.")
    print("Engineering ranges are chosen manually in carrybox_locomotion_config.py.")
    samples["pooled"] = {key: np.concatenate([v[key] for v in samples.values()])
                         for key in next(iter(samples.values()))}
    for name, values in samples.items():
        print("\n" + name)
        for key, value in values.items():
            print(key, "min", np.round(value.min(0), 4), "max", np.round(value.max(0), 4),
                  "P5/P95", np.round(np.percentile(value, [5, 95], axis=0), 4))


if __name__ == "__main__":
    main()
