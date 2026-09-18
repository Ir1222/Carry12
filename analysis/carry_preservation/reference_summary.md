# Generated CarryWith reference summary

Reproduce with `python -m experiments.carry_preservation.analyze` from the repository root.
Full distributions, histograms, covariances and conditional correlations are in `statistics.json`.
Source hashes and runtime parameters are in `legged_gym/resources/config/carry_preservation.json`.

All three clips are used (773 frames at 60 Hz); differentiation never crosses clip boundaries.
Root/base and pelvis are the same frame. Torso transforms are reconstructed using the current URDF.
Stored palm positions agree with FK to within a few millimetres; that discrepancy is retained in the statistics.

The files do not contain reference box dimensions or measured box rotations. Box center is exactly
the stored hand midpoint; MotionLib reconstructs box heading from the pelvis. Nominal-size normalized
coordinates describe a 0.35 x 0.35 x 0.30 m scenario, not measured reference-object metadata.

| Motion | Raw grasp | Arm | Position | Orientation | Motion (50 Hz) | Surface-retargeted grasp |
|---|---:|---:|---:|---:|---:|---:|
| carrywith1.pt | 0.0170 | 0.9753 | 0.8273 | 0.8614 | 0.8667 | 0.9471 |
| carrywith2.pt | 0.0146 | 0.9999 | 0.9994 | 0.8636 | 0.9996 | 0.9989 |
| carrywith3.pt | 0.0135 | 0.9999 | 0.9994 | 0.8104 | 0.9996 | 0.9992 |

Raw grasp scores are low because the raw palms are inside the nominal box. Surface retargeting
intersects each measured hand ray with its designated side face. It preserves direction but is
only a geometry test, not a feasible-pose, contact, IK, or simulator validation. No reset is retargeted.

Torso-relative position standard deviation (m): [0.005161, 0.008416, 0.007995].

Pelvis-relative position standard deviation (m): [0.008753, 0.060124, 0.020555].

Calibration widths use the largest per-motion P95 residual, multiplied by 1.5 and floored.
Centers give equal influence to each motion. A zero pooled MAD is never used as a reward width.
The full technical interpretation, tests and training instructions are in REPORT.md.
