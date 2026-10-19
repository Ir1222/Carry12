# Shared model snapshots

This directory contains explicitly published model and TensorBoard-event snapshots for transferring between machines.  The active training-output directory, `legged_gym/logs/`, remains ignored by Git so future training runs are not staged or uploaded accidentally.

## Snapshot: 2026-10-19

- `carry_track/`: `model_33999.pt` and its TensorBoard event file.
- `track_vel_fina/`: `model_19999 (2).pt` and its TensorBoard event file.

The binary artifacts are stored with Git LFS.  On another machine, install Git LFS and run:

```powershell
git pull origin velocity-tracking
git lfs pull
```
