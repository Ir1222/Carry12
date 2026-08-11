# carrybox_perturb_debug 控制评估说明

这个目录是 `carrybox_perturb` 的实验层，用于两类事情：

- `play_debug.py`：调试外力链路是否触发、提交、绘制、真实施加。
- `evaluate.py`：做受控 rollout 评估，用同一组外力条件比较不同 Actor checkpoint 的响应。

这里不会修改 baseline 任务、Actor observation、Critic、PPO、训练流程，也不会一次加载多个策略。每次命令只加载一个 Actor checkpoint。

## Ubuntu 运行前提

所有命令都假设你在仓库根目录运行：

```bash
cd /path/to/PhysHSI
```

如果你的环境里 `python` 指向正确的 Conda/Isaac Gym Python，也可以把下面命令里的 `python3` 替换成 `python`。

如果你还没有激活 Isaac Gym 环境，先激活你的环境，例如：

```bash
conda activate <your_isaacgym_env>
```

注意：方向参数里 `-box_x`、`-box_y` 以 `-` 开头，在 Bash 里必须用等号写法：

```bash
--direction=-box_y
```

不要写成：

```bash
--direction -box_y
```

后者会被 argparse 当成新的命令行选项，导致命令不可用。

## 当前执行路径

`evaluate.py` 的执行路径是：

```text
evaluate.py
-> 读取 carrybox_perturb baseline cfg
-> apply_evaluation_config()
-> 临时注册 carrybox_perturb_eval
-> 创建 env
-> 创建 PPO runner，但不 resume Critic
-> load_actor_only_for_inference()
-> 只加载 checkpoint 里的 actor.* 和 std
-> rollout 一个或多个受控 trial
```

物理外力路径仍复用 baseline 的箱体 COM 外力逻辑：

```python
gym.apply_rigid_body_force_tensors(
    ...,
    space=gymapi.CoordinateSpace.GLOBAL_SPACE,
)
```

外力施加点是 box center of mass，不添加偏心力矩。

## Debug 和 Evaluation 的区别

`play_debug.py` 适合回答：

- 外力有没有触发？
- viewer 里箭头方向对不对？
- carry gate 为什么没过？
- force tensor 有没有真实写入？

`evaluate.py` 适合回答：

- 同一个外力条件下，不同 Actor checkpoint 的响应是否不同？
- perturbation recovery 怎么样？
- sustained object-mediated force following 响应怎么样？
- 箱子、机器人、手-箱耦合的原始指标是多少？

正式比较 checkpoint 时，用 `evaluate.py`，不要用 `play_debug.py`。

## 受控场景

评估模式保留当前 fixed-scene 设计：

- `num_envs = 1`
- 初始机器人 pose 固定
- box placement 固定
- goal placement 固定
- seed 可控
- legacy robot disturbance 关闭
- delay 关闭
- `push_robots` 关闭
- box random properties 关闭

同一个 seed 和同一个 force condition，换不同 checkpoint 时，非策略实验条件应保持一致。

启用 `--save_csv` 时，`summary.csv` 会保存 initial-state signature，包括：

- robot root pose
- robot root linear/angular velocity
- DOF position
- DOF velocity
- box pose
- box linear/angular velocity
- goal
- box mass

这个 signature 只用于评估记录，不会加入 policy observation。

## Trial 定义

一个 trial 等于一个 episode，也等于一个 force condition：

```text
RESET
-> APPROACH / PICKUP
-> WAIT_CONFIRMED_CARRY
-> PRE_FORCE
-> APPLY ONE CONTROLLED FORCE CONDITION
-> POST_FORCE
-> TRIAL_END
```

一个 episode 内不会 sweep 多个不同外力条件，避免前一次扰动污染下一次评估。

外力释放后默认继续运行：

```text
POST_FORCE_OBSERVATION_S = 2.0
```

如果已有物理失败 termination 先发生，则提前结束并记录失败原因。

## Confirmed Carry 触发

外力不能在第一次看到 `carry_phase=True` 时立刻施加。评估器使用：

```text
stable_confirmed_carry_policy_steps = 10
PRE_FORCE_DELAY_S = 0.20
```

实际触发逻辑：

```text
confirmed carry 连续满足 10 个 policy step
-> CONFIRMED_CARRY
-> 再稳定等待 0.20 s
-> FORCE ON
```

如果 pre-force delay 中 confirmed carry 丢失，则回到 `WAIT_CONFIRMED_CARRY`，不会盲目施加外力。

## 方向语义

默认评估方向：

```text
+box_x
-box_x
+box_y
-box_y
```

box-local 方向会在 force commit 时用 `q_box(t0)` 转成 world direction，并且只转换一次。之后整个 force event 中 `direction_world` 冻结不变，即使箱子旋转也不会重新计算方向。

## 力大小

力大小使用归一化定义：

```text
F_peak = beta * m_box * g
```

默认 sweep beta：

```text
0.1, 0.3, 0.5, 0.7, 0.9, 1.1
```

可以用 `--betas=...` 指定其他 beta 组。

## 外力 profile

### half_sine

短脉冲扰动，用于 perturbation robustness / recovery：

```text
F(t) = F_peak * sin(pi * t / T) * direction_world
```

默认：

```text
T = 0.10 s
```

对应参数：

```bash
--profile half_sine --pulse_duration 0.10
```

### smooth_hold

持续平滑外力，用于 object-mediated force following：

```text
0 -> half-cosine ramp up -> constant hold -> half-cosine ramp down -> 0
```

默认：

```text
ramp_up = 0.15 s
hold_duration = 1.0 s
ramp_down = 0.15 s
```

注意：`--hold_duration 1.0` 表示中间恒定 hold 的时长，不包括 ramp。总物理外力时长是：

```text
0.15 + 1.0 + 0.15 = 1.30 s
```

对应参数：

```bash
--profile smooth_hold --hold_duration 1.0 --ramp_up 0.15 --ramp_down 0.15
```

## 命令 1：Viewer 单条件可视化

用途：打开 Isaac Gym viewer，人工检查一个 checkpoint 在一个外力条件下的响应。

特点：

- 只跑一个 trial
- 不自动 sweep
- 默认不写 CSV
- viewer 箭头显示真实瞬时外力
- 适合人工观察 official / retrained / PI checkpoint 的行为差异

命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile smooth_hold \
  --direction=-box_y \
  --beta 0.5 \
  --hold_duration 1.0 \
  --seed 1
```

什么时候用：

- 想看机器人是否被外力带动、是否保持抓箱、是否摔倒。
- 想确认 smooth_hold 箭头是否 ramp up、hold、ramp down。
- 想手动比较不同 checkpoint，但每次仍然只加载一个 checkpoint。

## 命令 2：Headless 单条件并保存 CSV

用途：无 viewer 跑一个 trial，并保存 trace CSV 和 summary CSV。

特点：

- 只跑一个 trial
- 不打开 viewer
- 保存机器可读结果
- 适合先验证 CSV 输出和 initial-state signature

命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile smooth_hold \
  --direction=-box_y \
  --beta 0.5 \
  --hold_duration 1.0 \
  --seed 1 \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/official_single
```

什么时候用：

- 想快速检查 evaluator 是否能跑通。
- 想检查 `summary.csv` 和 `traces/T0001.csv` 是否生成。
- 想验证同一个 seed 重复运行时 initial-state signature 是否一致。

## 命令 3：Headless 默认 smooth_hold sweep

用途：对一个 checkpoint 跑默认 force-follow 矩阵。

特点：

- 一个命令仍然只加载一个 checkpoint
- 自动生成多个 deterministic trial
- Trial ID 不包含 checkpoint 名，因此不同 checkpoint 的 `T0001`、`T0002` 可配对比较
- 默认保存 CSV

默认矩阵：

```text
directions: +box_x, -box_x, +box_y, -box_y
betas:      0.1, 0.3, 0.5, 0.7, 0.9, 1.1
holds:      0.5, 1.0, 2.0
seeds:      1, 2, 3
```

总 trial 数：

```text
4 * 6 * 3 * 3 = 216
```

命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile smooth_hold \
  --sweep \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/official
```

什么时候用：

- 正式跑 official checkpoint 的完整 force-follow 评估。
- 之后用同一条命令只替换 `--resume_path` 和 `--output_dir`，分别跑 retrained / PI checkpoint。

## 命令 4：Headless 小 sweep 测试

用途：正式大 sweep 之前先跑一个小矩阵，确认 trial 数、CSV、force timing 都正常。

特点：

- 只跑 8 个 trial
- 适合调试 evaluator
- 比默认 216 trial 快很多

命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile smooth_hold \
  --sweep \
  --directions=+box_x,-box_x \
  --betas=0.1,0.3 \
  --seeds=1,2 \
  --hold_durations=1.0 \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/smoke_sweep
```

矩阵大小：

```text
2 directions * 2 betas * 2 seeds * 1 hold = 8 trials
```

什么时候用：

- 第一次在 Ubuntu 上验证命令。
- 修改 evaluator 后做 smoke test。
- 检查 deterministic Trial ID 是否正确。

## 命令 5：half_sine 短扰动单条件

用途：测试短时扰动恢复，而不是持续 force following。

特点：

- 使用 half-sine pulse
- `--pulse_duration` 控制整个短脉冲时长
- 不使用 `--hold_duration`

Viewer 命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile half_sine \
  --direction=+box_x \
  --beta 0.5 \
  --pulse_duration 0.10 \
  --seed 1
```

Headless CSV 命令：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile half_sine \
  --direction=+box_x \
  --beta 0.5 \
  --pulse_duration 0.10 \
  --seed 1 \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/half_sine_single
```

什么时候用：

- 想测局部冲击后的恢复。
- 想和旧的 half-sine perturbation 行为保持可比。

## 命令 6：三个 checkpoint 的配对评估方式

评估器不会在一个进程里加载三个 policy。你需要分别运行三次。

official：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path legged_gym/resources/ckpt/carrybox.pt \
  --profile smooth_hold \
  --sweep \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/official
```

retrained：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path <retrained_checkpoint.pt> \
  --profile smooth_hold \
  --sweep \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/retrained
```

PI-trained：

```bash
python3 experiments/carrybox_perturb_debug/evaluate.py \
  --task carrybox_perturb \
  --resume_path <pi_checkpoint.pt> \
  --profile smooth_hold \
  --sweep \
  --headless \
  --save_csv \
  --output_dir experiments/carrybox_perturb_debug/results/pi
```

这三次运行的条件矩阵相同，Trial ID 相同，只有 Actor 参数不同。后续比较时按 `trial_id` 配对即可。

## 常用参数说明

`--task carrybox_perturb`

指定 baseline config 来源。当前 evaluator 只支持 `carrybox_perturb`。

`--resume_path <checkpoint.pt>`

指定要加载的 Actor checkpoint。只加载 `actor.*` 和 `std`，Critic 跳过。

`--profile half_sine`

短脉冲扰动，用 `--pulse_duration` 控制持续时间。

`--profile smooth_hold`

持续外力跟随测试，用 `--hold_duration` 控制恒定 hold 时间。

`--direction=+box_x`

box-local 正 x 方向。正方向不一定需要等号，但建议统一使用等号。

`--direction=-box_y`

box-local 负 y 方向。负方向必须使用等号。

`--beta 0.5`

力大小系数，实际峰值力为 `0.5 * m_box * g`。

`--seed 1`

单 trial 使用的随机种子。

`--headless`

不打开 viewer。适合服务器或批量评估。

`--save_csv`

保存 trace CSV 和 summary CSV。

`--sweep`

自动生成条件矩阵。仍然只加载一个 checkpoint。

`--directions=+box_x,-box_x`

自定义 sweep 方向列表。

`--betas=0.1,0.3`

自定义 sweep beta 列表。

`--seeds=1,2`

自定义 sweep seed 列表。

`--hold_durations=0.5,1.0,2.0`

自定义 smooth_hold 的 hold duration 列表。

`--output_dir <dir>`

指定输出目录。

`--verbose`

打开详细 debug 输出，包括 fixed-scene dump、CarryGate、force viewer check 等。默认不要开，避免刷屏。

## 默认终端输出

默认输出保持安静，只显示 trial 状态切换和结果摘要，类似：

```text
============================================================
Trial T0001
profile=smooth_hold
direction=-box_y
beta=0.500
hold=1.000s
seed=1
============================================================
[STATE]
WAIT_CARRY -> CONFIRMED_CARRY
[STATE]
CONFIRMED_CARRY -> PRE_FORCE
[FORCE ON]
direction=-box_y
world_direction=(...)
beta=0.500
box_mass=...
target_force=...N
ramp_up=0.150s
hold=1.000s
ramp_down=0.150s
[FORCE OFF]
duration=...
impulse=...Ns
[RESULT]
physical_failure=no
contact_loss=...
box_displacement_along_force=...
robot_displacement_along_force=...
max_hand_box_relative_speed=...
final_confirmed_carry=...
```

## CSV 输出

启用 `--save_csv` 后，输出结构为：

```text
<output_dir>/
  traces/
    T0001.csv
    T0002.csv
    ...
  summary.csv
```

`summary.csv` 至少包含：

- `trial_id`
- `checkpoint`
- `seed`
- `profile`
- `direction`
- `beta`
- `hold_duration`
- `pulse_duration`
- `box_mass`
- `peak_force_N`
- `impulse_Ns`
- `physical_failure`
- `termination_reason`
- `contact_loss`
- `max_hand_box_relative_speed`
- `box_displacement_along_force`
- `robot_displacement_along_force`
- `final_confirmed_carry`
- `task_success`
- `object2goal_distance_final`
- `initial_state_signature_sha1`
- `initial_state_signature_json`

`traces/Txxxx.csv` 保存每个物理 substep 的外力、箱体响应、接触、手-箱相对运动和已有 force/contact instrumentation。

## Viewer 箭头语义

viewer 里的红色箭头表示真实瞬时物理外力：

- 方向等于真实施加的 `force_world`
- 长度只受可视化比例 `debug_force_draw_scale_m_per_N` 影响
- 改变箭头比例不会改变物理外力
- `smooth_hold` 下箭头会自然 ramp up、hold、ramp down

评估 viewer 默认不使用 force 结束后的假箭头 hold。

## 不修改的 baseline 文件

这个 evaluator 隔离在 `experiments/carrybox_perturb_debug/` 下，不需要修改：

- `legged_gym/legged_gym/envs/g1/carrybox.py`
- `legged_gym/legged_gym/envs/g1/carrybox_PI.py`
- `legged_gym/legged_gym/envs/g1/carrybox_boxperturb.py`
- `legged_gym/legged_gym/envs/g1/carrybox_boxperturb_config.py`
- `legged_gym/legged_gym/envs/g1/carrybox_config.py`
- `legged_gym/legged_gym/envs/g1/carrybox_config_PI.py`
- `legged_gym/legged_gym/scripts/play.py`
- `legged_gym/legged_gym/scripts/play_ActorOnly.py`
- PPO / Critic / training pipeline

## 最常见错误

### 1. `--direction -box_y` 报错

原因：`-box_y` 以 `-` 开头，被 Bash/argparse 当成 option。

正确写法：

```bash
--direction=-box_y
```

### 2. `python3: command not found`

使用你的环境里的 Python：

```bash
python experiments/carrybox_perturb_debug/evaluate.py ...
```

或者先激活 Conda 环境。

### 3. `ModuleNotFoundError: isaacgym`

说明当前 Python 环境不是 Isaac Gym 环境。先激活安装了 Isaac Gym 的环境。

### 4. 服务器没有显示器

使用：

```bash
--headless
```

如果需要保存结果，同时加：

```bash
--save_csv --output_dir <dir>
```
