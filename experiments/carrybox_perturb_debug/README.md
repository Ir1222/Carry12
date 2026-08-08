# carrybox_perturb_debug 使用指南

这个目录是 `carrybox_perturb` 的独立 debug/evaluation layer，用来检查 box perturbation 是否真正 trigger、schedule、commit、apply。它不复制 `carrybox_PI.py` / `carrybox_boxperturb.py`，也不要求修改正式 task 的核心实现。

## 当前结构

```text
experiments/carrybox_perturb_debug/
├── play_debug.py
├── configs/
│   ├── __init__.py
│   └── debug_config.py
└── envs/
    ├── __init__.py
    └── carrybox_perturb_debug.py
```

继承关系：

```text
carrybox_PI.LeggedRobot
-> carrybox_boxperturb.LeggedRobot
-> experiments/carrybox_perturb_debug/envs/carrybox_perturb_debug.LeggedRobot
```

`play_debug.py` 会在当前 Python 进程里临时注册 `carrybox_perturb_debug` task。这个注册不是写入 `legged_gym/legged_gym/envs/__init__.py` 的永久注册，所以删除整个 `experiments/carrybox_perturb_debug/` 后，不会留下新的 debug task coupling。

## 标准运行命令

PI-trained Actor：

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/logs/Jul09_from_55500/model_73500.pt
```

official CarryBox Actor：

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt
```

也可以写成：

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb_debug --resume_path legged_gym/resources/ckpt/carrybox.pt
```

`--task carrybox_perturb` 和 `--task carrybox_perturb_debug` 在这个脚本里都会使用 debug subclass。保留 `--task carrybox_perturb` 是为了命令习惯和 baseline config 来源清晰。

## 当前默认 debug 配置

默认配置集中在 `configs/debug_config.py`：

```python
DEBUG_NUM_ENVS = 1
DEBUG_EPISODE_LENGTH_S = 15
DEBUG_STABLE_CARRY_STEPS = 5
DEBUG_FORCE_EVENT = True
DEBUG_DRAW_FORCE = True
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 5
```

`apply_debug_config(env_cfg)` 会在 `task_registry.make_env()` 之前执行。这个时机很重要，因为 `carrybox_boxperturb._init_buffers()` 里会读取：

```python
self.debug_viz = bool(self.cfg.box_perturbation.debug_draw_force)
```

所以 `debug_draw_force` 必须在 env 初始化前设置。

当前 override：

```text
env.num_envs = 1
env.episode_length_s = 15
env.test = True
domain_rand.disturbance = False
domain_rand.delay = False
domain_rand.push_robots = False
asset.box.random_props = False
asset.box.reset_mode = "default"
box_perturbation.enabled = True
box_perturbation.debug_force_event = True
box_perturbation.debug_draw_force = True
box_perturbation.stable_confirmed_carry_policy_steps = 5
box_perturbation.debug_carry_gate_log_interval_policy_steps = 5
```

## 日志怎么看

debug subclass 只做 diagnostics / instrumentation，不改 perturb 算法。

### `[CarryGate]`

每隔 `DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS` 个 policy step 打印一次 env 0 的 carry gate 状态：

```text
[CarryGate]
step=...
env=0
carry_phase=...
confirmed=...
streak=...
projected_streak_before_update=...
eligible_before_decision=...
decision_made=...
event_count=...
stage=...
probability=...
clearance=...
height_gate=...
static_gate=...
left_contact=...
right_contact=...
left_contact_norm_N=...
right_contact_norm_N=...
rel_lin_vel=...
box_ang_vel=...
```

如果一直没有 perturb，先看：

```text
height_gate 是否 True
static_gate 是否 True
left_contact / right_contact 是否同时 True
confirmed 是否 True
streak 是否达到 threshold
eligible_before_decision 是否 True
```

当前 confirmed carry 的原始链条是：

```text
box clearance / height gate
+ static gate
-> carry_phase_buf

left hand contact
+ right hand contact
-> both_contact

carry_phase_buf & both_contact
-> confirmed_carry_buf
-> confirmed_carry_streak
```

### `[PerturbSchedule]`

env 0 被 scheduler 选中时立即打印：

```text
[PerturbSchedule]
step=...
env=0
eligible=True
stage=...
probability=...
decision_made=...
event_count=...
threshold=...
debug_force_event=...
scheduled=True
```

如果 `[CarryGate]` 里 `eligible_before_decision=True`，但没有 `[PerturbSchedule]`，通常说明 `debug_force_event=False` 且 Bernoulli 没采中，或者 env 0 不是被 schedule 的 env。

### `[PerturbCommit]`

`_commit_box_perturbation()` 后立即打印：

```text
[PerturbCommit]
step=...
env=0
label=...
event_count=...
beta=...
peak_force_N=...
direction_world=[...]
applied_force_world=[...]
applied_magnitude_N=...
elapsed_steps=...
remaining_steps=...
```

注意：commit 刚结束时，`applied_force_world` 可能还是 0，因为真正的 force 是在后续 physics substep 里由 `_apply_box_perturbation_force()` 写入。

### `[PerturbApplied]`

force pulse 开始和结束时打印：

```text
[PerturbApplied]
phase=start 或 end
step=...
env=0
force_world=[...]
magnitude_N=...
peak_N=...
beta=...
direction_world=[...]
elapsed_steps=...
remaining_steps=...
```

这个日志用于确认物理力是否真的写进 `box_perturb_force_tensor` 并通过 Isaac Gym force tensor API apply。

## 可以通过命令行改变的内容

这些适合临时切换，不需要改代码。

### 换 checkpoint

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path <checkpoint>
```

只改变 Actor checkpoint，debug env/config 不变。脚本复用 `play_ActorOnly.load_actor_only_for_inference()`，只加载：

```text
actor.*
std
```

不会加载 checkpoint critic。

### 指定随机种子

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt --seed 1
```

用于复现实验中的随机 stage direction / beta / Bernoulli sample。

### 临时改变 env 数量

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt --num_envs 4
```

注意：虽然 `debug_config.py` 默认 `DEBUG_NUM_ENVS = 1`，但 `task_registry.make_env()` 内部还会调用 `update_cfg_from_args()`，所以命令行 `--num_envs` 会覆盖 debug config。

当前日志主要盯 env 0。多 env 可以用来提高触发概率或观察并行行为，但日志不会逐个 env 全量打印。

### headless 运行

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt --headless
```

用于只看日志，不看 viewer arrow。`debug_draw_force=True` 仍然可以保留，但 headless 下不会显示 viewer。

### 切换设备

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt --rl_device cuda:0
```

或 CPU 调试：

```powershell
python experiments/carrybox_perturb_debug/play_debug.py --task carrybox_perturb --resume_path legged_gym/resources/ckpt/carrybox.pt --rl_device cpu
```

CPU 是否可用取决于本机 Isaac Gym / PhysX 环境。

## 可以通过 debug_config.py 改变的测试效果

这些是 experiment-only 固定配置，适合写在 `configs/debug_config.py`。

### 1. 更快确认 trigger pipeline

当前就是这个模式：

```python
DEBUG_STABLE_CARRY_STEPS = 5
DEBUG_FORCE_EVENT = True
```

含义：

```text
confirmed_carry_streak >= 5 后 eligible
eligible 后 probability 强制为 1.0
```

用途：验证 trigger -> schedule -> commit -> apply 是否通。

### 2. 接近正式配置

把 threshold 改回正式值：

```python
DEBUG_STABLE_CARRY_STEPS = 20
```

用途：确认正式 gating 下是否能自然触发。

### 3. 测 Bernoulli / stage probability

关闭强制 event：

```python
DEBUG_FORCE_EVENT = False
```

此时 probability 来自当前 stage：

```text
C1: 0.25
C2: 0.40
C3: 0.50
C4: 0.60
```

用途：检查不是“强制触发”时，真实 staged schedule 是否能触发。

### 4. 固定测试某个 stage

可以在 `apply_debug_config()` 里加入：

```python
env_cfg.box_perturbation.manual_stage_override = "C1"
```

或：

```python
env_cfg.box_perturbation.manual_stage_override = "C4"
```

用途：

```text
C1: 只测 box local x 方向，较小 beta
C4: 测 x/y/z_world 混合方向，更大 beta 范围
```

### 5. 调大/调小力强度分布

可以只在 experiment config 里改 stage beta，例如：

```python
env_cfg.box_perturbation.stages["C1"]["beta"] = {"x": (0.20, 0.30)}
```

用途：如果 viewer arrow 太小或物理效果不明显，可以先在 debug experiment 里放大 beta。这个会改变 perturb 分布，只建议用于 debug/evaluation，不要回写 baseline config。

### 6. 改 pulse duration

在 `apply_debug_config()` 里加入：

```python
env_cfg.box_perturbation.pulse_duration_s = 0.20
```

用途：让 force pulse 持续更久，更容易在 viewer 和日志里确认。

### 7. 只看日志，不画 arrow

```python
DEBUG_DRAW_FORCE = False
```

用途：避免 viewer line 干扰，只通过 `[PerturbApplied]` 判断是否真实 apply。

### 8. 改 viewer arrow 显示尺度

在 `apply_debug_config()` 里加入：

```python
env_cfg.box_perturbation.debug_force_draw_scale_m_per_N = 0.20
env_cfg.box_perturbation.debug_force_arrow_hold_s = 2.0
```

用途：物理 force 不变，只让 viewer arrow 更长、停留更久。

### 9. 改日志频率

```python
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 1
```

每个 policy step 都打印 carry gate，适合短时精查。日志会很多。

```python
DEBUG_CARRY_GATE_LOG_INTERVAL_POLICY_STEPS = 20
```

降低刷屏。

### 10. 改 episode 长度

```python
DEBUG_EPISODE_LENGTH_S = 30
```

用途：如果 15 秒内经常还没进入 confirmed carry，可以拉长 episode。

## 可以通过 play_debug.py 改变的测试效果

这些属于运行脚本行为，不属于 env 算法。

### 改 commanded velocity

当前固定：

```python
env.commands[:, 0] = 0.8
env.commands[:, 1] = 0.0
env.commands[:, 2] = 0.0
```

如果想测试慢走：

```python
env.commands[:, 0] = 0.3
```

如果想测试转向或侧向：

```python
env.commands[:, 1] = 0.2
env.commands[:, 2] = 0.3
```

用途：不同 command 可能改变是否稳定 carry、是否更容易丢箱、perturb 后是否恢复。

### 改 rollout 时长倍数

当前：

```python
for i in range(10 * int(env.max_episode_length)):
```

这表示最多跑 10 个 episode length 的 step 数。可以改成：

```python
for i in range(2 * int(env.max_episode_length)):
```

用于短跑快速看 trigger。

## 不建议在这个 experiment layer 做的事

除非明确要做新的实验，不建议改：

```text
Actor observation definition
privileged observation definition
critic architecture
PPO runner / training pipeline
carry phase 原始判定逻辑
box perturb force generator 实现
GLOBAL_SPACE / COM force apply 逻辑
baseline task registry 的永久注册
```

这个目录的定位是：隔离地改变 config、增加日志、做 actor-only playback evaluation。

## 常见判断路径

### 情况 A：viewer 里看不到 arrow

先看日志：

```text
有没有 [PerturbSchedule]
有没有 [PerturbCommit]
有没有 [PerturbApplied]
```

如果有 `[PerturbApplied]` 且 `magnitude_N > 0`，说明物理 force 已经 apply。viewer 看不到通常是：

```text
arrow scale 太小
pulse 太短
headless
debug_draw_force 没在 make_env 前设置
viewer camera 没看到 box
```

可以尝试：

```python
env_cfg.box_perturbation.debug_force_draw_scale_m_per_N = 0.20
env_cfg.box_perturbation.debug_force_arrow_hold_s = 2.0
```

### 情况 B：一直没有 schedule

看 `[CarryGate]`：

```text
height_gate=False      -> 箱子没有达到 carry clearance
static_gate=False      -> box/robot 相对速度或 box 角速度太大
left_contact=False     -> 左手接触不足
right_contact=False    -> 右手接触不足
confirmed=False        -> carry_phase 或 both_contact 没满足
streak 太小            -> threshold 没到
decision_made=True     -> 本 episode 已经做过 perturb decision
event_count>=max       -> 本 episode event 数已达上限
```

### 情况 C：schedule/commit 有，但 apply force 为 0

重点看：

```text
remaining_steps
peak_force_N
direction_world
[PerturbApplied] phase=start
```

commit 当下 `applied_force_world` 可以是 0，这是正常的，因为 force 在 physics substep 里写入。

## 最小删除恢复原则

这个 experiment layer 的目标是可删除。删除：

```text
experiments/carrybox_perturb_debug/
```

后，debug subclass、debug config、debug player 都会消失。只要没有手动把这里的改动复制回 `legged_gym/legged_gym/envs/__init__.py` 或核心 env 文件，baseline `carrybox_perturb` 行为不会因为这个目录残留而改变。
