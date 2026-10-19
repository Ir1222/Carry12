# Carry12 Teacher Warm-up + Smooth Handoff

分支：`research/teacher-warmup-handoff`，基于本地 `velocity-tracking`。本实现保留两个 Actor，不做蒸馏，不训练 Teacher，不启用 AMP 优化器。当前交付是实现与 CPU 验证；**尚未在 Isaac Gym 中训练出或验证新的物理策略**。

## 1. 审计和实际 checkpoint

本地实际文件：

| Actor | checkpoint | Actor MLP | 原 Critic | AMP |
|---|---|---|---|---|
| Full Teacher | `shared_artifacts/2026-10-19/carry_track/model_33999.pt` | 738,512,256,256,29 | 126D | 有 |
| Original Loco | `shared_artifacts/2026-10-19/track_vel_fina/model_19999 (2).pt` | 738,512,256,256,29 | 126D | 无 |

注意本地 Student 文件含空格和 `(2)`；没有重命名或覆盖它。`checkpoint_audit.json` 包含实际 SHA-256、std 范围、全部关节名及 observation/control contract。

一个 Actor frame 是 108D proprio + 15D task，连续六帧共 738D：angular velocity(3)、gravity(3)、q-default(29)、dq(29)、end effectors(15)、上一执行 action(29)；task 为 box position(3)、box orientation 6D(6)、size(3)、command(3)。历史始终由原 Full CarryBox 的 observation builder 更新。vy 在 Teacher 阶段保持原来的清零语义；Student 控制后恢复 raw vx/vy/yaw。vx/vy 是 pelvis yaw frame，yaw rate 是 world Z。

源代码使用 ELU；历史 state_dict 没有 activation、normalization 或关节语义元数据，因此权重尺寸只能验证结构，不能独立证明历史语义。加载器接受 Sequential ELU MLP、严格检查全部 actor 参数，并从 checkpoint 推断 hidden widths；Teacher 和 Student 可具有不同宽度。运行时还将 Gym DOF 顺序与 `resources/config/joint_id.txt` 逐项比对。

PD control：simulation dt=0.005s，decimation=4，policy dt=0.02s（50Hz）。`q_target = default_q + 0.25 * applied_action`。新任务明确关闭 actuation delay；原两个任务的配置不变。

## 2. 文件清单

新增环境与配置：

- `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup.py`
- `legged_gym/legged_gym/envs/g1/carrybox_teacher_warmup_config.py`

新增学习和控制模块：

- `rsl_rl/rsl_rl/teacher_handoff.py`：独立 Actor loader、joint groups、PD target adapter。
- `rsl_rl/rsl_rl/teacher_handoff_reference.py`：真实姿态 reference schema 校验。
- `rsl_rl/rsl_rl/storage/teacher_warmup_rollout_storage.py`：mask、GAE、有效 minibatches。
- `rsl_rl/rsl_rl/algorithms/teacher_warmup_ppo.py`：复用 CarryLocomotionPPO 的 PPO update，加入 imitation。
- `rsl_rl/rsl_rl/runners/teacher_warmup_on_policy_runner.py`：异步收集、TensorBoard、save/resume。

新增 scripts：`train_teacher_warmup_loco.py`、`eval_teacher_warmup_loco.py`、`analyze_teacher_warmup.py`、`build_teacher_posture_reference.py`、`audit_teacher_handoff.py`，均位于 `legged_gym/legged_gym/scripts/`。

新增实验文件：本说明、`checkpoint_audit.json`、`config_4090.json`、`VERIFICATION.md`、`tests/test_teacher_warmup.py`。

修改五个既有文件：

- `envs/__init__.py`：增加独立 task registration。
- `utils/task_registry.py`：阻止新任务误用 generic AMP runner，要求专用 masked PPO 入口；旧任务不受此 guard 影响。
- `algorithms/carry_locomotion_ppo.py`：增加默认返回 0 的 auxiliary-loss hook；原 loss 行为不变。
- `algorithms/him_ppo.py`：把 Muon import 移入启用 Muon 的分支。
- `utils/utils.py`：把 GitPython import 移入 `store_code_state`，便于独立 CPU 测试。

原 Full/Loco 环境、配置、训练脚本、ActorCritic、旧 storage 和旧 evaluator 未修改。

## 3. 架构、控制权和数据流

新环境通过现有 Locomotion 类继承 Full CarryBox，复用 locomotion 的 geometry/reward helpers；显式调用 Full 的 actor/box reset 和 carry-stage logic。每个 episode 从 Full 的 default reset 开始，Teacher 在真实仿真中 approach/pickup。该 reset 会随机放置箱体；**交接期间不调用任何 simulator state setter**。

每个环境独立维护：

1. `TEACHER_WARMUP`：Frozen Teacher mean 控制，累计稳定门限，不进入 PPO policy/value/entropy/smoothness loss。
2. `HANDOFF_TRANSITION`：Student 已控制，adapter 可启用，属于有效 PPO transition。稳定至少 1s 后进入 carry；不稳定则等待，最多 3s，超时作为失败终止。
3. `STUDENT_CARRY`：Student 全程控制，保留 velocity/contact/preservation/posture/regularization rewards。45s 独立 carry horizon，支持更长 evaluation。

门限包括双手接触、箱底相对平台顶部/地面的 lift、box tilt、pelvis-box world relative velocity、box angular velocity 和 robot-box distance。严格门限积累稳定时间；在 1.25 倍宽松门限内暂停积累，越过宽松门限清零。交接后不因门限抖动反复切回 Teacher。hand contact 使用现有 rubber-hand net contact force，因此它仍不是精确的 hand-box contact-pair 估计。

Actor 仍是 738D。新 Critic 是 **160D**：126D 原 privileged obs + phase one-hot(3) + phase age(1) + handoff age(1) + previous applied action(29)。Critic 和 Adam 从新初始化，Student Actor 和 29D std 从 Loco checkpoint 恢复。Teacher 通过独立 loader 冻结，AMP state 不加载到优化器。

每个 transition 存 raw sampled action、其原始 Normal log probability/mean/std、实际 applied action、PD target、control phase、teacher mean、imitation 时间权重和 Student-valid mask。确定性 rate limiter 属于环境动力学；PPO likelihood ratio 始终针对 raw sample，不把裁剪后的 action 误当 raw sample。

adapter 唯一递推状态是上一执行 action，它已在现有 observation history 中。每步先裁剪 raw action，再按 `rate * dt / action_scale` 限制 action 差；转换后的 action 交给原 PD 路径。arms/waist 的变化可比 legs 更慢。limiter 贯穿 Student 控制，训练与评估一致。没有实现可选 smooth_blend。

GAE 使用显式 next-state value；真正失败 bootstrap=0，成功 carry timeout 使用**reset 前 terminal privileged observation** bootstrap。done、Teacher slot 和 rollout 边界都不串接 Student GAE。末步也参与 PPO。minibatches 只 gather 有效 Student slots，所有 PPO/entropy/smoothness/auxiliary losses 都不会接触 Teacher slots。有效数量低于阈值则清空该轮 buffer、跳过 Adam 更新。没有随机 episode age 初始化。

resume 恢复 Actor、Critic、std、Adam、iteration 和 Python/NumPy/Torch/CUDA RNG，校验 teacher/reference 内容哈希、动作接口、loss/reward 和环境配置。它从新的 Teacher episode 开始；**不保存 PhysX solver/contact cache，不是物理轨迹的逐位续接**。允许改变并行环境数，拒绝悄悄更改 adapter 或 loss。

## 4. 新参数与默认值

`--handoff_config file.json` 可覆盖 `handoff`、`algorithm`、`rewards` 三个 section；`rewards` 是 reward scales。未知 key 会报错。所有有效配置写入新 run 的 `config.json` 和 checkpoint contract。

| 参数 | 默认 | 含义 |
|---|---:|---|
| warmup_timeout_s | 45 | Full approach/pickup 最大时间；失败终止 |
| transition_duration_s / transition_timeout_s | 1 / 3 | Student 过渡最短稳定时长 / 最长预算 |
| student_timeout_s | 45 | 独立 Student carry horizon；成功截断 |
| stable_s / hysteresis_ratio | 0.3 / 1.25 | 准入稳定积累时间 / 抖动容差 |
| lift_min_m | 0.10 | 箱底超过支撑面的高度 |
| tilt_max_deg | 20 | handoff 允许 tilt |
| relative_velocity_max / box_omega_max | 0.30m/s / 1rad/s | handoff 允许相对速度 / 箱体角速度 |
| warmup_vx / fixed_command | 0.4 / [0.4,0,0] | Teacher command / 初始 Student command |
| command_suite | fixed | mixed 时采样 stand/vx/vy/yaw/mixed，沿用 Loco 概率和范围，每 4–6s 重采样 |
| adapter | hard | A5 为 rate_limited；A1 frozen baseline 也可启用 |
| target_rate_arms / waist / legs | 2 / 1.5 / 12 rad/s | 分组 PD target rate；可用 joint_target_rates 的 29D 数组覆盖 |
| imitation_window_s | 2 | handoff 后线性衰减至 0 |
| imitation_max_vy / yaw / vx | 0.15 / 0.20 / 0.8 | 仅正常前进且门限稳定时监督；负 vx 不监督 |
| handoff_imitation_coef | 0.5（A3+） | PPO minibatch 中的附加 loss 系数 |
| imitation_arms / waist / legs_weight | 2 / 2 / 0.25 | weighted MSE；可用 imitation_joint_weights 的 29D 数组覆盖 |
| carry_upper_body_pose scale | 0（A4+ 为 0.5） | 持续 posture penalty 系数 |
| posture_box_size_tolerance_m | 0.04 | reference size coverage 的额外容差；超出 coverage 不施加该 prior |
| posture_penalty_cap | 10 | 单步归一化姿态 penalty 上限 |
| learning_rate | 5e-5 | 新 Adam 学习率 |
| min_valid_samples | 64 | 少于此有效数量不更新；4090 示例改为 2048 |
| num_steps_per_env / learning epochs / minibatches | 100 / 5 / 4 | 复用原 Loco PPO |

实际 lower-body 和 hand-box preservation terms 延续 Loco 配置。姿态 prior 采用真实 Teacher joint quantile interval；区间内为零 penalty，超出区间按 softness 渐进惩罚，不改变 URDF arm limits。它在整个 Student carry 生效，不随 imitation 时间窗口消失。当前 reference 是按 box-size coverage 验证的 pooled 分布，不是按 command/grasp 条件训练的回归器；不同箱体/抓取应独立采集和训练比较。

## 5. 环境准备、CPU 验证和小规模 smoke

以下训练/Viewer 命令在**已有可运行 Isaac Gym 的 Linux/CUDA 环境**、仓库根目录执行。当前 Windows Python 3.11 CPU 环境不能运行 Isaac Gym。先使用原项目已配置好的训练环境。

```bash
export PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl:$PYTHONPATH"
FULL="shared_artifacts/2026-10-19/carry_track/model_33999.pt"
LOCO="shared_artifacts/2026-10-19/track_vel_fina/model_19999 (2).pt"
python -m unittest discover -s experiments/teacher_warmup/tests -v
```

先 4 env / 3 iterations 检查启动和有限值。仅 6s 模拟时间，Teacher 若尚未完成 pickup，`valid=0, skipped=1` 是正确结果，不是 PPO 故障。

```bash
python legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --student_checkpoint "$LOCO" --ablation A2 --num_envs 4 \
  --max_iterations 3 --seed 1 --rl_device cuda:0 --headless
```

启动通过后，以相同命令改为 `--max_iterations 50`，覆盖约 100s 模拟时间，检查完整 pickup→handoff→student→reset、有效 PPO 更新、reason counters 和有限 value loss。这些命令尚未在本机 Isaac Gym 执行成功。

## 6. 从实际 checkpoint 训练与 resume

先固定 `[0.4,0,0]`，分别比较 A2/A3；4090 建议先 256 env 检查显存和 contact，再 512，最后测试 1024。`config_4090.json` 只是待实测起点，并未测出吞吐或最优超参。

```bash
python legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --student_checkpoint "$LOCO" --ablation A3 --command_suite fixed \
  --handoff_config experiments/teacher_warmup/config_4090.json \
  --num_envs 512 --max_iterations 5000 --seed 1 --rl_device cuda:0 --headless
```

新 run 路径由 `[Run] ...` 打印，输出在 `legged_gym/logs/carrybox_teacher_warmup/`。所有训练均新建唯一目录，save 拒绝覆盖已存在的 checkpoint。

设置 `RUN` 为实际打印的目录；以下恢复 A3、同一 config。**不要同时传 student_checkpoint**。`max_iterations` 表示额外执行的迭代数。

```bash
RUN="legged_gym/logs/carrybox_teacher_warmup/实际目录_A3_fixed"
python legged_gym/legged_gym/scripts/train_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --resume --resume_path "$RUN/model_100.pt" --ablation A3 --command_suite fixed \
  --handoff_config experiments/teacher_warmup/config_4090.json \
  --num_envs 512 --max_iterations 1000 --seed 1 --rl_device cuda:0 --headless
```

读取 TensorBoard：

```bash
tensorboard --logdir legged_gym/logs/carrybox_teacher_warmup
```

包括 `Phase/*`、`Handoff/valid_ratio, valid_samples, skipped_update, handoff_imitation_loss, imitation_active_ratio, gap_arms, gap_waist, gap_legs`、`Loss/*`、`TrackingRMSE/*`、`Physics/*` 和完整 episode reason/pickup/warmup counters。action gap 单位是 normalized action；乘以 0.25 可得到对应 target rad 差。无 reference 时不伪报 posture error。

## 7. 收集真实 Teacher 姿态分布

A4/A5 必须先完成采集。Teacher-only evaluator 仍使用同一真实 pickup、门限和 episode 流程，交接后继续调用 Full mean。统计工具仅接收 `full_teacher_physical_pickup` manifest，只选择完整成功 episode 中 handoff 后 ≥1s 且满足严格稳定门限的样本。

```bash
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --student_checkpoint "$LOCO" --ablation A0 --collect_teacher \
  --num_envs 4 --episodes 20 --carry_seconds 45 --seed 1 --headless \
  --output experiments/teacher_warmup/results/teacher_seed1
python legged_gym/legged_gym/scripts/build_teacher_posture_reference.py \
  experiments/teacher_warmup/results/teacher_seed1 \
  --output experiments/teacher_warmup/references/full_teacher_upper.json \
  --margin 0.05 --softness 0.20
```

生成的是全部 17 个 waist/shoulder/elbow/wrist joints 的 P05/P95 + 0.05rad margin；区间外 softness=0.20rad。最少 3 个成功 episode、100 个稳定样本；拒绝 student 数据、NaN、错误关节、错误 Teacher hash 和覆盖现有输出。没有达标数据时应改善采集设置或调查 Teacher 失败原因，不能填入假统计。可调 margin/softness 后重新采集比较，保留手腕/肩肘接触调整自由度。

A4/A5 的训练命令在第 6 节基础上改 `--ablation A4` 或 `A5`，并增加：

```bash
--posture_reference experiments/teacher_warmup/references/full_teacher_upper.json
```

先独立从原 Loco 初始化做受控 ablation。选定配方后，再通过 `--student_checkpoint 上一阶段模型.pt` 开新的 fine-tuning run 扩展 mixed suite；这个操作只继承 Actor/std，会重新初始化 Critic/Adam。需要保留整个训练状态时使用 resume，且配置必须一致。

## 8. Viewer、batch evaluation、A0–A5

Viewer A0（不加 headless）：

```bash
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --student_checkpoint "$LOCO" --ablation A0 --num_envs 1 --episodes 3 \
  --carry_seconds 45 --seed 1 --output experiments/teacher_warmup/results/A0_viewer
```

Viewer A5：将 Student 换成实际新训练模型，传训练时使用的相同 config 和 reference。

```bash
MODEL="legged_gym/logs/carrybox_teacher_warmup/实际目录_A5_fixed/model_5000.pt"
python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
  --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
  --student_checkpoint "$MODEL" --ablation A5 --num_envs 1 --episodes 3 \
  --handoff_config experiments/teacher_warmup/config_4090.json \
  --posture_reference experiments/teacher_warmup/references/full_teacher_upper.json \
  --carry_seconds 45 --seed 1 --output experiments/teacher_warmup/results/A5_viewer
```

Batch baseline，固定 seed 1/2/3，每个 seed 从 pickup 开始 100 episodes：

```bash
for SEED in 1 2 3; do
  for ABLATION in A0 A1; do
    python legged_gym/legged_gym/scripts/eval_teacher_warmup_loco.py \
      --task carrybox_teacher_warmup --teacher_checkpoint "$FULL" \
      --student_checkpoint "$LOCO" --ablation "$ABLATION" --num_envs 16 \
      --episodes 100 --carry_seconds 45 --seed "$SEED" --headless \
      --output "experiments/teacher_warmup/results/${ABLATION}_seed${SEED}"
  done
done
```

Batch 新模型使用上面的 A5 命令，加 `--headless`、`--num_envs 16 --episodes 100`，每个 seed 使用独立 output。长 carry 将 `--carry_seconds` 改为 120。该参数是 Student carry phase 的预算，另外还有最多 3s transition；survival 的计时始终从第一次 Student action 的 handoff 时刻开始。

| Ablation | Student checkpoint | PPO | imitation | persistent prior | adapter |
|---|---|---|---|---|---|
| A0 | original Loco | 不训练 | 无 | 无 | hard |
| A1 | original Loco | 不训练 | 无 | 无 | rate_limited |
| A2 | 独立 A2 run | 有 | 无 | 无 | hard |
| A3 | 独立 A3 run | 有 | 有 | 无 | hard |
| A4 | 独立 A4 run | 有 | 有 | 有 | hard |
| A5 | 独立 A5 run | 有 | 有 | 有 | rate_limited |

A2–A5 使用相同 seed、原 Loco 初始化、命令 suite 和训练预算；reference 使用同一个已验证文件。A0/A1 绝不能换成微调后的 Student。评估会检查新 checkpoint 的 Teacher hash、joint mapping、action scale、adapter 和 rate limits，避免训练和测试的执行路径不一致。

固定命令测试分别传 `--fixed_command 0 0 0`、`0.4 0 0`、`0 0.3 0`、`0 0 0.4`、`0.4 0.2 0.3`。扩展训练与随机测试用 `--command_suite mixed`，沿用 Loco 原 stand/vx/vy/yaw/mixed sampling。Teacher 阶段始终使用 warmup_vx，不向 pre-carry 输入 lateral command。

每次 evaluation 输出：

- `manifest.json`：seed、两个 checkpoint hash、配置、关节名、reference 信息。
- `steps.csv`：每步 q/dq/raw student action/applied action/实际 PD target、控制权、box/contact/velocity、command/actual、action group gaps、posture error 和失败原因。每个 episode 的 terminal pre-reset 状态也保留。
- `summary.json`：完整 pickup success、eligibility、conditional survival/completion、end-to-end completion、3/15/30s survival、vx/vy/yaw RMSE、双手 contact ratio、tilt/angular/relative velocity、每关节 target jump、相对 handoff 时间窗口和 student sample ratio。

确定性 evaluation 没有采样分布，CSV 的 `raw_log_probability` 留空；训练的真实 raw sample log probability 存于 rollout storage，与 applied action/PD target 分开记录。posture reference 未加载时 CSV posture error 留空。

```bash
python legged_gym/legged_gym/scripts/analyze_teacher_warmup.py \
  experiments/teacher_warmup/results/A0_seed1
```

成功率分母包含所有请求的完整 pickup episodes。终止原因：1=physical failure，2=warmup timeout，3=transition timeout，4=成功 carry horizon timeout。未结束的 CSV 会被分析器拒绝。不同 seed 的 summary 应报告逐 seed 均值/离散度；不把只成功交接的样本当总体成功率。

## 9. 验证、限制和下一步顺序

详细测试结果见 `VERIFICATION.md`。CPU 测试覆盖真实 checkpoint 前向、原始 raw action density、异步 GAE/timeout、mask-only minibatches、imitation 梯度、adapter rad/s bound、history 保留、单环境 reset、persistent prior、数据采集过滤及 runner 三次更新→save→resume→再更新。

尚未验证：PhysX rollout、实际 pickup gate 到达率、4090 显存/吞吐、reward 权重是否改善姿态、3DoF tracking 是否保持、rate limiter 对腿部平衡和 contact impulse 的影响。该 limiter 限制输入 target 的离散变化，不保证接触动力学零冲击。参考分布是 pooled box-size family，新的 grasp/尺寸条件可能需要独立 reference；旧 Loco waist prior 与 Teacher prior 的冲突也应在数据上检查。

建议顺序：CPU 检查 → 4-env 启动/50-iteration 物理 smoke → A0/A1 固定 vx baseline → 真实 Teacher reference 采集 → A2/A3 固定 vx 训练与比较 → A4/A5 固定 vx 比较 → 最佳配方扩展 mixed → 3 seeds、45/120s 全流程评价。只有 rollout 结果通过后，再把该 Dual Policy 作为未来单 Actor 蒸馏的 Teacher。
