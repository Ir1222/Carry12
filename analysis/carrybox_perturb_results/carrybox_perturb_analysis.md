# CarryBox 扰动评估数据分析报告

## 分析范围

原始结果目录：`D:\D_Downloads\results`。

这份报告分析的是 `experiments/carrybox_perturb_debug/evaluate.py` 跑出来的三组结果：`official`、`retrained`、`pi`。评估流程不是普通 play，而是一个受控扰动实验：单环境运行，先等机器人进入 `confirmed_carry`，再延迟 0.20 s，然后在箱体 COM 施加一次外力，外力结束后继续观察 2.0 s。

`summary.csv` 的核心指标来自 `experiments/carrybox_perturb_debug/evaluation/metrics.py`；每个 trial 的逐物理步细节来自 `traces/Txxxx.csv`。也就是说，这里的结论不是只看最终 reward，而是结合了失败、手箱接触、外力响应、箱体位移和 trace 级接触/速度指标。

默认 `smooth_hold` sweep 的完整矩阵是：

`3 个 seed x 4 个方向 x 6 个 beta x 3 个 hold duration = 216 trials`

其中 `official` 和 `pi` 都是完整的 216 条 `smooth_hold` 结果，可以严格对比。`retrained` 只有 72 条，并且 profile 是 `half_sine`、`hold_duration=0.0`，所以它不是同一个 sustained force 条件，只能作为参考，不能直接拿来证明 smooth-hold 抗扰更强。

## 核心结果

| 策略 | profile | trial 数 | 成功施加外力比例 | 物理失败率 | 接触丢失率 | 最终确认搬运率 | 原任务成功率 | 最终箱体到目标距离中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| official carrybox.pt | smooth_hold | 216 | 50.5% | 50.0% | 44.9% | 11.1% | 32.9% | 3.356 m |
| PI policy | smooth_hold | 216 | 67.1% | 33.3% | 43.5% | 38.4% | 6.9% | 0.590 m |
| retrained.pt | half_sine | 72 | 88.9% | 11.1% | 69.4% | 47.2% | 0.0% | 0.462 m |

严格只看 `official` 和 `pi` 的 `smooth_hold` 对比：

- PI 的物理失败率从 50.0% 降到 33.3%，绝对下降 16.7%。
- PI 的最终确认搬运率从 11.1% 提升到 38.4%，绝对提升 27.3%。
- 接触丢失率几乎没有改善：44.9% -> 43.5%。
- 原始 task success 反而下降：32.9% -> 6.9%。这里要注意，`task_success` 仍然是原 carrybox 的“箱体到目标点并放置成功”指标，不是专门为抗扰恢复设计的成功指标。

结论：PI 的主要收益是“更容易进入/维持搬运状态，并减少跌倒/重置类物理失败”；但它还没有真正解决 sustained external force 下的双手稳定夹持问题。

## 指标判定条件

下面四个结果指标来自 `experiments/carrybox_perturb_debug/evaluation/metrics.py`。它们衡量的是不同层级的行为，不是互斥的成功/失败标签。同一条 trial 可以同时发生接触丢失、最终恢复搬运且没有物理失败。

### `physical_failure`：环境终止

在 `evaluate.py` 的 rollout 中，只要环境返回 `done=True`，该 trial 就记为 `physical_failure=1`。触发 `done` 的条件来自 `carrybox_PI.py::check_termination()`，主要包括：

- 指定终止碰撞部位的净接触力超过 `10 N`；
- episode 超时；
- 头部高度低于 `0.6 m`，或 base 高度低于 `0.2 m`；
- `|roll| > 0.5 rad`，或 `|pitch| > 1.1 rad`；
- 指定刚体的水平速度超过 `3.0 m/s`；
- 任一 hip-yaw 刚体高度低于 `0.15 m`；
- 启用箱体终止条件时，箱体严重倾倒。

因此 `physical_failure` 更准确地表示“环境终止”，不只表示机器人摔倒。当前实现也把 `timeout` 计为物理失败。图中的物理失败率是 `physical_failure` 这个 0/1 字段在对应 trial 集合中的平均值，越低越好。

### `contact_loss`：响应窗口内曾丢失任一只手接触

评估器只在外力已经开始后的 `FORCE` 和 `POST_FORCE` 阶段采样左右手接触。每个 policy step 分别计算左右手刚体的净接触力范数：

`left_contact = ||F_left|| > 1.0 N`

`right_contact = ||F_right|| > 1.0 N`

只要响应窗口内存在一个采样点满足 `left_contact=0` 或 `right_contact=0`，整条 trial 就记为 `contact_loss=1`。所以它是严格的双手持续接触指标：单手短暂脱离后重新接触，仍然算发生过 contact loss；但这不等于箱子一定掉落，也不等于机器人一定终止。

这里使用的是手部刚体的 simulator net contact force，并没有进一步分离具体接触对象。因此它主要表示“手部有效接触是否中断”，不是严格的手-箱 pairwise 接触测量。越低越好。

### `final_confirmed_carry`：响应结束时仍满足有效搬运状态

该字段读取 `FORCE/POST_FORCE` 响应窗口最后一个采样点上的 `env.confirmed_carry_buf`。`confirmed_carry=1` 必须同时满足：

- 箱子底部相对平台顶面或支撑面的净空高度大于 `0.05 m`；
- 箱子相对机器人 base 的线速度小于 `1.0 m/s`；
- 箱子角速度小于 `3.0 rad/s`；
- 左右手净接触力都大于 `1.0 N`。

也就是：`confirmed_carry = lifted AND motion_stable AND left_contact AND right_contact`。

因此 `final_confirmed_carry=1` 表示扰动响应结束时，箱子仍被抬起、没有相对机器人剧烈运动或旋转，并且双手都有有效接触。它不要求箱子已经到达目标位置，也只检查最后一个采样点，不要求连续稳定若干步。若 trial 在施力前终止、没有响应采样，该字段默认记为 `0`。越高越好。

### `task_success`：原始 CarryBox 目标完成

评估配置使用 `env.test=True`。在测试模式下，`task_success=1` 要同时满足：

- `object2goal_dist_xyz < 0.1 m`，即箱子与三维目标点的距离小于 `10 cm`；
- `||projected_gravity_box_xy|| < 0.2`，即箱体倾斜程度在允许范围内。

它不要求双手始终保持接触，也不要求最终仍处于 `confirmed_carry`。所以它衡量的是原始 CarryBox 的到达/放置目标能力，而不是专门的抗扰恢复成功。越高越好。

### 比例统计的共同口径

`01_outcome_rates.png` 对每种 policy 的全部 trial 求 0/1 平均值；`02_beta_curves.png` 则先按 `beta` 分组，再对不同方向、seed 和 hold duration 下的 trial 求平均。

需要特别注意：没有达到 `confirmed_carry`、因而没有真正施加外力的 trial 也包含在总分母中。此时因为没有响应采样，`contact_loss` 默认是 `0`，而 `final_confirmed_carry` 默认是 `0`。这会压低总体接触丢失率，因此解释 `contact_loss` 时必须同时参考 `Force scheduled rate`；若要单独评价受力后的抓持鲁棒性，应另外计算“仅在成功施力 trial 中的条件接触丢失率”。

## 每张图怎么看

### 1. `01_outcome_rates.png`：三种策略的总体结果柱状图

![总体结果柱状图](01_outcome_rates.png)

这张图横轴是三种策略，纵轴是比例，范围 0 到 1。每组柱子表示一个指标：

- `Physical failure rate`：物理失败率，越低越好。它对应 episode 中出现 head_low、base_tilt、box_tilt、timeout 等导致 reset/失败的情况。
- `Contact loss rate`：接触丢失率，越低越好。只要外力响应期间左手或右手有一侧掉接触，就计为 contact loss。
- `Final confirmed-carry rate`：扰动后最终仍然满足 confirmed carry 的比例，越高越好。
- `Task success rate`：原始任务成功率，越高越好，但它不是抗扰专用成功指标。
- `Force scheduled rate`：评估器实际施加外力的比例，越高说明 policy 更常达到 `confirmed_carry` 门槛。

这张图最重要的读法是：PI 的物理失败率明显低于 official，final confirmed-carry 明显高于 official，但 task success 低于 official，contact loss 变化不大。`retrained` 因为不是 smooth-hold 条件，只能看趋势，不能直接和另外两组比较。

### 2. `02_beta_curves.png`：不同外力强度 beta 下的性能曲线

![beta 曲线](02_beta_curves.png)

横轴 `beta` 是外力强度系数，评估代码里峰值外力近似是：

`peak_force = beta * box_mass * 9.81`

四个子图分别是物理失败率、接触丢失率、最终确认搬运率、原任务成功率。纵轴都是比例。理论上 beta 越大，外力越强，性能可能越差；但实际曲线不是单调的，说明主导失败的不只是外力大小，还包括施力时刻、当时步态相位、箱体姿态、双手接触几何和 policy 当前动作。

PI 在大多数 beta 上物理失败率低于 official，最终确认搬运率高于 official；但 contact loss 仍然在多个 beta 上很高。

## 结合具体代码的解释

### 评估代码做了什么

`experiments/carrybox_perturb_debug/evaluate.py` 的 `run_trial()` 会先 reset 到指定 seed，然后不断执行 policy。只有当 `confirmed_carry_streak` 达到阈值后，才进入 `PRE_FORCE`，再等待 `pre_force_delay=0.20 s`，然后调用 `env.schedule_evaluation_force()` 施加外力。

对于 `smooth_hold`，外力 profile 是：

1. 半余弦 ramp up；
2. 常值 hold；
3. 半余弦 ramp down。

外力方向是箱体局部坐标下的 `+box_x`、`-box_x`、`+box_y`、`-box_y`，峰值力约等于 `beta * box_mass * 9.81`。评估配置里关闭了 disturbance、delay、push_robots 和随机箱体属性，所以这批结果主要反映 policy 本身和接触动力学的差异。

### PI policy 实现改了什么

PI 版本保持 actor 输入兼容：actor observation 仍是 738 维。因此 checkpoint 能用 actor-only 方式加载评估。

真正变化主要在训练侧和环境侧：

- critic privileged observation 变成 143 维，其中末尾加入 17 维 interaction privileged 信息；
- 这 17 维包括箱体线速度、箱体角速度、左右手接触力、箱体接触力、左右手接触 flag；
- 新增 carry-phase 检测：箱体离平台高度、箱体相对机器人速度、箱体角速度、左右手接触共同决定 `carry_phase_buf` 和 `confirmed_carry_buf`；
- reward shaping 加入 `bimanual_contact=0.35`、`single_hand_contact=0.05`、`hand_box_relative_motion=-0.15`；
- hand-box relative motion penalty 使用 0.35 m/s deadband，超过后惩罚相对滑动。

这解释了为什么 PI 的 final confirmed-carry 和物理稳定性变好：训练信号确实在鼓励双手接触和低相对滑动。但 contact loss 仍然高，说明奖励和 privileged 信息还没有转化成足够可靠的力闭合策略。

## retrained.pt 的注意事项

`retrained.pt` 目录不是同一套 `smooth_hold` sweep。它是 72 条 `half_sine` trial，`hold_duration=0.0`。`half_sine` 是短脉冲，和 `smooth_hold` 的持续外力不是同一个难度。

因此，`retrained.pt` 的物理失败率低 (11.1%) 不能直接说明它在持续外力下更强。它的接触丢失率很高 (69.4%)，并且原任务成功率是 0，说明它更像是“短脉冲下不容易摔”，但没有稳定完成任务，也没有保持可靠双手接触。

## 总结判断

PI policy 的方向是有效的，但目前效果偏向 gross stability 和 carry-state robustness，而不是完整的 grasp robustness。下一步如果要继续提升，建议重点优化：

- 把 contact loss 作为主指标，而不是只看 physical failure；
- 增强外力方向下的双手法向力和切向抗滑指标；
- 对 sustained force 单独设计 recovery success，而不是继续使用原始 object-to-goal task success；
- 让训练阶段见到更接近 evaluation 的 smooth-hold sustained perturbation，而不是只靠短扰动或隐式 contact shaping。

## 生成文件

- `combined_summary.csv`：三组 summary 合并表。
- `trace_derived_metrics.csv`：从每个 trace CSV 里提取的响应指标。
- `combined_summary_with_trace_metrics.csv`：summary 和 trace 派生指标合并表。
- `overview.csv`：每个策略的总览指标。
- `by_beta.csv`：按 beta 聚合的指标。
- `by_direction.csv`：按外力方向聚合的指标。
- `smooth_trace_overview.csv`：smooth-hold trace 指标总览。
- `01_outcome_rates.png`：总体结果柱状图。
- `02_beta_curves.png`：beta 强度曲线。
