# carryWith AMP reference 与 velocity tracking 审查

审查日期：2026-09-19。范围：当前工作区 `--task carrybox` 的代码和三个 carryWith `.pt` 文件；没有启动 Isaac Gym、训练策略或据此宣称训练因果关系。训练源码、配置和原始数据均未修改。

## 结论

**有速度信息，而且确实进入 AMP 判别器。** 不仅 `.pt` 保存速度字段，MotionLib 还从根位置和姿态重新计算速度。当前 AMP 是包含绝对运动速度的动作先验，并非完全与速度无关的姿态先验。它不会覆盖 policy 的 command observation，但可能在优化目标上与 velocity tracking 竞争。

最有依据的风险是：AMP 与 tracking 的实际奖励量级不同；carrywith2 按当前坐标定义主要后退，而所有 vx command 为正；carryWith 转动分布较宽，而 command 多数要求零转动或缓慢转动。另有确定的角速度定义问题：欧拉角导数被当成 world angular velocity 再转到 body frame。

这些证据说明值得优先进行消融，不能单独证明 AMP 是当前 tracking 不佳的唯一或主要原因。

## 文件内容与有效数据源

三个文件均为 dict，全部 tensor 为 float32，未发现 NaN/Inf。无 command、target velocity、fps 或 timestamp 字段。60 Hz 来自当前配置；保存的线速度、关节速度与 60 Hz 前向差分一致，支持这一解释。

| 字段 | 形状 | 当前路径的实际用途 |
|---|---|---|
| base_height | N | AMP 基座高度 |
| base_position | N×3 | 重新差分生成 world linear velocity；RSI 位置 |
| base_quat | N×4 | 姿态、速度坐标转换、重新生成角速度 |
| base_linear_velocity | N×3 | 文件内存在，但当前 MotionLib 不读取该键 |
| base_angular_velocity | N×3 | 文件内存在，但当前 MotionLib 不读取该键 |
| joint_position | N×29 | AMP 关节位置及 RSI |
| joint_velocity | N×29 | 读入 motion_dof_vel，用于 RSI；不直接拼入 AMP |
| link_position | N×6×3 | loader 取前五个末端位置，再转到 body frame |
| box_pos_local | N×3 | loader 再转到 body frame，进入 AMP |
| box_height_global | N | 当前 MotionLib 不读取该键 |

`base_linear_velocity` 与 `60 * diff(base_position)` 的 RMSE 依次为 1.66e-6、3.40e-6、1.95e-6 m/s（排除没有后继帧的末帧）。因此原始线速度数值与世界坐标位置差分一致，不能直接视为机器人前向速度。

关节速度均非零，三份文件的全部关节元素范围分别为 [-9.659, 11.881]、[-3.756, 6.011]、[-10.607, 15.617] rad/s；与 60 Hz 关节位置前向差分的 RMSE 均小于 5e-7 rad/s。

## 实际训练路径

`envs/__init__.py:36-55` 将 carrybox 注册到 `carrybox.py` 与 `carrybox_config.py`。当前实际使用 `carrybox_no_relocation.yaml`，而不是同目录的 `carrybox.yaml`；前者没有 putDown。

关键源码：

- [配置与 AMP 尺寸](D:/AAAProject/0806/PhysHSI/legged_gym/legged_gym/envs/g1/carrybox_config.py)：347-359 行。
- [reference 预处理与专家窗口采样](D:/AAAProject/0806/PhysHSI/legged_gym/legged_gym/envs/motionlib/motionlib_carrybox.py)：109-163、297-326 行。
- [policy AMP 特征](D:/AAAProject/0806/PhysHSI/legged_gym/legged_gym/envs/g1/carrybox.py)：157-185 行。
- [判别器训练取样](D:/AAAProject/0806/PhysHSI/rsl_rl/rsl_rl/algorithms/him_ppo.py)：250-257 行。

一帧 AMP observation 为 60 维，下标为 Python 左闭右开：

| 切片 | 特征 | 维数 |
|---|---|---:|
| 0:1 | base_height | 1 |
| 1:30 | dof_pos | 29 |
| 30:45 | end_effector_pos | 15 |
| 45:48 | box_pos | 3 |
| 48:51 | base_lin_vel，完整 body frame | 3 |
| 51:54 | base_ang_vel，完整 body frame | 3 |
| 54:60 | 去除 heading 的姿态 6D 表示 | 6 |

10 帧共 600 维。policy 为 50 Hz，窗口首尾相距 0.18 秒；reference 按 60/50 的比例插值，另有 0.95～1.05 倍时间采样随机化。即使删除显式速度，关节和末端位置的时间序列仍提供动作速率、步频等信息。

速度命令没有进入判别器；`get_expert_obs(batch_size)` 也不接收命令或当前阶段。clip 权重总和 35：pickUp 为 12/35=34.29%，carryWith 为 12/35=34.29%，loco 为 11/35=31.43%。carryWith 内部权重 5:2:5，分别为 41.67%、16.67%、41.67%。这是按 clip 权重抽样，而非按帧数。

混合技能本身不证明错误：判别器可利用姿态、箱子相对位置区分动作状态。但它没有显式区分不同 command 下应当采用什么速度分布。

## 实测速度分布

下面前向速度采用与 tracking 相同的 yaw frame：世界位置差分后，只去掉 yaw。时间为首帧至末帧 `(N-1)/60`，末帧速度按 loader 复制前一帧。

| 文件 | 帧数 | 时间/s | vx 均值/(m/s) | vx P05～P95 | vx 最小～最大 |
|---|---:|---:|---:|---|---|
| carrywith1.pt | 256 | 4.250 | 0.738 | 0.048～1.151 | -0.003～1.219 |
| carrywith2.pt | 194 | 3.217 | -0.361 | -0.678～-0.002 | -0.776～0.033 |
| carrywith3.pt | 323 | 5.367 | 0.684 | 0.217～1.038 | 0.178～1.094 |

**carrywith2 有 95.88% 的帧 vx<0。** 这是在当前姿态约定下的后退，不能通过世界坐标某个轴的正负推断，也不能把它描述成约 +0.36 m/s 的前进。

当前命令：vx 每个 episode 采样一次，范围 [0.1,1.2]；vy 恒为 0；进入 carry 时及其后每 1.5～3 秒重采样 yaw。每次 yaw 抽样 60% 概率为 0，40% 概率为正或负的 [0.1,0.4] rad/s。60% 是抽样概率，实际 rollout 时间比例还受状态和 episode 长度影响。

AMP 实际 body wz 的均值、P05/P95 分别为：

| 文件 | 均值/(rad/s) | P05～P95/(rad/s) | abs(wz)>0.4 的帧比例 |
|---|---:|---|---:|
| carrywith1.pt | 0.036 | -0.929～0.889 | 50.00% |
| carrywith2.pt | -0.004 | -0.833～0.875 | 54.64% |
| carrywith3.pt | 0.504 | -1.030～2.719 | 68.73% |

注意：AMP body wz 和 tracking world omega-z 不是完全相同的量。图中另用 quaternion 相对旋转计算 world omega-z，确认明显转动并非只由坐标变换引起。

![reference 速度曲线](D:/AAAProject/0806/PhysHSI/analysis/carrywith_amp_audit/reference_velocity.png)

调用实际 `get_expert_obs`，仅临时在审查进程中保留 carryWith 的 5:2:5 权重，固定随机种子 19，采样 20,000 个窗口，得到：

- AMP body vx 均值 0.543，中位数 0.747，P05/P95=-0.475/1.101 m/s。
- 16.25% 的采样帧 body vx<0；41.15% >0.8 m/s；13.04% >1.0 m/s。
- AMP body wz 中位数 0.097，P05/P95=-0.953/2.356 rad/s。
- 仅 7.91% 的采样帧 abs(body wz)<0.1；58.32% >0.4 rad/s。

以上不是全训练 expert pool 的统计，也不是实际策略 rollout。数据覆盖了较高前进速度，因此不能简单归因于“reference 没有高速”；问题更接近速度、转动与特定动作状态的联合分布不适配所有 command。

## 两处角速度问题需要区分

1. 原始字段的跳变尖峰：carrywith3 的第 38 帧（从 0 计数）`base_angular_velocity[38,2] = -375.1377` rad/s。原始角速度与未 unwrap 的欧拉角前向差分 RMSE 约 4.94e-6，说明它包含 yaw 跨 ±π 引起的差分跳变。当前 loader 忽略该字段并 unwrap 重算，该处变为 +1.8534 rad/s。quaternion 相对旋转得到的 world omega-z 为 +1.9018 rad/s。因此 -375 尖峰没有按当前路径直接进入 AMP。

2. 当前 loader 的定义不一致：`motion_global_ang_vel` 实际是 `[roll_dot,pitch_dot,yaw_dot]`，随后被当作 world angular velocity 进行 `R(q)^T` 变换。欧拉角导数不是世界坐标物理角速度。policy 端来自模拟器 rigid-body angular velocity，两端定义不一致。

用 `q_next * inverse(q_current)` 的最短旋转向量除以 dt 计算 world omega，再转 body frame，与当前 loader 对比，三维角速度的逐元素 RMSE 分别为 0.630、0.321、0.481 rad/s；只比较 body z 的 RMSE 分别为 0.048、0.048、0.065 rad/s（均排除复制的末帧）。误差主要不在 z，因此不能把它直接等同于 yaw tracking 的全部误差。

建议 reference 使用 quaternion 差分并明确 world/body 约定，随后重新训练或适配判别器。不能拿旧判别器对新特征分布做无控制对比。

## AMP 的实际奖励量级

`carrybox.py:1267` 把每项环境 reward scale 乘 dt。这里 dt=0.005×4=0.02。`him_on_policy_runner.py:153` 将 AMP 原始 style reward 乘 0.5；`amp.py:124-131` 再按 amp_coef=0.25 混合，没有对 AMP 同样乘 dt。

令 S 为 clamp 后的 style reward（0～1），L、Y 为已启用的线速度、yaw 跟踪指数 reward（0～1），最终每步相关贡献为：

`r = 0.125*S + 0.015*L + 0.01125*Y + 0.75*其他环境奖励`

所以 AMP 最大贡献为两项 tracking 最大贡献之和的约 4.76 倍；与单独线速度项相比为 8.33 倍。它不是“AMP 只占最终有效优化影响的 25%”。这比较的是单步系数上界，不是已测梯度比例或真实累计贡献。

例如，vy=0 时，从 vx=0.8 改为正确的 vx_cmd=0.1，线速度 reward 最多改善约 0.01289/步；只要 style score S 同时下降约 0.103，就足以抵消这一收益。这个例子说明可能的竞争条件，并不声称策略实际发生了这样的 style score 变化。

## velocity cmd / obs / tracking 接线

- `carrybox.py:451-474`：`carry_policy_commands[:,:3]` 同时进入 actor 和 critic 的 task observations。
- 每步 actor 观测为 108 proprio +15 task =123 维，command 在该步末三维（120:123），6 帧历史合计 738 维。
- `carrybox.py:768-803`：传给 actor 的 yaw 为原始采样值；heading_ref/error 仅为诊断量。
- `carrybox.py:2093-2115`：线速度 reward 比较 command 与 pelvis yaw-frame velocity；角速度 reward 比较 command 与 pelvis world omega-z。两者与 AMP 所用完整 body frame 有意区分，不能直接把 AMP 的 vx、wz 当作 tracking 指标。
- `cfg.control.upper_body_link='pelvis'` 才是这里用于选择 body 的配置，不是同名 `cfg.asset.upper_body_link='torso_link'`。
- actor 没有显式 base linear velocity，critic 有。当前 ActorCritic 是直接使用历史观测的 MLP，没有独立显式 velocity estimator。这不等于无法 tracking，但会使速度反馈依赖历史推断，尤其在接触变化、滑移或负载变化时值得做观测消融。
- tracking 不是 `is_stage_carry` 一成立就启用，而是等箱高>0.67 m、双手净接触力均>1 N、robot-box 水平距离<0.7 m 同时成立后，锁存 `carry_tracking_started=True`。此后直到 reset 都保持启用。净接触力不区分接触对象，不能将该判据直接解释为已验证双手都接触箱子。
- RSI 中 root reference 速度在赋值后又被 [-0.2,0.2] 随机速度覆盖；joint_velocity 仍用于关节初始化。不能说策略每次 reset 都继承 demo 的行走速度。

由此可见，reference 不会直接替换 command，也没有证据表明 cmd 没有进 observation；更主要的疑点在先验与任务奖励竞争、反馈信息和奖励启用条件。

## 建议的最小验证顺序

1. 固定 command 网格、初始状态和种子，按 carry_tracking_started=True 的有效时间统计 vx/vy/yaw MAE、style reward、两项 tracking reward，以及箱子掉落率和稳定搬运持续时间。按未触发 gate 的 episode 单独统计，避免把零 tracking reward 误判成跟踪失败。
2. 单变量降低 AMP 系数，或单独提高 tracking 系数，做短程续训对照；不要一次改多项。若减少 style 相对影响后 command 响应明显改善，才获得因果证据。仅在 inference 时关闭 reward 不会改变已经训练好的策略。
3. 确认只需要前进搬运时，把 carrywith2 权重置零做单独对照。再评估/分段处理强转动 reference，保留抱箱姿态与接触相关数据。
4. 做“判别器不看直接平面速度/yaw 速度”的对称特征消融：expert 与 policy 两侧同时处理，并匹配 AMP history 与 reset 初始化路径。可以先保留尺寸并一致 mask，但旧判别器需重新训练/适配。只改原始 `.pt` 的 base velocity 字段无效，因为 loader 重算；只改 expert 一侧会制造判别捷径。
5. 修复 reference quaternion 角速度定义，单独评估；长期增加慢速直行、多速度和不同转弯半径的搬运动作覆盖，或加入与 demo 实测速度一致的 command conditioning。不能给 expert 随机拼接与动作无关的 command。

去掉显式速度不等于速度完全解耦：10 帧姿态/末端序列仍约束步频。现有 ±5% 采样率随机化也不等于实现完整速度增强，而且代码改变采样间隔时没有同步缩放显式速度特征；这是另一项较小的时序一致性问题。

AMP 用任务 reward 指定目标、用动作数据构造 style reward 的一般机制可参考[作者项目页](https://xbpeng.github.io/projects/AMP/)。上面的具体数值与问题判断均来自当前仓库和本次本地测量。

## 复现与产物

执行 `python analysis/carrywith_amp_audit/audit_carrywith.py`。需要 CPU PyTorch、NumPy、PyYAML、tqdm、Matplotlib，不需要 Isaac Gym。

- `audit_carrywith.py`：通过最小 import shim 加载仓库实际 MotionLib 和 torch_utils，使用 joint_id 顺序作为审查时的 dof_names；不声称模拟器实际 joint 顺序已经运行验证。
- `summary.json`：全部字段形状、有效性、速度统计、差分一致性、原始尖峰和 sampler 统计。
- `per_frame.csv`：三个文件全部 773 帧的速度明细。
- `reference_velocity.png`：前向速度和转动曲线，绿色带为当前 command 范围。

已完成全部 `.pt` 安全权重模式读取、实际 CPU MotionLib 预处理、20,000 个 carryWith 专家窗口审查，以及恢复原权重后 1,000 个混合专家窗口的形状/finite 检查（1000×600）。未做仿真训练，因此需要上述消融来确认因果贡献。
