import json
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SRC = Path('D:/D_Downloads/Sep18_loco_11000.csv')
d = pd.read_csv(SRC)
axes = ['vx', 'vy', 'yaw_rate']
valid = d.measurement_steps.gt(0)
complete = d.trial_completed.eq(1)
assert d.trial_id.is_unique
assert (d.loc[complete, 'measurement_steps'] == 150).all()
for a in axes:
    v = d.loc[valid]
    assert np.allclose(v[f'{a}_rmse']**2, v[f'{a}_bias']**2 + v[f'{a}_actual_std']**2, atol=1e-6)

rows = []
for name, mask in [('all_observed', valid), ('completed', complete), ('failed_observed', valid & ~complete)]:
    g = d.loc[mask]
    n = g.measurement_steps
    for a in axes:
        rows.append(dict(group=name, axis=a, trials=len(g), steps=int(n.sum()),
            mae=float(np.average(g[f'{a}_mae'], weights=n)),
            rmse=float(np.sqrt(np.average(g[f'{a}_rmse']**2, weights=n))),
            bias=float(np.average(g[f'{a}_bias'], weights=n))))
pd.DataFrame(rows).to_csv(OUT / 'tracking_metrics.csv', index=False)
mode = d.groupby('mode', sort=False).agg(trials=('trial_id','size'), completed=('trial_completed','sum'))
mode['completion_rate'] = mode.completed / mode.trials
mode.to_csv(OUT / 'completion_by_mode.csv')
data = {'source': str(SRC), 'records': json.loads(d.to_json(orient='records')), 'metrics': rows}
(OUT / 'plot_data.json').write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

lines = ['# Sep18 model_11000 速度跟踪分析', '',
    '结论：前进速度已有一定跟踪能力，低速侧移与转向响应较弱；搬运稳定性尚不可靠。', '',
    f'数据来源：{SRC.name}；43 个工况，单一 seed=1、carry_motion_id=0、carry_phase=0.5。预热 0.2 s，计划测量 3 s（150 步）。',
    '本文件只有逐试验汇总，没有时间序列。实际均值 ± 标准差表示该次测量窗口内的速度波动，并非置信区间。不能据此计算延迟、超调或收敛时间。', '',
    '## 完成率与失效', '',
    '完成 32/43（74.4%）；倾斜超限 8 次，箱体掉落 3 次。所有失败试验均在累计 0.90 s 内终止。',
    'T0015、T0022 的测量步数为 0：计入完成率分母，跟踪指标保留缺失，不补零。其他失败试验的测量窗口仅 0.02–0.68 s。', '',
    '| 模式 | 完成/总数 | 完成率 |', '|---|---:|---:|']
for m,r in mode.iterrows():
    lines.append(f'| {m} | {int(r.completed)}/{int(r.trials)} | {r.completion_rate:.1%} |')
lines += ['', '## 跟踪指标', '',
    '误差为 actual − target。跨试验 MAE、bias 按 measurement_steps 加权；RMSE = sqrt(sum(n × RMSE²)/sum(n))。不对试验 P95 再平均并称其为总体 P95。',
    '完成组指标仅描述完成试验；所有有效数据的指标也会因提前终止而受到截断偏差影响，必须结合完成率阅读。', '',
    '| 数据组 | 轴 | MAE | RMSE | Bias |', '|---|---|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['group']} | {r['axis']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['bias']:+.4f} |")
lines += ['', 'vx、vy 单位为 m/s；yaw_rate 单位为 rad/s，不能直接比较不同单位的绝对误差大小。', '',
    '## 具体表现', '',
    '- 纯前进：指令 0.1、0.4、1.2 m/s 的实际均值分别为 0.143、0.375、1.058 m/s，均完成；0.9 m/s 却在 0.26 s 终止，说明可用性并非随速度单调变化，不能认定整个前进区间稳定。',
    '- 纯后退：−0.5、−0.25 m/s 均提前倾斜终止；短窗口实际均值分别为 +0.043、+0.092 m/s。只能判定这些试验没有建立可持续后退，不能将其当作后退稳态速度。',
    '- 纯侧移：+0.2 → +0.009 m/s、−0.1 → −0.016 m/s，低速响应不足；+0.4 → +0.380、−0.4 → −0.306 m/s 相对更好。纯侧移完成试验同时出现 +0.063 至 +0.178 m/s 的非指令前向漂移。',
    '- 纯转向：+0.1 → +0.0067、+0.25 → +0.0013、−0.5 → −0.154 rad/s；−0.1 → +0.0115 rad/s。多数完成工况也没有充分跟随目标转速。+0.5 rad/s 工况提前掉落，不能视为稳态反向响应。',
    '- 混合指令的 19 个完成试验中，vx 通常比 yaw 更接近目标；例如 T0025=(0.96, −0.32, +0.4)，实际均值=(0.812, −0.237, +0.100)。单个较好工况 T0024 的 yaw 为 −0.4 → −0.376，说明转向能力依赖运动组合，不能简单归为完全没有转向能力。',
    '- 静止工况也有前向均速 0.0818 m/s；测量窗口首末位置距离 0.260 m、航向变化约 12.1°，静止保持仍存在漂移。', '',
    '## 判断边界与下一步', '',
    '当前结果支持“yaw 指令响应不足 + 横向低速响应不足 + 搬运早期失败”的判断，但不能仅凭汇总表确定 reward、动作限制、初始接触或坐标系中的哪项是根因。',
    '建议下一轮优先记录逐步 command/actual 三轴速度、box tilt、双手接触、termination，并对相同指令重复多个 seed 和 carry phase。重点复测纯 yaw、低速 vy、纯后退，以及表现不一致的 vx=0.9/1.2。',
    '图中没有将不同试验连接为时间曲线；没有将未完成工况删除或用 0 替代缺失。当前仅一个检查点，无法判断训练收敛趋势。', '',
    '## 图件', '',
    '- tracking_overview.png / .svg：逐试验三轴指令与实际均值 ± 标准差。',
    '- command_response.png / .svg：单轴与混合指令响应散点。',
    '- stability_diagnostics.png / .svg：完成率、误差与搬运稳定性。']
(OUT / 'analysis_zh.md').write_text('\n'.join(lines), encoding='utf-8')
print(pd.DataFrame(rows).to_string(index=False))
print(mode.to_string())
