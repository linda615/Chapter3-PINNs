# 简支—固支混合边界板

单位正方形薄板，D=1、ν=0.3、q=1。x=0,1为简支边，y=0,1为固支边。挠度包络为`x(1-x)y²(1-y)²`，简支弯矩采用软约束。参考解采用40个奇数模态的Lévy级数。

```bash
python -m experiments.simply_clamped_uniform_load.ablation --name Mixed-6B --seed 2052
```

| 模型 | 参数量 | 最高空间求导阶数 |
|---|---:|---:|
| W-PINN | 16897 | 4 |
| MO4-PINN | 33862 | 4 |
| Mixed-1B | 17222 | 2 |
| Mixed-3B | 21286 | 2 |
| Mixed-6B | 33862 | 2 |
| Mixed-1B-Wide | 33576 | 2 |

`--name`选择模型，`--seed`选择2051、2052或2053。各运行训练20000轮，每200轮验证，保存验证选优和最终轮次模型，并在200×200单元中心网格上评价。结果位于`results_ablation/runs/<模型>_seed<种子>`。

三个种子的六场平均误差为：W-PINN 10.005±0.142%、MO4-PINN 16.140±1.194%、Mixed-1B 5.146±2.259%、Mixed-3B 6.266±1.612%、Mixed-6B 3.309±0.168%、Mixed-1B-Wide 3.464±0.120%。完整统计见根目录`data/paper/scsc_ablation.csv`。

本目录其他训练入口用于单次试验；论文消融结果使用上述ablation入口。

## Alpha/Beta 尺度敏感性实验

尺度实验固定使用上述 Mixed-6B、20000 epochs、三随机种子及相同的采样、学习率、损失权重和 PCGrad 设置，只改变输出尺度系数 alpha 与残差尺度系数 beta：

| 方案 | 输出尺度 alpha | 残差尺度 beta |
|---|---|---|
| `unit_alpha_beta` | 全部取 1，仅保留量纲尺度 | 物理与边界残差全部取 1，仅保留量纲尺度 |
| `alpha_half` | 当前六个 alpha 同时乘 0.5 | 保持当前 beta 不变 |
| `alpha_double` | 当前六个 alpha 同时乘 2 | 保持当前 beta 不变 |
| `alpha_quadruple` | 当前六个 alpha 同时乘 4 | 保持当前 beta 不变 |

单位板中 `q0=D=L=1`，因此 `unit_alpha_beta` 的实际输出尺度和残差尺度均为 1。每次运行会在结果目录保存 `scale_audit.json`，明确记录实际 alpha、beta 和尺度值。

先检查四组配置和一次训练步骤：

```bash
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant all --seed 2051 --check
```

正式训练的单次命令为：

```bash
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant unit_alpha_beta --seed 2051
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant alpha_half --seed 2051
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant alpha_double --seed 2051
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant alpha_quadruple --seed 2051
```

将 `--seed` 分别设为 `2051`、`2052` 和 `2053`，即可完成 12 次运行。结果相互隔离地保存在 `results_scale_ablation/runs/<方案>_seed<种子>`；已存在完整 `result.json` 的运行会自动跳过，中断运行则从 TensorFlow 状态 checkpoint 继续。

也可以用一条命令顺序运行全部 12 组（不并发占用显存）：

```bash
python -m experiments.simply_clamped_uniform_load.scale_ablation --variant all --seed all
```

全部运行完成后，汇总原 Mixed-6B 基线及四组尺度实验：

```bash
python -m experiments.simply_clamped_uniform_load.summarize_scale_ablation
```

汇总表默认写入 `data/paper/scsc_scale_ablation.csv`，包含 best/final 的六场 RelL2、六场平均值、三次运行均值和样本标准差。
