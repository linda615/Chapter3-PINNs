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
