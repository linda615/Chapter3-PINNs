# 非均匀支承全固支顶板

单位正方形Kirchhoff薄板，D=1、ν=0.25，采用非均匀载荷和空间变化Winkler地基。挠度变换`x²(1-x)²y²(1-y)² S_w tanh(raw_w)`满足四周零挠度与零法向转角；内力分支不乘边界包络。

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.run
python -m experiments.clamped_rock_roof_nonuniform_support.evaluate --save-figures
python -m experiments.clamped_rock_roof_nonuniform_support.plot_loss_history
```

结果目录为`results_bounded_output_rms_loss`，默认评价`best_fdm_validation.weights.h5`。误差评价采用241×241参考网格，输出场CSV默认采用101×101网格。最新六场相对L₂误差（%）依次为0.1046、1.2183、1.2342、7.5747、7.3385、7.6728，平均4.1905。

输出与残差尺度见`config.py`。损失图左侧采用各阶段实际权重，右侧为未加权归一化残差。本例无软边界损失，加权物理损失等于总损失；左图用对称对数坐标显示零值边界项。

微调与无地基比较为可选试验。评价程序同时输出从挠度恢复内力的诊断结果（from_w），论文误差采用网络直接输出。
