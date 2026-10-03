# 论文结果

误差统计均以百分数表示，训练日志中的物理损失不转换为百分数。

- `scsc_ablation.csv`：三次独立运行的均值与样本标准差，分别报告验证选优和最终轮次结果。
- `scsc_scale_ablation_seed2051.csv`：随机种子2051的四组尺度设置结果，包含第20000轮最终模型和验证选优模型。
- `cccc_errors.csv`：2026年9月26日训练的验证选优模型在241×241有限差分网格上的直接输出误差。
- `cccc_loss.csv`、`cccc_fdm_validation.csv`、`cccc_config.json`：该次训练记录与配置。

跨硬件和TensorFlow版本的运行可能存在浮点差异。历史试验结果不作为当前论文结果。
