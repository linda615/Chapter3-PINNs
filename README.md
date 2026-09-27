# Mixed-variable PINNs for Kirchhoff plate bending

博士论文第三章实验代码。网络联合预测挠度w、弯矩Mx/My/Mxy和剪力Qx/Qy，训练损失由混合物理关系、平衡方程及边界约束构成。参考解用于尺度设置、模型选择和误差评价，不作为逐点拟合标签。

## 环境与运行

论文实验使用Python 3.9、TensorFlow 2.10.1；分布式比较另使用Python 3.8、TensorFlow 2.10.0。消融入口需使用GPU。在仓库根目录运行：

```bash
python -m pip install -r requirements-paper.txt
python -m experiments.simply_supported_sinusoidal.run
python -m experiments.simply_clamped_uniform_load.ablation --name Mixed-6B --seed 2052
python -m experiments.clamped_rock_roof_nonuniform_support.run
```

第一、第三个算例训练后运行对应的`evaluate --save-figures`模块。第三例损失图由`experiments.clamped_rock_roof_nonuniform_support.plot_loss_history`生成。消融入口完成训练后自动评价。

## 实验设置

| 算例 | 输出映射 | 参考解 |
|---|---|---|
| SSSS，正弦载荷 | 线性输出、单位输出尺度 | 解析解 |
| SCSC，均布载荷 | tanh及分量输出尺度 | 40个奇数模态的Lévy级数 |
| CCCC，非均匀地基与载荷 | tanh及分量输出尺度 | 241×241有限差分参考解 |

三个算例均训练20000轮，域内配置点数为2048。SSSS每边20个边界点；SCSC每边80个候选点；CCCC通过挠度输出变换满足全固支边界，不设置对应软边界损失。PCGrad自第5001轮启用，仅处理共享参数的任务梯度。

消融模型为W-PINN、MO4-PINN、Mixed-1B、Mixed-3B、Mixed-6B和Mixed-1B-Wide，各使用2051、2052、2053三个种子。每200轮在101×101网格上选优，在200×200单元中心网格上测试，采用float32并关闭TF32。跨硬件标准差包含种子与执行环境差异，耗时只在同硬件下比较。

最新CCCC六场平均相对L₂误差为**4.1905%**。结果表及该次训练记录见[data/paper](data/paper)。运行配置保存为`config.json`，权重、场数组和图片保存在各算例的`results*`目录，不纳入版本控制。SCSC消融使用`results_ablation`，CCCC使用`results_bounded_output_rms_loss`。

## 方法来源

- Raissi et al., Journal of Computational Physics, 2019, 378: 686–707. DOI: 10.1016/j.jcp.2018.10.045.
- Yu et al., Gradient Surgery for Multi-Task Learning, NeurIPS, 2020.
- Timoshenko and Woinowsky-Krieger, Theory of Plates and Shells, 2nd ed., 1959.

本仓库为作者博士论文相关研究代码，论文引用信息将在定稿后补充。
