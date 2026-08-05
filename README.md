# Mixed-variable PINNs for Kirchhoff plate bending

本仓库整理了博士论文第 3 章的可复现实验代码。模型使用共享主干与六个物理场分支，同时预测薄板挠度 `w`、弯矩 `Mx/My/Mxy` 和剪力 `Qx/Qy`，并通过 Kirchhoff 薄板方程、物理量递推关系及边界条件进行无监督训练。

> 说明：参考解仅用于训练后的精度评估（岩层顶板算例中也用于独立的检查点监控），不作为监督标签加入物理损失。

## 论文实验

| 论文算例 | 目录 | 边界与载荷 | 参考解 |
| --- | --- | --- | --- |
| 四边简支薄板 | `experiments/simply_supported_sinusoidal` | 四边简支，正弦分布载荷 | 显式解析解 |
| 混合边界薄板 | `experiments/simply_clamped_uniform_load` | 两边简支、两边固支，均布载荷 | Levy 级数解 |
| 岩层顶板 | `experiments/clamped_rock_roof_nonuniform_support` | 四边固支，非均匀载荷与 Winkler 支承 | 独立有限差分解 |

核心实现按职责划分为：`models/`（网络）、`physics/`（控制方程和自动微分）、`boundary/`（边界条件）、`sampling/`（配置点采样）、`losses/`（归一化物理损失）和 `trainer/`（Adam、PCGrad、L-BFGS）。

## 环境

建议使用 Python 3.10 或 3.11，并在仓库根目录创建独立环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需 GPU，请根据本机 CUDA 环境和 TensorFlow 官方兼容表安装对应版本。

## 快速验证

以下命令只运行很短的训练，用于验证安装与数据流，不代表论文结果：

```bash
python -m experiments.simply_supported_sinusoidal.run --epochs 1
python -m experiments.simply_clamped_uniform_load.run --epochs 1
python -m experiments.clamped_rock_roof_nonuniform_support.run --epochs 1 --fdm-evaluation-every 0
```

运行自动测试：

```bash
python -m pytest -q
```

## 复现正式实验

### 1. 四边简支正弦载荷

```bash
python -m experiments.simply_supported_sinusoidal.run
python -m experiments.simply_supported_sinusoidal.evaluate --save-figures
python -m experiments.simply_supported_sinusoidal.plot_loss_history
```

### 2. 简支/固支混合边界均布载荷

```bash
python -m experiments.simply_clamped_uniform_load.run
python -m experiments.simply_clamped_uniform_load.evaluate --save-figures
```

该目录还包含 W-PINN、MO4-PINN 和全共享网络等对比入口，详见 [`experiments/simply_clamped_uniform_load/README.md`](experiments/simply_clamped_uniform_load/README.md)。

### 3. 非均匀支承岩层顶板

仓库保留了正式算例所需的有限差分参考数据。重新生成参考解、训练和评估的命令如下：

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference
python -m experiments.clamped_rock_roof_nonuniform_support.reference.grid_convergence
python -m experiments.clamped_rock_roof_nonuniform_support.run
python -m experiments.clamped_rock_roof_nonuniform_support.evaluate --save-figures
```

完整的二阶段微调、诊断和外部求解器导出方法见 [`experiments/clamped_rock_roof_nonuniform_support/README.md`](experiments/clamped_rock_roof_nonuniform_support/README.md)。

## 输出与版本控制

训练会在各实验目录的 `results*` 下生成权重、历史记录、CSV 和图片。这些文件体积较大且可以重新生成，因此默认不会提交到 Git。根目录的 `outputs/`、缓存、临时论文文件和本机编辑器配置也已排除。岩层顶板的两份 `reference/*.npz` 是复现实验所需输入，已明确保留。

随机种子、采样点数、学习率阶段、损失权重和输出目录都集中在各实验的 `config.py` 中；每次运行还会在结果目录写出 `config.json`，用于记录实际配置。

## 引用与许可

本仓库包含作者博士论文第 3 章相关实验的研究代码。论文目前尚未最终定稿，正式的论文题目、出版年份及推荐引用格式将在论文完成后更新。