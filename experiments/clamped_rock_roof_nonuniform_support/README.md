# Nonuniform-Support Rock-Roof Example

This experiment models a rectangular rock roof as a small-deflection,
homogeneous, isotropic Kirchhoff thin plate on the nondimensional unit square.
The default problem uses a spatially varying Winkler foundation and a smooth
nonuniform transverse load.
All four edges are treated as clamped boundaries:

```text
w = 0
dw/dn = 0
```

The hard output transform is applied only to the deflection field:

```text
B(x,y) = x^2(1-x)^2y^2(1-y)^2
w(x,y) = B(x,y) S_w tanh(raw_w(x,y))
```

Both `w=0` and `dw/dn=0` are imposed by the transform. The transform is not
multiplied into `Mx`, `My`, `Mxy`, `Qx`, or `Qy`. Every Head uses a bounded
`tanh` output followed by its own physics-derived physical scale.

Physics residuals are divided by characteristic moment, shear, and load
scales before squaring. These scales use only the known load, plate rigidity,
unit-square geometry, and clamped fundamental wave number; the FDM reference
is not used to construct the training loss.

## Engineering Interpretation

The optional nonuniform Winkler foundation represents spatially varying
support from intact surrounding rock, coal pillars, or backfill. It is enabled
by default. A local Gaussian load concentration remains active to retain a
meaningful rock-roof loading pattern.

This is an engineering-equivalent plate model. It is not a full three-dimensional
discontinuous rock-mass analysis.

## Governing Residual Convention

Without a foundation, the equilibrium residual is:

```text
r_eq = Qx_x + Qy_y + q
```

With the default nonuniform foundation, it becomes:

```text
r_eq = Qx_x + Qy_y + q - k*w
```

The foundation reaction is kept separate from the external load function.

## Finite-Difference Reference

No analytical manufactured solution is used for training or checkpoint
selection in this example. The independent 13-point finite-difference solver
uses:

```text
D * biharmonic(w) + k(x,y) * w = q(x,y)
w = 0 and dw/dn = 0 on all four outer edges
```

The default nonuniform-foundation reference is saved to:

```text
reference/rock_roof_reference.npz
```

Required fields:

```text
x, y, w, Mx, My, Mxy, Qx, Qy, q, k, D, nu
```

`Mx`, `My`, `Mxy`, `Qx`, and `Qy` use exactly the same definitions and signs
as `physics/plate_residuals.py`. This result is consistently called the
finite-difference reference or FDM reference.

Generate the default 241x241 smooth-foundation FDM reference:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference
```

Generate a two-region foundation reference. The interface at `x=0.45` is
aligned with the N=241 grid:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference --foundation-mode two-region --k1 5.0 --k2 0.2 --interface-x 0.45 --output experiments/clamped_rock_roof_nonuniform_support/reference/rock_roof_reference.npz
```

Run the required N=61, 121, 241 grid-convergence analysis:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.reference.grid_convergence
```

If the reference file is absent, `evaluate.py` still exports PINN predictions
but skips relative L2 metrics.

## Commands

Default nonuniform-support training:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.run
```

Optional no-foundation ablation:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.run --no-foundation
```

The default mixed PINN matches the simply-supported uniform-load experiment:

```text
shared trunk: 64 x 4
three branches: 16 x 2
interior points: 2048
boundary points per side: 20
epochs: 20000
```

Adam uses fixed learning-rate transitions:

```text
epochs 1-3000:      1e-3
epochs 3001-7000:   5e-4
epochs 7001-12000:  1e-4
epochs 12001-20000: 3e-5
```

All normalized residual equations use unit aggregation weights:

```text
epochs 1-20000: moment=1, shear=1, equilibrium=1, boundary=1
```

Shared-trunk PCGrad is enabled from epoch 2001. Branch-specific parameters
continue to use the ordinary total-loss gradient.

During training, the six PINN fields are compared with the pre-generated FDM
reference every 200 epochs. The console reports each field's relative L2 error
and their arithmetic mean. The history is saved to:

```text
results_bounded_output_rms_loss/history/fdm_validation.csv
```

Whenever the mean six-field relative L2 score reaches a new minimum, the model
weights are saved independently from the physics-loss checkpoint:

```text
results_bounded_output_rms_loss/checkpoints/best_fdm_validation.weights.h5
```

The interval and inference batch size can be changed without altering the
training collocation points:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.run --fdm-evaluation-every 200 --fdm-evaluation-batch-size 8192
```

Set `--fdm-evaluation-every 0` to disable this diagnostic. The FDM file must
use the same `D`, `nu`, `q`, and `k` as the active training configuration.

## Adam and L-BFGS Fine Tuning

After the initial run has produced `best_fdm_validation.weights.h5`, run the
second-stage Adam optimization and third-stage L-BFGS optimization with:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.finetune
```

The default sequence is:

```text
best_fdm_validation.weights.h5
    -> Adam: lr=3e-5, at most 1500 epochs
    -> best_fdm_finetune.weights.h5
    -> fixed-collocation L-BFGS: at most 500 accepted iterations
    -> best_fdm_lbfgs.weights.h5
```

Adam fine tuning uses unit loss weights. It checks the FDM score every 200
epochs and stops after three
checks without an improvement of at least `1e-4`. Each stage evaluates and
saves its input model as iteration zero, so its best checkpoint cannot be
worse than the preceding stage.

Run only the resumed Adam stage:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.finetune --stage adam
```

Run only L-BFGS after Adam fine tuning:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.finetune --stage lbfgs
```

The principal outputs are:

```text
results/checkpoints/best_fdm_finetune.weights.h5
results/checkpoints/final_adam_finetune.weights.h5
results/checkpoints/best_fdm_lbfgs.weights.h5
results/checkpoints/final_lbfgs_iter_500.weights.h5
results/history/loss_adam_finetune.csv
results/history/fdm_validation_adam_finetune.csv
results/history/loss_lbfgs_finetune.csv
results/history/fdm_validation_lbfgs.csv
```

Evaluate the best model selected during L-BFGS with:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.evaluate --checkpoint experiments/clamped_rock_roof_nonuniform_support/results/checkpoints/best_fdm_lbfgs.weights.h5 --save-figures
```

Diagnostics:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.diagnose
```

Evaluation:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.evaluate --checkpoint experiments/clamped_rock_roof_nonuniform_support/results/checkpoints/best_fdm_validation.weights.h5 --save-figures
```

Compare the directly predicted mixed fields with fields reconstructed from
the best checkpoint's deflection derivatives:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.evaluate_auxiliary_fields
```

The command evaluates the complete FDM grid and writes:

```text
results/auxiliary_fields_from_w.csv
results/auxiliary_fields_from_w_metrics.csv
```

For each of `Mx`, `My`, `Mxy`, `Qx`, and `Qy`, the summary reports the direct
PINN error, the error after reconstruction from `w`, and the consistency error
between those two PINN representations.

Problem export for external solvers:

```bash
python -m experiments.clamped_rock_roof_nonuniform_support.export_problem
```
