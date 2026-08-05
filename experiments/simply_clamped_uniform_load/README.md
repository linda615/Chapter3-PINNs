# Opposite Simple/Clamped Edges Under Uniform Load

This experiment solves a unit-square Kirchhoff plate under
`q(x, y) = q0`.

Boundary assignment:

- `x = 0` and `x = 1`: simply supported, `w = 0`, `Mx = 0`.
- `y = 0` and `y = 1`: clamped, `w = 0`, `dw/dy = 0`.

The default deflection transform `x(1-x)y^2(1-y)^2` enforces zero
deflection on all four edges and zero normal slope on the two horizontal
clamped edges. Set `hard_clamped_slope=False` for the former soft-slope
ablation. If the edge types are swapped, the model automatically uses
`x^2(1-x)^2y(1-y)` instead. The simple-edge moment remains a soft
boundary loss by default. Set `hard_simple_moment=True` to enable the
optional normalized transform `Mx = 4x(1-x) Mx_raw` on the two vertical
simply supported edges.

Physics residuals are divided by the corresponding analytical-field RMS
before squaring. The equilibrium residual uses the load RMS. The active
model maps every raw Head through `tanh` before hard transforms and physical
scaling. Its physical output scales are 1.1 times the analytical peak, so
the bounded Heads retain enough range while loss denominators remain the
analytical RMS. The run is saved under `results_bounded_output_rms_loss` so
checkpoints with different loss and output semantics are not mixed.
The analytical reference is a Levy odd-sine series in the x direction.

Run a short smoke test:

```bash
python -m experiments.simply_clamped_uniform_load.run --epochs 10
```

Run the default experiment:

```bash
python -m experiments.simply_clamped_uniform_load.run
```

Evaluate the best checkpoint:

```bash
python -m experiments.simply_clamped_uniform_load.evaluate
```

## PINN comparison variants

The experiment exposes three independent training/evaluation pipelines. They
share the same mixed edge assignment, collocation settings, learning-rate
schedule, Levy reference, and six-field validation score:

- **Mixed PINN**: six directly predicted fields and the low-order mixed system.
- **All-shared Mixed PINN**: one `64 x 5` MLP predicts all six fields under
  the same low-order mixed system. With PCGrad enabled, the complete MLP is
  treated as shared parameters.
- **W-PINN**: only `w` is predicted; all force fields are reconstructed from
  second- and third-order derivatives and equilibrium uses the fourth-order
  plate equation.
- **MO4-PINN**: six fields are predicted directly, while equilibrium is closed
  by the fourth-order equation in `w`.

```bash
python -m experiments.simply_clamped_uniform_load.run_mixed_pinn
python -m experiments.simply_clamped_uniform_load.run_all_shared_mixed_pinn
python -m experiments.simply_clamped_uniform_load.run_w_pinn
python -m experiments.simply_clamped_uniform_load.run_mo4_pinn
```

```bash
python -m experiments.simply_clamped_uniform_load.evaluate_mixed_pinn --save-figures
python -m experiments.simply_clamped_uniform_load.evaluate_all_shared_mixed_pinn --save-figures
python -m experiments.simply_clamped_uniform_load.evaluate_w_pinn --save-figures
python -m experiments.simply_clamped_uniform_load.evaluate_mo4_pinn --save-figures
```

W-PINN and MO4-PINN write to `results_w_pinn/` and `results_mo4_pinn/`.
The existing Mixed-PINN result directory is unchanged.

Every completed run writes `runtime.json`. Logged loss rows also contain the
mean sampling/gradient-update time for the preceding interval. Compare all
available runs with:

```bash
python -m experiments.simply_clamped_uniform_load.compare_runtimes
```
