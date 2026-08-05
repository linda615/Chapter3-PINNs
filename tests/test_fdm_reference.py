"""Tests for the independent 13-point finite-difference reference solver."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from experiments.clamped_rock_roof_nonuniform_support.config import DEFAULT_CONFIG
from experiments.clamped_rock_roof_nonuniform_support.evaluate import load_reference
from experiments.clamped_rock_roof_nonuniform_support.problem import (
    foundation_stiffness,
    rock_roof_load,
)
from experiments.clamped_rock_roof_nonuniform_support.reference.fdm_solver import (
    assemble_clamped_plate_system,
    recover_plate_fields,
    two_region_foundation,
)
from experiments.clamped_rock_roof_nonuniform_support.reference.generate_reference import (
    save_reference,
    solve_reference,
)
from experiments.clamped_rock_roof_nonuniform_support.reference.grid_convergence import (
    run_manufactured_convergence,
)


def test_biharmonic_matrix_has_thirteen_point_interior_stencil() -> None:
    """A node two layers from every edge must retain all 13 stencil entries."""
    grid_size = 9
    matrix, _, _, _, _, _ = assemble_clamped_plate_system(
        grid_size=grid_size,
        D=1.0,
        load=1.0,
        foundation=0.0,
    )
    interior_size = grid_size - 2
    center_row = 3 * interior_size + 3

    assert matrix.shape == (interior_size**2, interior_size**2)
    assert matrix.getrow(center_row).nnz == 13
    assert (matrix - matrix.T).nnz == 0


def test_two_region_foundation_requires_grid_aligned_interface() -> None:
    """The material interface must coincide with one node on every target grid."""
    for grid_size in (61, 121, 241):
        x = np.linspace(0.0, 1.0, grid_size)
        y = np.linspace(0.0, 1.0, grid_size)
        k = two_region_foundation(x, y, k1=5.0, k2=0.2, interface_x=0.45)
        interface_index = int(np.flatnonzero(np.isclose(x, 0.45))[0])

        assert k.shape == (grid_size, grid_size)
        np.testing.assert_allclose(k[:, : interface_index + 1], 5.0)
        np.testing.assert_allclose(k[:, interface_index + 1 :], 0.2)

    with pytest.raises(ValueError, match="coincide with one x-grid node"):
        two_region_foundation(
            np.linspace(0.0, 1.0, 61),
            np.linspace(0.0, 1.0, 61),
            k1=5.0,
            k2=0.2,
            interface_x=0.451,
        )


def test_clamped_polynomial_manufactured_solution_is_second_order() -> None:
    """The ghost-point clamped discretization must show second-order convergence."""
    rows = run_manufactured_convergence(
        grid_sizes=(17, 33, 65),
        D=1.0,
        nu=0.25,
        foundation=lambda X, Y: 1.0 + 0.5 * X,
    )

    assert rows[0]["w_relative_l2"] > rows[1]["w_relative_l2"]
    assert rows[1]["w_relative_l2"] > rows[2]["w_relative_l2"]
    assert rows[1]["observed_order"] >= 1.9
    assert rows[2]["observed_order"] >= 1.9
    assert rows[2]["linear_system_relative_residual"] < 1e-8


def test_recovered_fields_follow_pinn_sign_convention() -> None:
    """Recovered moments and shears must match the mixed PINN definitions."""
    grid_size = 65
    x = np.linspace(0.0, 1.0, grid_size)
    y = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(x, y, indexing="xy")
    D = 1.0
    nu = 0.25

    factor = lambda values: values**2 * (1.0 - values) ** 2
    first = lambda values: 2.0 * values - 6.0 * values**2 + 4.0 * values**3
    second = lambda values: 2.0 - 12.0 * values + 12.0 * values**2
    third = lambda values: -12.0 + 24.0 * values
    w = factor(X) * factor(Y)
    fields = recover_plate_fields(w, x, y, D=D, nu=nu)

    exact = {
        "Mx": -D * (second(X) * factor(Y) + nu * factor(X) * second(Y)),
        "My": -D * (factor(X) * second(Y) + nu * second(X) * factor(Y)),
        "Mxy": -D * (1.0 - nu) * first(X) * first(Y),
        "Qx": -D * (third(X) * factor(Y) + first(X) * second(Y)),
        "Qy": -D * (factor(X) * third(Y) + second(X) * first(Y)),
    }

    interior = np.s_[2:-2, 2:-2]
    for name, exact_values in exact.items():
        relative_error = np.linalg.norm(
            fields[name][interior] - exact_values[interior]
        ) / np.linalg.norm(exact_values[interior])
        assert relative_error < 5e-3, f"{name} relative error was {relative_error}."


def test_reference_file_contains_required_fields_and_matches_problem(tmp_path) -> None:
    """Generated FDM data must be readable by the PINN evaluation module."""
    result = solve_reference(grid_size=9, foundation_mode="smooth")
    output_path = tmp_path / "rock_roof_reference.npz"
    save_reference(result, output_path, foundation_mode="smooth")
    reference = load_reference(output_path)

    assert reference is not None
    for name in ("x", "y", "w", "Mx", "My", "Mxy", "Qx", "Qy", "q", "k"):
        assert reference[name].shape == (81, 1)
    assert float(reference["D"]) == DEFAULT_CONFIG.plate.D
    assert float(reference["nu"]) == DEFAULT_CONFIG.plate.nu

    xy = tf.concat(
        (
            tf.convert_to_tensor(reference["x"], dtype=tf.float64),
            tf.convert_to_tensor(reference["y"], dtype=tf.float64),
        ),
        axis=1,
    )
    np.testing.assert_allclose(reference["q"], rock_roof_load(xy).numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        reference["k"],
        foundation_stiffness(xy).numpy(),
        rtol=1e-12,
    )
