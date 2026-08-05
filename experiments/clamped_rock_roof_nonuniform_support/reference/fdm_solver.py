"""Thirteen-point finite-difference solver for a clamped Kirchhoff plate.

The solver is intentionally independent of TensorFlow and the PINN training
code. It solves

    D * biharmonic(w) + k(x, y) * w = q(x, y)

on a uniformly spaced square grid. Boundary deflection is eliminated with
``w = 0``. The clamped rotation condition is imposed with centered ghost
points, for example ``w[-1, j] = w[1, j]`` at ``x = 0``. This is the standard
second-order virtual-point discretization of ``dw/dn = 0`` and preserves the
13-point biharmonic stencil next to the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg


Array = np.ndarray
ScalarField = Union[float, Array, Callable[[Array, Array], Array]]

_BIHARMONIC_STENCIL = (
    (0, 0, 20.0),
    (1, 0, -8.0),
    (-1, 0, -8.0),
    (0, 1, -8.0),
    (0, -1, -8.0),
    (1, 1, 2.0),
    (1, -1, 2.0),
    (-1, 1, 2.0),
    (-1, -1, 2.0),
    (2, 0, 1.0),
    (-2, 0, 1.0),
    (0, 2, 1.0),
    (0, -2, 1.0),
)


@dataclass(frozen=True)
class FDMResult:
    """Finite-difference solution and recovered Kirchhoff physical fields."""

    x: Array
    y: Array
    w: Array
    Mx: Array
    My: Array
    Mxy: Array
    Qx: Array
    Qy: Array
    q: Array
    k: Array
    D: float
    nu: float
    relative_residual: float

    @property
    def grid_size(self) -> int:
        """Return the number of nodes along one coordinate direction."""
        return int(self.x.size)

    @property
    def fields(self) -> Dict[str, Array]:
        """Return the six fields using the same names as ``MultiSubNetPINN``."""
        return {
            "w": self.w,
            "Mx": self.Mx,
            "My": self.My,
            "Mxy": self.Mxy,
            "Qx": self.Qx,
            "Qy": self.Qy,
        }


def assemble_clamped_plate_system(
    grid_size: int,
    D: float,
    load: ScalarField,
    foundation: ScalarField = 0.0,
    x_limits: Tuple[float, float] = (0.0, 1.0),
    y_limits: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[sparse.csr_matrix, Array, Array, Array, Array, Array]:
    """Assemble the 13-point clamped-plate sparse linear system.

    Returns:
        ``(matrix, rhs, x, y, q, k)``. The unknown vector contains only the
        ``(grid_size - 2)^2`` interior deflections in row-major order.
    """
    grid_size = _validate_problem_definition(grid_size, D, x_limits, y_limits)
    x = np.linspace(x_limits[0], x_limits[1], grid_size, dtype=np.float64)
    y = np.linspace(y_limits[0], y_limits[1], grid_size, dtype=np.float64)
    spacing = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing="xy")
    q = _evaluate_scalar_field("load", load, X, Y, require_nonnegative=False)
    k = _evaluate_scalar_field("foundation", foundation, X, Y, require_nonnegative=True)

    interior_size = grid_size - 2
    unknown_count = interior_size * interior_size
    rows = []
    columns = []
    values = []
    stencil_scale = float(D) / spacing**4

    def unknown_index(i: int, j: int) -> int:
        return (j - 1) * interior_size + (i - 1)

    for j in range(1, grid_size - 1):
        for i in range(1, grid_size - 1):
            row = unknown_index(i, j)
            for di, dj, coefficient in _BIHARMONIC_STENCIL:
                target_i = i + di
                target_j = j + dj

                if (
                    1 <= target_i <= grid_size - 2
                    and 1 <= target_j <= grid_size - 2
                ):
                    column = unknown_index(target_i, target_j)
                else:
                    ghost = _map_clamped_ghost(
                        target_i,
                        target_j,
                        grid_size,
                    )
                    if ghost is None:
                        # A physical boundary node has prescribed w = 0.
                        continue
                    column = unknown_index(ghost[0], ghost[1])

                rows.append(row)
                columns.append(column)
                values.append(stencil_scale * coefficient)

            rows.append(row)
            columns.append(row)
            values.append(float(k[j, i]))

    matrix = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(unknown_count, unknown_count),
        dtype=np.float64,
    ).tocsr()
    matrix.sum_duplicates()
    rhs = np.ascontiguousarray(q[1:-1, 1:-1].reshape(-1))
    return matrix, rhs, x, y, q, k


def solve_clamped_plate(
    grid_size: int,
    D: float,
    nu: float,
    load: ScalarField,
    foundation: ScalarField = 0.0,
    x_limits: Tuple[float, float] = (0.0, 1.0),
    y_limits: Tuple[float, float] = (0.0, 1.0),
) -> FDMResult:
    """Solve the clamped plate and recover all six PINN physical fields."""
    if not np.isfinite(nu) or not (-1.0 < float(nu) < 0.5):
        raise ValueError(f"nu must be finite and satisfy -1 < nu < 0.5, got {nu}.")

    matrix, rhs, x, y, q, k = assemble_clamped_plate_system(
        grid_size=grid_size,
        D=D,
        load=load,
        foundation=foundation,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    interior_solution = sparse_linalg.spsolve(matrix, rhs)
    if not np.all(np.isfinite(interior_solution)):
        raise RuntimeError("The finite-difference linear solve returned non-finite values.")

    denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).eps)
    relative_residual = float(
        np.linalg.norm(matrix @ interior_solution - rhs) / denominator
    )

    w = np.zeros((grid_size, grid_size), dtype=np.float64)
    w[1:-1, 1:-1] = interior_solution.reshape((grid_size - 2, grid_size - 2))
    fields = recover_plate_fields(w, x, y, D=float(D), nu=float(nu))
    return FDMResult(
        x=x,
        y=y,
        w=w,
        Mx=fields["Mx"],
        My=fields["My"],
        Mxy=fields["Mxy"],
        Qx=fields["Qx"],
        Qy=fields["Qy"],
        q=q,
        k=k,
        D=float(D),
        nu=float(nu),
        relative_residual=relative_residual,
    )


def recover_plate_fields(
    w: Array,
    x: Array,
    y: Array,
    D: float,
    nu: float,
) -> Dict[str, Array]:
    """Recover moments and shears using the PINN sign convention.

    The definitions exactly match ``physics/plate_residuals.py``:

    ``Mx = -D * (w_xx + nu*w_yy)``
    ``My = -D * (w_yy + nu*w_xx)``
    ``Mxy = -D * (1-nu) * w_xy``
    ``Qx = Mx_x + Mxy_y``
    ``Qy = My_y + Mxy_x``
    """
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if w.shape != (y.size, x.size):
        raise ValueError(
            f"w must have shape (len(y), len(x)), got {w.shape} for "
            f"len(y)={y.size}, len(x)={x.size}."
        )
    if x.size < 5 or y.size < 5:
        raise ValueError("At least five grid nodes per direction are required.")

    hx = _uniform_spacing("x", x)
    hy = _uniform_spacing("y", y)
    w_xx = _second_derivative(w, hx, axis=1)
    w_yy = _second_derivative(w, hy, axis=0)
    w_xy = _first_derivative(_first_derivative(w, hx, axis=1), hy, axis=0)

    Mx = -float(D) * (w_xx + float(nu) * w_yy)
    My = -float(D) * (w_yy + float(nu) * w_xx)
    Mxy = -float(D) * (1.0 - float(nu)) * w_xy
    Qx = _first_derivative(Mx, hx, axis=1) + _first_derivative(Mxy, hy, axis=0)
    Qy = _first_derivative(My, hy, axis=0) + _first_derivative(Mxy, hx, axis=1)

    return {
        "w": w,
        "Mx": Mx,
        "My": My,
        "Mxy": Mxy,
        "Qx": Qx,
        "Qy": Qy,
    }


def two_region_foundation(
    x: Array,
    y: Array,
    k1: float,
    k2: float,
    interface_x: float,
) -> Array:
    """Create a grid-aligned two-region piecewise-constant foundation.

    Region 1 includes nodes with ``x <= interface_x`` and region 2 contains
    nodes with ``x > interface_x``. The internal interface must coincide with
    an actual x-grid node; otherwise a ``ValueError`` is raised.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional coordinate arrays.")
    for name, value in (("k1", k1), ("k2", k2)):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative, got {value}.")

    spacing = _uniform_spacing("x", x)
    tolerance = max(32.0 * np.finfo(np.float64).eps, abs(spacing) * 1e-10)
    aligned_indices = np.flatnonzero(np.isclose(x, interface_x, rtol=0.0, atol=tolerance))
    if aligned_indices.size != 1:
        raise ValueError(
            f"interface_x={interface_x} must coincide with one x-grid node. "
            f"Use interface_x = x[i] for the selected grid size."
        )
    interface_index = int(aligned_indices[0])
    if interface_index == 0 or interface_index == x.size - 1:
        raise ValueError("The two-region interface must lie strictly inside the domain.")

    values = np.where(x <= x[interface_index], float(k1), float(k2))
    return np.broadcast_to(values[np.newaxis, :], (y.size, x.size)).copy()


def _map_clamped_ghost(
    i: int,
    j: int,
    grid_size: int,
) -> Union[Tuple[int, int], None]:
    """Map one-layer virtual nodes using the centered zero-slope condition."""
    if i == -1 and 1 <= j <= grid_size - 2:
        return 1, j
    if i == grid_size and 1 <= j <= grid_size - 2:
        return grid_size - 2, j
    if j == -1 and 1 <= i <= grid_size - 2:
        return i, 1
    if j == grid_size and 1 <= i <= grid_size - 2:
        return i, grid_size - 2
    return None


def _evaluate_scalar_field(
    name: str,
    field: ScalarField,
    X: Array,
    Y: Array,
    require_nonnegative: bool,
) -> Array:
    """Evaluate and validate a scalar field on the full grid."""
    if callable(field):
        values = field(X, Y)
    else:
        values = field
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full_like(X, float(values), dtype=np.float64)
    try:
        values = np.broadcast_to(values, X.shape).copy()
    except ValueError as error:
        raise ValueError(
            f"{name} must be scalar or broadcastable to grid shape {X.shape}, "
            f"got {values.shape}."
        ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if require_nonnegative and np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative everywhere.")
    return values


def _validate_problem_definition(
    grid_size: int,
    D: float,
    x_limits: Tuple[float, float],
    y_limits: Tuple[float, float],
) -> int:
    """Validate grid and plate parameters used during assembly."""
    if isinstance(grid_size, bool) or int(grid_size) != grid_size or grid_size < 5:
        raise ValueError(f"grid_size must be an integer >= 5, got {grid_size}.")
    if not np.isfinite(D) or D <= 0.0:
        raise ValueError(f"D must be finite and positive, got {D}.")
    for name, limits in (("x_limits", x_limits), ("y_limits", y_limits)):
        if len(limits) != 2 or not np.all(np.isfinite(limits)) or limits[1] <= limits[0]:
            raise ValueError(f"{name} must contain two finite increasing values, got {limits}.")

    hx = (x_limits[1] - x_limits[0]) / (int(grid_size) - 1)
    hy = (y_limits[1] - y_limits[0]) / (int(grid_size) - 1)
    if not np.isclose(hx, hy, rtol=1e-12, atol=1e-15):
        raise ValueError(
            "The isotropic 13-point stencil requires equal x and y spacing; "
            f"got hx={hx} and hy={hy}."
        )
    return int(grid_size)


def _uniform_spacing(name: str, coordinates: Array) -> float:
    """Return spacing after checking that coordinates form a uniform grid."""
    differences = np.diff(coordinates)
    if differences.size == 0 or np.any(differences <= 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing.")
    spacing = float(differences[0])
    if not np.allclose(differences, spacing, rtol=1e-12, atol=1e-15):
        raise ValueError(f"{name} coordinates must be uniformly spaced.")
    return spacing


def _first_derivative(values: Array, spacing: float, axis: int) -> Array:
    """Second-order first derivative, including one-sided boundary formulas."""
    return np.gradient(values, spacing, axis=axis, edge_order=2)


def _second_derivative(values: Array, spacing: float, axis: int) -> Array:
    """Second-order second derivative with one-sided boundary formulas."""
    moved = np.moveaxis(values, axis, -1)
    derivative = np.empty_like(moved)
    derivative[..., 1:-1] = (
        moved[..., 2:] - 2.0 * moved[..., 1:-1] + moved[..., :-2]
    ) / spacing**2
    derivative[..., 0] = (
        2.0 * moved[..., 0]
        - 5.0 * moved[..., 1]
        + 4.0 * moved[..., 2]
        - moved[..., 3]
    ) / spacing**2
    derivative[..., -1] = (
        2.0 * moved[..., -1]
        - 5.0 * moved[..., -2]
        + 4.0 * moved[..., -3]
        - moved[..., -4]
    ) / spacing**2
    return np.moveaxis(derivative, -1, axis)
