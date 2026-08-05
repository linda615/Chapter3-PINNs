"""Boundary condition residual definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from physics.autodiff import deflection_derivatives

from .data import BoundaryBatch, BoundaryResiduals
from .operators import normal_moment, normal_shear, normal_slope


class BoundaryCondition(ABC):
    """Base interface for Kirchhoff plate boundary residuals."""

    @abstractmethod
    def compute_residuals(
        self,
        model: tf.keras.Model,
        batch: BoundaryBatch,
        training: bool = False,
    ) -> BoundaryResiduals:
        """Compute boundary residual tensors."""


class SimplySupportedBoundary(BoundaryCondition):
    """Simply supported boundary: ``w = 0`` and ``M_n = 0``."""

    def compute_residuals(
        self,
        model: tf.keras.Model,
        batch: BoundaryBatch,
        training: bool = False,
    ) -> BoundaryResiduals:
        batch = batch.validated()
        fields = model(batch.xy, training=training)
        return BoundaryResiduals(
            deflection=fields["w"],
            moment=normal_moment(fields, batch.normals),
        )


class ClampedBoundary(BoundaryCondition):
    """Clamped boundary: ``w = 0`` and ``dw/dn = 0``."""

    def compute_residuals(
        self,
        model: tf.keras.Model,
        batch: BoundaryBatch,
        training: bool = False,
    ) -> BoundaryResiduals:
        batch = batch.validated()
        w_derivatives = deflection_derivatives(model, batch.xy, training=training)
        return BoundaryResiduals(
            deflection=w_derivatives["w"],
            slope=normal_slope(w_derivatives, batch.normals),
        )


class FreeBoundary(BoundaryCondition):
    """Free boundary: ``M_n = 0`` and ``Q_n = 0``.

    The natural free-edge convention is centralized here. If the dissertation
    sign convention or effective shear definition changes, this class should be
    updated rather than the loss or trainer code.
    """

    def compute_residuals(
        self,
        model: tf.keras.Model,
        batch: BoundaryBatch,
        training: bool = False,
    ) -> BoundaryResiduals:
        batch = batch.validated()
        fields = model(batch.xy, training=training)
        return BoundaryResiduals(
            moment=normal_moment(fields, batch.normals),
            shear=normal_shear(fields, batch.normals),
        )
