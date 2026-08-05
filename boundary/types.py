"""Boundary condition type definitions."""

from __future__ import annotations

from enum import Enum


class BoundaryType(str, Enum):
    """Supported Kirchhoff thin plate boundary condition families."""

    SIMPLE = "simple"
    CLAMPED = "clamped"
    FREE = "free"
