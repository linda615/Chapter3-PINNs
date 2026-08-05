"""Independent multi-subnetwork PINN model for Kirchhoff thin plates."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math

import tensorflow as tf

from .subnet import IndependentSubNet


FIELD_NAMES = ("w", "Mx", "My", "Mxy", "Qx", "Qy")
OUTPUT_ACTIVATION_LINEAR = "linear"
OUTPUT_ACTIVATION_TANH = "tanh"
OUTPUT_ACTIVATION_NAMES = (
    OUTPUT_ACTIVATION_LINEAR,
    OUTPUT_ACTIVATION_TANH,
)


@dataclass(frozen=True)
class FieldOutputScales:
    """Positive scales mapping dimensionless Head outputs to physical fields."""

    w: float = 1.0
    Mx: float = 1.0
    My: float = 1.0
    Mxy: float = 1.0
    Qx: float = 1.0
    Qy: float = 1.0

    def __post_init__(self) -> None:
        """Require finite positive scales for every physical field."""
        for name, value in self.as_dict().items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Field output scale {name} must be finite and positive, "
                    f"got {value}."
                )

    def as_dict(self) -> dict[str, float]:
        """Return scales in the canonical field order."""
        return {
            "w": self.w,
            "Mx": self.Mx,
            "My": self.My,
            "Mxy": self.Mxy,
            "Qx": self.Qx,
            "Qy": self.Qy,
        }
DEFLECTION_TRANSFORM_NONE = "none"
DEFLECTION_TRANSFORM_CUSTOM = "custom"
DEFLECTION_TRANSFORM_UNIT_SQUARE = "unit_square_zero_boundary"
DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE = "unit_square_clamped"
DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y = "unit_square_simple_x_clamped_y"
DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y = "unit_square_clamped_x_simple_y"
DEFLECTION_TRANSFORM_NAMES = (
    DEFLECTION_TRANSFORM_NONE,
    DEFLECTION_TRANSFORM_CUSTOM,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE,
    DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
    DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
)
MOMENT_TRANSFORM_NONE = "none"
MOMENT_TRANSFORM_SIMPLE_X_EDGES = "simply_supported_x_edges"
MOMENT_TRANSFORM_SIMPLE_Y_EDGES = "simply_supported_y_edges"
MOMENT_TRANSFORM_SIMPLE_ALL_EDGES = "simply_supported_all_edges"
MOMENT_TRANSFORM_NAMES = (
    MOMENT_TRANSFORM_NONE,
    MOMENT_TRANSFORM_SIMPLE_X_EDGES,
    MOMENT_TRANSFORM_SIMPLE_Y_EDGES,
    MOMENT_TRANSFORM_SIMPLE_ALL_EDGES,
)

GROUPINGS: dict[str, tuple[tuple[str, ...], ...]] = {
    "one": (FIELD_NAMES,),
    "two": (("w", "Mx", "My", "Mxy"), ("Qx", "Qy")),
    "three": (("w",), ("Mx", "My", "Mxy"), ("Qx", "Qy")),
    "six": (("w",), ("Mx",), ("My",), ("Mxy",), ("Qx",), ("Qy",)),
}

GROUP_NET_NAMES: dict[str, tuple[str, ...]] = {
    "one": ("net_all",),
    "two": ("net_w_moment", "net_shear"),
    "three": ("net_w", "net_moment", "net_shear"),
    "six": ("net_w", "net_Mx", "net_My", "net_Mxy", "net_Qx", "net_Qy"),
}


class MultiSubNetPINN(tf.keras.Model):
    """A configurable PINN made of fully independent field subnetworks."""

    def __init__(
        self,
        grouping: str = "six",
        hidden_width: int = 64,
        hidden_depth: int = 4,
        activation: str | tf.keras.layers.Layer = "tanh",
        hidden_units: Sequence[int] | None = None,
        use_shared_trunk: bool | None = None,
        shared_width: int | None = None,
        shared_depth: int = 2,
        shared_units: Sequence[int] | None = None,
        apply_deflection_transform: bool = False,
        deflection_transform_name: str = DEFLECTION_TRANSFORM_NONE,
        deflection_transform: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None = None,
        apply_moment_transform: bool = False,
        moment_transform_name: str = MOMENT_TRANSFORM_NONE,
        field_output_scales: FieldOutputScales | Mapping[str, float] | None = None,
        output_activation: str = OUTPUT_ACTIVATION_LINEAR,
        name: str = "multi_subnet_pinn",
    ) -> None:
        """Create the independent multi-subnetwork PINN."""
        if grouping not in GROUPINGS:
            valid = ", ".join(f'"{key}"' for key in GROUPINGS)
            raise ValueError(f"Unsupported grouping {grouping!r}. Expected one of: {valid}.")
        if deflection_transform_name not in DEFLECTION_TRANSFORM_NAMES:
            valid = ", ".join(f'"{key}"' for key in DEFLECTION_TRANSFORM_NAMES)
            raise ValueError(
                "Unsupported deflection_transform_name "
                f"{deflection_transform_name!r}. Expected one of: {valid}."
            )
        if moment_transform_name not in MOMENT_TRANSFORM_NAMES:
            valid = ", ".join(f'"{key}"' for key in MOMENT_TRANSFORM_NAMES)
            raise ValueError(
                "Unsupported moment_transform_name "
                f"{moment_transform_name!r}. Expected one of: {valid}."
            )
        if output_activation not in OUTPUT_ACTIVATION_NAMES:
            valid = ", ".join(f'"{key}"' for key in OUTPUT_ACTIVATION_NAMES)
            raise ValueError(
                f"Unsupported output_activation {output_activation!r}. "
                f"Expected one of: {valid}."
            )
        if apply_moment_transform and moment_transform_name == MOMENT_TRANSFORM_NONE:
            raise ValueError(
                "A built-in moment_transform_name must be selected when "
                "apply_moment_transform=True."
            )
        if use_shared_trunk is None:
            use_shared_trunk = len(GROUPINGS[grouping]) > 1
        if shared_width is None:
            shared_width = hidden_width
        if use_shared_trunk:
            if shared_width <= 0:
                raise ValueError(f"shared_width must be positive, got {shared_width}.")
            if shared_depth < 0:
                raise ValueError(f"shared_depth must be non-negative, got {shared_depth}.")
            if shared_units is not None and any(unit <= 0 for unit in shared_units):
                raise ValueError("All shared layer widths must be positive.")
        if deflection_transform is not None and deflection_transform_name == DEFLECTION_TRANSFORM_NONE:
            deflection_transform_name = DEFLECTION_TRANSFORM_CUSTOM
        if (
            deflection_transform is not None
            and deflection_transform_name != DEFLECTION_TRANSFORM_CUSTOM
        ):
            raise ValueError(
                "Use deflection_transform_name='custom' when passing a custom "
                "deflection_transform callable."
            )
        if apply_deflection_transform and deflection_transform is None:
            deflection_transform = MultiSubNetPINN._get_named_deflection_transform(
                deflection_transform_name
            )
        if apply_deflection_transform and deflection_transform is None:
            raise ValueError(
                "deflection_transform must be provided when "
                "apply_deflection_transform=True, or use a built-in "
                "deflection_transform_name."
            )

        super().__init__(name=name)
        self.grouping = grouping
        self.hidden_width = hidden_width
        self.hidden_depth = hidden_depth
        self.activation = activation
        self.hidden_units = None if hidden_units is None else tuple(hidden_units)
        self.use_shared_trunk = bool(use_shared_trunk)
        self.shared_width = shared_width
        self.shared_depth = shared_depth
        self.shared_units = None if shared_units is None else tuple(shared_units)
        self.apply_deflection_transform = apply_deflection_transform
        self.deflection_transform_name = deflection_transform_name
        self.deflection_transform = deflection_transform
        self.apply_moment_transform = apply_moment_transform
        self.moment_transform_name = moment_transform_name
        self.field_output_scales = _normalize_field_output_scales(
            field_output_scales
        )
        self.output_activation = output_activation
        self.field_names = FIELD_NAMES
        self.output_groups = GROUPINGS[grouping]

        self.shared_trunk: IndependentSubNet | None = None
        branch_input_dim = 2
        if self.use_shared_trunk:
            trunk_hidden_units = _resolve_units(
                width=shared_width,
                depth=shared_depth,
                units=shared_units,
            )
            shared_feature_dim = trunk_hidden_units[-1] if trunk_hidden_units else shared_width
            self.shared_trunk = IndependentSubNet(
                output_dim=shared_feature_dim,
                hidden_width=shared_width,
                hidden_depth=shared_depth,
                activation=activation,
                hidden_units=trunk_hidden_units,
                input_dim=2,
                name="shared_trunk",
            )
            branch_input_dim = shared_feature_dim

        self.subnets: list[IndependentSubNet] = []
        for net_name, fields in zip(GROUP_NET_NAMES[grouping], self.output_groups):
            subnet = IndependentSubNet(
                output_dim=len(fields),
                hidden_width=hidden_width,
                hidden_depth=hidden_depth,
                activation=activation,
                hidden_units=hidden_units,
                input_dim=branch_input_dim,
                name=net_name,
            )
            setattr(self, net_name, subnet)
            self.subnets.append(subnet)

    def call(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        """Return final Kirchhoff plate fields as ``(batch_size, 1)`` tensors."""
        raw_outputs = self.call_raw(xy, training=training)
        xy = self._validate_xy(xy)
        return self.apply_output_transforms(xy, raw_outputs)

    def call_raw(self, xy: tf.Tensor, training: bool = False) -> dict[str, tf.Tensor]:
        """Return untransformed subnetwork outputs for debugging and analysis."""
        xy = self._validate_xy(xy)
        branch_inputs = xy
        if self.shared_trunk is not None:
            branch_inputs = self.shared_trunk(xy, training=training)
        outputs: dict[str, tf.Tensor] = {}

        for fields, subnet in zip(self.output_groups, self.subnets):
            group_output = subnet(branch_inputs, training=training)
            split_outputs = tf.split(group_output, len(fields), axis=-1)
            outputs.update(dict(zip(fields, split_outputs)))

        return {field: outputs[field] for field in self.field_names}

    def apply_output_transforms(
        self,
        xy: tf.Tensor,
        raw_outputs: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Apply configured output transforms to raw field predictions."""
        outputs = self.apply_head_output_activation(raw_outputs)

        if self.apply_deflection_transform:
            outputs["w"] = self._apply_deflection_output_transform(xy, outputs["w"])
        if self.apply_moment_transform:
            outputs = self._apply_moment_output_transform(xy, outputs)

        return {
            field: outputs[field]
            * tf.cast(self.field_output_scales[field], outputs[field].dtype)
            for field in self.field_names
        }

    def predict_fields(self, xy: tf.Tensor) -> dict[str, tf.Tensor]:
        """Predict all physical fields with inference-mode behavior."""
        return self(xy, training=False)

    def predict_raw_fields(self, xy: tf.Tensor) -> dict[str, tf.Tensor]:
        """Predict unscaled, untransformed Head outputs for diagnostics."""
        return self.call_raw(xy, training=False)

    def predict_normalized_fields(self, xy: tf.Tensor) -> dict[str, tf.Tensor]:
        """Return activated Head outputs before physical and hard transforms."""
        return self.apply_head_output_activation(
            self.call_raw(xy, training=False)
        )

    def apply_head_output_activation(
        self,
        raw_outputs: Mapping[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Map raw Head values to their configured dimensionless range."""
        outputs = {
            field: tf.convert_to_tensor(raw_outputs[field])
            for field in self.field_names
        }
        if self.output_activation == OUTPUT_ACTIVATION_LINEAR:
            return outputs
        if self.output_activation == OUTPUT_ACTIVATION_TANH:
            return {field: tf.tanh(value) for field, value in outputs.items()}
        raise ValueError(
            f"Unsupported output_activation {self.output_activation!r}."
        )

    def get_grouping(self) -> str:
        """Return the active output grouping name."""
        return self.grouping

    def get_field_output_scales(self) -> dict[str, float]:
        """Return the physical output scales in canonical field order."""
        return dict(self.field_output_scales)

    def uses_deflection_transform(self) -> bool:
        """Return whether the deflection output transform is enabled."""
        return self.apply_deflection_transform

    def uses_moment_transform(self) -> bool:
        """Return whether a simply supported moment transform is enabled."""
        return self.apply_moment_transform

    def get_trainable_parameter_count(self) -> int:
        """Return the total number of trainable scalar parameters."""
        return int(sum(tf.size(variable).numpy() for variable in self.trainable_variables))

    @staticmethod
    def _validate_xy(xy: tf.Tensor) -> tf.Tensor:
        """Validate and return the coordinate tensor."""
        xy = tf.convert_to_tensor(xy)

        if xy.shape.rank is not None:
            if xy.shape.rank != 2:
                raise ValueError(f"xy must be a 2D tensor, got rank {xy.shape.rank}.")
            if xy.shape[-1] != 2:
                raise ValueError(f"xy must have shape (batch_size, 2), got {xy.shape}.")
            return xy

        tf.debugging.assert_equal(tf.rank(xy), 2, message="xy must be a 2D tensor.")
        tf.debugging.assert_equal(
            tf.shape(xy)[-1],
            2,
            message="xy must have shape (batch_size, 2).",
        )
        return xy

    def _apply_deflection_output_transform(
        self,
        xy: tf.Tensor,
        raw_w: tf.Tensor,
    ) -> tf.Tensor:
        """Apply the user-provided transform to the deflection field only."""
        if self.deflection_transform is None:
            raise ValueError("deflection_transform is not configured.")

        transformed_w = tf.convert_to_tensor(self.deflection_transform(xy, raw_w))
        if transformed_w.shape.rank is not None:
            if transformed_w.shape.rank != 2:
                raise ValueError(
                    "deflection_transform output must be a 2D tensor, "
                    f"got rank {transformed_w.shape.rank}."
                )
            if transformed_w.shape[-1] != 1:
                raise ValueError(
                    "deflection_transform output must have shape "
                    f"(batch_size, 1), got {transformed_w.shape}."
                )
        else:
            tf.debugging.assert_equal(
                tf.rank(transformed_w),
                2,
                message="deflection_transform output must be 2D.",
            )
            tf.debugging.assert_equal(
                tf.shape(transformed_w)[-1],
                1,
                message="deflection_transform output must have shape (batch_size, 1).",
            )
        return transformed_w

    @staticmethod
    def _get_named_deflection_transform(
        name: str,
    ) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor] | None:
        """Return a built-in serializable deflection transform by name."""
        if name in (DEFLECTION_TRANSFORM_NONE, DEFLECTION_TRANSFORM_CUSTOM):
            return None
        if name == DEFLECTION_TRANSFORM_UNIT_SQUARE:
            return _unit_square_zero_boundary_transform
        if name == DEFLECTION_TRANSFORM_CLAMPED_UNIT_SQUARE:
            return _unit_square_clamped_transform
        if name == DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y:
            return _unit_square_simple_x_clamped_y_transform
        if name == DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y:
            return _unit_square_clamped_x_simple_y_transform
        raise ValueError(f"Unsupported deflection_transform_name {name!r}.")

    def _apply_moment_output_transform(
        self,
        xy: tf.Tensor,
        outputs: dict[str, tf.Tensor],
    ) -> dict[str, tf.Tensor]:
        """Hard-enforce normal moment on selected unit-square simple edges.

        On vertical axis-aligned edges the normal moment is ``Mx``; on
        horizontal edges it is ``My``. ``Mxy`` must remain unconstrained.
        """
        x = tf.cast(xy[:, 0:1], outputs["Mx"].dtype)
        y = tf.cast(xy[:, 1:2], outputs["My"].dtype)
        transformed = dict(outputs)
        if self.moment_transform_name in (
            MOMENT_TRANSFORM_SIMPLE_X_EDGES,
            MOMENT_TRANSFORM_SIMPLE_ALL_EDGES,
        ):
            transformed["Mx"] = 4.0 * x * (1.0 - x) * transformed["Mx"]
        if self.moment_transform_name in (
            MOMENT_TRANSFORM_SIMPLE_Y_EDGES,
            MOMENT_TRANSFORM_SIMPLE_ALL_EDGES,
        ):
            transformed["My"] = 4.0 * y * (1.0 - y) * transformed["My"]
        return transformed

    def get_config(self) -> dict:
        """Return a serializable configuration dictionary."""
        if (
            self.apply_deflection_transform
            and self.deflection_transform_name == DEFLECTION_TRANSFORM_CUSTOM
        ):
            raise ValueError(
                "Models using a custom deflection_transform callable cannot be "
                "fully restored from get_config(). Use a built-in "
                "deflection_transform_name or recreate the callable manually."
            )

        return {
            "grouping": self.grouping,
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "activation": self.activation,
            "hidden_units": self.hidden_units,
            "use_shared_trunk": self.use_shared_trunk,
            "shared_width": self.shared_width,
            "shared_depth": self.shared_depth,
            "shared_units": self.shared_units,
            "apply_deflection_transform": self.apply_deflection_transform,
            "deflection_transform_name": self.deflection_transform_name,
            "apply_moment_transform": self.apply_moment_transform,
            "moment_transform_name": self.moment_transform_name,
            "field_output_scales": dict(self.field_output_scales),
            "output_activation": self.output_activation,
            "name": self.name,
        }

    @classmethod
    def from_config(cls, config: dict) -> "MultiSubNetPINN":
        """Restore a model from ``get_config`` output."""
        return cls(**config)


def _unit_square_zero_boundary_transform(
    xy: tf.Tensor,
    raw_w: tf.Tensor,
) -> tf.Tensor:
    """Apply a unit-square zero-boundary transform to the deflection field."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return x * (1.0 - x) * y * (1.0 - y) * raw_w


def _unit_square_clamped_transform(
    xy: tf.Tensor,
    raw_w: tf.Tensor,
) -> tf.Tensor:
    """Enforce zero deflection and zero normal slope on all square edges."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    return (
        tf.square(x * (1.0 - x))
        * tf.square(y * (1.0 - y))
        * raw_w
    )


def _unit_square_simple_x_clamped_y_transform(
    xy: tf.Tensor,
    raw_w: tf.Tensor,
) -> tf.Tensor:
    """Enforce w=0 on all edges and w_y=0 on y=0 and y=1."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    y_factor = y * (1.0 - y)
    return x * (1.0 - x) * tf.square(y_factor) * raw_w


def _unit_square_clamped_x_simple_y_transform(
    xy: tf.Tensor,
    raw_w: tf.Tensor,
) -> tf.Tensor:
    """Enforce w=0 on all edges and w_x=0 on x=0 and x=1."""
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    x_factor = x * (1.0 - x)
    return tf.square(x_factor) * y * (1.0 - y) * raw_w


def _resolve_units(
    width: int,
    depth: int,
    units: Sequence[int] | None,
) -> tuple[int, ...]:
    """Resolve explicit layer widths or repeat ``width`` ``depth`` times."""
    if units is not None:
        return tuple(units)
    return tuple(width for _ in range(depth))


def _normalize_field_output_scales(
    scales: FieldOutputScales | Mapping[str, float] | None,
) -> dict[str, float]:
    """Validate and normalize field scales accepted by the public model API."""
    if scales is None:
        return FieldOutputScales().as_dict()
    if isinstance(scales, FieldOutputScales):
        return scales.as_dict()
    if not isinstance(scales, Mapping):
        raise TypeError(
            "field_output_scales must be FieldOutputScales, a mapping, or None."
        )
    missing = set(FIELD_NAMES).difference(scales)
    extra = set(scales).difference(FIELD_NAMES)
    if missing or extra:
        raise ValueError(
            "field_output_scales must contain exactly "
            f"{FIELD_NAMES}; missing={sorted(missing)}, extra={sorted(extra)}."
        )
    return FieldOutputScales(
        **{name: float(scales[name]) for name in FIELD_NAMES}
    ).as_dict()
