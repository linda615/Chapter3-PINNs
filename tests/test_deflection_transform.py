"""Tests for optional deflection output transforms."""

import pytest
import tensorflow as tf

from models import (
    DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
    DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
    DEFLECTION_TRANSFORM_UNIT_SQUARE,
    FIELD_NAMES,
    FieldOutputScales,
    MOMENT_TRANSFORM_SIMPLE_X_EDGES,
    MultiSubNetPINN,
    OUTPUT_ACTIVATION_TANH,
)


def test_deflection_transform_changes_only_w_output() -> None:
    """The optional transform should be applied only to the deflection field."""
    xy = tf.ones((3, 2), dtype=tf.float32)
    base_model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        name="base_model",
    )
    transformed_model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        apply_deflection_transform=True,
        deflection_transform=lambda coords, raw_w: raw_w + coords[:, 0:1],
        name="transformed_model",
    )
    base_outputs = base_model(xy)
    transformed_model(xy)
    transformed_model.set_weights(base_model.get_weights())

    transformed_outputs = transformed_model(xy)

    assert tuple(transformed_outputs.keys()) == FIELD_NAMES
    tf.debugging.assert_near(transformed_outputs["w"], base_outputs["w"] + xy[:, 0:1])
    for field in ("Mx", "My", "Mxy", "Qx", "Qy"):
        tf.debugging.assert_near(transformed_outputs[field], base_outputs[field])


def test_raw_and_final_outputs_are_separated() -> None:
    """Raw outputs should remain available when final outputs are transformed."""
    xy = tf.constant([[0.5, 0.25], [0.25, 0.75]], dtype=tf.float32)
    model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        apply_deflection_transform=True,
        deflection_transform=lambda coords, raw_w: raw_w + 2.0 * coords[:, 0:1],
    )

    raw_outputs = model.call_raw(xy)
    final_outputs = model(xy)
    predicted_raw_outputs = model.predict_raw_fields(xy)

    tf.debugging.assert_near(raw_outputs["w"], predicted_raw_outputs["w"])
    tf.debugging.assert_near(final_outputs["w"], raw_outputs["w"] + 2.0 * xy[:, 0:1])
    for field in ("Mx", "My", "Mxy", "Qx", "Qy"):
        tf.debugging.assert_near(final_outputs[field], raw_outputs[field])


def test_deflection_transform_requires_callable_when_enabled() -> None:
    """Enabling the option without a transform function should fail clearly."""
    with pytest.raises(ValueError, match="deflection_transform must be provided"):
        MultiSubNetPINN(apply_deflection_transform=True)


def test_get_config_restores_model_without_deflection_transform() -> None:
    """A default model should be restorable from its config."""
    model = MultiSubNetPINN(grouping="three", hidden_width=5, hidden_depth=2)

    restored = MultiSubNetPINN.from_config(model.get_config())

    assert restored.get_grouping() == "three"
    assert restored.hidden_width == 5
    assert restored.hidden_depth == 2
    assert restored.uses_deflection_transform() is False


def test_get_config_restores_builtin_deflection_transform() -> None:
    """Built-in named deflection transforms should survive config round-trips."""
    xy = tf.constant([[0.5, 0.5], [0.25, 0.75]], dtype=tf.float32)
    model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        apply_deflection_transform=True,
        deflection_transform_name=DEFLECTION_TRANSFORM_UNIT_SQUARE,
    )
    restored = MultiSubNetPINN.from_config(model.get_config())

    model(xy)
    restored(xy)
    restored.set_weights(model.get_weights())

    outputs = model(xy)
    restored_outputs = restored(xy)

    assert restored.uses_deflection_transform() is True
    assert restored.deflection_transform_name == DEFLECTION_TRANSFORM_UNIT_SQUARE
    for field in FIELD_NAMES:
        tf.debugging.assert_near(restored_outputs[field], outputs[field])


@pytest.mark.parametrize(
    ("transform_name", "xy", "normal_axis"),
    (
        (
            DEFLECTION_TRANSFORM_SIMPLE_X_CLAMPED_Y,
            ((0.2, 0.0), (0.7, 1.0)),
            1,
        ),
        (
            DEFLECTION_TRANSFORM_CLAMPED_X_SIMPLE_Y,
            ((0.0, 0.2), (1.0, 0.7)),
            0,
        ),
    ),
)
def test_mixed_edge_transform_hard_enforces_clamped_normal_slope(
    transform_name: str,
    xy: tuple[tuple[float, float], ...],
    normal_axis: int,
) -> None:
    """Named mixed-edge transforms enforce both clamped conditions exactly."""
    coordinates = tf.Variable(xy, dtype=tf.float32)
    model = MultiSubNetPINN(
        grouping="one",
        hidden_width=5,
        hidden_depth=1,
        apply_deflection_transform=True,
        deflection_transform_name=transform_name,
    )

    with tf.GradientTape() as tape:
        w = model(coordinates, training=False)["w"]
        summed_w = tf.reduce_sum(w)
    gradient = tape.gradient(summed_w, coordinates)

    tf.debugging.assert_near(w, tf.zeros_like(w), atol=1e-12)
    tf.debugging.assert_near(
        gradient[:, normal_axis : normal_axis + 1],
        tf.zeros_like(w),
        atol=1e-12,
    )

    restored = MultiSubNetPINN.from_config(model.get_config())
    assert restored.deflection_transform_name == transform_name


def test_custom_deflection_transform_config_fails_clearly() -> None:
    """Arbitrary Python callables are not fully serializable via get_config."""
    model = MultiSubNetPINN(
        apply_deflection_transform=True,
        deflection_transform=lambda coords, raw_w: raw_w + coords[:, 0:1],
    )

    with pytest.raises(ValueError, match="custom deflection_transform callable"):
        model.get_config()


def test_simple_x_edge_moment_transform_changes_only_mx() -> None:
    """Vertical-edge hard constraints multiply only Mx by 4x(1-x)."""
    xy = tf.constant(
        [[0.0, 0.25], [0.25, 0.0], [0.5, 0.5], [1.0, 0.75]],
        dtype=tf.float32,
    )
    model = MultiSubNetPINN(
        grouping="one",
        hidden_width=5,
        hidden_depth=1,
        apply_moment_transform=True,
        moment_transform_name=MOMENT_TRANSFORM_SIMPLE_X_EDGES,
    )

    raw = model.predict_raw_fields(xy)
    final = model.predict_fields(xy)
    x = xy[:, 0:1]

    tf.debugging.assert_near(final["Mx"], 4.0 * x * (1.0 - x) * raw["Mx"])
    for field in ("w", "My", "Mxy", "Qx", "Qy"):
        tf.debugging.assert_near(final[field], raw[field])


def test_get_config_restores_builtin_moment_transform() -> None:
    """The selected hard moment transform should survive a config round-trip."""
    model = MultiSubNetPINN(
        grouping="three",
        hidden_width=5,
        hidden_depth=1,
        apply_moment_transform=True,
        moment_transform_name=MOMENT_TRANSFORM_SIMPLE_X_EDGES,
    )

    restored = MultiSubNetPINN.from_config(model.get_config())

    assert restored.uses_moment_transform() is True
    assert restored.moment_transform_name == MOMENT_TRANSFORM_SIMPLE_X_EDGES


def test_field_output_scales_map_raw_heads_to_physical_fields() -> None:
    """Final fields are scaled while raw dimensionless Head outputs stay intact."""
    xy = tf.constant(((0.25, 0.5), (0.75, 0.25)), dtype=tf.float32)
    scales = FieldOutputScales(
        w=2.0,
        Mx=3.0,
        My=4.0,
        Mxy=5.0,
        Qx=6.0,
        Qy=7.0,
    )
    model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        field_output_scales=scales,
    )

    raw = model.predict_raw_fields(xy)
    final = model.predict_fields(xy)

    for name, scale in scales.as_dict().items():
        tf.debugging.assert_near(final[name], raw[name] * scale)


def test_deflection_transform_precedes_physical_output_scaling() -> None:
    """The hard w envelope acts on the dimensionless output before scaling."""
    xy = tf.constant(((0.25, 0.5), (0.75, 0.25)), dtype=tf.float32)
    model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        apply_deflection_transform=True,
        deflection_transform_name=DEFLECTION_TRANSFORM_UNIT_SQUARE,
        field_output_scales=FieldOutputScales(w=2.5),
    )

    raw = model.predict_raw_fields(xy)
    final = model.predict_fields(xy)
    envelope = (
        xy[:, 0:1]
        * (1.0 - xy[:, 0:1])
        * xy[:, 1:2]
        * (1.0 - xy[:, 1:2])
    )

    tf.debugging.assert_near(final["w"], envelope * raw["w"] * 2.5)


def test_field_output_scales_survive_config_round_trip() -> None:
    """Scale-aware models remain fully restorable through get_config()."""
    scales = FieldOutputScales(w=0.2, Mx=0.3, My=0.4, Mxy=0.5, Qx=0.6, Qy=0.7)
    model = MultiSubNetPINN(field_output_scales=scales)

    restored = MultiSubNetPINN.from_config(model.get_config())

    assert restored.get_field_output_scales() == scales.as_dict()


def test_tanh_head_outputs_are_bounded_before_physical_scaling() -> None:
    """Bounded Head values are scaled only after the tanh activation."""
    xy = tf.constant(((0.2, 0.3), (0.8, 0.7)), dtype=tf.float32)
    scales = FieldOutputScales(
        w=2.0,
        Mx=3.0,
        My=4.0,
        Mxy=5.0,
        Qx=6.0,
        Qy=7.0,
    )
    model = MultiSubNetPINN(
        grouping="six",
        hidden_width=4,
        hidden_depth=1,
        output_activation=OUTPUT_ACTIVATION_TANH,
        field_output_scales=scales,
    )

    raw = model.predict_raw_fields(xy)
    normalized = model.predict_normalized_fields(xy)
    physical = model.predict_fields(xy)

    for name, scale in scales.as_dict().items():
        tf.debugging.assert_near(normalized[name], tf.tanh(raw[name]))
        tf.debugging.assert_less_equal(normalized[name], 1.0)
        tf.debugging.assert_greater_equal(normalized[name], -1.0)
        tf.debugging.assert_near(physical[name], normalized[name] * scale)

    restored = MultiSubNetPINN.from_config(model.get_config())
    assert restored.output_activation == OUTPUT_ACTIVATION_TANH


@pytest.mark.parametrize("bad_scale", (-1.0, 0.0, float("nan"), float("inf")))
def test_invalid_field_output_scale_is_rejected(bad_scale: float) -> None:
    """Non-positive or non-finite output scales fail at construction time."""
    with pytest.raises(ValueError, match="finite and positive"):
        FieldOutputScales(w=bad_scale)
