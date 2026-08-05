"""Forward-pass tests for the independent multi-subnetwork PINN."""

import pytest
import tensorflow as tf

from models import FIELD_NAMES, MultiSubNetPINN


FIELD_VALUES = {
    "w": 1.0,
    "Mx": 2.0,
    "My": 3.0,
    "Mxy": 4.0,
    "Qx": 5.0,
    "Qy": 6.0,
}


class ConstantGroupSubNet(tf.keras.Model):
    """Fake subnet that returns one constant column for each grouped field."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        super().__init__()
        self.fields = fields

    def call(self, xy: tf.Tensor, training: bool = False) -> tf.Tensor:
        del training
        columns = [
            tf.ones((tf.shape(xy)[0], 1), dtype=xy.dtype) * FIELD_VALUES[field]
            for field in self.fields
        ]
        return tf.concat(columns, axis=1)


@pytest.mark.parametrize("grouping", ["one", "two", "three", "six"])
def test_model_forward_returns_all_fields(grouping: str) -> None:
    """All grouping modes should return six ``(10, 1)`` field tensors."""
    model = MultiSubNetPINN(grouping=grouping, hidden_width=8, hidden_depth=2)
    xy = tf.zeros((10, 2), dtype=tf.float32)

    outputs = model(xy)

    assert tuple(outputs.keys()) == FIELD_NAMES
    for field in FIELD_NAMES:
        assert outputs[field].shape == (10, 1)


@pytest.mark.parametrize("grouping", ["two", "three", "six"])
def test_multibranch_model_uses_shared_trunk_by_default(grouping: str) -> None:
    """Multi-branch groupings should pass coordinates through a shared trunk first."""
    model = MultiSubNetPINN(
        grouping=grouping,
        hidden_width=8,
        hidden_depth=1,
        shared_width=6,
        shared_depth=1,
    )
    xy = tf.zeros((5, 2), dtype=tf.float32)

    outputs = model(xy)

    assert model.use_shared_trunk is True
    assert model.shared_trunk is not None
    assert model.shared_trunk.output_dim == 6
    assert all(subnet.input_dim == 6 for subnet in model.subnets)
    for field in FIELD_NAMES:
        assert outputs[field].shape == (5, 1)


def test_single_branch_model_does_not_use_shared_trunk_by_default() -> None:
    """A one-branch model should keep the direct coordinate-to-output network."""
    model = MultiSubNetPINN(grouping="one", hidden_width=8, hidden_depth=1)

    assert model.use_shared_trunk is False
    assert model.shared_trunk is None
    assert all(subnet.input_dim == 2 for subnet in model.subnets)


def test_multibranch_shared_trunk_can_be_disabled() -> None:
    """The shared trunk can be disabled for independent-subnet ablation runs."""
    model = MultiSubNetPINN(
        grouping="three",
        hidden_width=8,
        hidden_depth=1,
        use_shared_trunk=False,
    )

    assert model.use_shared_trunk is False
    assert model.shared_trunk is None
    assert all(subnet.input_dim == 2 for subnet in model.subnets)


@pytest.mark.parametrize("grouping", ["one", "two", "three", "six"])
def test_model_output_binding_preserves_field_order_for_each_grouping(grouping: str) -> None:
    """Grouped subnet columns should bind to the intended physical fields."""
    model = MultiSubNetPINN(grouping=grouping, hidden_width=4, hidden_depth=1)
    fake_subnets = []
    for index, fields in enumerate(model.output_groups):
        fake_subnet = ConstantGroupSubNet(fields)
        fake_subnets.append(fake_subnet)
        setattr(model, model.subnets[index].name, fake_subnet)
    model.subnets = fake_subnets

    outputs = model(tf.zeros((3, 2), dtype=tf.float32), training=False)

    assert tuple(outputs.keys()) == FIELD_NAMES
    for field in FIELD_NAMES:
        expected = tf.ones((3, 1), dtype=tf.float32) * FIELD_VALUES[field]
        tf.debugging.assert_near(outputs[field], expected)
