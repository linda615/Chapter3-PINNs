"""Tests for shared-trunk loss-gradient diagnostics."""

import tensorflow as tf

from models import MultiSubNetPINN
from trainer import compute_shared_gradient_diagnostics


def test_shared_gradient_diagnostics_return_norms_and_cosines() -> None:
    """Four weighted losses should produce ten scalar diagnostics."""
    tf.keras.utils.set_random_seed(83)
    model = MultiSubNetPINN(
        grouping="three",
        hidden_width=4,
        hidden_depth=1,
        use_shared_trunk=True,
        shared_width=5,
        shared_depth=1,
    )
    xy = tf.constant(
        [[0.2, 0.3], [0.4, 0.7], [0.8, 0.6]],
        dtype=tf.float32,
    )

    def loss_fn() -> dict[str, tf.Tensor]:
        fields = model(xy, training=False)
        return {
            "weighted_moment_loss": tf.reduce_mean(
                tf.square(fields["Mx"] + fields["My"])
            ),
            "weighted_shear_loss": tf.reduce_mean(
                tf.square(fields["Qx"] - fields["Qy"])
            ),
            "weighted_equilibrium_loss": tf.reduce_mean(
                tf.square(fields["Qx"] + fields["Qy"] + 1.0)
            ),
            "weighted_boundary_loss": tf.reduce_mean(tf.square(fields["w"])),
        }

    diagnostics = compute_shared_gradient_diagnostics(model, loss_fn)

    assert len(diagnostics) == 10
    for name, value in diagnostics.items():
        assert value.shape == (), name
        tf.debugging.assert_all_finite(value, f"{name} must be finite.")
        if "_cos_" in name:
            assert float(value.numpy()) >= -1.000001
            assert float(value.numpy()) <= 1.000001
