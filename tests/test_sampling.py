"""Tests for sampling utilities."""

import tensorflow as tf

from sampling import (
    BoundarySide,
    RectangularBoundarySampler,
    RectangularInteriorSampler,
    constant_load,
    sinusoidal_load,
)


def test_rectangular_interior_sampler_shapes_and_bounds() -> None:
    """Interior samples should lie inside the configured rectangle."""
    sampler = RectangularInteriorSampler(x_min=-1.0, x_max=2.0, y_min=0.5, y_max=3.0)

    xy = sampler.sample(16, seed=123)

    assert xy.shape == (16, 2)
    tf.debugging.assert_greater_equal(xy[:, 0], tf.constant(-1.0, dtype=xy.dtype))
    tf.debugging.assert_less_equal(xy[:, 0], tf.constant(2.0, dtype=xy.dtype))
    tf.debugging.assert_greater_equal(xy[:, 1], tf.constant(0.5, dtype=xy.dtype))
    tf.debugging.assert_less_equal(xy[:, 1], tf.constant(3.0, dtype=xy.dtype))


def test_rectangular_boundary_sampler_side_normals() -> None:
    """Boundary side samples should carry the expected outward normals."""
    sampler = RectangularBoundarySampler()

    xy, normals = sampler.sample_side(BoundarySide.LEFT, 5, seed=123)

    assert xy.shape == (5, 2)
    assert normals.shape == (5, 2)
    tf.debugging.assert_near(xy[:, 0:1], tf.zeros((5, 1), dtype=xy.dtype))
    tf.debugging.assert_near(
        normals,
        tf.tile(tf.constant([[-1.0, 0.0]], dtype=normals.dtype), [5, 1]),
    )


def test_rectangular_boundary_sampler_all_sides_shape() -> None:
    """Sampling all sides should concatenate four side batches."""
    sampler = RectangularBoundarySampler()

    xy, normals = sampler.sample_all_sides(points_per_side=4, seed=42)

    assert xy.shape == (16, 2)
    assert normals.shape == (16, 2)


def test_rectangular_interior_sampler_reuses_seed_deterministically() -> None:
    """Interior sampling should not depend on global RNG state when seed is provided."""
    sampler = RectangularInteriorSampler()

    first = sampler.sample(8, seed=321)
    _ = tf.random.uniform((32, 2))
    second = sampler.sample(8, seed=321)
    third = sampler.sample(8, seed=322)

    tf.debugging.assert_equal(first, second)
    difference = tf.reduce_max(tf.abs(first - third))
    tf.debugging.assert_greater(difference, tf.constant(0.0, dtype=difference.dtype))


def test_rectangular_boundary_sampler_reuses_seed_deterministically() -> None:
    """Boundary sampling should not depend on global RNG state when seed is provided."""
    sampler = RectangularBoundarySampler()

    first_xy, first_normals = sampler.sample_all_sides(points_per_side=3, seed=654)
    _ = tf.random.uniform((32, 2))
    second_xy, second_normals = sampler.sample_all_sides(points_per_side=3, seed=654)
    third_xy, _ = sampler.sample_all_sides(points_per_side=3, seed=655)

    tf.debugging.assert_equal(first_xy, second_xy)
    tf.debugging.assert_equal(first_normals, second_normals)
    difference = tf.reduce_max(tf.abs(first_xy - third_xy))
    tf.debugging.assert_greater(difference, tf.constant(0.0, dtype=difference.dtype))


def test_load_generators_return_column_vectors() -> None:
    """Generated load functions should return ``(batch_size, 1)`` tensors."""
    xy = tf.constant([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=tf.float32)

    q_const = constant_load(2.5)(xy)
    q_sine = sinusoidal_load(amplitude=1.0)(xy)

    assert q_const.shape == (3, 1)
    assert q_sine.shape == (3, 1)
    tf.debugging.assert_near(q_const, tf.ones((3, 1), dtype=tf.float32) * 2.5)
