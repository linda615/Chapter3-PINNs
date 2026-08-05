"""Tests for shared-trunk PCGrad variants."""

import tensorflow as tf

from trainer.pcgrad import (
    BoundaryAnchoredPCGradConfig,
    SharedTrunkPCGradConfig,
    apply_boundary_anchored_pcgrad,
    apply_shared_trunk_pcgrad,
)


def test_conflicting_moment_gradient_is_projected_from_boundary() -> None:
    """A negative moment-boundary dot product should become orthogonal."""
    shared = tf.Variable([1.0, 1.0], dtype=tf.float32)
    branch = tf.Variable(2.0, dtype=tf.float32)
    variables = (shared, branch)
    config = BoundaryAnchoredPCGradConfig(enabled=True)

    with tf.GradientTape(persistent=True) as tape:
        moment = shared[0] + branch
        boundary = -shared[0] + shared[1]
        shear = tf.square(branch)
        equilibrium = tf.reduce_sum(tf.square(shared)) * 0.0
        loss_dict = {
            "weighted_moment_loss": moment,
            "weighted_boundary_loss": boundary,
            "weighted_shear_loss": shear,
            "weighted_equilibrium_loss": equilibrium,
        }
        total = moment + boundary + shear + equilibrium

    total_gradients = tape.gradient(total, variables)
    projected, diagnostics = apply_boundary_anchored_pcgrad(
        tape=tape,
        loss_dict=loss_dict,
        all_variables=variables,
        shared_variables=(shared,),
        total_gradients=total_gradients,
        config=config,
    )
    del tape

    tf.debugging.assert_near(
        projected[0],
        tf.constant([-0.5, 1.5], dtype=tf.float32),
    )
    tf.debugging.assert_near(projected[1], total_gradients[1])
    tf.debugging.assert_equal(
        diagnostics["pcgrad_moment_boundary_applied"],
        tf.constant(1.0),
    )
    tf.debugging.assert_near(
        diagnostics["pcgrad_moment_boundary_cos_after"],
        tf.constant(0.0),
        atol=1e-6,
    )


def test_aligned_moment_gradient_is_not_changed() -> None:
    """PCGrad should be an identity operation without a conflict."""
    shared = tf.Variable([1.0, 1.0], dtype=tf.float32)
    config = BoundaryAnchoredPCGradConfig(enabled=True)

    with tf.GradientTape(persistent=True) as tape:
        moment = shared[0]
        boundary = shared[0] + shared[1]
        loss_dict = {
            "weighted_moment_loss": moment,
            "weighted_boundary_loss": boundary,
        }
        total = moment + boundary

    total_gradients = tape.gradient(total, (shared,))
    projected, diagnostics = apply_boundary_anchored_pcgrad(
        tape=tape,
        loss_dict=loss_dict,
        all_variables=(shared,),
        shared_variables=(shared,),
        total_gradients=total_gradients,
        config=config,
    )
    del tape

    tf.debugging.assert_near(projected[0], total_gradients[0])
    tf.debugging.assert_equal(
        diagnostics["pcgrad_moment_boundary_applied"],
        tf.constant(0.0),
    )
    assert float(
        diagnostics["pcgrad_moment_boundary_cos_before"].numpy()
    ) > 0.0


def test_dynamic_pcgrad_projects_all_current_conflicting_task_pairs() -> None:
    """Four-task PCGrad should detect conflicts from the current gradients."""
    shared = tf.Variable([1.0, 1.0], dtype=tf.float32)
    branch = tf.Variable(2.0, dtype=tf.float32)
    variables = (shared, branch)
    config = SharedTrunkPCGradConfig(enabled=True)

    with tf.GradientTape(persistent=True) as tape:
        moment = shared[0] + branch
        shear = -shared[0] + shared[1] + 2.0 * branch
        equilibrium = -shared[1] + 3.0 * branch
        boundary = shared[0] + shared[1] + 4.0 * branch
        loss_dict = {
            "weighted_moment_loss": moment,
            "weighted_shear_loss": shear,
            "weighted_equilibrium_loss": equilibrium,
            "weighted_boundary_loss": boundary,
        }
        total = moment + shear + equilibrium + boundary

    total_gradients = tape.gradient(total, variables)
    projected, diagnostics = apply_shared_trunk_pcgrad(
        tape=tape,
        loss_dict=loss_dict,
        all_variables=variables,
        shared_variables=(shared,),
        total_gradients=total_gradients,
        config=config,
    )
    del tape

    tf.debugging.assert_near(projected[1], total_gradients[1])
    assert bool(
        tf.reduce_any(
            tf.not_equal(projected[0], total_gradients[0])
        ).numpy()
    )
    tf.debugging.assert_equal(
        diagnostics["pcgrad_conflict_count"],
        tf.constant(3.0),
    )
    tf.debugging.assert_equal(
        diagnostics["pcgrad_conflict_moment_shear"],
        tf.constant(1.0),
    )
    tf.debugging.assert_equal(
        diagnostics["pcgrad_conflict_shear_equilibrium"],
        tf.constant(1.0),
    )
    tf.debugging.assert_equal(
        diagnostics["pcgrad_conflict_equilibrium_boundary"],
        tf.constant(1.0),
    )
    tf.debugging.assert_greater(
        diagnostics["pcgrad_projection_count"],
        tf.constant(0.0),
    )


def test_dynamic_pcgrad_is_identity_when_all_tasks_align() -> None:
    """No shared or branch gradient should change without a conflict."""
    shared = tf.Variable([1.0, 1.0], dtype=tf.float32)
    branch = tf.Variable(2.0, dtype=tf.float32)
    variables = (shared, branch)
    config = SharedTrunkPCGradConfig(enabled=True)

    with tf.GradientTape(persistent=True) as tape:
        base = shared[0] + shared[1]
        loss_dict = {
            "weighted_moment_loss": base + branch,
            "weighted_shear_loss": 2.0 * base + branch,
            "weighted_equilibrium_loss": 3.0 * base + branch,
            "weighted_boundary_loss": 4.0 * base + branch,
        }
        total = tf.add_n(list(loss_dict.values()))

    total_gradients = tape.gradient(total, variables)
    projected, diagnostics = apply_shared_trunk_pcgrad(
        tape=tape,
        loss_dict=loss_dict,
        all_variables=variables,
        shared_variables=(shared,),
        total_gradients=total_gradients,
        config=config,
    )
    del tape

    for projected_gradient, total_gradient in zip(
        projected,
        total_gradients,
    ):
        tf.debugging.assert_near(projected_gradient, total_gradient)
    tf.debugging.assert_equal(
        diagnostics["pcgrad_conflict_count"],
        tf.constant(0.0),
    )
    tf.debugging.assert_equal(
        diagnostics["pcgrad_projection_count"],
        tf.constant(0.0),
    )


def test_dynamic_pcgrad_runs_inside_tf_function() -> None:
    """Dynamic shared projection should remain graph compatible."""
    shared = tf.Variable([1.0, 1.0], dtype=tf.float32)
    branch = tf.Variable(2.0, dtype=tf.float32)
    config = SharedTrunkPCGradConfig(enabled=True)

    @tf.function
    def project():
        with tf.GradientTape(persistent=True) as tape:
            loss_dict = {
                "weighted_moment_loss": shared[0] + branch,
                "weighted_shear_loss": -shared[0] + shared[1],
                "weighted_equilibrium_loss": -shared[1],
                "weighted_boundary_loss": shared[0] + shared[1],
            }
            total = tf.add_n(list(loss_dict.values()))
        total_gradients = tape.gradient(total, (shared, branch))
        projected, diagnostics = apply_shared_trunk_pcgrad(
            tape=tape,
            loss_dict=loss_dict,
            all_variables=(shared, branch),
            shared_variables=(shared,),
            total_gradients=total_gradients,
            config=config,
        )
        del tape
        return (
            projected[0],
            projected[1],
            diagnostics["pcgrad_conflict_count"],
        )

    shared_gradient, branch_gradient, conflict_count = project()

    tf.debugging.assert_all_finite(
        shared_gradient,
        "Projected shared gradient must be finite.",
    )
    tf.debugging.assert_all_finite(
        branch_gradient,
        "Ordinary branch gradient must be finite.",
    )
    tf.debugging.assert_greater(
        conflict_count,
        tf.constant(0.0),
    )
