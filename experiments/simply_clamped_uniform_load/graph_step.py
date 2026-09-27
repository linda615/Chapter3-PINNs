"""Compile the existing training step without changing its mathematics."""
import tensorflow as tf
from boundary import BoundaryBatch, BoundaryConditionBatch

def install(trainer):
    original = trainer.train_step
    cache = {}
    def step(interior_xy, boundary_batch, loss_weights=None, epoch=1):
        if any(g.batch.weights is not None for g in boundary_batch):
            raise ValueError('Weighted boundary batches need an explicit graph signature')
        active = trainer.shared_pcgrad_config.enabled and epoch >= trainer.shared_pcgrad_config.start_epoch
        key = (loss_weights, active, tuple(g.boundary_type for g in boundary_batch))
        if key not in cache:
            representative_epoch = trainer.shared_pcgrad_config.start_epoch if active else 1
            types = key[2]
            @tf.function(autograph=False, input_signature=[tf.TensorSpec((None,2),tf.float32)]*5)
            def compiled(ix,x0,n0,x1,n1):
                groups=tuple(BoundaryConditionBatch(kind,BoundaryBatch(x,n)) for kind,x,n in zip(types,(x0,x1),(n0,n1)))
                return original(ix,groups,loss_weights=loss_weights,epoch=representative_epoch)
            cache[key] = compiled
        return cache[key](interior_xy,boundary_batch[0].batch.xy,boundary_batch[0].batch.normals,
                          boundary_batch[1].batch.xy,boundary_batch[1].batch.normals)
    trainer.train_step=step
    trainer.compiled_steps=cache
