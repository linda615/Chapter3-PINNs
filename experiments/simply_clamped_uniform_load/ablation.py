"""Train and evaluate the six SCSC ablation models."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
import json
import math
import sys
import time
from pathlib import Path
from dataclasses import replace
import numpy as np
import tensorflow as tf

ROOT = Path(os.environ.get('CH3_OUTPUT_ROOT', str(Path(__file__).resolve().parent / 'results_ablation')))
GRAPH_MODE = os.environ.get('CH3_GRAPH_MODE', '1') == '1'
SEEDS = (2051, 2052, 2053)
NAMES = ('Mixed-6B', 'MO4-PINN', 'Mixed-1B-Wide', 'Mixed-1B', 'Mixed-3B', 'W-PINN')
COUNTS = {'W-PINN':16897, 'MO4-PINN':33862, 'Mixed-1B':17222,
          'Mixed-3B':21286, 'Mixed-6B':33862, 'Mixed-1B-Wide':33576}
from configs import save_experiment_config
from models import FIELD_NAMES
from losses import resolve_loss_weights
from experiments.simply_clamped_uniform_load.config import DEFAULT_CONFIG
from experiments.simply_clamped_uniform_load.w_pinn_config import W_PINN_CONFIG
from experiments.simply_clamped_uniform_load.mo4_config import MO4_PINN_CONFIG
from experiments.simply_clamped_uniform_load.all_shared_mixed_config import ALL_SHARED_MIXED_CONFIG
from experiments.simply_clamped_uniform_load.problem import create_model, create_load_fn, create_interior_sampler, create_boundary_sampler
from experiments.simply_clamped_uniform_load.w_pinn_problem import create_w_pinn_model
from experiments.simply_clamped_uniform_load.trainer import MixedBoundaryPINNTrainer
from experiments.simply_clamped_uniform_load.comparison_trainers import MixedBoundaryMO4PINNTrainer, MixedBoundaryWPINNTrainer
from experiments.simply_clamped_uniform_load.run import set_reproducible_seed, create_learning_rate_schedule
from experiments.simply_clamped_uniform_load.analytical import analytical_fields
from experiments.simply_clamped_uniform_load.evaluate import make_test_grid
from physics import compute_w_pinn_fields

def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def setup():
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, True)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A TensorFlow-compatible GPU is required for the ablation runs.')

def build(name, seed, pilot=False):
    base = DEFAULT_CONFIG
    if name == 'W-PINN': base = W_PINN_CONFIG
    elif name == 'MO4-PINN': base = MO4_PINN_CONFIG
    elif name.startswith('Mixed-1B'):
        base = replace(ALL_SHARED_MIXED_CONFIG, hidden_width=90 if name.endswith('Wide') else 64)
    elif name == 'Mixed-3B': base = replace(DEFAULT_CONFIG, model_grouping='three')
    cfg = replace(base, seed=seed, training=replace(base.training, seed=seed),
                  gradient_diagnostics_enabled=False, initial_checkpoint_path=None,
                  results_dir=ROOT / ('pilot' if pilot else 'runs') / f'{name}_seed{seed}')
    set_reproducible_seed(seed)
    model = create_w_pinn_model(cfg) if name == 'W-PINN' else create_model(cfg)
    model(tf.zeros((1,2)), training=False)
    assert model.count_params() == COUNTS[name], (name, model.count_params())
    cls = MixedBoundaryWPINNTrainer if name == 'W-PINN' else (MixedBoundaryMO4PINNTrainer if name == 'MO4-PINN' else MixedBoundaryPINNTrainer)
    optimizer = tf.keras.optimizers.Adam(create_learning_rate_schedule(cfg))
    trainer = cls(model=model, optimizer=optimizer, interior_sampler=create_interior_sampler(),
        boundary_sampler=create_boundary_sampler(), load_fn=create_load_fn(cfg), parameters=cfg.plate,
        config=cfg.training, loss_weights=cfg.loss_weights, physics_loss_config=cfg.physics_loss_config,
        boundary_loss_config=cfg.boundary_loss_config, mxy_consistency_config=cfg.mxy_consistency_config,
        shared_pcgrad_config=cfg.shared_pcgrad, experiment_config=cfg)
    # MixedBoundaryPINNTrainer's super().__init__ overwrites its earlier assignment.
    # Set the declared config explicitly for every model, including the comparators.
    original_boundary_config = trainer.boundary_loss_config
    trainer.boundary_loss_config = cfg.boundary_loss_config
    assert trainer.boundary_loss_config.normalize_residuals
    if GRAPH_MODE:
        from .graph_step import install
        install(trainer)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    dump(cfg.results_dir/'configuration_audit.json', {
        'model':name, 'parameters':model.count_params(), 'seed':seed,
        'boundary_config_was_none':original_boundary_config is None,
        'boundary_config_restored':True, 'TF32':False,
        'execution_mode':'tf.function' if GRAPH_MODE else 'eager',
        'tensorflow':tf.__version__, 'python':sys.version, 'executable':sys.executable,
        'GPU':[str(d) for d in tf.config.list_physical_devices('GPU')]})
    save_experiment_config(cfg, run_entry='experiments.simply_clamped_uniform_load.ablation')
    return cfg, model, trainer

def fields(model, xy, cfg, name):
    parts = {k:[] for k in FIELD_NAMES}
    for start in range(0, len(xy), 1024):
        batch = xy[start:start+1024]
        pred = compute_w_pinn_fields(model=model, xy=batch, parameters=cfg.plate, training=False) if name=='W-PINN' else model(batch, training=False)
        for k in FIELD_NAMES: parts[k].append(pred[k].numpy().reshape(-1))
    return {k:np.concatenate(v).astype(np.float64) for k,v in parts.items()}

def reference(xy, cfg, modes=None):
    result = analytical_fields(xy=xy, parameters=cfg.plate, q0=cfg.q0, mode_count=modes or cfg.analytical_modes)
    return {k:v.numpy().reshape(-1).astype(np.float64) for k,v in result.items()}

def errors(pred, ref, mask=None):
    result = {}
    for k in FIELD_NAMES:
        p,r = pred[k],ref[k]
        if mask is not None: p,r=p[mask],r[mask]
        result[k] = float(np.linalg.norm(p-r) / max(np.linalg.norm(r), 1e-12) * 100)
    result['six_field_mean'] = float(np.mean(list(result.values())))
    return result

def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()

def state_checkpoint(model, trainer, directory):
    if hasattr(trainer.optimizer, 'build'):
        trainer.optimizer.build(model.trainable_variables)
    else:
        trainer.optimizer._create_all_weights(model.trainable_variables)
    ck = tf.train.Checkpoint(model=model, optimizer=trainer.optimizer,
        epoch=tf.Variable(0,dtype=tf.int64), best=tf.Variable(1e30,dtype=tf.float64),
        best_epoch=tf.Variable(0,dtype=tf.int64),
        train_seconds=tf.Variable(0.,dtype=tf.float64), validation_seconds=tf.Variable(0.,dtype=tf.float64))
    manager=tf.train.CheckpointManager(ck, str(directory/'resume'), max_to_keep=2)
    if manager.latest_checkpoint:
        ck.restore(manager.latest_checkpoint).assert_existing_objects_matched()
    return ck, manager

def test_run(name):
    setup(); cfg,model,tr = build(name,2051,True)
    checks=[]
    for epoch in [1,5001]:
        xy,bc=tr.sample_epoch(epoch); started=time.perf_counter()
        loss=tr.train_step(xy,bc,loss_weights=resolve_loss_weights(cfg.loss_weights,epoch),epoch=epoch)
        vals={k:float(v.numpy()) for k,v in loss.items()}
        assert all(math.isfinite(v) for v in vals.values())
        assert bool(vals.get('pcgrad_active',0)) == bool(name!='W-PINN' and epoch>=5001)
        checks.append({'epoch_tested':epoch,'seconds':time.perf_counter()-started,'loss':vals['total_loss']})
    xy=make_test_grid(9)
    before=fields(model,xy,cfg,name)
    ck,manager=state_checkpoint(model,tr,cfg.results_dir)
    ck.epoch.assign(2);manager.save(checkpoint_number=2)
    expected_iterations=int(tr.optimizer.iterations.numpy())
    cfg2,model2,tr2=build(name,2051,True)
    ck2,_=state_checkpoint(model2,tr2,cfg.results_dir)
    after=fields(model2,xy,cfg2,name)
    assert int(tr2.optimizer.iterations.numpy())==expected_iterations
    assert all(np.array_equal(before[k],after[k]) for k in FIELD_NAMES)
    a,b=tr.sample_epoch(5002)
    loss1=tr.train_step(a,b,loss_weights=resolve_loss_weights(cfg.loss_weights,5002),epoch=5002)
    loss2=tr2.train_step(a,b,loss_weights=resolve_loss_weights(cfg.loss_weights,5002),epoch=5002)
    assert all(np.allclose(x.numpy(),y.numpy(),rtol=1e-6,atol=1e-7) for x,y in zip(model.weights,model2.weights))
    dump(cfg.results_dir/'passed.json', {'name':name,'checks':checks,'resume_next_step_verified':True})
    print('PILOT PASSED',name,checks,flush=True)

def evaluate_final(name,cfg,model,tr,ck):
    values=(np.arange(200,dtype=np.float32)+0.5)/200
    x,y=np.meshgrid(values,values,indexing='xy')
    xy=tf.constant(np.stack([x.ravel(),y.ravel()],axis=1))
    ref=reference(xy,cfg)
    border=np.minimum.reduce([x.ravel(),y.ravel(),1-x.ravel(),1-y.ravel()])<.1
    result={'model':name,'seed':cfg.seed,'parameters':model.count_params(),
        'execution_mode':'tf.function' if GRAPH_MODE else 'eager',
        'epochs':int(ck.epoch.numpy()),'best_epoch':int(ck.best_epoch.numpy()),
        'train_seconds':float(ck.train_seconds.numpy()),'validation_seconds':float(ck.validation_seconds.numpy()),
        'test_grid':'200x200 cell centres, independent of validation grid', 'reference_modes':40}
    for label,path in [('final',cfg.final_weights_path),('best',cfg.best_validation_loss_weights_path)]:
        model.load_weights(str(path));pred=fields(model,xy,cfg,name)
        result[label]=errors(pred,ref)
        result[label+'_boundary_strip']=errors(pred,ref,border)
        np.savez_compressed(cfg.results_dir/f'{label}_test_fields.npz',xy=xy.numpy(),**pred,**{'reference_'+k:v for k,v in ref.items()})
        # Independent fixed collocation sample, shared by all configurations/seeds.
        old=tr.config;tr.config=replace(old,seed=99173,interior_points=4096,boundary_points_per_side=160)
        ix,bc=tr.sample_epoch(1)
        loss=tr.compute_loss(ix,bc,loss_weights=resolve_loss_weights(cfg.loss_weights,20000),training=False)
        tr.config=old
        result[label+'_physical_loss']={k:float(v.numpy()) for k,v in loss.items()}
    result['reference_40_vs_80_percent']=errors(ref,reference(xy,cfg,80))
    dump(cfg.results_dir/'result.json',result)

def run(name,seed):
    setup();cfg,model,tr=build(name,seed)
    if (cfg.results_dir/'result.json').exists(): return
    ck,manager=state_checkpoint(model,tr,cfg.results_dir)
    xy=make_test_grid(101);ref=reference(xy,cfg)
    for epoch in range(int(ck.epoch.numpy())+1,20001):
        started=time.perf_counter();ix,bc=tr.sample_epoch(epoch)
        loss=tr.train_step(ix,bc,loss_weights=resolve_loss_weights(cfg.loss_weights,epoch),epoch=epoch)
        total=float(loss['total_loss'].numpy())
        if not math.isfinite(total): raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
        ck.train_seconds.assign_add(time.perf_counter()-started)
        if epoch==1 or epoch%100==0:
            row={'epoch':epoch,'model':name,'seed':seed,'train_seconds':float(ck.train_seconds.numpy()),
                 **{k:float(v.numpy()) for k,v in loss.items()}}
            append_json(cfg.results_dir/'training.jsonl',row)
            dump(ROOT/'current_run.json',row)
            print(f'{name} seed={seed} epoch={epoch}/20000 loss={total:.7g}',flush=True)
        if epoch%200==0:
            started=time.perf_counter();err=errors(fields(model,xy,cfg,name),ref)
            score=err['six_field_mean']
            if not math.isfinite(score):raise FloatingPointError('Nonfinite validation')
            if score<float(ck.best.numpy()):
                cfg.checkpoints_dir.mkdir(parents=True,exist_ok=True)
                model.save_weights(str(cfg.best_validation_loss_weights_path))
                ck.best.assign(score);ck.best_epoch.assign(epoch)
            ck.validation_seconds.assign_add(time.perf_counter()-started)
            ck.epoch.assign(epoch);manager.save(checkpoint_number=epoch)
            append_json(cfg.results_dir/'validation.jsonl',{'epoch':epoch,**err})
            print('VALIDATION',epoch,score,flush=True)
    model.save_weights(str(cfg.final_weights_path))
    evaluate_final(name,cfg,model,tr,ck)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=NAMES, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, default=2051)
    parser.add_argument('--check', action='store_true', help='Check training and checkpoint restoration without a full run.')
    args = parser.parse_args()
    if args.check:
        test_run(args.name)
    else:
        run(args.name, args.seed)

if __name__ == '__main__':
    main()
