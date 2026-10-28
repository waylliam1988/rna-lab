"""Freeze/run fixed-budget CNN and prior-residual ablations; no test access."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import torch
from torch.utils.data import DataLoader
from rna2d.evidence import pairing_evidence_issues
from rna2d.experiment_data import file_sha256, validate_split_manifest
from rna2d.losses import balanced_pair_bce
from rna2d.reporting import group_macro_f1
from rna2d.sampling import group_balanced_sample_indices, sample_schedule_sha256
from rna2d.sequence_encoders import SequencePairModel
from rna2d.thermodynamics import SETTINGS, PriorStore, ThermodynamicModel
from rna2d.thermo_fusion import ThermoResidualModel
from rna2d.training import Records, collate, evaluate, seed_all

PROTOCOL = ROOT/'data/manifests/thermo_comparison_protocol_v1.json'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2)+'\n', encoding='utf-8', newline='\n')


def state_hash(state):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def environment():
    return {'python':platform.python_version(),
            'packages':{name:importlib.metadata.version(name) for name in
                        ('torch','numpy','scipy','scikit-learn','ViennaRNA')},
            'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}


def load_datasets():
    folder=ROOT/'data/experimental/joint_split_v2'
    manifest=validate_split_manifest(folder,ROOT/'data/manifests/joint_split_v2_validation.json',['train','validation'])
    datasets={s:Records(folder/f'{s}.jsonl') for s in ('train','validation')}
    groups={s:{r['split_group_id'] for r in ds.rows} for s,ds in datasets.items()}
    if groups['train'] & groups['validation']:
        raise ValueError('Train/validation groups overlap')
    for ds in datasets.values():
        for row in ds.rows:
            if pairing_evidence_issues(row):raise ValueError(f"Experimental evidence rejected: {row['id']}")
    return datasets,manifest


def checked_store(protocol):
    index_path=ROOT/'runs/thermodynamics_v1/step1/cache_index.json'
    if file_sha256(index_path)!=protocol['cache_index_sha256']:raise ValueError('Prior cache index changed')
    store=PriorStore(ROOT/'data/processed/thermodynamic_prior_v1')
    for name,expected in json.loads(index_path.read_text()).items():
        if Path(name).name != name or file_sha256(store.folder/name)!=expected:
            raise ValueError('Prior cache content changed')
    return store


def freeze():
    if PROTOCOL.exists():raise FileExistsError('Do not overwrite a frozen protocol')
    datasets,manifest=load_datasets()
    historical_path=ROOT/'data/manifests/data_first_loss_protocol_v2.json'
    old=json.loads(historical_path.read_text())
    baseline_path=ROOT/'data/manifests/thermodynamic_baseline_summary_v1.json'
    baseline=json.loads(baseline_path.read_text())
    folder=ROOT/'data/experimental/joint_split_v2'
    data_hashes={s:file_sha256(folder/f'{s}.jsonl') for s in datasets}
    if data_hashes!=old['data_sha256']:raise ValueError('Historical data differs')
    for seed in old['seeds']:
        seed_all(seed)
        cfg=old['model'];model=SequencePairModel(cfg['kind'],cfg['width'],cfg['hidden'],cfg['depth'])
        if state_hash(model.state_dict())!=old['initial_state_sha256'][str(seed)]:
            raise ValueError('Initialization differs from historical paired seeds')
        schedule=group_balanced_sample_indices([r['split_group_id'] for r in datasets['train'].rows],
                    updates=550,batch_size=4,seed=seed)
        if sample_schedule_sha256(schedule,[r['id'] for r in datasets['train'].rows])!=old['sample_order_sha256'][str(seed)]:
            raise ValueError('Sampling differs from historical paired schedule')
    sources=['scripts/run_thermo_comparison.py','scripts/verify_thermo_comparison.py',
             'src/rna2d/thermo_fusion.py','src/rna2d/thermodynamics.py','src/rna2d/sequence_encoders.py',
             'src/rna2d/training.py','src/rna2d/data.py','src/rna2d/losses.py','src/rna2d/sampling.py',
             'src/rna2d/evidence.py','src/rna2d/reporting.py','requirements-shared-common.txt']
    p={'protocol':'thermo-residual-comparison-v1','status':'frozen_before_training',
       'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'data_sha256':data_hashes,'split_manifest_sha256':manifest,
       'split_counts':{s:len(ds) for s,ds in datasets.items()},
       'split_groups':{s:len({r['split_group_id'] for r in ds.rows}) for s,ds in datasets.items()},
       'settings':SETTINGS,'epsilon':1e-4,'conditions':{'cnn':0.0,'hybrid_a025':0.25,'hybrid_a1':1.0},
       'primary_hybrid':'hybrid_a025','sensitivity_only':'hybrid_a1',
       'initial_state_sha256':old['initial_state_sha256'],'sample_order_sha256':old['sample_order_sha256'],
       'model':old['model'],'optimizer':old['optimizer'],'batch_size':4,'optimizer_updates':550,
       'epochs':25,'seeds':[42,43,44],'run_limit':9,'attempts_per_condition_seed':1,
       'threshold':0.5,'evaluation_batch_size':4,'torch_num_threads':1,
       'checkpoint':'final update only; no early stopping or validation checkpoint selection',
       'supervision':'unchanged balanced BCE over existing canonical separation>=4 observed pair mask only',
       'test_access':'prohibited; only train.jsonl and validation.jsonl opened',
       'independent_test_available':False,
       'source_sha256':{f:file_sha256(ROOT/f) for f in sources},
       'environment':environment(),'prior_role':'computed detached input, not labels or auxiliary supervision',
       'cache_index_sha256':baseline['cache_index_sha256'],
       'baseline_summary_sha256':file_sha256(baseline_path),
       'historical_protocol_sha256':file_sha256(historical_path),
       'thermo_validation_group_macro_f1':baseline['splits']['validation']['thermo']['group_macro_f1'],
       'gate':{'comparator':'thermo alone AND paired fresh CNN',
               'primary_mean_gain_over_thermo_min':0.02,'all_three_seeds_beat_both':True,
               'minimum_groups_improved_over_thermo':4,'maximum_group_regression':0.10,
               'meaning':'engineering continuation threshold, not statistical significance',
               'sensitivity_cannot_rescue_primary_failure':True,
               'default_change':False,'no_metric_driven_retries_or_extra_search':True},
       'timing':'training includes cached feature lookup; prior generation reported separately; inference includes cached lookup, excludes generation',
       'limitations':['10 development RNAs/7 groups, already inspected; no independent generalization claim.',
                      'Same isolated-sequence prior settings despite heterogeneous experimental complex environments.',
                      'Fixed offset experiment, not a learned energy model or cotranscriptional dynamics.']}
    checked_store(p)
    write(PROTOCOL,p)
    print(json.dumps({'frozen':True,'sha256':file_sha256(PROTOCOL),'runs':9,'updates_per_run':550}),flush=True)


def checked_protocol():
    p=json.loads(PROTOCOL.read_text())
    if p['status']!='frozen_before_training':raise ValueError('Protocol not frozen')
    if p['environment']!=environment():raise ValueError('Frozen runtime changed')
    if p['settings']!=SETTINGS:raise ValueError('Prior settings changed')
    for f,expected in p['source_sha256'].items():
        if file_sha256(ROOT/f)!=expected:raise ValueError(f'Frozen source changed: {f}')
    for f,key in [('thermodynamic_baseline_summary_v1.json','baseline_summary_sha256'),
                  ('data_first_loss_protocol_v2.json','historical_protocol_sha256')]:
        if file_sha256(ROOT/'data/manifests'/f)!=p[key]:raise ValueError('Prior baseline/protocol changed')
    datasets,manifest=load_datasets()
    if manifest!=p['split_manifest_sha256']:raise ValueError('Split manifest changed')
    for s,ds in datasets.items():
        if len(ds)!=p['split_counts'][s] or file_sha256(ROOT/f'data/experimental/joint_split_v2/{s}.jsonl')!=p['data_sha256'][s]:
            raise ValueError('Frozen data changed')
    return p,datasets


def configure_torch():
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def make_model(protocol,store,alpha,device):
    cfg=protocol['model']
    return ThermoResidualModel(SequencePairModel(cfg['kind'],cfg['width'],cfg['hidden'],cfg['depth']).to(device),
                              store,alpha=alpha,epsilon=protocol['epsilon'])


def assess(model,dataset,device):
    loader=DataLoader(dataset,batch_size=4,shuffle=False,collate_fn=collate,num_workers=0)
    torch.cuda.synchronize(device)
    start=time.perf_counter()
    metrics,details=evaluate(model,loader,device,0.5)
    torch.cuda.synchronize(device)
    elapsed=time.perf_counter()-start
    macro,groups=group_macro_f1(details,dataset.rows)
    return {'metrics':metrics,'group_macro_f1':macro,'per_group':groups,'details':details,
            'evaluation_seconds_including_decode':elapsed}


def run(condition,seed):
    configure_torch()
    p,datasets=checked_protocol()
    if condition not in p['conditions'] or seed not in p['seeds']:raise ValueError('Unplanned condition/seed')
    output=ROOT/f'runs/thermodynamics_v1/step3/{condition}/seed{seed}'
    output.mkdir(parents=True,exist_ok=False)
    write(output/'started.json',{'protocol_sha256':file_sha256(PROTOCOL),'condition':condition,'seed':seed,
                                'started_at_utc':datetime.now(timezone.utc).isoformat()})
    device=torch.device('cuda:0')
    store=checked_store(p)
    # Validate/preload priors for both splits; only sequence features, no labels.
    for ds in datasets.values():
        for row in ds.rows:store.get(row['sequence'])
    seed_all(seed)
    model=make_model(p,store,p['conditions'][condition],device)
    initial_hash=state_hash(model.cnn.state_dict())
    if initial_hash!=p['initial_state_sha256'][str(seed)]:raise ValueError('Initial state changed')
    schedule=group_balanced_sample_indices([r['split_group_id'] for r in datasets['train'].rows],
                                          updates=550,batch_size=4,seed=seed)
    schedule_hash=sample_schedule_sha256(schedule,[r['id'] for r in datasets['train'].rows])
    if schedule_hash!=p['sample_order_sha256'][str(seed)]:raise ValueError('Sampling changed')
    opt=torch.optim.AdamW(model.parameters(),lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter();losses=[]
    with (output/'steps.jsonl').open('w',encoding='utf-8',newline='\n') as log:
        for update,indices in enumerate(schedule,1):
            model.train()
            tokens,labels,mask,_=collate([datasets['train'].rows[i] for i in indices])
            tokens,labels,mask=tokens.to(device),labels.to(device),mask.to(device)
            opt.zero_grad(set_to_none=True)
            loss=balanced_pair_bce(model(tokens),labels,mask)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['gradient_clip'],error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
            log.write(json.dumps({'update':update,'loss':losses[-1]})+'\n')
    torch.cuda.synchronize(device)
    seconds=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(device)/1024**2
    config={'condition':condition,'seed':seed,'alpha':p['conditions'][condition],'epsilon':p['epsilon'],
            'model':p['model'],'protocol_sha256':file_sha256(PROTOCOL),'data_sha256':p['data_sha256'],
            'data_manifest_sha256':p['split_manifest_sha256'],'threshold':0.5,'settings':SETTINGS}
    checkpoint=output/'checkpoint.pt'
    torch.save({'model':model.state_dict(),'config':config,'updates':len(losses)},checkpoint)
    result={'condition':condition,'seed':seed,'config':config,'updates':len(losses),
            'initial_state_sha256':initial_hash,'sample_order_sha256':schedule_hash,
            'parameters':sum(x.numel() for x in model.parameters()),'final_cnn_state_sha256':state_hash(model.cnn.state_dict()),
            'checkpoint_sha256':file_sha256(checkpoint),'training_seconds':seconds,'peak_gpu_allocated_MiB':peak,
            'epoch_mean_loss':[sum(losses[i:i+22])/22 for i in range(0,550,22)],
            'splits':{s:assess(model,ds,device) for s,ds in datasets.items()},'test_accessed':False}
    if condition=='cnn':
        historic=ROOT/f'runs/data_first_loss_v2/bce/seed{seed}/checkpoint.pt'
        old=torch.load(historic,map_location='cpu',weights_only=True)
        result['historical_cnn_state_exactly_reproduced']=state_hash(old['model'])==result['final_cnn_state_sha256']
        if not result['historical_cnn_state_exactly_reproduced']:
            raise ValueError('Fresh alpha=0 final CNN weights differ from identical historical budget')
    write(output/'result.json',result)
    print(json.dumps({'condition':condition,'seed':seed,'seconds':seconds,
                      'train_group_f1':result['splits']['train']['group_macro_f1'],
                      'validation_group_f1':result['splits']['validation']['group_macro_f1']}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--condition',choices=['cnn','hybrid_a025','hybrid_a1'])
    parser.add_argument('--seed',type=int,choices=[42,43,44])
    args=parser.parse_args()
    if args.freeze:
        if args.condition or args.seed:parser.error('--freeze is separate from training')
        freeze()
    else:
        if args.condition is None or args.seed is None:parser.error('condition and seed required')
        run(args.condition,args.seed)


if __name__=='__main__':main()
