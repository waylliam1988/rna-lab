"""Independent reload of every frozen run, then report all predeclared conditions."""
import json
from pathlib import Path
import statistics

from run_thermo_comparison import (ROOT,PROTOCOL,assess,checked_protocol,checked_store,
                                  configure_torch,file_sha256,make_model,write)
import torch
from rna2d.thermodynamics import ThermodynamicModel, pair_complementarity


def main():
    configure_torch()
    p,datasets=checked_protocol();store=checked_store(p);device=torch.device('cuda:0')
    for ds in datasets.values():
        for row in ds.rows:store.get(row['sequence'])
    baseline=assess(ThermodynamicModel(store),datasets['validation'],device)
    if abs(baseline['group_macro_f1']-p['thermo_validation_group_macro_f1'])>1e-9:
        raise ValueError('Thermodynamic baseline did not reproduce')
    runs={};verification=[]
    for condition,alpha in p['conditions'].items():
        runs[condition]=[]
        for seed in p['seeds']:
            folder=ROOT/f'runs/thermodynamics_v1/step3/{condition}/seed{seed}'
            result=json.loads((folder/'result.json').read_text())
            checkpoint_path=folder/'checkpoint.pt'
            if file_sha256(checkpoint_path)!=result['checkpoint_sha256']:raise ValueError('Checkpoint hash changed')
            checkpoint=torch.load(checkpoint_path,map_location=device,weights_only=True)
            cfg=checkpoint['config']
            if cfg!=result['config'] or cfg['protocol_sha256']!=file_sha256(PROTOCOL) or cfg['alpha']!=alpha:
                raise ValueError('Checkpoint config differs')
            if result['updates']!=550 or checkpoint['updates']!=550:raise ValueError('Budget differs')
            if result['initial_state_sha256']!=p['initial_state_sha256'][str(seed)] or result['sample_order_sha256']!=p['sample_order_sha256'][str(seed)]:
                raise ValueError('Paired initialization/schedule differs')
            model=make_model(p,store,alpha,device)
            model.load_state_dict(checkpoint['model'],strict=True)
            for split,ds in datasets.items():
                reproduced=assess(model,ds,device);old=result['splits'][split]
                if reproduced['metrics']!=old['metrics'] or reproduced['per_group']!=old['per_group'] or reproduced['details']!=old['details']:
                    raise ValueError(f'Reload results differ: {condition}/{seed}/{split}')
            runs[condition].append(result)
            verification.append({'condition':condition,'seed':seed,'checkpoint_sha256':result['checkpoint_sha256'],
                                 'train_and_validation_exactly_reproduced':True})
    summary={'protocol_sha256':file_sha256(PROTOCOL),'all_reload_checks_passed':True,
             'reloaded_checkpoints':9,'verification':verification,'test_accessed':False,
             'thermo':{k:v for k,v in baseline.items() if k!='details'},'conditions':{},'primary_gate':{}}
    by_thermo={x['id']:x for x in baseline['details']}
    for condition,rows in runs.items():
        f1s=[r['splits']['validation']['group_macro_f1'] for r in rows]
        groups={g:statistics.mean(r['splits']['validation']['per_group'][g] for r in rows)
                for g in baseline['per_group']}
        per_seed=[]
        for row in rows:
            audits=[pair_complementarity(r,by_thermo[r['id']]['predicted_pairs'],d['predicted_pairs'])
                    for r,d in zip(datasets['validation'].rows,row['splits']['validation']['details'])]
            overlap={key:sum(a['counts'][key] for a in audits) for key in audits[0]['counts']}
            per_seed.append({'seed':row['seed'],'group_macro_f1':row['splits']['validation']['group_macro_f1'],
                             'delta_vs_thermo':row['splits']['validation']['group_macro_f1']-baseline['group_macro_f1'],
                             'overlap_with_thermo':overlap})
        summary['conditions'][condition]={'alpha':p['conditions'][condition],'group_macro_f1_seeds':f1s,
            'group_macro_f1_mean':statistics.mean(f1s),'group_macro_f1_stdev':statistics.stdev(f1s),
            'rna_macro_f1_mean':statistics.mean(r['splits']['validation']['metrics']['macro_f1'] for r in rows),
            'train_group_macro_f1_mean':statistics.mean(r['splits']['train']['group_macro_f1'] for r in rows),
            'mean_training_seconds':statistics.mean(r['training_seconds'] for r in rows),
            'mean_peak_gpu_allocated_MiB':statistics.mean(r['peak_gpu_allocated_MiB'] for r in rows),
            'validation_eval_seconds_mean':statistics.mean(r['splits']['validation']['evaluation_seconds_including_decode'] for r in rows),
            'mean_per_group_f1':groups,'per_seed':per_seed,
            'length_strata':{k:{'samples':v['samples'],'mean_macro_f1':statistics.mean(
                r['splits']['validation']['metrics']['length_bins'][k]['macro_f1'] for r in rows)}
                for k,v in rows[0]['splits']['validation']['metrics']['length_bins'].items()},
            'all_historical_cnn_states_exact':all(r.get('historical_cnn_state_exactly_reproduced',False) for r in rows) if condition=='cnn' else None}
    primary=summary['conditions'][p['primary_hybrid']];cnn=summary['conditions']['cnn']
    delta_groups={g:primary['mean_per_group_f1'][g]-v for g,v in baseline['per_group'].items()}
    rules={'all_seeds_beat_thermo':all(f>baseline['group_macro_f1'] for f in primary['group_macro_f1_seeds']),
           'all_seeds_beat_paired_cnn':all(a>b for a,b in zip(primary['group_macro_f1_seeds'],cnn['group_macro_f1_seeds'])),
           'mean_gain_at_least_0_02':primary['group_macro_f1_mean']-baseline['group_macro_f1']>=0.02,
           'at_least_4_groups_improve':sum(d>0 for d in delta_groups.values())>=4,
           'no_group_drops_over_0_10':all(d>=-0.10 for d in delta_groups.values())}
    summary['primary_gate']={'condition':p['primary_hybrid'],'passed':all(rules.values()),'rules':rules,
                             'mean_gain_vs_thermo':primary['group_macro_f1_mean']-baseline['group_macro_f1'],
                             'per_group_delta_vs_thermo':delta_groups,
                             'sensitivity_only_cannot_rescue_failure':True}
    summary['decision']='primary eligible for further independent validation; no automatic default change' if all(rules.values()) else 'primary development gate failed; do not adopt as improvement or retune on this validation set'
    summary['limitations']=p['limitations']
    summary['prior_generation_seconds_separate']=json.loads((ROOT/'data/manifests/thermodynamic_baseline_summary_v1.json').read_text())['generation_seconds_98_records']
    write(ROOT/'data/manifests/thermo_comparison_summary_v1.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k in ('all_reload_checks_passed','primary_gate','decision')},indent=2))


if __name__=='__main__':main()
