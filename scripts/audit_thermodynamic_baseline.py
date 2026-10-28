"""Step 1: fixed-condition ViennaRNA BPP versus all three frozen BCE CNNs."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import RNA
import torch
from torch.utils.data import DataLoader
from rna2d.evidence import pairing_evidence_issues
from rna2d.experiment_data import file_sha256, validate_split_manifest
from rna2d.reporting import group_macro_f1
from rna2d.sequence_encoders import SequencePairModel
from rna2d.thermodynamics import SETTINGS, PriorStore, ThermodynamicModel, pair_complementarity
from rna2d.training import Records, collate, evaluate


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2) + '\n', encoding='utf-8')


def stratum(rows, details, field):
    by_id = {r['id']: r for r in rows}
    values = defaultdict(list)
    for d in details:
        row = by_id[d['id']]
        if field == 'length':
            n = len(row['sequence'])
            key = next(f'{lo}-{hi}' for lo, hi in [(0,64),(65,128),(129,256),(257,512)] if lo <= n <= hi)
        else:
            key = row['split_group_id']
        values[key].append(d['f1'])
    return {k: {'samples': len(v), 'macro_f1': statistics.mean(v)} for k, v in values.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / 'runs/thermodynamics_v1/step1'
    output.mkdir(parents=True, exist_ok=False)
    dataset_dir = ROOT / 'data/experimental/joint_split_v2'
    manifest_hash = validate_split_manifest(dataset_dir, ROOT / 'data/manifests/joint_split_v2_validation.json', ['train', 'validation'])
    old_protocol_path = ROOT / 'data/manifests/data_first_loss_protocol_v2.json'
    old_protocol = json.loads(old_protocol_path.read_text())
    if any(file_sha256(dataset_dir / f'{s}.jsonl') != old_protocol['data_sha256'][s]
           for s in ('train', 'validation')):
        raise ValueError('Dataset differs from the historical CNN training/validation data')
    checkpoints = {}
    for seed in (42,43,44):
        checkpoint = ROOT / f'runs/data_first_loss_v2/bce/seed{seed}/checkpoint.pt'
        expected = json.loads(checkpoint.with_name('result.json').read_text())['checkpoint_sha256']
        if file_sha256(checkpoint) != expected:
            raise ValueError('Historical CNN checkpoint fingerprint changed')
        checkpoints[str(seed)] = {'path': checkpoint.relative_to(ROOT).as_posix(), 'sha256': expected}
    protocol = {'stage': 'thermodynamic-baseline-v1', 'status': 'frozen_before_prior_generation_and_evaluation',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'settings': SETTINGS,
                'source': 'ViennaRNA Python API fold_compound, same RNAfold partition-function engine',
                'source_urls': ['https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/usage/man/RNAfold.html',
                                'https://www.tbi.univie.ac.at/RNA/ViennaRNA/doc/html/api/python.html'],
                'data_sha256': {s: file_sha256(dataset_dir / f'{s}.jsonl') for s in ('train','validation')},
                'split_manifest_sha256': manifest_hash, 'historical_protocol_sha256': file_sha256(old_protocol_path),
                'checkpoints': checkpoints, 'threshold': 0.5, 'batch_size': 4, 'test_accessed': False,
                'decode': 'existing canonical AU/GC/GU greedy, min separation 4; crossing edges allowed; decode before observed-mask',
                'family_scope': 'No verified Rfam family field: use isolation/homology groups, not invented family labels',
                'analysis': 'per-split, per-seed positive recovery overlap and FP overlap; group/length/pair-type strata; no tuning',
                'source_sha256': {f: file_sha256(ROOT/f) for f in ('scripts/audit_thermodynamic_baseline.py',
                    'src/rna2d/thermodynamics.py','src/rna2d/training.py','src/rna2d/data.py','src/rna2d/sequence_encoders.py')}}
    protocol_path = ROOT / 'data/manifests/thermodynamic_baseline_protocol_v1.json'
    if protocol_path.exists():
        raise ValueError('Protocol already exists; do not overwrite a completed/failed attempt')
    write(protocol_path, protocol)
    store = PriorStore(ROOT / 'data/processed/thermodynamic_prior_v1')
    start = time.perf_counter()
    datasets = {s: Records(dataset_dir / f'{s}.jsonl') for s in ('train','validation')}
    for dataset in datasets.values():
        for row in dataset.rows:
            if pairing_evidence_issues(row):
                raise ValueError(f"Evidence gate rejected {row['id']}")
            store.get(row['sequence'], create=True)
    RNA.params_save(str(output / 'turner2004.par'))
    generation_seconds = time.perf_counter() - start
    cache_files = {p.name: file_sha256(p) for p in sorted(store.folder.iterdir()) if p.suffix in ('.npy','.json')}
    write(output / 'cache_index.json', cache_files)
    summary = {'protocol_sha256': file_sha256(protocol_path), 'settings': SETTINGS,
               'viennarna_import_path': RNA.__file__, 'test_accessed': False,
               'generation_seconds_98_records': generation_seconds, 'unique_sequence_priors': len(store.memo),
               'cache_index_sha256': file_sha256(output/'cache_index.json'),
               'parameter_file_sha256': file_sha256(output/'turner2004.par'), 'splits': {},
               'environment': {n: importlib.metadata.version(n) for n in ('torch','numpy','ViennaRNA','scipy','scikit-learn')}}
    for split, dataset in datasets.items():
        loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate)
        tm, td = evaluate(ThermodynamicModel(store), loader, args.device, 0.5)
        tg, tpg = group_macro_f1(td, dataset.rows)
        split_result = {'samples': len(dataset), 'groups': len(tpg), 'thermo': {'metrics': tm, 'group_macro_f1': tg,
                        'per_group': tpg, 'length_strata': stratum(dataset.rows, td, 'length')}, 'cnn_seeds': {}}
        t_by_id = {d['id']:d for d in td}
        for seed, checkpoint_info in checkpoints.items():
            checkpoint = torch.load(ROOT / checkpoint_info['path'], map_location=args.device, weights_only=True)
            if checkpoint['protocol_sha256'] != file_sha256(old_protocol_path):
                raise ValueError('Historical checkpoint protocol differs')
            cfg = old_protocol['model']
            model = SequencePairModel(cfg['kind'], cfg['width'], cfg['hidden'], cfg['depth']).to(args.device)
            model.load_state_dict(checkpoint['model'], strict=True)
            cm, cd = evaluate(model, loader, args.device, 0.5)
            cg, cpg = group_macro_f1(cd, dataset.rows)
            if split == 'validation':
                old = json.loads((ROOT / checkpoint_info['path']).with_name('result.json').read_text())
                if abs(cg-old['validation_group_macro_f1']) > 1e-9 or any(
                    x['predicted_pairs'] != y['predicted_pairs'] for x,y in zip(cd,old['details'])):
                    raise ValueError('Historical CNN validation predictions did not reproduce')
            audits=[]
            total=Counter(); by_type=defaultdict(Counter); by_group=defaultdict(Counter); by_length=defaultdict(Counter)
            for row, detail in zip(dataset.rows, cd):
                audit=pair_complementarity(row,t_by_id[row['id']]['predicted_pairs'],detail['predicted_pairs'])
                total.update(audit['counts']);by_group[row['split_group_id']].update(audit['counts'])
                length=next(k for k in tm['length_bins'] if int(k.split('-')[0]) <= len(row['sequence']) <= int(k.split('-')[1]))
                by_length[length].update(audit['counts'])
                for k, counts in audit['by_pair_type'].items():by_type[k].update(counts)
                audits.append({'id':row['id'],'length':len(row['sequence']), 'group':row['split_group_id'], **audit})
            write(output / f'{split}_seed{seed}_details.json', {'thermo':td,'cnn':cd,'complementarity':audits})
            split_result['cnn_seeds'][seed]={'metrics':cm,'group_macro_f1':cg,'per_group':cpg,
                'length_strata':stratum(dataset.rows,cd,'length'),'overlap_counts':dict(total),
                'overlap_by_type':{k:dict(v) for k,v in by_type.items()},
                'overlap_by_group':{k:dict(v) for k,v in by_group.items()},
                'overlap_by_length':{k:dict(v) for k,v in by_length.items()},
                'historical_predictions_reproduced':split=='validation'}
            print(split,seed,'thermo',round(tg,6),'cnn',round(cg,6),'overlap',dict(total),flush=True)
        split_result['cnn_mean_group_macro_f1']=statistics.mean(v['group_macro_f1'] for v in split_result['cnn_seeds'].values())
        summary['splits'][split]=split_result
    summary['limitations']=['Development split already inspected; no independent test or broad family-generalization claim.',
        'Prior treats sequence as isolated RNA, unlike some coordinate-label complex environments.',
        'Standard partition ensemble excludes pseudoknots; marginal decoding does not enforce a noncrossing constraint.',
        'TP overlaps are true-edge recovery, not all-candidate accuracy; false positives reported separately.',
        'Oracle union recovery is descriptive, not an implementable selector or proof of fusion gains.']
    write(ROOT/'data/manifests/thermodynamic_baseline_summary_v1.json',summary)


if __name__ == '__main__':
    main()
