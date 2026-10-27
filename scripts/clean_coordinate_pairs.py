"""Local atom QC, conservative interfaces, and Barnaba/FR3D cross-checks.

Creates evidence pools, not train/test splits. Modified bases stay quarantined.
"""
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import zipfile
from acquire_experimental_corpus import ROOT,now,write_json,digest_file

sys.path.insert(0,str(ROOT/'src'))
import rna2d
import gemmi
import numpy as np
from scipy.spatial import cKDTree
from rna2d.coordinate_qc import BASE_HEAVY,values,rows
from rna2d.annotation import annotate_chain
from rna2d.fine_qc import residue_geometry,consensus

_ZIP=None;_FR=None


def initialize():
    global _ZIP,_FR
    config=json.loads((ROOT/'data/experimental/coordinate_qc_v1/run_config.json').read_text())
    _ZIP=zipfile.ZipFile(config['archive'])
    _FR=sqlite3.connect(f'file:{(ROOT/"data/experimental/fr3d_index_v1/contacts.sqlite3").as_posix()}?mode=ro',uri=True)


def process(entry):
    pdb=entry['entry']['pdb_id'];accepted_1d=[];accepted_2d=[];audit=[]
    try:
        if entry['source_kind']=='official_cif':
            path=ROOT/entry['path']
            data=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
        else: data=_ZIP.read(entry['member'])
        if hashlib.sha256(data).hexdigest()!=entry['entry']['cif_sha256']: raise ValueError('coordinate_payload_changed')
        candidates=[c for c in entry['chains'] if c['coordinate_screen_pass'] and 20<=c['length']<=10000]
        if not candidates: return dict(pdb_id=pdb,one_d=[],two_d=[],audit=[],error=None)
        block=gemmi.cif.read_string(data.decode('utf-8')).sole_block()
        structure=gemmi.make_structure_from_block(block)
        model=next(m for m in structure if m.num==candidates[0]['model_id'])
        types=dict(rows(block,['_entity_poly.entity_id','_entity_poly.type']))
        nucleic={c for c,e in rows(block,['_struct_asym.id','_struct_asym.entity_id']) if 'ribo' in types.get(e,'')}
        atommap={};global_xyz=[];global_owner=[]
        for chain in model:
            for res in chain:
                if res.subchain not in nucleic or res.label_seq is None: continue
                groups={}
                for atom in res:
                    alt=atom.altloc if atom.altloc!='\x00' else '.'
                    groups.setdefault(alt,{})[atom.name.replace('*',"'")]=atom
                shared=groups.get('.',{})
                opts=[shared|a for alt,a in sorted(groups.items()) if alt!='.'] or [shared]
                selected=max(opts,key=lambda a:sum(x.occ for x in a.values()))
                atoms={name:([a.pos.x,a.pos.y,a.pos.z],a.occ,a.b_iso) for name,a in selected.items() if a.element.name not in ('H','D')}
                atommap[(res.subchain,res.label_seq)]=atoms
                for xyz,occ,b in atoms.values():
                    if occ>0.1 and np.isfinite(xyz).all():
                        global_xyz.append(xyz);global_owner.append((res.subchain,res.label_seq))
        xyz=np.array(global_xyz);tree=cKDTree(xyz)
        clashes=set()
        for a,b in tree.query_pairs(1.0):
            if global_owner[a]!=global_owner[b]: clashes.update((global_owner[a],global_owner[b]))
        contacts=_FR.execute('SELECT u1,u2,code FROM contacts WHERE pdb=?',(pdb,)).fetchall()
        for original in candidates:
            record=dict(original);mask=list(record['observed_mask']);reasons=Counter();interfaces=set()
            for i,base in enumerate(record['sequence']):
                key=(record['label_asym_id'],i+1)
                atoms=atommap.get(key,{})
                issue=residue_geometry(base,atoms)
                if key in clashes: issue='severe_atom_clash'
                if issue: mask[i]=False;reasons[issue]+=1
                # Mask any residue close to another nucleic-acid chain, even if
                # a historical annotation omitted its cross-chain interaction.
                for name in BASE_HEAVY[base]:
                    if name not in atoms: continue
                    if any(global_owner[j][0]!=key[0] for j in tree.query_ball_point(atoms[name][0],5.0)):
                        interfaces.add(i);break
            local_coverage=sum(mask)/len(mask)
            item=dict(id=record['id'],local_residue_exclusions=dict(reasons),local_coverage=local_coverage,
                      interface_positions_masked=len(interfaces))
            if local_coverage<0.95:
                item['reason']='local_quality_coverage_below0_95';audit.append(item);continue
            context='complex' if record['contains_protein'] or record['contains_dna_or_hybrid'] or record['rna_chain_count_in_entry']>1 else 'single_RNA'
            one=dict(id=record['id'],sequence=record['sequence'],sequence_sha256=record['sequence_sha256'],
                     pdb_id=pdb,label_asym_id=record['label_asym_id'],context=context,
                     source_kind=entry['source_kind'],cif_sha256=record['cif_sha256'],local_coverage=local_coverage,
                     evidence_class='experimental_coordinate_sequence',pair_labels_available=False,
                     modified_bases_admitted=False)
            accepted_1d.append(one)
            if record['length']>512:
                item['reason']='pair_length_outside20_512';audit.append(item);continue
            try:
                record['observed_mask']=mask
                with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    annotation=annotate_chain(data,record)
                if annotation['annotation_reasons']: raise ValueError('Barnaba:'+','.join(annotation['annotation_reasons']))
                pair_mask=list(mask)
                for i in interfaces: pair_mask[i]=False
                pairs,label_mask,check=consensus(record['sequence'],pair_mask,annotation['pairs'],annotation['observed_mask'],contacts,record)
                record.update(dict(pairs=pairs,observed_mask=label_mask,context=context,
                                   independent_annotation=check,annotators=['Barnaba','FR3D'],
                                   all_pair_annotations=annotation['all_pair_annotations'],
                                   local_quality=dict(residue_exclusions=dict(reasons),coverage=local_coverage),
                                   interface_positions_masked=sorted(interfaces),
                                   evidence_class='experimental_coordinate_annotation',
                                   provenance_verified=entry['source_kind']=='official_cif',quality_verified=True,contains_model_predictions=False,
                                   pair_labels_available=True,source_kind=entry['source_kind'],
                                   experimental_condition_text=values(block,'_exptl_crystal_grow.details'),
                                   nonpolymer_components=values(block,'_pdbx_entity_nonpoly.comp_id'),
                                   quality_protocol='fine-coordinate-v1: canonical unmodified atoms, occupancy, B<=150, base bonds/planes, clashes, 5A inter-chain mask, Barnaba/FR3D consensus',
                                   negative_label_scope='No consensus canonical contact among unmasked observed residues; unsupported/interface/disagreement endpoints excluded.',
                                   training_split_ready=False))
                accepted_2d.append(record);item['reason']='accepted_consensus_pairs'
            except Exception as exc: item['reason']=str(exc)
            audit.append(item)
        return dict(pdb_id=pdb,one_d=accepted_1d,two_d=accepted_2d,audit=audit,error=None)
    except Exception as exc: return dict(pdb_id=pdb,one_d=[],two_d=[],audit=[],error=str(exc))


def main():
    out=ROOT/'data/experimental/fine_qc_v1';spool=out/'fine_entry_results.jsonl'
    config=dict(protocol='fine-coordinate-v1',code_sha256=digest_file(Path(__file__)),
                module_sha256=digest_file(ROOT/'src/rna2d/fine_qc.py'),
                coordinate_audit_sha256=digest_file(out/'coordinate_entries.jsonl'),
                fr3d_sha256=json.loads((ROOT/'data/experimental/fr3d_index_v1/summary.json').read_text())['sha256'])
    config_path=out/'fine_config.json'
    if config_path.exists() and json.loads(config_path.read_text())!=config: raise ValueError('Inputs/code changed; choose a new version')
    write_json(config_path,config)
    done=set()
    if spool.exists():
        with spool.open(encoding='utf-8') as f: done={json.loads(s)['pdb_id'] for s in f}
    entries=[]
    with (out/'coordinate_entries.jsonl').open(encoding='utf-8') as f:
        for line in f:
            entry=json.loads(line)
            if entry['entry']['pdb_id'] not in done: entries.append(entry)
    with spool.open('a',encoding='utf-8') as f,ProcessPoolExecutor(max_workers=2,initializer=initialize) as pool:
        for n,result in enumerate(pool.map(process,entries,chunksize=1),1):
            f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
            if n%100==0: print('Fine coordinate entries',n,'/',len(entries),flush=True)
    sequences={};paired=defaultdict(list);counts=Counter();errors=[]
    with spool.open(encoding='utf-8') as f, (out/'sequence_records.jsonl').open('w',encoding='utf-8') as seqout:
        for line in f:
            r=json.loads(line)
            if r['error']: errors.append(dict(pdb_id=r['pdb_id'],error=r['error']))
            for row in r['one_d']:
                sequences.setdefault(row['sequence_sha256'],row)
                seqout.write(json.dumps(row)+'\n');counts['usable_sequence_records']+=1
            for row in r['two_d']: paired[row['sequence_sha256']].append(row)
            counts.update('pair_audit:'+a['reason'] for a in r['audit'])
    dedup=[];conflicts=[]
    for h,variants in paired.items():
        shared=[all(v['observed_mask'][i] for v in variants) for i in range(len(variants[0]['sequence']))]
        signatures={tuple(tuple(p) for p in v['pairs'] if shared[p[0]] and shared[p[1]]) for v in variants}
        if len(signatures)!=1 or sum(shared)/len(shared)<0.5 or not next(iter(signatures)):
            conflicts.append(dict(sequence_sha256=h,ids=[v['id'] for v in variants],reason='context_or_conformer_disagreement_or_shared_coverage'))
            continue
        best=dict(min(variants,key=lambda v:(v['resolution_A'],v['id'])))
        best.update(observed_mask=shared,pairs=[list(p) for p in next(iter(signatures))],alternative_structure_ids=[v['id'] for v in variants])
        dedup.append(best)
    with (out/'paired_deduplicated.jsonl').open('w',encoding='utf-8') as f:
        for row in dedup: f.write(json.dumps(row,allow_nan=False)+'\n')
    with (out/'sequences.fasta').open('w',encoding='utf-8') as f:
        for h,row in sorted(sequences.items()): f.write('>'+h+'\n'+row['sequence']+'\n')
    write_json(out/'sequence_conflicts.json',conflicts)
    summary=dict(created_at_utc=now(),coordinate_entries=sum(1 for _ in spool.open(encoding='utf-8')),
                 not_an_approved_release=True,
                 unique_1d_sequences=len(sequences),unique_2d_sequences=len(dedup),
                 unique_2d_single_RNA_sequences=sum(r['context']=='single_RNA' for r in dedup),
                 unique_2d_complex_sequences=sum(r['context']=='complex' for r in dedup),
                 paired_sequence_conflicts=len(conflicts),counts=dict(counts),errors=errors,
                 modified_bases_admitted=0,training_split_ready=False,
                 limitations=['Machine-screened protocol, not manual experimental map re-refinement.',
                              'Modified bases and NMR remain quarantined; unknown is never unpaired.',
                              'FR3D snapshot is historical; absent independent annotations are excluded from this consensus pool.',
                              'Complex-context labels are retained explicitly; do not interpret them as isolated-RNA folding labels.',
                              'Homology/study/construct leakage control must precede model training.'])
    write_json(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
