"""Joint 3D/2D study, construct and known-homology isolation; no model scores."""
from collections import defaultdict, Counter
import gzip
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from acquire_experimental_corpus import ROOT, write_json, digest_file
from build_joint_splits import load, study_map, sequence_edges
import gemmi


def pose_from_residues(sequence, residues):
    """C4' positions in nm, base-ring plane normals from one conformer.

    Chemical atom ordering fixes the normal sign convention. This is not a
    measured angular velocity or a validated orientation-energy model.
    """
    pose=[]
    for i,base in enumerate(sequence,1):
        residue=residues.get(i)
        if residue is None:return None
        groups=defaultdict(dict)
        for atom in residue:groups[atom.altloc][atom.name.replace('*',"'")]=atom
        shared=groups.get('\x00',{})
        options=[shared|v for k,v in sorted(groups.items()) if k!='\x00'] or [shared]
        ring=['N9','C8','N7'] if base in 'AG' else ['N1','C2','N3']
        for option in options:
            needed=["C4'"]+ring
            if not all(n in option for n in needed):continue
            atoms=[option[n] for n in needed]
            if any(a.occ<.9-1e-6 or not 0<=a.b_iso<=150 for a in atoms):continue
            xyz=np.array([[a.pos.x,a.pos.y,a.pos.z] for a in atoms])
            if not np.isfinite(xyz).all():continue
            normal=np.cross(xyz[2]-xyz[1],xyz[3]-xyz[1]);size=np.linalg.norm(normal)
            if size<.5:continue
            pose.append((xyz[0]*.1).tolist()+(normal/size).tolist());break
        else:return None
    return pose


def main():
    out=ROOT/'data/experimental/three_d_isolated_v1';out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Version already frozen')
    source=ROOT/'data/experimental/three_d_pilot_v1/static_anchors.jsonl'
    pilot=load(source)
    # Include quarantined 2D records as bridges, not just previously assigned ones.
    old=[];oldrole={}
    for split in ('train','validation','test','quarantine'):
        for r in load(ROOT/'data/experimental/joint_split_v2'/(split+'.jsonl')):
            if split!='quarantine':oldrole[r['sequence_sha256']]=split
            old.append(r)
    unique={}
    for r in old+pilot:
        h=r['sequence_sha256']
        if h not in unique:unique[h]=dict(r,alternative_structure_ids=[])
        unique[h]['alternative_structure_ids']=sorted(set(unique[h]['alternative_structure_ids']+r.get('alternative_structure_ids',[r['id']])))
    records=list(unique.values());studies=study_map(records);components=defaultdict(set)
    for r in load(ROOT/'data/experimental/rna3db_all_chain_metadata.jsonl'):
        c=r.get('homology_component')
        if c and c!='component_0':components[r['sequence']].add(c)
    parent=list(range(len(records)));owners={}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):parent[find(b)]=find(a)
    for i,r in enumerate(records):
        pdbs=sorted({a.split('_')[0] for a in r['alternative_structure_ids']})
        r['source_pdb_ids']=pdbs;r['study_ids']=sorted({s for p in pdbs for s in studies.get(p,[])})
        r['study_metadata_complete']=all(studies.get(p) for p in pdbs)
        r['homology_components']=sorted(components[r['sequence']])
        for key in ['seq:'+r['sequence_sha256']]+['pdb:'+p for p in pdbs]+r['study_ids']+r['homology_components']:
            if key in owners:union(i,owners[key])
            owners[key]=i
    index={r['id']:i for i,r in enumerate(records)};edges=list(sequence_edges(records))
    for e in edges:union(index[e['a']],index[e['b']])
    groups=defaultdict(list)
    for i,r in enumerate(records):groups[find(i)].append(r)
    info={};newgroups=[]
    for group in groups.values():
        gid=hashlib.sha256('\n'.join(sorted(r['sequence_sha256'] for r in group)).encode()).hexdigest()
        roles={oldrole[r['sequence_sha256']] for r in group if r['sequence_sha256'] in oldrole}
        if 'test' in roles:role='quarantine_previously_exposed_test_group'
        elif len(roles)>1:role='quarantine_bridges_old_train_validation'
        elif roles:role=next(iter(roles))
        else:role='new';newgroups.append(gid)
        for r in group:info[r['sequence_sha256']]=(gid,role,r)
    # Entirely new groups, if available, get deterministic roles before any fitting.
    newgroups=sorted(set(newgroups),key=lambda g:hashlib.sha256(('3d-v1:'+g).encode()).hexdigest())
    newroles={g:('test' if i==0 and len(newgroups)>=3 else 'validation' if i==1 and len(newgroups)>=3 else 'train') for i,g in enumerate(newgroups)}
    assigned=defaultdict(list);excluded=[]
    for row in pilot:
        gid,role,meta=info[row['sequence_sha256']]
        if role=='new':role=newroles[gid]
        if not meta['homology_components'] or not meta['study_metadata_complete']:role='quarantine_unverified_homology_or_study'
        if not all(row['observed_3d_mask']):role='quarantine_incomplete_anchor_chain'
        if role.startswith('quarantine'):
            excluded.append(dict(id=row['id'],reason=role,group=gid));continue
        receipt=row['source_download_receipt'];path=ROOT/receipt['path'] if 'path' in receipt else ROOT/'data/raw/experimental_cif_v1'/(row['pdb_id'].lower()+'.cif')
        data=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['cif_sha256']:raise ValueError('Coordinate source changed')
        structure=gemmi.make_structure_from_block(gemmi.cif.read_string(data.decode()).sole_block())
        model=next(m for m in structure if m.num==row['model_id'])
        residues={r.label_seq:r for c in model for r in c if r.subchain==row['label_asym_id'] and r.label_seq is not None}
        pose=pose_from_residues(row['sequence'],residues)
        if pose is None:excluded.append(dict(id=row['id'],reason='incomplete_base_plane',group=gid));continue
        result=dict(row,split=role,split_group_id=gid,training_split_ready=True,q_nm=pose,orientation_source='experimental_base_ring_plane_with_chemical_sign_convention',homology_components=meta['homology_components'],study_ids=meta['study_ids'])
        assigned[role].append(result)
    # Retain conformers in the corpus but train once per sequence to avoid alias weighting.
    for role in ('train','validation','test'):
        rs=sorted(assigned[role],key=lambda r:r['id'])
        (out/(role+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rs),encoding='utf-8',newline='\n')
    (out/'quarantine.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in excluded),encoding='utf-8',newline='\n')
    seen=defaultdict(set)
    for role,rs in assigned.items():
        for r in rs:
            for key in [r['sequence_sha256'],r['split_group_id']]+r['study_ids']+r['homology_components']:seen[key].add(role)
    assert all(len(v)==1 for v in seen.values())
    summary=dict(source_sha256=digest_file(source),splits={s:dict(structures=len(assigned[s]),sequences=len({r['sequence_sha256'] for r in assigned[s]}),groups=len({r['split_group_id'] for r in assigned[s]})) for s in ('train','validation','test')},excluded=dict(Counter(r['reason'] for r in excluded)),output_sha256={s:digest_file(out/(s+'.jsonl')) for s in ('train','validation','test')},validation_passed=True,one_d_pretraining_enabled=False,folding_trajectory_labels=False,new_group_count=len(newgroups),limits=['Known homology/study and local alignment controls are conservative, not a proof against undiscovered remote homology.','Old test-related groups are quarantined; old validation remains development-only.','Static structures supervise geometry only; any simulated transitions must be labeled synthetic.','C4 position/base-plane coarse graining differs from undocumented old CSV atom conventions; no atom-equivalence claim.'])
    write_json(out/'study_map.json',studies);write_json(out/'sequence_edges.json',edges)
    write_json(out/'summary.json',summary);write_json(ROOT/'data/manifests/three_d_isolated_v1.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
