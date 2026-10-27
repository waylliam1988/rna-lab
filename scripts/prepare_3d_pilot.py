"""Prepare genuine static 3D anchor observations, not folding trajectories.

Pilot scope: already quality-screened, unmodified single RNA, length20-256.
C4', C1' and base glycosidic N anchors remain in original Angstrom coordinates.
This does not certify an all-atom force field or create dynamic labels.
"""
from collections import Counter
import gzip
import hashlib
import json
import sys
from acquire_experimental_corpus import ROOT,write_json,digest_file
sys.path.insert(0,str(ROOT/'src'))
import rna2d
import gemmi
import numpy as np


def extract_anchors(sequence,residues):
    anchors=[];mask=[];reasons=Counter()
    for i,base in enumerate(sequence,1):
        residue=residues.get(i);selected=None
        if residue is not None:
            groups={}
            for atom in residue:
                alt=atom.altloc if atom.altloc!='\x00' else '.'
                groups.setdefault(alt,{})[atom.name.replace('*',"'")]=atom
            shared=groups.get('.',{})
            options=[shared|a for alt,a in sorted(groups.items()) if alt!='.'] or [shared]
            needed=["C4'","C1'",'N9' if base in 'AG' else 'N1']
            for option in options:
                if not all(name in option for name in needed):continue
                atoms=[option[name] for name in needed]
                if any(a.occ<0.9-1e-6 or not 0<=a.b_iso<=150 for a in atoms):continue
                xyz=np.array([[a.pos.x,a.pos.y,a.pos.z] for a in atoms])
                if not np.isfinite(xyz).all():continue
                if np.linalg.norm(np.cross(xyz[0]-xyz[1],xyz[2]-xyz[1]))<0.5:continue
                selected=xyz.tolist();break
        anchors.append(selected);mask.append(selected is not None)
        if selected is None:reasons['missing_or_ambiguous_anchor_geometry']+=1
    # Broad continuity screen only; not a calibrated RNA potential.
    for i in range(len(sequence)-1):
        if anchors[i] is None or anchors[i+1] is None:continue
        distance=np.linalg.norm(np.array(anchors[i][0])-np.array(anchors[i+1][0]))
        if not 3<=distance<=9:
            mask[i]=mask[i+1]=False;reasons['C4_neighbor_distance_outside3_9A']+=1
    return anchors,mask,dict(reasons)


def main():
    out=ROOT/'data/experimental/three_d_pilot_v1';out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise ValueError('Pilot version already frozen')
    source=ROOT/'data/experimental/clean_release_v1/coordinate_sequence_records.jsonl'
    with source.open(encoding='utf-8') as f:
        candidates=[json.loads(l) for l in f]
    candidates=[r for r in candidates if r['context']=='single_RNA' and 20<=len(r['sequence'])<=256]
    accepted=[];audit=[]
    for row in candidates:
        try:
            receipt=row['source_download_receipt']
            path=ROOT/receipt['path'] if 'path' in receipt else ROOT/'data/raw/experimental_cif_v1'/(row['pdb_id'].lower()+'.cif')
            data=gzip.open(path,'rb').read() if path.suffix=='.gz' else path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['cif_sha256']:raise ValueError('Official coordinate hash mismatch')
            block=gemmi.cif.read_string(data.decode('utf-8')).sole_block();structure=gemmi.make_structure_from_block(block)
            model=min(structure,key=lambda m:m.num)
            residues={}
            for chain in model:
                for residue in chain:
                    if residue.subchain==row['label_asym_id'] and residue.label_seq is not None:
                        if residue.label_seq in residues:raise ValueError('Nonunique residue mapping')
                        residues[residue.label_seq]=residue
            anchors,mask,reasons=extract_anchors(row['sequence'],residues)
            coverage=sum(mask)/len(mask)
            audit.append(dict(id=row['id'],coverage=coverage,reasons=reasons))
            if coverage<0.95:continue
            accepted.append(dict(row,anchors_A=anchors,anchor_order=['C4prime','C1prime','N9_or_N1'],
                                 observed_3d_mask=mask,model_id=model.num,coordinate_units='Angstrom',
                                 evidence_class='experimental_static_3d_anchor_observations',
                                 folding_trajectory_labels=False,training_split_ready=False))
        except Exception as exc:audit.append(dict(id=row['id'],error=str(exc)))
    path=out/'static_anchors.jsonl'
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in accepted),encoding='utf-8',newline='\n')
    (out/'audit.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in audit),encoding='utf-8',newline='\n')
    summary=dict(candidate_structures=len(candidates),retained_static_structures=len(accepted),
                 unique_sequences=len({r['sequence_sha256'] for r in accepted}),
                 source_sha256=digest_file(source),output_sha256=digest_file(path),
                 errors=[r for r in audit if 'error' in r],coordinate_units='Angstrom',
                 folding_trajectory_labels=False,training_split_ready=False,model_trained=False,
                 limitations=['Static experimental anchor observations only, not time-ordered states or thermodynamic occupancies.',
                              'Additional anchor and broad continuity screening; not full backbone/all-atom geometry or density validation.',
                              'Conformers/constructs must stay in the same homology/study group before any 3D training.',
                              'Old Nezha physics-unit/force/thermostat contracts must be audited before adapting these coordinates.'])
    write_json(out/'summary.json',summary);write_json(ROOT/'data/manifests/three_d_pilot_v1.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
