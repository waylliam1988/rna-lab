"""Conservative local geometry and consensus-label gates for canonical RNA."""
from collections import Counter
import numpy as np
from .coordinate_qc import BASE_HEAVY
from .data import CANONICAL

RINGS={
 'A':'N9-C8 C8-N7 N7-C5 C5-C6 C6-N1 N1-C2 C2-N3 N3-C4 C4-C5 C4-N9 C6-N6',
 'G':'N9-C8 C8-N7 N7-C5 C5-C6 C6-N1 N1-C2 C2-N3 N3-C4 C4-C5 C4-N9 C6-O6 C2-N2',
 'C':'N1-C2 C2-N3 N3-C4 C4-C5 C5-C6 C6-N1 C2-O2 C4-N4',
 'U':'N1-C2 C2-N3 N3-C4 C4-C5 C5-C6 C6-N1 C2-O2 C4-O4'}


def residue_geometry(base, atoms):
    required=BASE_HEAVY[base]|{"C1'"}
    if not required<=set(atoms): return 'incomplete_base'
    xyz={k:np.asarray(atoms[k][0],dtype=float) for k in required}
    if any(not np.isfinite(v).all() for v in xyz.values()): return 'nonfinite_coordinates'
    if any(atoms[k][1]<0.9-1e-6 for k in required): return 'low_occupancy'
    if any(not np.isfinite(atoms[k][2]) or atoms[k][2]<0 or atoms[k][2]>150 for k in required):
        return 'b_factor_outside0_150'
    for bond in RINGS[base].split()+["C1'-"+('N9' if base in 'AG' else 'N1')]:
        a,b=bond.split('-')
        if not 1.05<=np.linalg.norm(xyz[a]-xyz[b])<=1.85: return 'base_bond_distance'
    points=np.asarray([xyz[k] for k in BASE_HEAVY[base]])
    singular=np.linalg.svd(points-points.mean(0),compute_uv=False)
    if singular[-1]/np.sqrt(len(points))>0.20: return 'base_nonplanarity'
    return None


def unit_key(text):
    fields=text.split('|')
    if len(fields)<5: return None
    try: model=int(fields[1])
    except ValueError: return None
    insertion=fields[7] if len(fields)>7 else ''
    unusual=(len(fields)>5 and any(fields[5:7])) or (len(fields)>8 and fields[8] not in ('','1_555'))
    return model,fields[2],fields[4],insertion,unusual


def consensus(sequence,local_mask,barnaba_pairs,barnaba_mask,fr3d_rows,record):
    mask=[a and b for a,b in zip(local_mask,barnaba_mask)]
    mapping={}
    for i,item in enumerate(record['residue_auth_mapping']):
        if item:
            key=(str(item[0]),str(item[1] or ''))
            if key in mapping: raise ValueError('nonunique_author_residue_mapping')
            mapping[key]=i
    fr_pairs=set();seen=0;cross=0;unsupported=0
    for u1,u2,code in fr3d_rows:
        k1,k2=unit_key(u1),unit_key(u2)
        if not k1 or not k2: continue
        for first,second in ((k1,k2),(k2,k1)):
            if first[0]!=record['model_id'] or first[1]!=record['auth_asym_id']: continue
            i=mapping.get((first[2],first[3]))
            if i is None: continue
            seen+=1
            if first[4] or second[4] or second[0]!=first[0] or second[1]!=first[1]:
                mask[i]=False;cross+=1;continue
            j=mapping.get((second[2],second[3]))
            if j is None: mask[i]=False;continue
            if code=='cWW' and sequence[i]+sequence[j] in CANONICAL and abs(j-i)>=4:
                fr_pairs.add(tuple(sorted((i,j))))
            else:
                mask[i]=mask[j]=False;unsupported+=1
    if not seen: raise ValueError('independent_annotation_missing_for_chain')
    bb={tuple(sorted(p)) for p in barnaba_pairs}
    eligible=lambda pair: local_mask[pair[0]] and local_mask[pair[1]]
    a={p for p in bb if eligible(p)};b={p for p in fr_pairs if eligible(p)}
    agreement=len(a&b)/len(a|b) if a|b else 0
    if agreement<0.9: raise ValueError('independent_canonical_pair_jaccard_below0_9')
    for i,j in a^b: mask[i]=mask[j]=False
    # If an endpoint loses its supported partner, it cannot become a negative.
    all_pairs=bb|fr_pairs
    changed=True
    while changed:
        changed=False
        for i,j in all_pairs:
            if mask[i]!=mask[j]: mask[i]=mask[j]=False;changed=True
    pairs=sorted([list(p) for p in a&b if mask[p[0]] and mask[p[1]]])
    mult=Counter(i for p in pairs for i in p)
    if any(v>1 for v in mult.values()): raise ValueError('multiple_canonical_partners')
    if not pairs or sum(mask)/len(mask)<0.5: raise ValueError('insufficient_consensus_label_coverage')
    return pairs,mask,dict(canonical_pair_jaccard=agreement,fr3d_cross_chain_or_symmetry_endpoints=cross,
                           fr3d_unsupported_endpoints=unsupported)
