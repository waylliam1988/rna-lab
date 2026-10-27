"""Adapter to Barnaba's published geometry annotator; no fold prediction."""
from collections import Counter
from importlib.metadata import version
import inspect
import hashlib

import gemmi
import mdtraj as md
import numpy as np
import barnaba as bb

from .coordinate_qc import BASE_HEAVY
from .data import CANONICAL


def trajectory_for_chain(data, chain_record):
    block = gemmi.cif.read_string(data.decode('utf-8')).sole_block()
    structure = gemmi.make_structure_from_block(block)
    model = next((m for m in structure if m.num == chain_record['model_id']), None)
    if model is None:
        raise ValueError('Coordinate model not found')
    by_position = {}
    for chain in model:
        for residue in chain:
            if residue.subchain == chain_record['label_asym_id'] and residue.label_seq is not None:
                if residue.label_seq in by_position:
                    raise ValueError('Multiple residue objects at one label_seq_id')
                by_position[residue.label_seq] = residue
    top = md.Topology()
    chain = top.add_chain()
    coords, mapping = [], []
    for position, (base, observed) in enumerate(zip(chain_record['sequence'], chain_record['observed_mask']), 1):
        if not observed:
            continue
        residue = by_position.get(position)
        if residue is None or residue.name != base:
            raise ValueError('Residue name/mapping mismatch')
        groups = {}
        for atom in residue:
            alt = atom.altloc if atom.altloc != '\x00' else '.'
            groups.setdefault(alt, {})[atom.name.replace('*', "'")] = atom
        shared = groups.get('.', {})
        options = [shared | atoms for alt, atoms in sorted(groups.items()) if alt != '.'] or [shared]
        # Gemmi stores occupancy as float32; 0.9 may round to 0.899999976.
        atoms = max(options, key=lambda x: (sum(a.occ >= 0.9 - 1e-6 for a in x.values()), sum(a.occ for a in x.values())))
        required = BASE_HEAVY[base] | {"C1'"}
        if not required <= set(atoms) or any(atoms[name].occ < 0.9 - 1e-6 for name in required):
            raise ValueError('Observed residue lost required atoms in adapter')
        md_residue = top.add_residue(base, chain, resSeq=position)
        for name in sorted(required):
            atom = atoms[name]
            top.add_atom(name, md.element.get_by_symbol(name[0]), md_residue)
            coords.append([atom.pos.x / 10, atom.pos.y / 10, atom.pos.z / 10])  # Angstrom -> nm
        mapping.append(position - 1)
    if len(mapping) < 2:
        raise ValueError('Insufficient observable residues')
    return md.Trajectory(np.array([coords], dtype=np.float32), top), mapping


def classify_pair_annotations(sequence, mapping, raw_pairs, codes, observed_mask):
    pairs, all_annotations = set(), []
    mask = list(observed_mask)
    reasons = []
    for indices, code in zip(raw_pairs, codes):
        i, j = sorted((mapping[int(indices[0])], mapping[int(indices[1])]))
        all_annotations.append({'i': i, 'j': j, 'barnaba_code': str(code)})
        if code in {'WCc', 'GUc'}:
            if sequence[i] + sequence[j] not in CANONICAL:
                reasons.append('canonical_annotation_sequence_mismatch')
            elif j - i < 4:
                mask[i] = mask[j] = False
            else:
                pairs.add((i, j))
        elif code != 'XXX':
            # Preserve noncanonical/weak/unknown-orientation contacts, but do not
            # turn their endpoints into negatives in the canonical-only task.
            mask[i] = mask[j] = False
    multiplicity = Counter(x for pair in pairs for x in pair)
    if any(count > 1 for count in multiplicity.values()):
        reasons.append('multiple_canonical_partners')
    pairs = sorted([list(pair) for pair in pairs if mask[pair[0]] and mask[pair[1]]])
    if not pairs:
        reasons.append('no_unambiguous_canonical_pairs')
    return pairs, mask, all_annotations, sorted(set(reasons))


def annotate_chain(data, chain_record):
    traj, mapping = trajectory_for_chain(data, chain_record)
    _, pairings, residues = bb.annotate_traj(traj)
    if len(residues) != len(mapping):
        raise ValueError('Barnaba skipped residues; annotation index mapping would be unsafe')
    pairs, mask, annotations, reasons = classify_pair_annotations(
        chain_record['sequence'], mapping, pairings[0][0], pairings[0][1], chain_record['observed_mask'])
    return {'pairs': pairs, 'observed_mask': mask, 'all_pair_annotations': annotations,
            'annotation_reasons': reasons, 'annotator': 'Barnaba', 'annotator_version': version('barnaba'),
            'annotator_function_sha256': hashlib.sha256(inspect.getsource(bb.annotate_traj).encode()).hexdigest(),
            'annotation_policy': 'strict WCc/GUc; canonical AU/GC/GU; >=4 index separation; unsupported contact endpoints masked',
            'negative_label_scope': 'No strict canonical interaction annotated between observable, unmasked residues; not proof of absence of all interactions.'}
