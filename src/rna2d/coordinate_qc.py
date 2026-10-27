"""Conservative residue-level quality screening of experimental mmCIF files."""
from collections import defaultdict
import hashlib
import math
import re

import gemmi

BASE_HEAVY = {
    'A': set('N9 C8 N7 C5 C6 N6 N1 C2 N3 C4'.split()),
    'G': set('N9 C8 N7 C5 C6 O6 N1 C2 N2 N3 C4'.split()),
    'C': set('N1 C2 O2 N3 C4 N4 C5 C6'.split()),
    'U': set('N1 C2 O2 N3 C4 O4 C5 C6'.split()),
}


def values(block, tag):
    return [gemmi.cif.as_string(x) for x in block.find_values(tag)]


def rows(block, tags):
    table = block.find(tags)
    return ([gemmi.cif.as_string(x) for x in row] for row in table)


def finite_number(value):
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def resolution_screen(methods, resolution):
    if any('THEORETICAL' in m.upper() for m in methods):
        return ['theoretical_or_mixed_theoretical_method']
    if methods == ['X-RAY DIFFRACTION']:
        return [] if resolution is not None and 0 < resolution <= 3 else ['xray_resolution_gt3_or_missing']
    if methods == ['ELECTRON MICROSCOPY']:
        return [] if resolution is not None and 0 < resolution <= 3.5 else ['em_resolution_gt3_5_or_missing']
    if any('NMR' in m.upper() for m in methods):
        return ['nmr_requires_separate_restraint_ensemble_qc']
    return ['experimental_method_requires_review']


def base_observation(sequence, atom_sets, occupancy_cutoff=0.9):
    mask, counts = [], []
    for index, base in enumerate(sequence, 1):
        required = BASE_HEAVY.get(base, set()) | {"C1'"}
        atoms = atom_sets.get(index, {})
        mask.append(base in BASE_HEAVY and all(atoms.get(atom, 0) >= occupancy_cutoff for atom in required))
        counts.append(len(set(atoms) & required))
    return mask, counts


def select_conformer(groups):
    shared = groups.get('.', {}) | groups.get('?', {}) | groups.get('', {})
    alternatives = [alt for alt in groups if alt not in {'.', '?', ''}]
    if not alternatives:
        return shared
    options = [shared | groups[alt] for alt in sorted(alternatives)]
    return max(options, key=lambda atoms: (sum(v >= 0.9 for v in atoms.values()), sum(atoms.values())))


def parse_coordinate_entry(data, source_member, current_ids=None):
    block = gemmi.cif.read_string(data.decode('utf-8')).sole_block()
    entry_ids = values(block, '_entry.id')
    pdb_id = entry_ids[0].upper() if entry_ids else source_member.rsplit('/', 1)[-1].split('.')[0].upper()
    methods = values(block, '_exptl.method')
    resolutions = [finite_number(x) for tag in ('_refine.ls_d_res_high', '_em_3d_reconstruction.resolution') for x in values(block, tag)]
    resolutions = [x for x in resolutions if x is not None and x > 0]
    resolution = max(resolutions) if resolutions else None
    reasons = resolution_screen(methods, resolution)
    if current_ids is not None and pdb_id not in current_ids:
        reasons.append('not_in_frozen_current_experimental_rna_search')
    filename_id = source_member.rsplit('/', 1)[-1].split('.')[0].upper()
    if pdb_id != filename_id:
        reasons.append('entry_filename_id_mismatch')
    entity_types = dict(rows(block, ['_entity_poly.entity_id', '_entity_poly.type']))
    sequences = {eid: re.sub(r'\s+', '', seq).upper() for eid, seq in rows(block, ['_entity_poly.entity_id', '_entity_poly.pdbx_seq_one_letter_code_can'])}
    rna_entities = {eid for eid, kind in entity_types.items() if kind == 'polyribonucleotide'}
    chains = [(chain, eid) for chain, eid in rows(block, ['_struct_asym.id', '_struct_asym.entity_id']) if eid in rna_entities]
    has_protein = any('polypeptide' in kind for kind in entity_types.values())
    has_dna_or_hybrid = any('deoxyribo' in kind or 'hybrid' in kind for kind in entity_types.values())
    entity_monomers = defaultdict(dict)
    for eid, num, mon in rows(block, ['_entity_poly_seq.entity_id', '_entity_poly_seq.num', '_entity_poly_seq.mon_id']):
        if eid in rna_entities and num.isdigit():
            entity_monomers[eid].setdefault(int(num), set()).add(mon)
    info = {'pdb_id': pdb_id, 'source_member': source_member, 'cif_sha256': hashlib.sha256(data).hexdigest(),
            'experimental_methods': methods, 'resolution_A': resolution, 'entry_screen_reasons': reasons,
            'rna_chains': len(chains), 'contains_protein': has_protein, 'contains_dna_or_hybrid': has_dna_or_hybrid,
            'deposition_date': (values(block, '_pdbx_database_status.recvd_initial_deposition_date') or [None])[0],
            'local_quality_note': 'Global resolution only; local map/geometry validation not available here.'}
    # Keep metadata for rejected entries; do not manufacture residue coverage.
    atom_sets = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    auth_mapping, auth_chains, atom_models = defaultdict(dict), {}, set()
    b_factors = defaultdict(list)
    if chains and not reasons:
        tags = ['_atom_site.label_asym_id', '_atom_site.label_seq_id', '_atom_site.label_atom_id',
                '_atom_site.occupancy', '_atom_site.Cartn_x', '_atom_site.Cartn_y', '_atom_site.Cartn_z',
                '_atom_site.pdbx_PDB_model_num', '_atom_site.auth_asym_id', '_atom_site.auth_seq_id',
                '_atom_site.pdbx_PDB_ins_code', '_atom_site.B_iso_or_equiv', '_atom_site.label_alt_id']
        table = block.find(tags)
        if not table:
            reasons.append('atom_site_fields_missing')
        else:
            model_numbers = values(block, '_atom_site.pdbx_PDB_model_num')
            model = min((int(x) for x in model_numbers if x.isdigit()), default=1)
            rna_chain_set = {c for c, _ in chains}
            for row in table:
                chain = gemmi.cif.as_string(row[0])
                if chain not in rna_chain_set or row[7] != str(model):
                    continue
                seq_id = gemmi.cif.as_string(row[1])
                if not seq_id.isdigit():
                    continue
                idx = int(seq_id)
                atom = gemmi.cif.as_string(row[2]).replace('*', "'")
                occ = finite_number(row[3])
                xyz = [finite_number(row[k]) for k in (4, 5, 6)]
                if occ is None or not 0 < occ <= 1.001 or any(v is None or abs(v) > 1e6 for v in xyz):
                    continue
                # Alternate conformers cannot be combined to invent a complete base.
                # High occupancy threshold conservatively masks split-occupancy atoms.
                alt = gemmi.cif.as_string(row[12])
                atom_sets[chain][idx][alt][atom] = max(atom_sets[chain][idx][alt].get(atom, 0), occ)
                auth_chains[chain] = gemmi.cif.as_string(row[8])
                auth_mapping[chain][idx] = [gemmi.cif.as_string(row[9]), gemmi.cif.as_string(row[10])]
                b = finite_number(row[11])
                if b is not None:
                    b_factors[chain].append(b)
            atom_models.add(model)
    records = []
    for chain, eid in chains:
        seq = sequences.get(eid, '')
        chain_reasons = list(reasons)
        modifications = [i - 1 for i, monomers in entity_monomers[eid].items() if len(monomers) != 1 or next(iter(monomers)) not in BASE_HEAVY]
        if not seq or set(seq) - set('ACGU'):
            chain_reasons.append('canonical_sequence_ambiguous_or_missing')
        if set(entity_monomers[eid]) != set(range(1, len(seq) + 1)):
            chain_reasons.append('entity_sequence_numbering_inconsistent')
        if modifications:
            chain_reasons.append('modified_or_microheterogeneous_residues_require_review')
        observed = None
        coverage = None
        if not reasons:
            consistent_atoms = {idx: select_conformer(groups) for idx, groups in atom_sets[chain].items()}
            observed, _ = base_observation(seq, consistent_atoms)
            coverage = sum(observed) / len(seq) if seq else 0
            if coverage < 0.95:
                chain_reasons.append('complete_base_coverage_below_0_95')
        records.append({'id': f'{pdb_id}_{chain}', 'pdb_id': pdb_id, 'label_asym_id': chain,
                        'auth_asym_id': auth_chains.get(chain), 'entity_id': eid,
                        'sequence': seq, 'length': len(seq), 'sequence_sha256': hashlib.sha256(seq.encode()).hexdigest(),
                        'observed_mask': observed, 'complete_base_coverage': coverage,
                        'modified_positions_0based': modifications,
                        'residue_auth_mapping': [auth_mapping[chain].get(i) for i in range(1, len(seq) + 1)] if observed is not None else None,
                        'rna_chain_count_in_entry': len(chains), 'contains_protein': has_protein,
                        'contains_dna_or_hybrid': has_dna_or_hybrid,
                        'model_id': min(atom_models) if atom_models else None,
                        'experimental_methods': methods, 'resolution_A': resolution,
                        'source_member': source_member, 'cif_sha256': info['cif_sha256'],
                        'quality_screen_reasons': chain_reasons, 'coordinate_screen_pass': not chain_reasons,
                        'evidence_class': 'experimental_coordinate_candidate',
                        'provenance_verified': False, 'quality_verified': False,
                        'pair_labels_available': False, 'contains_model_predictions': False,
                        'note': 'Observed mask requires complete canonical base heavy atoms plus C1-prime, occupancy >=0.9. No pairs inferred.'})
    return info, records
