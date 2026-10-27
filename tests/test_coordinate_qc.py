from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.coordinate_qc import BASE_HEAVY, base_observation, parse_coordinate_entry, resolution_screen, select_conformer


def fixture():
    text = '''data_1ABC
_entry.id 1ABC
_exptl.method 'X-RAY DIFFRACTION'
_refine.ls_d_res_high 2.0
_entity_poly.entity_id 1
_entity_poly.type polyribonucleotide
_entity_poly.pdbx_seq_one_letter_code_can AC
_struct_asym.id X
_struct_asym.entity_id 1
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.num
_entity_poly_seq.mon_id
1 1 A
1 2 C
loop_
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.label_atom_id
_atom_site.occupancy
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.B_iso_or_equiv
_atom_site.label_alt_id
'''
    for pos, base in enumerate('AC', 1):
        atoms = BASE_HEAVY[base] | ({"C1'"} if pos == 1 else set())
        for atom in sorted(atoms):
            text += f'X {pos} {atom} 1.0 0 0 0 1 Z {100 + pos} ? 20 .\n'
    return text.encode()


class CoordinateQCTests(unittest.TestCase):
    def test_missing_base_is_unknown_and_auth_mapping_retained(self):
        entry, records = parse_coordinate_entry(fixture(), '1abc.cif', {'1ABC'})
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r['sequence'], 'AC')
        self.assertEqual(r['observed_mask'], [True, False])
        self.assertEqual(r['label_asym_id'], 'X')
        self.assertEqual(r['auth_asym_id'], 'Z')
        self.assertEqual(r['residue_auth_mapping'][0][0], '101')
        self.assertEqual(r['complete_base_coverage'], 0.5)
        self.assertFalse(r['quality_verified'])
        self.assertFalse(r['pair_labels_available'])

    def test_alternate_conformers_are_not_combined(self):
        groups = {'A': {a: 1 for a in BASE_HEAVY['A']}, 'B': {"C1'": 1}}
        selected = select_conformer(groups)
        self.assertFalse(base_observation('A', {1: selected})[0][0])

    def test_low_occupancy_is_masked(self):
        atoms = {a: 1 for a in BASE_HEAVY['U'] | {"C1'"}}
        atoms['O4'] = 0.5
        self.assertEqual(base_observation('U', {1: atoms})[0], [False])

    def test_resolution_and_theoretical_filters(self):
        self.assertTrue(resolution_screen(['X-RAY DIFFRACTION'], 4))
        self.assertEqual(resolution_screen(['ELECTRON MICROSCOPY'], 3), [])
        self.assertTrue(resolution_screen(['SOLUTION NMR'], None))
        self.assertTrue(resolution_screen(['SOLUTION NMR', 'THEORETICAL MODEL'], 2))

    def test_current_pdb_snapshot_is_not_ignored(self):
        entry, records = parse_coordinate_entry(fixture(), '1abc.cif', {'2XYZ'})
        self.assertIn('not_in_frozen_current_experimental_rna_search', entry['entry_screen_reasons'])
        self.assertIsNone(records[0]['observed_mask'])


if __name__ == '__main__':
    unittest.main()
