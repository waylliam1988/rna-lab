from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.annotation import classify_pair_annotations, trajectory_for_chain
from rna2d.coordinate_qc import parse_coordinate_entry
from test_coordinate_qc import fixture


class AnnotationAdapterTests(unittest.TestCase):
    def test_noncanonical_contacts_mask_endpoints(self):
        pairs, mask, raw, reasons = classify_pair_annotations('GAAACAAU', list(range(8)),
                                                             [[0, 4], [1, 7]], ['WCc', 'WHt'], [True] * 8)
        self.assertEqual(pairs, [[0, 4]])
        self.assertFalse(mask[1])
        self.assertFalse(mask[7])
        self.assertEqual(len(raw), 2)
        self.assertEqual(reasons, [])

    def test_multiple_partners_quarantined_not_greedily_resolved(self):
        _, _, _, reasons = classify_pair_annotations('GAAACAAU', list(range(8)),
                                                     [[0, 4], [0, 7]], ['WCc', 'GUc'], [True] * 8)
        self.assertIn('multiple_canonical_partners', reasons)

    def test_canonical_code_must_match_sequence(self):
        _, _, _, reasons = classify_pair_annotations('GAAAG', list(range(5)), [[0, 4]], ['WCc'], [True] * 5)
        self.assertIn('canonical_annotation_sequence_mismatch', reasons)

    def test_adapter_preserves_indices_and_converts_angstrom_to_nm(self):
        data = fixture() + b"X 2 C1' 0.9 10 20 30 1 Z 102 ? 20 .\n"
        header, atom_rows = data.decode().split('_atom_site.label_alt_id\n')
        extra_tags = '_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n_atom_site.label_comp_id\n_atom_site.label_entity_id\n'
        full_rows = []
        for index, line in enumerate(atom_rows.splitlines(), 1):
            parts = line.split()
            full_rows.append(line + f' ATOM {index} {parts[2][0]} {"AC"[int(parts[1])-1]} 1')
        data = (header + '_atom_site.label_alt_id\n' + extra_tags + '\n'.join(full_rows) + '\n').encode()
        _, records = parse_coordinate_entry(data, '1abc.cif', {'1ABC'})
        self.assertEqual(records[0]['observed_mask'], [True, True])
        traj, mapping = trajectory_for_chain(data, records[0])
        self.assertEqual(mapping, [0, 1])
        index = next(a.index for a in traj.topology.atoms if a.residue.resSeq == 2 and a.name == "C1'")
        self.assertEqual(list(traj.xyz[0, index]), [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
