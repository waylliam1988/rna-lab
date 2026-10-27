import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_rfam_pilot_isolation import relation_decision, sequence_key


class FamilyIsolationContracts(unittest.TestCase):
    def test_positive_holdout_relation_blocks_even_supported_source(self):
        self.assertTrue(relation_decision(True,True,True).startswith('blocked_'))

    def test_no_exact_hit_not_independence(self):
        self.assertIn('unresolved',relation_decision(True,False))

    def test_unresolved_source_not_admitted(self):
        self.assertEqual(relation_decision(False,False,True),'source_provenance_unresolved')

    def test_ambiguous_sequence_not_exact_witness(self):
        self.assertIsNone(sequence_key('ACGN'))
        self.assertEqual(sequence_key('acgt'),sequence_key('ACGU'))
