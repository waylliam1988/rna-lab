import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from rna2d.evidence import pairing_evidence_issues, screen_rdat


class EvidenceTests(unittest.TestCase):
    def test_predictions_and_unverified_labels_cannot_enter_gold(self):
        self.assertTrue(pairing_evidence_issues({'evidence_class': 'predicted', 'provenance_verified': True,
                                                'quality_verified': True, 'contains_model_predictions': False}))
        self.assertTrue(pairing_evidence_issues({'evidence_class': 'experimental_coordinate_annotation'}))
        self.assertEqual(pairing_evidence_issues({'evidence_class': 'experimental_coordinate_annotation',
                                                'provenance_verified': True, 'quality_verified': True,
                                                'contains_model_predictions': False}), [])

    def test_nan_and_sentinels_are_not_observations(self):
        text = 'SEQUENCE ACGU\nSEQPOS A1 C2 G3 U4\nSTRUCTURE (..)\nREACTIVITY:1 NaN -999 0.1 0.2\nREACTIVITY_ERROR:1 0 0 0.01 0.02'
        row = screen_rdat(text)[0]
        self.assertEqual(row['measured_mask'], [False, False, True, True])
        self.assertEqual(row['coverage'], 0.5)
        self.assertFalse(row['preliminary_screen_pass'])
        self.assertFalse(row['structure_field_used_as_truth'])
        self.assertFalse(row['pair_labels_available'])

    def test_all_missing_profile_rejected(self):
        row = screen_rdat('SEQUENCE AC\nSEQPOS A1 C2\nREACTIVITY:1 NaN NaN\nREACTIVITY_ERROR:1 0 0')[0]
        self.assertIn('insufficient_measured_coverage', row['reasons'])

    def test_structure_does_not_supply_pair_truth(self):
        row = screen_rdat('SEQUENCE ACGU\nSEQPOS A1 C2 G3 U4\nSTRUCTURE (..)\nREACTIVITY:1 0.1 0.2 0.3 0.4\nREACTIVITY_ERROR:1 .1 .1 .1 .1')[0]
        self.assertTrue(row['preliminary_screen_pass'])
        self.assertFalse(row['quality_verified'])
        self.assertFalse(row['pair_labels_available'])

    def test_metadata_filter_keeps_nmr_for_separate_review(self):
        spec = importlib.util.spec_from_file_location('audit', ROOT / 'scripts/audit_experimental_sources.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.quality_reasons({'structure_method': 'x-ray diffraction', 'resolution': 2.1, 'sequence': 'ACGU'}), [])
        self.assertTrue(module.quality_reasons({'structure_method': 'x-ray diffraction', 'resolution': 5, 'sequence': 'ACGU'}))
        self.assertIn('nmr_requires_restraint_and_ensemble_quality_audit', module.quality_reasons({'structure_method': 'solution nmr', 'sequence': 'ACGU'}))


if __name__ == '__main__':
    unittest.main()
