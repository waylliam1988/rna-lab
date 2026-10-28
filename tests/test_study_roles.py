import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from rna2d.study_roles import candidate_review, effective_reference_role


class NextStudyRoleContracts(unittest.TestCase):
    def review(self, **changes):
        args=dict(provenance_verified=True,quality_verified=True,mapping_verified=True,
                  contains_model_predictions=False,supervision_scope='reviewed_binary_pair_scope',
                  active_holdout_relation=False,retired_evaluation_relation=False)
        args.update(changes)
        return candidate_review(**args)

    def test_retired_history_does_not_alone_block(self):
        self.assertEqual(self.review(retired_evaluation_relation=True)['status'], self.review()['status'])

    def test_candidate_is_not_automatic_training_admission(self):
        result=self.review(retired_evaluation_relation=True)
        self.assertTrue(result['can_enter_candidate_manifest'])
        self.assertFalse(result['training_allowed'])

    def test_active_holdout_relation_still_blocks(self):
        self.assertIn('related_to_active_development_or_test',self.review(active_holdout_relation=True,retired_evaluation_relation=True)['blocking_reasons'])

    def test_no_match_cannot_be_used_as_clearance(self):
        self.assertFalse(self.review(active_holdout_relation=None)['can_enter_candidate_manifest'])

    def test_unknown_origin_is_not_no_predictions(self):
        self.assertIn('label_origin_unresolved',self.review(contains_model_predictions=None)['blocking_reasons'])

    def test_model_predictions_remain_excluded(self):
        self.assertIn('model_predicted_or_imputed_labels_excluded',self.review(contains_model_predictions=True,retired_evaluation_relation=True)['blocking_reasons'])

    def test_positive_only_not_silently_binary(self):
        self.assertFalse(self.review(supervision_scope='positive_only')['can_enter_candidate_manifest'])

    def test_quality_and_mapping_not_waived(self):
        result=self.review(quality_verified=None,mapping_verified=False,retired_evaluation_relation=True)
        self.assertIn('quality_unresolved',result['blocking_reasons'])
        self.assertIn('sequence_label_mapping_unresolved',result['blocking_reasons'])

    def test_active_alias_protection_wins_over_retirement(self):
        self.assertEqual(effective_reference_role('retired_evaluation_reference',True),'protected_development_alias')
