from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.model_factory import build_pair_model, model_config_from_checkpoint, model_from_checkpoint
from rna2d.sequence_encoders import SequencePairModel
from rna2d.model import PairBaseline


class ModelFactoryTests(unittest.TestCase):
    def test_builds_frozen_cnn_config(self):
        config = {
            'architecture': 'sequence_pair_model', 'kind': 'cnn',
            'width': 113, 'hidden': 64, 'depth': 2,
        }
        model = build_pair_model(config)
        self.assertIsInstance(model, SequencePairModel)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 150019)

    def test_loads_modern_checkpoint_model_config(self):
        config = {
            'architecture': 'sequence_pair_model', 'kind': 'cnn',
            'width': 113, 'hidden': 64, 'depth': 2,
        }
        source = build_pair_model(config)
        checkpoint = {'model_config': config, 'model': source.state_dict()}
        target = model_from_checkpoint(checkpoint)
        target.load_state_dict(checkpoint['model'], strict=True)
        self.assertIsInstance(target, SequencePairModel)
        self.assertEqual(model_config_from_checkpoint(checkpoint), config)
        self.assertTrue(torch.equal(source.embedding.weight, target.embedding.weight))

    def test_loads_frozen_experiment_checkpoint_config_layout(self):
        config = {
            'class': 'rna2d.sequence_encoders.SequencePairModel',
            'kind': 'cnn', 'width': 113, 'hidden': 64, 'depth': 2,
            'parameters': 150019,
        }
        source = SequencePairModel('cnn', 113, 64, 2)
        checkpoint = {'config': {'model': config}, 'model': source.state_dict()}
        target = model_from_checkpoint(checkpoint)
        target.load_state_dict(checkpoint['model'], strict=True)
        self.assertIsInstance(target, SequencePairModel)
        self.assertEqual(target.kind, 'cnn')

    def test_loads_legacy_pairbaseline_checkpoint_layout(self):
        source = PairBaseline(hidden=16)
        checkpoint = {'hidden': 16, 'model': source.state_dict(), 'config': {'hidden': 16}}
        target = model_from_checkpoint(checkpoint)
        target.load_state_dict(checkpoint['model'], strict=True)
        self.assertIsInstance(target, PairBaseline)
        self.assertEqual(model_config_from_checkpoint(checkpoint), {
            'architecture': 'pair_baseline', 'hidden': 16,
        })

    def test_rejects_incomplete_model_config(self):
        with self.assertRaisesRegex(ValueError, 'does not contain a supported'):
            model_config_from_checkpoint({'model': {}})


if __name__ == '__main__':
    unittest.main()
