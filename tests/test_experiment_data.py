import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.experiment_data import validate_split_manifest


class ExperimentDataManifestTests(unittest.TestCase):
    def test_passing_manifest_must_match_selected_splits(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / 'data'
            data.mkdir()
            train = data / 'train.jsonl'
            validation = data / 'validation.jsonl'
            train.write_text('{"id":"train"}\n', encoding='utf-8')
            validation.write_text('{"id":"validation"}\n', encoding='utf-8')
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({
                'passed': True,
                'files': {'train': digest(train), 'validation': digest(validation)},
            }), encoding='utf-8')
            self.assertEqual(validate_split_manifest(data, manifest, ['train', 'validation']), digest(manifest))

    def test_changed_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / 'data'
            data.mkdir()
            validation = data / 'validation.jsonl'
            validation.write_text('original\n', encoding='utf-8')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({
                'passed': True,
                'files': {'validation': hashlib.sha256(validation.read_bytes()).hexdigest()},
            }), encoding='utf-8')
            validation.write_text('changed\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                validate_split_manifest(data, manifest, ['validation'])

    def test_failed_manifest_is_rejected_before_file_use(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            data = root / 'data'
            data.mkdir()
            validation = data / 'validation.jsonl'
            validation.write_text('data\n', encoding='utf-8')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'passed': False, 'files': {}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'did not pass'):
                validate_split_manifest(data, manifest, ['validation'])


if __name__ == '__main__':
    unittest.main()
