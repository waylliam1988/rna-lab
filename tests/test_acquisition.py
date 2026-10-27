"""Verify download integrity, quarantine, and safe resume without network access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/acquire_experimental_corpus.py'
spec = importlib.util.spec_from_file_location('acquisition', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(patch.stopall)
        patch.object(module, 'ROOT', self.root).start()
        patch.object(module.shutil, 'disk_usage', return_value=MagicMock(free=100 * 1024**3)).start()
        patch.object(module.time, 'sleep').start()
        self.client = patch.object(module, 'session').start().return_value
        self.payload = b'RDAT_VERSION 0.3\nSEQUENCE ACGU\n'
        self.job = dict(id='test:rdat', kind='rdat', url='https://example.invalid/data',
                        relative_path='example.rdat', expected_bytes=len(self.payload),
                        expected_sha256=hashlib.sha256(self.payload).hexdigest())

    def response(self, payload, status=200, headers=None):
        response = MagicMock(status_code=status, headers=headers or {})
        response.iter_content.side_effect = lambda *_: iter([payload])
        self.client.get.return_value = response

    def test_verified_receipt_is_not_label_approval_and_cache_is_checked(self):
        self.response(self.payload)
        result = module.acquire(self.job, self.root)
        self.assertEqual(result['status'], 'verified_download', result)
        self.assertFalse(result['quality_approved'])
        self.client.get.reset_mock()
        self.assertEqual(module.acquire(self.job, self.root)['status'], 'verified_existing')
        self.client.get.assert_not_called()
        (self.root/'example.rdat').write_bytes(b'corruption')
        self.assertEqual(module.acquire(self.job, self.root)['status'], 'quarantined')
        self.client.get.assert_not_called()

    def test_validated_range_resume(self):
        (self.root/'example.rdat.partial').write_bytes(self.payload[:8])
        (self.root/'example.rdat.partial.json').write_text(json.dumps({'validator':'"v1"'}))
        self.response(self.payload[8:], 206, {'Content-Range':f'bytes 8-{len(self.payload)-1}/{len(self.payload)}'})
        result = module.acquire(self.job, self.root)
        self.assertEqual(result['status'], 'verified_download', result)
        self.assertEqual(self.client.get.call_args.kwargs['headers'],
                         {'Range':'bytes=8-', 'If-Range':'"v1"'})

    def test_hash_mismatch_never_publishes_final(self):
        self.response(b'X' * len(self.payload))
        result = module.acquire(self.job, self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse((self.root/'example.rdat').exists())
        self.assertTrue(list(self.root.glob('*.invalid-*')))

    def test_parent_traversal_rejected_before_network(self):
        with self.assertRaises(ValueError):
            module.acquire(dict(self.job, relative_path='../outside.rdat'), self.root)
        self.client.get.assert_not_called()


if __name__ == '__main__':
    unittest.main()
