import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / 'data/processed/train.jsonl').exists(), 'Run prepare_data.py first')
class PreparedDataTests(unittest.TestCase):
    def test_exact_sequence_isolation_and_label_contract(self):
        seen = set()
        for name in ['train', 'validation', 'test', 'test_external', 'archiveII']:
            path = ROOT / f'data/processed/{name}.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            local = set()
            for row in rows:
                seq, pairs = row['sequence'], row['pairs']
                self.assertNotIn(seq, seen)
                self.assertNotIn(seq, local)
                local.add(seq)
                self.assertEqual(len(row['observed_mask']), len(seq))
                self.assertEqual(row['official_split'], name)
                partners = set()
                for i, j in pairs:
                    self.assertTrue(0 <= i < j < len(seq))
                    self.assertNotIn(i, partners)
                    self.assertNotIn(j, partners)
                    partners.update((i, j))
            seen.update(local)


if __name__ == '__main__':
    unittest.main()
