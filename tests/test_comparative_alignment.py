import tempfile
import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rna2d.comparative_alignment import read_stockholm, ungap


class StockholmContracts(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'seed.sto'
            path.write_text(text, encoding='utf-8')
            return read_stockholm(path)

    def fixture(self):
        return '# STOCKHOLM 1.0\n#=GF AC RF00001\n#=GF SQ 1\nAB1.1/1-4 AC-\n#=GC SS_cons <._\n\nAB1.1/1-4 GU\n#=GC SS_cons >.\n//\n'

    def test_interleaved_alignment(self):
        d = self.parse(self.fixture())
        self.assertEqual(d['sequences']['AB1.1/1-4'], 'AC-GU')
        self.assertEqual(d['annotations']['SS_cons'], '<._>.')

    def test_annotation_alignment_required(self):
        with self.assertRaises(ValueError):
            self.parse(self.fixture().replace('#=GC SS_cons >.', '#=GC SS_cons >'))

    def test_count_required(self):
        with self.assertRaises(ValueError):
            self.parse(self.fixture().replace('SQ 1', 'SQ 2'))

    def test_terminator_required(self):
        with self.assertRaises(ValueError):
            self.parse(self.fixture().replace('//', ''))

    def test_ungap_preserves_unknown(self):
        self.assertEqual(ungap('aT-N.r~'), 'AUNR')
