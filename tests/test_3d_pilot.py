import sys
from pathlib import Path
from types import SimpleNamespace as S
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_3d_pilot import extract_anchors


def atom(name, xyz, alt='\x00'):
    return S(name=name, pos=S(x=xyz[0], y=xyz[1], z=xyz[2]), altloc=alt, occ=1.0, b_iso=20)


def residue(offset=0):
    return [atom("C4'", (offset, 0, 0)), atom("C1'", (offset, 1, 0)), atom('N9', (offset, 1, 1))]


class AnchorScreenTests(unittest.TestCase):
    def test_missing_and_broken_chain_are_masked(self):
        _, mask, _ = extract_anchors('AAA', {1: residue(), 2: residue(20)})
        self.assertEqual(mask, [False, False, False])

    def test_different_alt_conformers_cannot_complete_one_anchor(self):
        r = residue()
        r[0].altloc = 'A'
        r[1].altloc = 'B'
        _, mask, _ = extract_anchors('A', {1: r})
        self.assertEqual(mask, [False])

    def test_valid_chain_and_shared_atoms(self):
        r = residue()
        r[0].altloc = 'A'
        _, mask, reasons = extract_anchors('AA', {1: r, 2: residue(6)})
        self.assertEqual(mask, [True, True])
        self.assertEqual(reasons, {})


if __name__ == '__main__':
    unittest.main()
