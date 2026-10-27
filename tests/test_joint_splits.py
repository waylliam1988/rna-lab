import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from build_joint_splits import sequence_edges


class JointSplitTests(unittest.TestCase):
    def test_embedded_construct_is_linked_despite_flanks(self):
        seq='AUGCUGACUAGCUACGGAUCUGAC'
        edges=list(sequence_edges([dict(id='core',sequence=seq),dict(id='library',sequence='AAAA'+seq+'CCCC')]))
        self.assertEqual(len(edges),1)
        self.assertGreaterEqual(edges[0]['shorter_coverage'],0.8)

    def test_unrelated_and_short_shared_motif_are_not_homology(self):
        records=[dict(id='a',sequence='A'*30),dict(id='b',sequence='C'*30),dict(id='c',sequence='C'*25+'AAAAA')]
        edges=list(sequence_edges(records))
        self.assertFalse(any(set((e['a'],e['b']))=={'a','c'} for e in edges))


if __name__=='__main__':unittest.main()
