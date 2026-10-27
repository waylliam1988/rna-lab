import sys
from pathlib import Path
from types import SimpleNamespace as S
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from isolate_3d_data import pose_from_residues


def atom(name,xyz,alt='\x00'):
    return S(name=name,pos=S(x=xyz[0],y=xyz[1],z=xyz[2]),altloc=alt,occ=1.,b_iso=20.)


class PoseTests(unittest.TestCase):
    def test_units_and_real_base_plane(self):
        r=[atom("C4'",(10,0,0)),atom('N9',(0,0,0)),atom('C8',(1,0,0)),atom('N7',(0,1,0))]
        self.assertTrue(np.allclose(pose_from_residues('A',{1:r}),[[1,0,0,0,0,1]]))

    def test_missing_or_mixed_conformer_is_rejected(self):
        r=[atom("C4'",(10,0,0)),atom('N9',(0,0,0),'A'),atom('C8',(1,0,0),'B'),atom('N7',(0,1,0))]
        self.assertIsNone(pose_from_residues('A',{1:r}))
        self.assertIsNone(pose_from_residues('A',{}))


if __name__=='__main__':unittest.main()
