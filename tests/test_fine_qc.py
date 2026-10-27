import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from rna2d.fine_qc import consensus,unit_key,residue_geometry
from rna2d.probing_qc import sequence_mapping,quality_metadata,measured_profile


class FineQCTests(unittest.TestCase):
    def test_partial_consensus_masks_every_disputed_partner(self):
        seq='GAAACGAAAC'
        record=dict(model_id=1,auth_asym_id='A',residue_auth_mapping=[[str(i+1),''] for i in range(10)])
        contacts=[('X|1|A|G|1','X|1|A|C|5','cWW')]
        with self.assertRaisesRegex(ValueError,'jaccard'):
            consensus(seq,[True]*10,[[0,4],[5,9]],[True]*10,contacts,record)
        pairs,mask,_=consensus(seq,[True]*10,[[0,4],[5,9]],[True]*10,contacts,record,min_pair_jaccard=0)
        self.assertEqual(pairs,[[0,4]])
        self.assertFalse(mask[5]);self.assertFalse(mask[9])

    def test_plausible_base_and_exploded_bond(self):
        positions={'N1':[1.4,0,0],'C2':[0.7,1.2124,0],'N3':[-0.7,1.2124,0],
                   'C4':[-1.4,0,0],'C5':[-0.7,-1.2124,0],'C6':[0.7,-1.2124,0],
                   'O2':[1.4,2.4248,0],'N4':[-2.8,0,0],"C1'":[2.85,0,0]}
        atoms={k:(v,1.0,20.0) for k,v in positions.items()}
        self.assertIsNone(residue_geometry('C',atoms))
        atoms['C2']=([50,50,50],1.0,20.0)
        self.assertEqual(residue_geometry('C',atoms),'base_bond_distance')

    def test_cross_chain_partner_is_not_negative(self):
        seq='GAAACGAAAC'
        record=dict(model_id=1,auth_asym_id='A',residue_auth_mapping=[[str(i+1),''] for i in range(len(seq))])
        pairs=[[0,4],[5,9]]
        contacts=[('X|1|A|G|1','X|1|A|C|5','cWW'),('X|1|A|G|6','X|1|A|C|10','cWW'),
                  ('X|1|A|G|1','X|1|B|C|3','cWW')]
        result,mask,_=consensus(seq,[True]*10,pairs,[True]*10,contacts,record)
        self.assertEqual(result,[[5,9]])
        self.assertFalse(mask[0])
        self.assertFalse(mask[4])  # Removed partner cannot become an unpaired label.

    def test_no_independent_annotation_is_rejected(self):
        record=dict(model_id=1,auth_asym_id='A',residue_auth_mapping=[[str(i+1),''] for i in range(5)])
        with self.assertRaisesRegex(ValueError,'missing'):
            consensus('GAAAC',[True]*5,[[0,4]],[True]*5,[],record)

    def test_symmetry_and_insertion_are_preserved(self):
        self.assertEqual(unit_key('X|1|A|G|25|||A|2_555'),(1,'A','25','A',True))

    def test_rdat_offset_and_override(self):
        seq,indices=sequence_mapping('XXXX',['sequence:ACGUACGU'],10,['X11','X18'])
        self.assertEqual((seq,indices),('ACGUACGU',[0,7]))
        with self.assertRaisesRegex(ValueError,'out_of_bounds'):
            sequence_mapping('ACGU',[],10,['A10'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            sequence_mapping('ACGU',[],0,['A1','A1'])

    def test_unknown_mutations_are_quarantined(self):
        with self.assertRaisesRegex(ValueError,'mutation_mapping'):
            sequence_mapping('ACGU',['mutation:deletion2'],0,['A1'])
        self.assertEqual(sequence_mapping('ACGU',['mutation:C2G'],0,['X2'])[0],'AGGU')

    def test_warning_and_low_snr_are_rejected(self):
        reasons,_=quality_metadata(['signal_to_noise:weak:1.2','warning:badQuality'],['modifier:DMS'])
        self.assertIn('source_warning',reasons)
        self.assertIn('snr_missing_or_below3',reasons)

    def test_missing_values_are_masked_not_imputed(self):
        values,errors,mask=measured_profile(' '.join(['1']*20+['-999']), ' '.join(['0.1']*21),21,['DMS'])
        self.assertEqual(int(mask.sum()),20)
        self.assertEqual(values[-1],-999)
        with self.assertRaisesRegex(ValueError,'length_mismatch'):
            measured_profile(' '.join(['1']*20+['oops']), ' '.join(['0.1']*20),20,['DMS'])


if __name__=='__main__': unittest.main()
