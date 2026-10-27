import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_egnn_geometry import run_audit


class EGNNGeometryAuditTests(unittest.TestCase):
    def test_real_batch_shapes_alignment_and_coordinate_path(self):
        report=run_audit()
        self.assertTrue(report['all_checks_passed'])
        self.assertFalse(report['repair_required'])


if __name__=='__main__':unittest.main()
