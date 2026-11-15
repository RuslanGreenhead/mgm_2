"""Analytic metric checks independent of all experimental target files."""
import math
import unittest
import torch
from .common import ManifoldOps
from .data_metrics import direction_distances,metrics


class MetricTests(unittest.TestCase):
    def test_sphere_normalization_and_antipodes(self):
        ops=ManifoldOps("sphere",3)
        x=torch.tensor([[1.,0,0]],dtype=torch.float64)
        y=torch.tensor([[4.,0,0],[-2.,0,0],[0.,3,0]],dtype=torch.float64)
        expected=torch.tensor([[0.,math.pi,math.pi/2]],dtype=torch.float64)
        torch.testing.assert_close(direction_distances(ops,x,y),expected)

    def test_torus_seam_and_periodicity(self):
        ops=ManifoldOps("torus",2)
        x=torch.tensor([[.01,0]],dtype=torch.float64)
        y=torch.tensor([[2*math.pi-.01,0]],dtype=torch.float64)
        self.assertAlmostEqual(float(direction_distances(ops,x,y)),.02)
        torch.testing.assert_close(direction_distances(ops,x,y),direction_distances(ops,x+2*math.pi,y-4*math.pi))

    def test_coincident_samples_ties_and_biased_mmd(self):
        ops=ManifoldOps("torus",2);x=torch.zeros(8,2)
        row=metrics(ops,x,x,{"bandwidth2":1.,"coverage_radius":.1})
        self.assertAlmostEqual(row["kernel_mmd2_v"],0.)
        self.assertEqual(row["one_nna"],.5)
        self.assertEqual(row["radius_coverage"],1.)
        self.assertEqual(row["nn_occupancy_coverage"],1/8)
        self.assertEqual(row["symmetric_nn"],0.)

    def test_known_separated_and_symmetric_metrics(self):
        ops=ManifoldOps("torus",2)
        x=torch.zeros(4,2);y=torch.zeros(4,2);y[:,0]=math.pi
        ref={"bandwidth2":1.,"coverage_radius":.1}
        row=metrics(ops,x,y,ref);reverse=metrics(ops,y,x,ref)
        self.assertGreater(row["kernel_mmd2_v"],1.9)
        self.assertEqual(row["one_nna"],1.)
        self.assertEqual(row["radius_coverage"],0.)
        self.assertAlmostEqual(row["symmetric_nn"],math.pi,places=6)
        for key in ["kernel_mmd2_v","symmetric_nn","one_nna"]:
            self.assertAlmostEqual(row[key],reverse[key])


if __name__=="__main__":unittest.main()
