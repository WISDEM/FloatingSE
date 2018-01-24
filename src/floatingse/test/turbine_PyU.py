import numpy as np
import numpy.testing as npt
import unittest
import floatingse.turbine as turbine
import commonse.Frustum as frustum
from commonse import gravity as g

class TestTurbine(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params = {}
        self.params['freeboard'] = 15.0
        self.params['tower_wind_force'] = 10.0
        self.params['rna_wind_force'] = 20.0
        self.params['tower_mass'] = 5e3
        self.params['rna_mass'] = 5e3
        self.params['tower_center_of_gravity'] = 10.0
        self.params['rna_center_of_gravity'] = 15.0
        self.params['rna_center_of_gravity_x'] = 0.5

        self.myturb = turbine.Turbine()
        
    def testAll(self):
        self.myturb.solve_nonlinear(self.params, self.unknowns, None)

        self.assertEqual(self.unknowns['total_mass'], 1e4)
        self.assertEqual(self.unknowns['z_center_of_gravity'], (5e3*(15+10) + 5e3*(15+15)) / 1e4)
        npt.assert_equal(self.unknowns['surge_force'], np.array([10, 20]))
        npt.assert_equal(self.unknowns['force_points'], np.array([10+15, 15+15]))
        self.assertEqual(self.unknowns['pitch_moment'], - 5e3*g*0.5)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestTurbine))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
