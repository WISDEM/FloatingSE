import numpy as np
import numpy.testing as npt
import unittest
import floatingse.sparGeometry as sparGeometry
from floatingse.floatingInstance import nodal2sectional

myones = np.ones((100,))

class TestSpar(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params = {}
        self.params['water_depth'] = 100.0
        self.params['wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['freeboard'] = 15.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0
        self.params['tower_base_radius'] = 7.0

        self.mysparG = sparGeometry.SparGeometry()
        
        
    def testNodal2Sectional(self):
        npt.assert_equal(nodal2sectional(np.array([8.0, 10.0, 12.0])), np.array([9.0, 11.0]))

    def testSetGeometry(self):
        self.mysparG.solve_nonlinear(self.params, self.unknowns, None)
        
        npt.assert_equal(self.unknowns['z_nodes'], np.array([-35.0, -15.0, 15.0]))
        self.assertEqual(self.params['freeboard'], self.unknowns['z_nodes'][-1])
        self.assertEqual(self.unknowns['draft'], np.sum(self.params['section_height'])-self.params['freeboard'])
        self.assertEqual(self.unknowns['draft'], 35.0)
        self.assertEqual(self.unknowns['draft'], np.abs(self.unknowns['z_nodes'][0]))
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0)
        self.assertEqual(self.unknowns['draft_depth_ratio'], 0.35)
        self.assertEqual(self.unknowns['fairlead_draft_ratio'], 10./35.)
        self.assertEqual(self.unknowns['transition_radius'], 3.0)
        npt.assert_equal(self.unknowns['z_section'], np.array([-25.0, 0.0]))

    def testTaperRatio(self):
        self.params['outer_radius'] = np.array([10.0, 9.0, 10.0])
    
        self.mysparG.solve_nonlinear(self.params, self.unknowns, None)
        npt.assert_equal(self.unknowns['taper_ratio'], np.array([0.1, 1./9.]))

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSpar))
    return suite

if __name__ == '__main__' and __package__ is None:
    __package__ = 'floatingse.test.sparGeometry_PyU'
    unittest.TextTestRunner().run(suite())
