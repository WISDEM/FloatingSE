import __init__
import numpy as np
import numpy.testing as npt
import unittest
import sparGeometry
import commonse.Frustum as frustum

myones = np.ones((100,))

class TestSpar(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params = {}
        self.params['wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['freeboard'] = 15.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0

        self.mysparG = sparGeometry.SparGeometry()
        self.mysparG.solve_nonlinear(self.params, self.unknowns, None)
        
        
    def testNodal2Sectional(self):
        npt.assert_equal(sparGeometry.nodal2sectional(np.array([8.0, 10.0, 12.0])), np.array([9.0, 11.0]))

    def testSetGeometry(self):
        
        npt.assert_equal(self.unknowns['z_nodes'], np.array([-35.0, -15.0, 15.0]))
        self.assertEqual(self.params['freeboard'], self.unknowns['z_nodes'][-1])
        self.assertEqual(self.unknowns['draft'], np.sum(self.params['section_height'])-self.params['freeboard'])
        self.assertEqual(self.unknowns['draft'], 35.0)
        self.assertEqual(self.unknowns['draft'], np.abs(self.unknowns['z_nodes'][0]))
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0)
        npt.assert_equal(self.unknowns['z_section'], np.array([-25.0, 0.0]))

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSpar))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
