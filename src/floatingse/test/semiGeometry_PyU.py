import numpy as np
import numpy.testing as npt
import unittest
import floatingse.semiGeometry as semiGeometry

myones = np.ones((100,))

class TestSemiGeom(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params = {}
        self.params['water_depth'] = 100.0
        
        self.params['base_wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['base_outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['base_section_height'] = np.array([20.0, 30.0])
        self.params['base_freeboard'] = 15.0

        self.params['ballast_wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['ballast_outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['ballast_section_height'] = np.array([20.0, 30.0])
        self.params['ballast_freeboard'] = 15.0

        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0
        self.params['radius_to_ballast_cylinder'] = 25.0
        
        self.params['tower_base_radius'] = 7.0

        self.mysemiG = semiGeometry.SemiGeometry(2)


    def testSetGeometry(self):
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)
        
        npt.assert_equal(self.unknowns['base_z_nodes'], np.array([-35.0, -15.0, 15.0]))
        npt.assert_equal(self.unknowns['ballast_z_nodes'], np.array([-35.0, -15.0, 15.0]))
        self.assertEqual(self.params['base_freeboard'], self.unknowns['base_z_nodes'][-1])
        self.assertEqual(self.params['ballast_freeboard'], self.unknowns['ballast_z_nodes'][-1])
        self.assertEqual(self.unknowns['base_draft'], np.sum(self.params['base_section_height'])-self.params['base_freeboard'])
        self.assertEqual(self.unknowns['ballast_draft'], np.sum(self.params['ballast_section_height'])-self.params['ballast_freeboard'])
        self.assertEqual(self.unknowns['base_draft'], 35.0)
        self.assertEqual(self.unknowns['ballast_draft'], 35.0)
        self.assertEqual(self.unknowns['base_draft'], np.abs(self.unknowns['base_z_nodes'][0]))
        self.assertEqual(self.unknowns['ballast_draft'], np.abs(self.unknowns['ballast_z_nodes'][0]))

        self.assertEqual(self.unknowns['fairlead_radius'], 11.0+25.0)
        self.assertEqual(self.unknowns['base_draft_depth_ratio'], 0.35)
        self.assertEqual(self.unknowns['ballast_draft_depth_ratio'], 0.35)
        self.assertEqual(self.unknowns['fairlead_draft_ratio'], 10./35.)
        self.assertEqual(self.unknowns['transition_radius'], 3.0)
        npt.assert_equal(self.unknowns['base_z_section'], np.array([-25.0, 0.0]))
        npt.assert_equal(self.unknowns['ballast_z_section'], np.array([-25.0, 0.0]))

        self.assertEqual(self.unknowns['base_ballast_spacing'], 20.0/25.0)
        
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSemiGeom))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
