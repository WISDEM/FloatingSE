import numpy as np
import numpy.testing as npt
import unittest
import floatingse.substructure as subs

from commonse import gravity as g
NSECTIONS = 5
NPTS = 100

class TestSubs(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resids = {}


        # From other components
        self.params['structural_mass'] = 1e4
        self.params['structure_center_of_mass'] = 40.0*np.ones(3)
        self.params['total_force'] = 26.0*np.ones(3)
        self.params['total_moment'] = 5e4*np.ones(3)
        self.params['total_displacement'] = 1e4
        
        self.params['mooring_mass'] = 20.0
        self.params['mooring_effective_mass'] = 15.0
        self.params['mooring_surge_restoring_force'] = 1e2
        self.params['mooring_pitch_restoring_force'] = 1e5 * np.ones((10,3))
        self.params['mooring_pitch_restoring_force'][3:,:] = 0.0
        self.params['mooring_cost'] = 256.0
        self.params['fairlead'] = 0.5
        self.params['fairlead_radius'] = 5.0
        self.params['max_heel'] = 10.0

        self.params['pontoon_cost'] = 512.0
        
        self.params['base_column_Iwaterplane'] = 150.0
        self.params['base_column_cost'] = 32.0
        self.params['base_freeboard'] = 10.0

        self.params['tower_base'] = 8.0
        
        self.params['auxiliary_column_Iwaterplane'] = 50.0
        self.params['auxiliary_column_Awaterplane'] = 9.0
        self.params['auxiliary_column_cost'] = 64.0

        self.params['number_of_auxiliary_columns'] = 3
        self.params['water_ballast_mass_vector'] = 1e9*np.arange(5)
        self.params['water_ballast_zpts_vector'] = np.array([-10, -9, -8, -7, -6])
        self.params['radius_to_auxiliary_column'] = 20.0
        self.params['z_center_of_buoyancy'] = -2.0

        self.params['water_density'] = 1e3

        self.mysemi = subs.SemiStable(NPTS)
        self.mysemiG = subs.SubstructureGeometry(2)

        
    def testSetGeometry(self):
        self.params['number_of_auxiliary_columns'] = 3
        self.params['base_outer_diameter'] = 2*np.array([10.0, 10.0, 10.0])
        self.params['auxiliary_outer_diameter'] = 2*np.array([10.0, 10.0, 10.0])
        self.params['auxiliary_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.params['base_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.params['radius_to_auxiliary_column'] = 25.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)

        # Semi
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0+25.0)
        self.assertEqual(self.unknowns['base_auxiliary_spacing'], 20.0/25.0)
        self.assertEqual(self.unknowns['transition_buffer'], 10-0.5*8)

        # Spar
        self.params['number_of_auxiliary_columns'] = 0
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0)
        self.assertEqual(self.unknowns['base_auxiliary_spacing'], 20.0/25.0)
        self.assertEqual(self.unknowns['transition_buffer'], 10-0.5*8)

        
    def testBalance(self):
        self.mysemi.balance_semi(self.params, self.unknowns)
        m_water = 1e3*1e4 - 1e4 - 15
        h_expect = np.interp(m_water, self.params['water_ballast_mass_vector'], self.params['water_ballast_zpts_vector']) + 10.0
        cg_expect_z = (1e4*40.0 + m_water*(-10 + 0.5*h_expect)) / (1e4+m_water)
        cg_expect_xy = 1e4*40.0/ (1e4+m_water)
        
        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.unknowns['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        
        self.params['number_of_auxiliary_columns'] = 0
        self.mysemi.balance_semi(self.params, self.unknowns)

        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height_ratio'], h_expect/4.0)
        npt.assert_almost_equal(self.unknowns['center_of_mass'], np.array([cg_expect_xy, cg_expect_xy, cg_expect_z]))
        

    def testStability(self):
        self.params['mooring_pitch_restoring_force'] = 0.0 * np.ones((10,3))
        self.unknowns['center_of_mass'] = np.array([0.0, 0.0, -1.0])
        self.mysemi.compute_stability(self.params, self.unknowns)

        I_expect = 150.0 + (50.0 + 9.0*(20.0*np.cos(np.deg2rad(np.array([0.0, 120., 240.0]))) )**2).sum()
        static_expect = -1.0 + 2.0
        meta_expect = I_expect/1e4 - static_expect
        self.assertEqual(self.unknowns['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.params['number_of_auxiliary_columns'] = 0
        self.mysemi.compute_stability(self.params, self.unknowns)

        I_expect = 150.0
        meta_expect = I_expect/1e4 - static_expect
        self.assertEqual(self.unknowns['buoyancy_to_gravity'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], 26.0/1e2)
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (5e4)/(1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

        self.params['fairlead'] = 1.0
        self.params['mooring_pitch_restoring_force'][:3,-1] = 1.0
        self.assertAlmostEqual(self.unknowns['heel_moment_ratio'], (5e4)/(1*5 + 1e4*g*1e3*np.sin(np.deg2rad(10))*np.abs(meta_expect)))

    def testCost(self):
        self.mysemi.compute_costs(self.params, self.unknowns)
        c_expect = 256.0 + 512.0 + 32.0 + 3*64.0
        self.assertEqual(self.unknowns['total_cost'], c_expect)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSubs))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
