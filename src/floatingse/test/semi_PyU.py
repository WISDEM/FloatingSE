import numpy as np
import numpy.testing as npt
import unittest
import floatingse.semi as semi

from commonse import gravity as g
NSECTIONS = 5
NPTS = 100

class TestSemi(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resids = {}

        # From other components
        self.params['turbine_mass'] = 1e4
        self.params['turbine_center_of_gravity'] = 40.0*np.ones(3)
        self.params['turbine_surge_force'] = 26.0
        self.params['turbine_pitch_moment'] = 5e4
        
        self.params['mooring_mass'] = 20.0
        self.params['mooring_effective_mass'] = 15.0
        self.params['mooring_surge_restoring_force'] = 1e2
        self.params['mooring_cost'] = 256.0

        self.params['pontoon_mass'] = 50.0
        self.params['pontoon_cost'] = 512.0
        self.params['pontoon_buoyancy'] = 2e6
        self.params['pontoon_center_of_buoyancy'] = -15.0
        self.params['pontoon_center_of_gravity'] = -10.0
        
        self.params['base_cylinder_mass'] = 2e4 * np.ones((NSECTIONS,))
        self.params['base_cylinder_displaced_volume'] = 1e4 * np.ones((NSECTIONS,))
        self.params['base_cylinder_center_of_buoyancy'] = -12.5
        self.params['base_cylinder_center_of_gravity'] = -20.0
        self.params['base_cylinder_Iwaterplane'] = 150.0
        self.params['base_cylinder_surge_force'] = np.array([11.0, 15.0]) 
        self.params['base_cylinder_force_points'] = np.array([-7.0, -4.0]) 
        self.params['base_cylinder_cost'] = 32.0
        self.params['base_freeboard'] = 10.0
        
        self.params['ballast_cylinder_mass'] = 1.5e4 * np.ones((NSECTIONS,))
        self.params['ballast_cylinder_displaced_volume'] = 5e3 * np.ones((NSECTIONS,))
        self.params['ballast_cylinder_center_of_buoyancy'] = -5.0
        self.params['ballast_cylinder_center_of_gravity'] = -1.0
        self.params['ballast_cylinder_Iwaterplane'] = 50.0
        self.params['ballast_cylinder_Awaterplane'] = 9.0
        self.params['ballast_cylinder_surge_force'] = np.array([2.0, 3.0])
        self.params['ballast_cylinder_force_points'] = np.array([-2.0, -3.0])
        self.params['ballast_cylinder_cost'] = 64.0
        
        self.params['number_of_ballast_cylinders'] = 3
        self.params['fairlead'] = 1.0
        self.params['water_ballast_mass_vector'] = 1e9*np.arange(5)
        self.params['water_ballast_zpts_vector'] = np.array([-10, -9, -8, -7, -6])
        self.params['radius_to_ballast_cylinder'] = 20.0

        self.params['water_density'] = 1e3

        self.mysemi = semi.Semi(NSECTIONS, NPTS)
        self.mysemiG = semi.SemiGeometry(2)

        
    def testSetGeometry(self):
        self.params['base_outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['ballast_outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['ballast_z_nodes'] = np.array([-35.0, -15.0, 15.0])
        self.params['radius_to_ballast_cylinder'] = 25.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 1.0
        self.mysemiG.solve_nonlinear(self.params, self.unknowns, None)
        
        self.assertEqual(self.unknowns['fairlead_radius'], 11.0+25.0)
        self.assertEqual(self.unknowns['base_ballast_spacing'], 20.0/25.0)

        
    def testBalance(self):
        self.mysemi.balance_semi(self.params, self.unknowns)
        m_expect = 1e4 + 15.0 + 50.0 + 5*2e4 + 3*1.5e4*5
        V_expect = 1e4*5 + 3*5*5e3 + 2e6/g/1e3
        m_water = 1e3*V_expect - m_expect
        h_expect = np.interp(m_water, self.params['water_ballast_mass_vector'], self.params['water_ballast_zpts_vector']) + 10.0
        cg_expect = (1e4*40.0 + 50.0*(-10.0) + 5*2e4*(-20.0) + 3*5*1.5e4*(-1.0) + 15.0*(-1.0) + m_water*(-10 + 0.5*h_expect)) / (m_expect+m_water)
        cb_expect = (2e6/g/1e3*(-15.0) + 5*1e4*(-12.5) + 3*5*5e3*(-5.0)) / V_expect
        
        self.assertEqual(self.unknowns['total_mass'], m_expect - 1e4 - 15.0 + 20.0)
        self.assertEqual(self.unknowns['total_displacement'], V_expect)
        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height'], h_expect)
        self.assertEqual(self.unknowns['z_center_of_buoyancy'], cb_expect)
        self.assertAlmostEqual(self.unknowns['z_center_of_gravity'], cg_expect)

        
    def testStability(self):
        self.unknowns['z_center_of_gravity'] = -1.0
        self.unknowns['z_center_of_buoyancy'] = -2.0
        self.unknowns['total_displacement'] = 1e4
        self.mysemi.compute_stability(self.params, self.unknowns)

        Fc = np.trapz(np.array([11.0, 15.0]), np.array([-7, -4.0])) + 3*np.trapz(np.array([2.0, 3.0]), np.array([-2, -3.0]))
        Mc = np.trapz(np.array([11.0, 15.0])*(np.array([-7, -4.0])+1), np.array([-7, -4.0]))
        m_expect = 2e4 + 50.0
        I_expect = 150.0 + (50.0 + 9.0*(20.0*np.cos(np.deg2rad(np.array([0.0, 120., 240.0]))) )**2).sum()
        static_expect = -2.0 + 1.0
        meta_expect = static_expect + I_expect/1e4
        self.assertEqual(self.unknowns['static_stability'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], (Fc+26)/1e2)
        #self.assertEqual(self.unknowns['heel_angle'], (180/np.pi)*(16+Mc)/(1e4*g*1e3*meta_expect))


    def testCost(self):
        self.mysemi.compute_costs(self.params, self.unknowns)
        c_expect = 256.0 + 512.0 + 32.0 + 3*64.0
        self.assertEqual(self.unknowns['total_cost'], c_expect)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSemi))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
