import numpy as np
import numpy.testing as npt
import unittest
import floatingse.spar as spar

from commonse import gravity as g
NSECTIONS = 5
NPTS = 100

class TestSpar(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resids = {}

        self.params['water_density'] = 1e3
        self.params['turbine_mass'] = 1e4
        self.params['mooring_mass'] = 50.0
        self.params['mooring_effective_mass'] = 40.0
        self.params['base_cylinder_mass'] = 2e4/NSECTIONS  * np.ones((NSECTIONS,))
        self.params['base_cylinder_displaced_volume'] = 2e2/NSECTIONS  * np.ones((NSECTIONS,))
        self.params['base_cylinder_center_of_buoyancy'] = -10.0
        self.params['base_cylinder_center_of_gravity'] = -8.0
        self.params['base_freeboard'] = 10.0
        self.params['water_ballast_mass_vector'] = 1e5*np.arange(5)
        self.params['water_ballast_zpts_vector'] = np.array([-10, -9, -8, -7, -6])
        self.params['turbine_center_of_gravity'] = 2.0*np.ones(3)
        self.params['fairlead'] = 1.0
        self.params['base_cylinder_Iwaterplane'] = 150.0
        self.params['base_cylinder_surge_force'] = np.array([11.0, 15.0]) 
        self.params['base_cylinder_force_points'] = np.array([-7.0, -4.0]) 
        self.params['turbine_surge_force'] = 13.0
        self.params['turbine_pitch_moment'] = 5.0
        self.params['mooring_surge_restoring_force'] = 200.0
        self.params['mooring_cost'] = 2.0
        self.params['base_cylinder_cost'] = 2.5

        self.myspar = spar.Spar(NSECTIONS, NPTS)

    def testBalance(self):
        self.myspar.balance_spar(self.params, self.unknowns)

        m_expect = 1e4 + 2e4 + 40.0
        m_water = 2e5 - m_expect
        h_expect = np.interp(m_water, self.params['water_ballast_mass_vector'], self.params['water_ballast_zpts_vector']) + 10
        cg_expect = (1e4*2 + 2e4*(-8) + 40*(-1) + m_water*(0.5*h_expect-10)) / (m_expect+m_water)
        self.assertEqual(self.unknowns['variable_ballast_mass'], m_water)
        self.assertEqual(self.unknowns['variable_ballast_height'], h_expect)
        self.assertEqual(self.unknowns['total_mass'], m_expect - 1e4 - 40.0 + 50.0)
        self.assertEqual(self.unknowns['z_center_of_gravity'], cg_expect)

        
    def testStability(self):
        self.unknowns['z_center_of_gravity'] = -1.0
        self.myspar.compute_stability(self.params, self.unknowns)

        Fc = np.trapz(np.array([11.0, 15.0]), np.array([-7, -4.0]))
        Mc = np.trapz(np.array([11.0, 15.0])*(np.array([-7, -4.0])+1), np.array([-7, -4.0]))
        m_expect = 2e4 + 50.0
        static_expect = -10 +1.0
        meta_expect = static_expect + 150/200.0 
        self.assertEqual(self.unknowns['static_stability'], static_expect)
        self.assertEqual(self.unknowns['metacentric_height'], meta_expect)
        self.assertEqual(self.unknowns['offset_force_ratio'], (Fc+13)/200.0)
        self.assertAlmostEqual(self.unknowns['heel_angle'], (180/np.pi)*(5+13*11+Mc)/(2e2*g*1e3*meta_expect))


    def testCost(self):
        self.myspar.compute_costs(self.params, self.unknowns)
        self.assertEqual(self.unknowns['total_cost'], 4.5)
        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSpar))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
    
    
