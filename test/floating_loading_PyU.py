import numpy as np
import numpy.testing as npt
import unittest
import floatingse.floating_loading as sP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from commonse import gravity as g


NSECTIONS = 5
NPTS = 11
myones = np.ones((NPTS,))

def DrawTruss(mytruss):
    mynodes = {}
    for k in xrange(len(mytruss.frame.nx)):
        mynodes[mytruss.frame.nnode[k]] = np.r_[mytruss.frame.nx[k], mytruss.frame.ny[k], mytruss.frame.nz[k]]
    myelem = []
    for k in xrange(len(mytruss.frame.eN1)):
        myelem.append( (mytruss.frame.eN1[k], mytruss.frame.eN2[k]) )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for e in myelem:
        xs = np.array( [ mynodes[e[0]][0], mynodes[e[1]][0] ] )
        ys = np.array( [ mynodes[e[0]][1], mynodes[e[1]][1] ] )
        zs = np.array( [ mynodes[e[0]][2], mynodes[e[1]][2] ] )
        ax.plot(xs, ys, zs)
    ax.auto_scale_xyz([-10, 10], [-10, 10], [-30, 50])
    plt.show()


class TestFrame(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['cross_attachment_pontoons'] = True
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = True
        self.params['lower_ring_pontoons'] = True
        self.params['upper_ring_pontoons'] = True
        self.params['outer_cross_pontoons'] = True

        self.params['number_of_auxiliary_columns'] = 3
        self.params['connection_ratio_max'] = 0.25

        self.params['fairlead'] = 7.5
        self.params['base_pontoon_attach_upper'] = 8.0
        self.params['base_pontoon_attach_lower'] = -14.0
        
        self.params['radius_to_auxiliary_column'] = 10.0
        self.params['pontoon_outer_diameter'] = 2.0
        self.params['pontoon_wall_thickness'] = 1.0

        self.params['base_z_full'] = np.array([-15.0, -12.5, -10.0, 0.0, 5.0, 10.0])
        self.params['base_d_full'] = 2*10.0 * np.ones((NSECTIONS+1,))
        self.params['base_t_full'] = 0.1 * np.ones((NSECTIONS+1,))
        self.params['base_column_mass'] = 1e2 * np.ones((NSECTIONS,))
        self.params['base_column_displaced_volume'] = 1e2 * np.ones((NSECTIONS,))
        self.params['base_column_center_of_buoyancy'] = -10.0
        self.params['base_column_center_of_mass'] = -6.0
        self.params['base_column_Px'] = 50.0 * np.ones((NSECTIONS+1,))
        self.params['base_column_Py'] = np.zeros((NSECTIONS+1,))
        self.params['base_column_Pz'] = np.zeros((NSECTIONS+1,))
        self.params['base_column_qdyn'] = 70.0 * np.ones((NSECTIONS+1,))

        self.params['auxiliary_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 10.0])
        self.params['auxiliary_d_full'] = 2*2.0 * np.ones((NSECTIONS+1,))
        self.params['auxiliary_t_full'] = 0.05 * np.ones((NSECTIONS+1,))
        self.params['auxiliary_column_mass'] = 1e1 * np.ones((NSECTIONS,))
        self.params['auxiliary_column_displaced_volume'] = 1e1 * np.ones((NSECTIONS,))
        self.params['auxiliary_column_center_of_buoyancy'] = -5.0
        self.params['auxiliary_column_center_of_mass'] = -3.0
        self.params['auxiliary_column_Px'] = 50.0 * np.ones((NSECTIONS+1,))
        self.params['auxiliary_column_Py'] = np.zeros((NSECTIONS+1,))
        self.params['auxiliary_column_Pz'] = np.zeros((NSECTIONS+1,))
        self.params['auxiliary_column_qdyn'] = 70.0 * np.ones((NSECTIONS+1,))

        self.params['tower_z_full'] = np.linspace(0, 90, NSECTIONS+1)
        self.params['tower_d_full'] = 2*7.0 * np.ones((NSECTIONS+1,))
        self.params['tower_t_full'] = 0.5 * np.ones((NSECTIONS+1,))
        self.params['tower_mass'] = 2e2 * np.ones((NSECTIONS,))
        self.params['tower_buckling_length'] = 25.0
        self.params['tower_center_of_mass'] = 50.0
        self.params['tower_Px'] = 50.0 * np.ones((NSECTIONS+1,))
        self.params['tower_Py'] = np.zeros((NSECTIONS+1,))
        self.params['tower_Pz'] = np.zeros((NSECTIONS+1,))
        self.params['tower_qdyn'] = 70.0 * np.ones((NSECTIONS+1,))
        
        self.params['E'] = 200e9
        self.params['G'] = 79.3e9
        self.params['material_density'] = 7850.0
        self.params['yield_stress'] = 345e6

        self.params['rna_force'] = 6e1*np.ones(3)
        self.params['rna_moment'] = 7e2*np.ones(3)
        self.params['rna_mass'] = 6e1
        self.params['rna_cg'] = np.array([3.05, 2.96, 2.13])
        self.params['rna_I'] = np.array([3.05284574e9, 2.96031642e9, 2.13639924e7, 0.0, 2.89884849e7, 0.0])

        self.params['water_density'] = 1025.0

        self.params['gamma_f'] = 1.35
        self.params['gamma_m'] = 1.1
        self.params['gamma_n'] = 1.0
        self.params['gamma_b'] = 1.1
        self.params['gamma_fatigue'] = 1.755

        self.params['pontoon_cost_rate'] = 6.250

        self.unknowns['pontoon_stress'] = np.zeros(50)
        
        self.mytruss = sP.FloatingFrame(NSECTIONS+1)


    def testStandard(self):
        self.params['auxiliary_z_full'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 3.0])
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        npt.assert_equal(self.unknowns['base_connection_ratio'], 0.25-0.1)
        npt.assert_equal(self.unknowns['auxiliary_connection_ratio'], 0.25-0.5)
        self.assertEqual(self.unknowns['pontoon_base_attach_upper'], 23.0/25.0)
        self.assertEqual(self.unknowns['pontoon_base_attach_lower'], 1.0/25.0)

        #DrawTruss(self.mytruss)
        

    def testSpar(self):
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['number_of_auxiliary_columns'] = 0
        
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        #DrawTruss(self.mytruss)
        
        
    def testOutputsIncremental(self):
        ncyl   = self.params['number_of_auxiliary_columns']
        R_semi = self.params['radius_to_auxiliary_column']
        Ro     = 0.5*self.params['pontoon_outer_diameter']
        Ri     = Ro - self.params['pontoon_wall_thickness']
        rho    = self.params['material_density']
        rhoW   = self.params['water_density']

        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['base_pontoon_attach_upper'] = 10.0
        self.params['base_pontoon_attach_lower'] = -15.0
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * R_semi * ncyl
        m = np.pi * (Ro*Ro-Ri*Ri) * R_semi * ncyl * rho
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)


        self.params['lower_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * (1 + np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)


        self.params['upper_attachment_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * (2 + np.sqrt(3))
        cg = ((-15)*(1 + np.sqrt(3)) + 10) / (2+np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], cg)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['upper_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * 2 * (1 + np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_mass'], -2.5)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['cross_attachment_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        L = np.sqrt(R_semi*R_semi + 25*25)
        k = 15. / 25.
        V = np.pi * Ro*Ro * ncyl * (k*L + R_semi * (1 + np.sqrt(3)))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (L + R_semi * 2 * (1 + np.sqrt(3)))
        self.assertAlmostEqual(self.unknowns['pontoon_displacement'], V)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        #self.params['outer_cross_pontoons'] = True
        #self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

    def testForces(self):
        self.params['auxiliary_z_full'] = np.linspace(-1e-4, 0.0, NSECTIONS+1)
        self.params['base_z_full'] = np.linspace(-1e-4, 0.0, NSECTIONS+1)
        self.params['tower_z_full'] = np.linspace(0, 1e-4, NSECTIONS+1)
        self.params['fairlead'] = 0.0
        self.params['number_of_auxiliary_columns'] = 0
        self.params['base_column_mass'] = 1.0 * np.ones((NSECTIONS,))
        self.params['base_column_center_of_mass'] = 0.0
        self.params['auxiliary_column_mass'] = 0.0 * np.ones((NSECTIONS,))
        self.params['auxiliary_column_center_of_mass'] = 0.0
        self.params['tower_mass'] = 1.0 * np.ones((NSECTIONS,))
        self.params['tower_center_of_mass'] = 0.0
        self.params['rna_mass'] = 1.0
        self.params['rna_force'] = 10.0*np.ones(3)
        self.params['rna_moment'] = 20.0*np.ones(3)
        self.params['rna_cg'] = np.array([0.0, 0.0, 0.0])
        self.params['rna_I'] = np.zeros(6)
        self.params['base_column_Px'] = 0.0 * np.ones((NSECTIONS,))
        self.params['auxiliary_column_Px'] = 0.0 * np.ones((NSECTIONS,))
        self.params['tower_Px'] = 0.0 * np.ones((NSECTIONS,))
        self.params['base_column_qdyn'] = 0.0 * np.ones((NSECTIONS+1,))
        self.params['auxiliary_column_qdyn'] = 0.0 * np.ones((NSECTIONS+1,))
        self.params['tower_qdyn'] = 0.0 * np.ones((NSECTIONS+1,))
        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = False
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.params['outer_cross_pontoons'] = False
        self.params['base_pontoon_attach_upper'] = 0.0
        self.params['base_pontoon_attach_lower'] = 0.0
        self.params['water_density'] = 1e-12
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        m = NSECTIONS*2 + 1
        self.assertAlmostEqual(self.unknowns['structural_mass'], m, 4)
        self.assertAlmostEqual(self.unknowns['substructure_mass'], NSECTIONS, 5)
        npt.assert_almost_equal(self.unknowns['total_force'], np.array([10.0, 10.0, 10-m*g]), decimal=1)
        npt.assert_almost_equal(self.unknowns['total_moment'], np.array([20.0, 20.0, 20.0]), decimal=2)

        self.params['rna_cg'] = np.array([5.0, 5.0, 5.0])
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        npt.assert_almost_equal(self.unknowns['total_force'], np.array([10.0, 10.0, 10-m*g]), decimal=1)
        self.assertEqual(self.unknowns['total_moment'][-1], 20.0)

        
    def testBadInput(self):
        self.params['number_of_auxiliary_columns'] = 1
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_auxiliary_columns'] = 2
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_auxiliary_columns'] = 8
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)

        self.params['number_of_auxiliary_columns'] = 3
        self.params['base_z_full'][-2] = self.params['base_z_full'][-3] + 1e-12
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['substructure_mass'], 1e30)
        
    def testCombinations(self):
        self.params['radius_to_auxiliary_column'] = 30.0
        
        for nc in [0, 3, 4, 5, 6, 7]:
            
            for cap in [True, False]:
                
                for lap in [True, False]:
                    
                    for uap in [True, False]:
                        
                        for lrp in [True, False]:
                            
                            for urp in [True, False]:
                                
                                for ocp in [True, False]:
                                    self.params['number_of_auxiliary_columns'] = nc
                                    self.params['cross_attachment_pontoons'] = cap
                                    self.params['lower_attachment_pontoons'] = lap
                                    self.params['upper_attachment_pontoons'] = uap
                                    self.params['lower_ring_pontoons'] = lrp
                                    self.params['upper_ring_pontoons'] = urp
                                    self.params['outer_cross_pontoons'] = ocp
                                    if (nc > 0) and (not cap) and (not lap) and (not uap): continue
        
                                    self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
                                    if self.unknowns['substructure_mass'] == 1e30:
                                        print nc, cap, lap, uap, lrp, urp, ocp
                                    self.assertNotEqual(self.unknowns['substructure_mass'], 1e30)

def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFrame))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
