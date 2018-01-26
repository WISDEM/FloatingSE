import numpy as np
import numpy.testing as npt
import unittest
import floatingse.semiPontoon as sP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from commonse import gravity as g
myones = np.ones((100,))

NSECTIONS = 5

class TestCylinder(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['cross_attachment_pontoons'] = True
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = True
        self.params['lower_ring_pontoons'] = True
        self.params['upper_ring_pontoons'] = True
        self.params['pontoon_cost_rate'] = 6.250

        self.params['radius_to_ballast_cylinder'] = 10.0
        self.params['pontoon_outer_diameter'] = 2.0
        self.params['pontoon_inner_diameter'] = 1.0
        self.params['base_outer_diameter'] = 2*10.0 * np.ones((NSECTIONS+1,))
        self.params['tower_diameter'] = 2*7.0 * np.ones((NSECTIONS+1,))
        self.params['ballast_outer_diameter'] = 2*2.0 * np.ones((NSECTIONS+1,))
        self.params['base_wall_thickness'] = 0.1 * np.ones((NSECTIONS+1,))
        self.params['ballast_wall_thickness'] = 0.05 * np.ones((NSECTIONS+1,))
        self.params['base_cylinder_mass'] = 1e2 * np.ones((NSECTIONS+1,))
        self.params['ballast_cylinder_mass'] = 1e1 * np.ones((NSECTIONS+1,))
        
        self.params['E'] = 200e9
        self.params['G'] = 79.3e9
        self.params['material_density'] = 7850.0
        self.params['yield_stress'] = 345e6
        self.params['number_of_ballast_cylinders'] = 3
        self.params['base_z_nodes'] = np.array([-15.0, -12.5, -10.0, 0.0, 5.0, 10.0])
        self.params['ballast_z_nodes'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 10.0])
        self.params['turbine_surge_force'] = 6e1
        self.params['turbine_pitch_moment'] = 7e2
        self.params['turbine_mass'] = 6e1
        self.params['water_density'] = 1025.0
        self.params['base_cylinder_displaced_volume'] = 1e2
        self.params['ballast_cylinder_displaced_volume'] = 1e1
        
        self.mytruss = sP.SemiPontoon(NSECTIONS)

        
    def testTubeProperties(self):
        Ax, As, I, J, S, C = sP.TubeProperties(5.0, 4.0)
        
        self.assertEqual(Ax, np.pi*9.0)
        self.assertEqual(I,  np.pi*369.0/4.0)
        self.assertEqual(J,  np.pi*369.0/2.0)
        self.assertEqual(S,  np.pi*369.0/4.0/5.0)
        self.assertEqual(C,  np.pi*369.0/2.0/5.0)

        # Test improper Ri>Ro
        Ax, As, I, J, S, C = sP.TubeProperties(4.0, 5.0)
        
        self.assertGreater(Ax, 0.0)
        self.assertGreater(As, 0.0)
        self.assertGreater(I, 0.0)
        self.assertGreater(J, 0.0)
        self.assertGreater(S, 0.0)
        self.assertGreater(C, 0.0)
    
    def testOutputsIncremental(self):
        ncyl   = self.params['number_of_ballast_cylinders']
        R_semi = self.params['radius_to_ballast_cylinder']
        Ro     = 0.5*self.params['pontoon_outer_diameter']
        Ri     = 0.5*self.params['pontoon_inner_diameter']
        rho    = self.params['material_density']
        rhoW   = self.params['water_density']

        self.params['cross_attachment_pontoons'] = False
        self.params['lower_attachment_pontoons'] = True
        self.params['upper_attachment_pontoons'] = False
        self.params['lower_ring_pontoons'] = False
        self.params['upper_ring_pontoons'] = False
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * R_semi * ncyl
        m = np.pi * (Ro*Ro-Ri*Ri) * R_semi * ncyl * rho
        self.assertAlmostEqual(self.unknowns['pontoon_buoyancy'], V*rhoW*g)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_gravity'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['lower_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * (1 + np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_buoyancy'], V*rhoW*g)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_gravity'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['upper_attachment_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * (2 + np.sqrt(3))
        cg = ((-15)*(1 + np.sqrt(3)) + 10) / (2+np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_buoyancy'], V*rhoW*g)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_gravity'], cg)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['upper_ring_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        V = np.pi * Ro*Ro * ncyl * R_semi * (1 + np.sqrt(3))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * R_semi * 2 * (1 + np.sqrt(3))
        self.assertAlmostEqual(self.unknowns['pontoon_buoyancy'], V*rhoW*g)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_buoyancy'], -15.0)
        self.assertAlmostEqual(self.unknowns['pontoon_center_of_gravity'], -2.5)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)

        
        self.params['cross_attachment_pontoons'] = True
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)

        L = np.sqrt(R_semi*R_semi + 25*25)
        k = 15. / 25.
        V = np.pi * Ro*Ro * ncyl * (k*L + R_semi * (1 + np.sqrt(3)))
        m = np.pi * (Ro*Ro-Ri*Ri) * ncyl * rho * (L + R_semi * 2 * (1 + np.sqrt(3)))
        self.assertAlmostEqual(self.unknowns['pontoon_buoyancy'], V*rhoW*g)
        self.assertAlmostEqual(self.unknowns['pontoon_mass'], m)
        self.assertAlmostEqual(self.unknowns['pontoon_cost'], m*6.25, 2)
        
        
    def testDrawTruss(self):
        self.params['ballast_z_nodes'] = np.array([-15.0, -10.0, -5.0, 0.0, 2.5, 3.0])
        self.mytruss.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['pontoon_radii_ratio'], 0.5)

        
        mynodes = {}
        for k in xrange(len(self.mytruss.frame.nx)):
            mynodes[self.mytruss.frame.nnode[k]] = np.r_[self.mytruss.frame.nx[k], self.mytruss.frame.ny[k], self.mytruss.frame.nz[k]]
        myelem = []
        for k in xrange(len(self.mytruss.frame.eN1)):
            myelem.append( (self.mytruss.frame.eN1[k], self.mytruss.frame.eN2[k]) )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for e in myelem:
            xs = np.array( [ mynodes[e[0]][0], mynodes[e[1]][0] ] )
            ys = np.array( [ mynodes[e[0]][1], mynodes[e[1]][1] ] )
            zs = np.array( [ mynodes[e[0]][2], mynodes[e[1]][2] ] )
            ax.plot(xs, ys, zs)
        ax.auto_scale_xyz([-10, 10], [-10, 10], [-30, 50])
        plt.show()


        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCylinder))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
