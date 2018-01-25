import numpy as np
import numpy.testing as npt
import unittest
import floatingse.cylinder as cylinder
import floatingse.sparGeometry as sparGeometry
import commonse.Frustum as frustum

from commonse import gravity as g
myones = np.ones((100,))

class TestCylinder(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['wall_thickness'] = np.array([0.5, 0.5, 0.5])
        self.params['outer_radius'] = np.array([10.0, 10.0, 10.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['freeboard'] = 15.0
        self.params['fairlead'] = 10.0
        self.params['fairlead_offset_from_shell'] = 0.0
        self.params['stack_mass_in'] = 0.0
        self.params['system_z_center_of_gravity'] = 25.0
        self.params['tower_base_radius'] = 5.0
        
        self.params['material_density'] = 5.0
        self.params['bulkhead_mass_factor'] = 1.5
        self.params['shell_mass_factor'] = 1.5
        self.params['ring_mass_factor'] = 1.25
        self.params['spar_mass_factor'] = 1.1
        self.params['outfitting_mass_fraction'] = 0.05

        self.params['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.params['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.params['stiffener_web_height']  = np.array([1.0, 1.0])
        self.params['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.params['stiffener_spacing'] =2.0
        self.params['bulkhead_nodes'] = [False, True, False]
        self.params['permanent_ballast_height'] = 1.0
        self.params['permanent_ballast_density'] = 2e3
        
        self.params['mooring_mass'] = 50.0
        self.params['mooring_vertical_load'] = 25.0
        self.params['mooring_restoring_force'] = 1e5
        self.params['mooring_cost'] = 1e4

        self.params['ballast_cost_rate'] = 10.0
        self.params['tapered_col_cost_rate'] = 100.0
        self.params['outfitting_cost_rate'] = 1.0
        
        self.params['wave_height'] = 2.
        self.params['wave_period'] = 50.0
        self.params['wind_reference_speed'] = 4.
        self.params['wind_reference_height'] = 1.
        self.params['alpha'] = 1.0
        self.params['water_density'] = 1e3
        self.params['water_viscosity'] = 1e-2
        self.params['water_depth'] = 1000.0
        self.params['air_density'] = 1.
        self.params['air_viscosity'] = 1e-5
        self.params['morison_mass_coefficient'] = 2.0

        self.set_geometry()

        self.myspar = cylinder.Cylinder(2)
        self.myspar.section_mass = np.zeros((2,))

        
    def set_geometry(self):
        sparGeom = sparGeometry.SparGeometry(2)
        tempUnknowns = {}
        sparGeom.solve_nonlinear(self.params, tempUnknowns, None)
        for pairs in tempUnknowns.items():
            self.params[pairs[0]] = pairs[1]
            
        
    def testCylinderForces(self):
        U   = 2.0
        A   = 4.0
        cm  = 1.0
        r   = 5.0
        rho = 0.5
        mu  = 1e-3

        Re = rho*U*2*r/mu
        q  = 0.5*rho*U*U
        cd = 1.11
        A  = 2*r
        D = q*A*cd

        Fi = rho * A * cm * np.pi * r*r
        Fp = Fi + D
        
        # Test drag only
        self.assertEqual(cylinder.cylinder_drag_per_length(U, r, rho, mu), D)
        npt.assert_equal(cylinder.cylinder_drag_per_length(U*myones, r*myones, rho*myones, mu*myones), D*myones)
        
        # Test full forces
        self.assertEqual(cylinder.cylinder_forces_per_length(U, A, r, rho, mu, cm), Fp)
        npt.assert_equal(cylinder.cylinder_forces_per_length(U*myones, A*myones, r*myones, rho*myones, mu*myones, cm*myones), Fp*myones)

    def testLinearWaves(self):
        Dwater = 30.0
        hmax   = 2.0
        rho    = 0.5
        z      = -2.0
        k      = 2.5

        omega = np.sqrt(g*k*np.tanh(k*Dwater))
        T = 2*np.pi/omega
        a     = 0.5*hmax
        U_exp = omega*a*np.cosh(k*(z+Dwater))/np.sinh(k*Dwater)
        A_exp = omega*omega*a*np.cosh(k*(z+Dwater))/np.sinh(k*Dwater)
        p_exp = -rho*g*(z - a*np.cosh(k*(z+Dwater))/np.cosh(k*Dwater))

        U_ans, A_ans, p_ans = cylinder.linear_waves(z, Dwater, hmax, T, rho)
        self.assertAlmostEqual(U_ans, U_exp)
        self.assertAlmostEqual(A_ans, A_exp)
        self.assertAlmostEqual(p_ans, p_exp)

        U_ans, A_ans, p_ans = cylinder.linear_waves(z*myones, Dwater, hmax, T, rho*myones)
        npt.assert_almost_equal(U_ans*myones, U_exp*myones)
        npt.assert_almost_equal(A_ans*myones, A_exp*myones)
        npt.assert_almost_equal(p_ans*myones, p_exp*myones)

    def testWindPowerLaw(self):
        uref = 5.0
        href = 3.0
        alpha = 2.0
        H = 9.0

        self.assertEqual(cylinder.wind_power_law(uref, href, alpha, H), 45.0)
        npt.assert_equal(cylinder.wind_power_law(uref, href, alpha, H*myones), 45.0*myones)
        
    def testBulkheadMass(self):
        self.params['wall_thickness'] = np.array([0.5, 1.0])
        self.params['outer_radius'] = np.array([10.0, 20.0])
        self.params['bulkhead_nodes'] = [False, True]

        expect = np.pi * 19.0*19.0 * 5.0 * 1.5
        actual = cylinder.compute_bulkhead_mass(self.params)
        self.assertEqual(actual.sum(), expect)
        npt.assert_equal(actual, np.array([0.0, expect]))
        
    def testShellMass(self):
        # Straight cylinder
        expect = 2.0*np.pi*9.75*0.5*5.0*1.5*np.array([20.0, 30.0])
        actual = cylinder.compute_shell_mass(self.params)
        self.assertAlmostEqual(actual.sum(), expect.sum())
        npt.assert_almost_equal(actual, expect)

        # Frustum shell
        self.params['wall_thickness'] = np.array([0.5, 0.4, 0.3])
        self.params['outer_radius'] = np.array([10.0, 8.0, 6.0])
        expect = np.pi/3.0*5.0*1.5*np.array([20.0, 30.0])*np.array([9.75*1.4+7.8*1.3, 7.8*1.1+5.85*1.0])
        actual = cylinder.compute_shell_mass(self.params)
        self.assertAlmostEqual(actual.sum(), expect.sum())
        npt.assert_almost_equal(actual, expect)
        
    def testStiffenerMass(self):
        self.params['wall_thickness'] = np.array([0.5, 0.4, 0.5])
        self.params['outer_radius'] = np.array([10.0, 8.0, 10.0])

        Rwo = 9-0.45
        Rwi = Rwo - 1.
        Rfi = Rwi - 0.3
        V1 = np.pi*(Rwo**2 - Rwi**2)*0.5
        V2 = np.pi*(Rwi**2 - Rfi**2)*2.0 
        V = V1+V2
        expect = 1.25*V*5.0 * np.array([20.0, 30.0])/2.0
        actual = cylinder.compute_stiffener_mass(self.params)

        self.assertAlmostEqual(actual.sum(), expect.sum())
        npt.assert_almost_equal(actual, expect)
        
    def testTBeam(self):
        h_web = 10.0
        w_flange = 8.0
        t_web = 3.0
        t_flange = 4.0

        area, y_cg, Ixx, Iyy = cylinder.TBeamProperties(h_web, t_web, w_flange, t_flange)
        self.assertEqual(area, 62.0)
        self.assertAlmostEqual(y_cg, 8.6129, 4)
        self.assertAlmostEqual(Iyy, 193.16666, 4)
        self.assertAlmostEqual(Ixx, 1051.37631867699, 4)

        area, y_cg, Ixx, Iyy = cylinder.TBeamProperties(h_web*myones, t_web*myones, w_flange*myones, t_flange*myones)
        npt.assert_equal(area, 62.0*myones)
        npt.assert_almost_equal(y_cg, 8.6129*myones, 1e-4)
        npt.assert_almost_equal(Iyy, 193.16666*myones, 1e-4)
        npt.assert_almost_equal(Ixx, 1051.37631867699*myones, 1e-4)

    def testPlasticityRF(self):
        Fy = 4.0

        Fe = 1.0
        Fi = Fe
        self.assertEqual(cylinder.plasticityRF(Fe, Fy), Fi)
        npt.assert_equal(cylinder.plasticityRF(Fe*myones, Fy), Fi*myones)
    
        Fe = 3.0
        Fr = 4.0/3.0
        Fi = Fe * Fr * (1.0 + 3.75*Fr**2)**(-0.25)
        self.assertEqual(cylinder.plasticityRF(Fe, Fy), Fi)
        npt.assert_equal(cylinder.plasticityRF(Fe*myones, Fy), Fi*myones)

    def testSafetyFactor(self):
        Fy = 100.0
        k = 1.25
        self.assertEqual(cylinder.safety_factor(25.0, Fy), k*1.2)
        npt.assert_equal(cylinder.safety_factor(25.0*myones, Fy), k*1.2*myones)
        self.assertEqual(cylinder.safety_factor(125.0, Fy), k*1.0)
        npt.assert_equal(cylinder.safety_factor(125.0*myones, Fy), k*1.0*myones)
        self.assertAlmostEqual(cylinder.safety_factor(80.0, Fy), k*1.08)
        npt.assert_almost_equal(cylinder.safety_factor(80.0*myones, Fy), k*1.08*myones)
    

    def testSparMassCG(self):
        m_spar, cg_spar = self.myspar.compute_spar_mass_cg(self.params, self.unknowns)

        bulk  = cylinder.compute_bulkhead_mass(self.params)
        stiff = cylinder.compute_stiffener_mass(self.params)
        shell = cylinder.compute_shell_mass(self.params)
        mycg  = 1.1*(np.dot(bulk, self.params['z_nodes']) + np.dot(stiff+shell, self.params['z_section']))/m_spar
        mysec = stiff+shell+bulk[:-1]
        mysec[-1] += bulk[-1]
        mysec *= 1.1
        
        self.assertEqual(self.unknowns['shell_mass'], shell.sum() )
        self.assertEqual(self.unknowns['stiffener_mass'], stiff.sum() )
        self.assertEqual(self.unknowns['bulkhead_mass'], bulk.sum() )
        self.assertAlmostEqual(self.unknowns['spar_mass'], 1.1*(bulk.sum()+stiff.sum()+shell.sum()) )
        self.assertAlmostEqual(self.unknowns['spar_mass'], m_spar )
        self.assertEqual(self.unknowns['outfitting_mass'], 0.05*m_spar )
        self.assertAlmostEqual(cg_spar, mycg )
        npt.assert_equal(self.myspar.section_mass, mysec)

    def testBallastMassCG(self):
        m_ballast, cg_ballast = self.myspar.compute_ballast_mass_cg(self.params, self.unknowns)

        area = np.pi * 9.5**2
        m_perm = area * 1.0 * 2e3
        cg_perm = self.params['z_nodes'][0] + 0.5

        h_expect = 1e6 / area / 1000.0
        m_expect = m_perm + 1e6
        cg_water = self.params['z_nodes'][0] + 1.0 + 0.5*h_expect
        cg_expect = (m_perm*cg_perm + 1e6*cg_water) / m_expect
        
        self.assertAlmostEqual(self.unknowns['ballast_mass'], m_perm)
        
        self.assertAlmostEqual(m_ballast, m_perm)
        self.assertAlmostEqual(cg_ballast, cg_perm)


    def testBalance(self):
        self.myspar.balance_cylinder(self.params, self.unknowns)
        m_spar, cg_spar = self.myspar.compute_spar_mass_cg(self.params, self.unknowns)
        m_ballast, cg_ballast = self.myspar.compute_ballast_mass_cg(self.params, self.unknowns)
        m_out = 0.05 * m_spar
        m_expect = m_spar + m_ballast + m_out
        cg_system = ((m_spar+m_out)*cg_spar + m_ballast*cg_ballast) / m_expect

        self.assertAlmostEqual(m_expect, self.unknowns['total_mass'].sum())
        self.assertAlmostEqual(cg_system, self.unknowns['z_center_of_gravity'])

        V_expect = np.pi * 100.0 * 35.0
        cb_expect = -17.5
        Ixx = 0.25 * np.pi * 1e4
        Axx = np.pi * 1e2
        self.assertAlmostEqual(self.unknowns['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.unknowns['z_center_of_buoyancy'], cb_expect)
        self.assertAlmostEqual(self.unknowns['Iwater'], Ixx)
        self.assertAlmostEqual(self.unknowns['Awater'], Axx)

        # Test if everything under water
        self.params['freeboard'] = -5.0
        self.set_geometry()
        self.myspar.balance_cylinder(self.params, self.unknowns)
        V_expect = np.pi * 100.0 * 50.0
        cb_expect = -25.0 - 5
        self.assertAlmostEqual(self.unknowns['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.unknowns['z_center_of_buoyancy'], cb_expect)
        

        
    def testSurgePitch(self):
        # No fluid forces
        self.params['wind_speed'] = 0.0
        self.params['wave_height'] = 0.0
        self.params['alpha'] = 1.0
        self.params['freeboard'] = 0.0
        self.params['tower_wind_force'] = 10.0
        self.params['tower_center_of_gravity'] = 5.0
        self.params['rna_wind_force'] = 50.0
        self.params['rna_center_of_gravity'] = 15.0
        self.params['rna_mass'] = 40.0
        self.params['rna_center_of_gravity_x'] = 0.5
        self.unknowns['metacentric_height'] = 20.0
        self.unknowns['total_mass'] = 100.0
        self.params['system_z_center_of_gravity'] = -25.0
        self.set_geometry()

        zpts = np.linspace(-50, 0, 100)
        self.myspar.compute_surge_pitch(self.params, self.unknowns)
        npt.assert_equal(self.unknowns['surge_force_vector'], np.zeros((100,)))
        npt.assert_equal(self.unknowns['surge_force_points'], zpts)

        # Only fluid forces, but constant along spar so no moment
        self.params['wind_reference_speed'] = 5.0
        self.params['wave_height'] = 1.0
        self.params['alpha'] = 0.0
        self.params['freeboard'] = 50.0
        self.params['tower_wind_force'] = 0.0
        self.params['tower_center_of_gravity'] = 5.0
        self.params['rna_wind_force'] = 0.0
        self.params['rna_center_of_gravity'] = 15.0
        self.params['rna_mass'] = 0.0
        self.params['rna_center_of_gravity_x'] = 0.5
        self.unknowns['metacentric_height'] = 20.0
        self.unknowns['total_mass'] = 100.0
        self.params['system_z_center_of_gravity'] = 25.0
        self.set_geometry()

        zpts = np.linspace(0, 50, 100)
        D = cylinder.cylinder_forces_per_length(5.0, 0.0, 10.0, self.params['air_density'], self.params['air_viscosity'], 0.0)
        F = D*50.0
        M = 0.0
        self.myspar.compute_surge_pitch(self.params, self.unknowns)
        npt.assert_equal(self.unknowns['surge_force_vector'], D*np.ones((100,)))
        npt.assert_equal(self.unknowns['surge_force_points'], zpts)


    def testAppliedHoop(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0

        R_od     = 0.5 * 600 * in_to_si
        t_wall   = 0.75 * in_to_si
        rho      = 64.0 * lbperft3_to_si
        z        = 60 * ft_to_si
        pressure = rho * g * z
        expect   = 1e-3 * 64. * 60. / 144. * ksi_to_si

        self.assertAlmostEqual(pressure, expect, -4)
        expect *= R_od/t_wall
        self.assertAlmostEqual(cylinder.compute_applied_hoop(pressure, R_od, t_wall), expect, -4)
        npt.assert_almost_equal(cylinder.compute_applied_hoop(pressure*myones, R_od*myones, t_wall*myones), expect*myones, decimal=-4)

    def testAppliedAxial(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        self.params['outer_radius'] = 0.5 * 600 * np.ones((4,)) * in_to_si
        self.params['wall_thickness'] = 0.75 * np.ones((4,)) * in_to_si
        self.params['stiffener_web_thickness'] = 5./8. * np.ones((3,)) * in_to_si
        self.params['stiffener_web_height'] = 14.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_thickness'] = 1.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_width'] = 10.0 * np.ones((3,)) * in_to_si
        self.params['section_height'] = 50.0 * np.ones((3,)) * ft_to_si
        self.params['stiffener_spacing'] = 5.0 * np.ones((3,)) * ft_to_si
        self.params['water_density'] = 64.0 * lbperft3_to_si
        self.params['E'] = 29e3 * ksi_to_si
        self.params['nu'] = 0.3
        self.params['yield_stress'] = 50 * ksi_to_si
        self.params['bulkhead_nodes'] = [False, False, False, False]
        self.params['wave_height'] = 0.0 # gives only static pressure
        self.params['stack_mass_in'] = 9000 * kip_to_si/g
        
        self.set_geometry()
        self.myspar.section_mass = np.zeros((3,))

        expect = 9000 * kip_to_si / (2*np.pi*(self.params['outer_radius'][0]-0.5*self.params['wall_thickness'][0])*self.params['wall_thickness'][0])
        npt.assert_almost_equal(cylinder.compute_applied_axial(self.params, self.myspar.section_mass), expect* np.ones((3,)), decimal=4)
        
    def testStiffenerFactors(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        self.params['outer_radius'] = 0.5 * 600 * np.ones((4,)) * in_to_si
        self.params['wall_thickness'] = 0.75 * np.ones((4,)) * in_to_si
        self.params['stiffener_web_thickness'] = 5./8. * np.ones((3,)) * in_to_si
        self.params['stiffener_web_height'] = 14.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_thickness'] = 1.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_width'] = 10.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_spacing'] = 5.0 * np.ones((3,)) * ft_to_si
        self.params['E'] = 29e3 * ksi_to_si
        self.params['nu'] = 0.3
        self.params['stack_mass_in'] = 9000 * kip_to_si / g

        pressure = 1e-3 * 64. * 60. / 144. * ksi_to_si
        axial    = 9000 * kip_to_si / (2*np.pi*(self.params['outer_radius'][0]-0.5*self.params['wall_thickness'][0])*self.params['wall_thickness'][0])
        self.assertAlmostEqual(axial, 0.5*9000/299.625/0.75/np.pi*ksi_to_si, -4)
        KthL, KthG = cylinder.compute_stiffener_factors(self.params, pressure, axial)
        npt.assert_almost_equal(KthL, 1.0*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(KthG, 0.5748*np.ones((3,)), decimal=4) #0.5642 if R_flange accounts for t_wall
    
    def testStressLimits(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        self.params['outer_radius'] = 0.5 * 600 * np.ones((4,)) * in_to_si
        self.params['wall_thickness'] = 0.75 * np.ones((4,)) * in_to_si
        self.params['stiffener_web_thickness'] = 5./8. * np.ones((3,)) * in_to_si
        self.params['stiffener_web_height'] = 14.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_thickness'] = 1.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_width'] = 10.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_spacing'] = 5.0 * np.ones((3,)) * ft_to_si
        self.params['section_height'] = 50.0 * np.ones((3,)) * ft_to_si
        self.params['E'] = 29e3 * ksi_to_si
        self.params['nu'] = 0.3
        self.params['yield_stress'] = 50 * ksi_to_si

        KthG = 0.5748
        FxeL, FreL, FxeG, FreG = cylinder.compute_elastic_stress_limits(self.params, KthG, loading='radial')
        npt.assert_almost_equal(FxeL, 16.074844135928885*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FreL, 19.80252150945599*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FxeG, 37.635953475479639*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FreG, 93.77314503852581*ksi_to_si*np.ones((3,)), decimal=1)

        FxcL = cylinder.plasticityRF(FxeL, self.params['yield_stress'])
        FxcG = cylinder.plasticityRF(FxeG, self.params['yield_stress'])
        FrcL = cylinder.plasticityRF(FreL, self.params['yield_stress'])
        FrcG = cylinder.plasticityRF(FreG, self.params['yield_stress'])
        npt.assert_almost_equal(FxcL, 1.0*16.074844135928885*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FrcL, 1.0*19.80252150945599*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FxcG, 0.799647237534*37.635953475479639*ksi_to_si*np.ones((3,)), decimal=1)
        npt.assert_almost_equal(FrcG, 0.444735273606*93.77314503852581*ksi_to_si*np.ones((3,)), decimal=1)
        
    def testCheckStresses(self):
        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3
        
        self.params['outer_radius'] = 0.5 * 600 * np.ones((4,)) * in_to_si
        self.params['wall_thickness'] = 0.75 * np.ones((4,)) * in_to_si
        self.params['stiffener_web_thickness'] = 5./8. * np.ones((3,)) * in_to_si
        self.params['stiffener_web_height'] = 14.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_thickness'] = 1.0 * np.ones((3,)) * in_to_si
        self.params['stiffener_flange_width'] = 10.0 * np.ones((3,)) * in_to_si
        self.params['section_height'] = 50.0 * np.ones((3,)) * ft_to_si
        self.params['stiffener_spacing'] = 5.0 * np.ones((3,)) * ft_to_si
        self.params['water_density'] = 64.0 * lbperft3_to_si
        self.params['E'] = 29e3 * ksi_to_si
        self.params['nu'] = 0.3
        self.params['yield_stress'] = 50 * ksi_to_si
        self.params['bulkhead_nodes'] = [False, False, False, False]
        self.params['wave_height'] = 0.0 # gives only static pressure
        self.params['stack_mass_in'] = 9000 * kip_to_si/g
        
        # Find pressure to give "head" of 60ft- put mid-point of middle section at this depth
        z = 60 * ft_to_si
        self.params['freeboard'] = np.sum(self.params['section_height']) - z - 1.5*(self.params['section_height'])[0]
        
        self.set_geometry()
        self.myspar.section_mass = np.zeros((3,))
        self.myspar.check_stresses(self.params, self.unknowns, loading='radial')
        
        npt.assert_almost_equal(self.unknowns['web_compactness'], 24.1/22.4 * np.ones((3,)), decimal=3)
        npt.assert_almost_equal(self.unknowns['flange_compactness'], 9.03/5.0 * np.ones((3,)), decimal=3)
        self.assertAlmostEqual(self.unknowns['axial_local_unity'][1], 1.07, 1)
        self.assertAlmostEqual(self.unknowns['axial_general_unity'][1], 0.34, 1)
        self.assertAlmostEqual(self.unknowns['external_local_unity'][1], 1.07, 1)
        self.assertAlmostEqual(self.unknowns['external_general_unity'][1], 0.59, 1)
        
    def testCheckCost(self):
        self.unknowns['ballast_mass'] = 50.0
        self.unknowns['spar_mass'] = 200.0
        self.unknowns['outfitting_mass'] = 25.0
        self.myspar.compute_cost(self.params, self.unknowns)

        self.assertEqual(self.unknowns['ballast_cost'], 10.0 * 50.0)
        self.assertEqual(self.unknowns['spar_cost'], 100.0 * 200.0)
        self.assertEqual(self.unknowns['outfitting_cost'], 1.0 * 25.0)
        self.assertEqual(self.unknowns['total_cost'], 10.0*50.0 + 100.0*200.0 + 1.0*25.0)
        self.assertEqual(self.myspar.cost, 10.0*50.0 + 100.0*200.0 + 1.0*25.0)
        self.assertEqual(self.unknowns['total_cost'], self.myspar.cost)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestCylinder))
    return suite

if __name__ == '__main__' and __package__ is None:
    __package__ = 'src.test.cylinder_PyU'
    unittest.TextTestRunner().run(suite())
