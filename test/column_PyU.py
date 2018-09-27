import numpy as np
import numpy.testing as npt
import unittest
import floatingse.column as column
from commonse.utilities import nodal2sectional

from commonse import gravity as g
NPTS = 11
NSEC = 2
myones = np.ones((NPTS,))
secones = np.ones((NPTS-1,))

class TestSectional(unittest.TestCase):
    def testAll(self):
        x = np.arange(0.0, 2.1, 0.5)
        y = np.array([-1.0, 1.0, -2.0, 2.0])
        xi = np.array([-0.1, 0.25, 0.9, 1.4, 1.5, 1.6, 2.1])
        yi = column.sectionalInterp(xi, x, y)

        y_expect = np.array([-1.0, -1.0, 1.0, -2.0, 2.0, 2.0, 2.0])
        npt.assert_array_equal(yi, y_expect)
        

class TestBulk(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['z_full'] = np.linspace(0, 1, NPTS)
        self.params['z_param'] = np.linspace(0, 1, 6)
        self.params['d_full'] = 10.0 * myones
        self.params['t_full'] = 0.5 * myones
        self.params['rho'] = 1e3
        self.params['bulkhead_mass_factor'] = 1.1
        self.params['bulkhead_thickness'] = 0.05 * np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0])

        self.bulk = column.BulkheadMass(5, NPTS)

    def testAll(self):
        self.bulk.solve_nonlinear(self.params, self.unknowns, self.resid)

        R_i = 0.5 * 10 - 0.5
        m_bulk = np.pi * 1e3 * 1.1 * R_i**2 * 0.05 
        expect = np.zeros( self.params['z_full'].shape )
        expect[[2,6]] = m_bulk
        npt.assert_almost_equal(self.unknowns['bulkhead_mass'], expect)

        J0 = 0.50 * m_bulk * R_i**2
        I0 = 0.25 * m_bulk * R_i**2

        I = np.zeros(6)
        I[2] = 2.0 * J0
        I[0] = I0 + m_bulk*self.params['z_param'][1]**2
        I[0] += I0 + m_bulk*self.params['z_param'][3]**2
        I[1] = I[0]
        npt.assert_almost_equal(self.unknowns['bulkhead_I_keel'], I)
    '''
    def testDeriv(self):
        self.bulk.solve_nonlinear(self.params, self.unknowns, self.resid)
        J = self.bulk.linearize(self.params, self.unknowns, self.resid)
        self.assertEqual(J['bulkhead_mass', 'd_full'].shape, (11,11))
        self.assertEqual(J['bulkhead_mass', 't_full'].shape, (11,11))
    '''


class TestBuoyancyTank(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['d_full'] = 10.0*myones
        self.params['z_full'] = np.linspace(0, 1, NPTS) - 0.5
        self.params['rho'] = 1e3
        
        self.params['buoyancy_tank_diameter'] = 12.0
        self.params['buoyancy_tank_height'] = 0.25
        self.params['buoyancy_tank_location'] = 0.0
        self.params['buoyancy_tank_mass_factor'] = 1.1

        self.box = column.BuoyancyTankProperties(NPTS)

    def testNormal(self):
        self.box.solve_nonlinear(self.params, self.unknowns, self.resid)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.25
        m_expect = (2*A_box + 0.25*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertEqual(self.unknowns['buoyancy_tank_mass'], m_expect)
        self.assertEqual(self.unknowns['buoyancy_tank_cg'], -0.5 + 0.5*0.25)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_displacement'], V_box)
        #self.assertEqual(self.unknowns['buoyancy_tank_I_keel'], 0.0)

    def testTopAbove(self):
        self.params['buoyancy_tank_height'] = 0.75
        self.box.solve_nonlinear(self.params, self.unknowns, self.resid)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.5
        m_expect = (2*A_box + 0.75*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_mass'], m_expect)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_cg'], -0.5 + 0.5*0.75)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_displacement'], V_box)

    def testBottomAbove(self):
        self.params['buoyancy_tank_location'] = 0.6
        self.box.solve_nonlinear(self.params, self.unknowns, self.resid)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.0
        m_expect = (2*A_box + 0.25*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_mass'], m_expect)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_cg'], 0.1 + 0.5*0.25)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_displacement'], V_box)

    def testTooNarrow(self):
        self.params['buoyancy_tank_diameter'] = 8.0
        self.box.solve_nonlinear(self.params, self.unknowns, self.resid)

        A_box = np.pi * (6*6 - 5*5)
        V_box = np.pi * (6*6 - 5*5) * 0.0
        m_expect = (2*A_box + 0.25*2*np.pi*6) * (6.0/50.0) * 1e3 * 1.1
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_mass'], 0.0)
        self.assertEqual(self.unknowns['buoyancy_tank_cg'], -0.5 + 0.5*0.25)
        self.assertAlmostEqual(self.unknowns['buoyancy_tank_displacement'], 0.0)


        
    
class TestStiff(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['t_web'] = 0.5*secones
        self.params['t_flange'] = 0.3*secones
        self.params['h_web']  = 1.0*secones
        self.params['w_flange'] = 2.0*secones
        self.params['L_stiffener'] = np.r_[0.1*np.ones(5), 0.05*np.ones(5)]
        self.params['rho'] = 1e3
        self.params['ring_mass_factor'] = 1.1

        self.params['t_full'] = 0.5*myones
        self.params['t_full'][1::2] = 0.4
        self.params['d_full'] = 2*10.0*myones
        self.params['d_full'][1::2] = 2*8.0
        self.params['z_full'] = np.linspace(0, 1, NPTS) - 0.5
        self.params['z_param'] = np.linspace(0, 1, NSEC+1) - 0.5

        self.stiff = column.StiffenerMass(NSEC, NPTS)

    def testAll(self):
        self.stiff.solve_nonlinear(self.params, self.unknowns, self.resid)

        Rwo = 9-0.45
        Rwi = Rwo - 1.
        Rfi = Rwi - 0.3
        V1 = np.pi*(Rwo**2 - Rwi**2)*0.5
        V2 = np.pi*(Rwi**2 - Rfi**2)*2.0 
        V = V1+V2
        expect = 1.1*V*1e3
        actual = self.unknowns['stiffener_mass']

        # Test Mass
        self.assertAlmostEqual(actual.sum(), expect*(0.5/0.1 + 0.5/0.05))

        # Test moment
        self.params['L_stiffener'] = 1.2*secones
        self.stiff.solve_nonlinear(self.params, self.unknowns, self.resid)
        I_web = column.I_tube(Rwi, Rwo, 0.5, V1*1e3*1.1)
        I_fl  = column.I_tube(Rfi, Rwi, 2.0, V2*1e3*1.1)
        I_sec = I_web + I_fl
        z_sec = 0.6 + 1e-6

        I = np.zeros(6)
        I[2] = I_sec[0,2]
        I[0] += I_sec[0,0] + expect*z_sec**2.0
        I[1] = I[0]
        
        npt.assert_almost_equal(self.unknowns['stiffener_I_keel'], I)
        npt.assert_equal(self.unknowns['flange_spacing_ratio'], 2*2.0/1.2)
        npt.assert_equal(self.unknowns['stiffener_radius_ratio'], 1.75/9.0)


class TestGeometry(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        self.params['z_full_in'] = np.linspace(0, 50.0, NPTS)
        self.params['z_param_in'] = np.array([0.0, 20.0, 50.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['section_center_of_mass'],_ = nodal2sectional( self.params['z_full_in'] )
        self.params['freeboard'] = 15.0
        self.params['water_depth'] = 100.0
        self.params['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.params['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.params['stiffener_web_height']  = np.array([1.0, 1.0])
        self.params['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.params['stiffener_spacing'] = np.array([0.1, 0.1])
        self.params['Hs'] = 5.0
        self.params['max_draft'] = 70.0

        self.geom = column.ColumnGeometry(NSEC, NPTS)

    def testAll(self):
        self.geom.solve_nonlinear(self.params, self.unknowns, self.resid)
        self.assertEqual(self.unknowns['draft'], 35.0)
        self.assertEqual(self.unknowns['draft'], np.sum(self.params['section_height'])-self.params['freeboard'])
        self.assertEqual(self.unknowns['draft'], -1*self.unknowns['z_full'][0])
        self.assertEqual(self.unknowns['draft'], -1*self.unknowns['z_param'][0])
        self.assertEqual(self.unknowns['draft_margin'], 0.5)
        npt.assert_equal(self.unknowns['z_param'], np.array([-35.0, -15.0, 15.0]) )
        npt.assert_equal(self.unknowns['z_full'], self.params['z_full_in']-35)
        npt.assert_equal(self.unknowns['z_section'], self.params['section_center_of_mass']-35)
        npt.assert_equal(self.unknowns['t_web'], 0.5*secones)
        npt.assert_equal(self.unknowns['t_flange'], 0.3*secones)
        npt.assert_equal(self.unknowns['h_web'], 1.0*secones)
        npt.assert_equal(self.unknowns['w_flange'], 2.0*secones)
        npt.assert_equal(self.unknowns['L_stiffener'], 0.1*secones)
        
        
        
class TestProperties(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        # For Geometry call
        self.params['z_full_in'] = np.linspace(0, 50.0, NPTS)
        self.params['z_param_in'] = np.array([0.0, 20.0, 50.0])
        self.params['section_height'] = np.array([20.0, 30.0])
        self.params['section_center_of_mass'],_ = nodal2sectional( self.params['z_full_in'] )
        self.params['freeboard'] = 15.0
        self.params['fairlead'] = 10.0
        self.params['water_depth'] = 100.0
        self.params['Hs'] = 5.0
        self.params['max_draft'] = 70.0
        
        self.params['t_full'] = 0.5*myones
        self.params['d_full'] = 2*10.0*myones

        self.params['stack_mass_in'] = 0.0

        self.params['shell_I_keel'] = 10.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['stiffener_I_keel'] = 20.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['bulkhead_I_keel'] = 30.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.params['buoyancy_tank_I_keel'] = 5.0 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

        self.params['buoyancy_tank_diameter'] = 15.0
        
        self.params['water_density'] = 1e3
        self.params['bulkhead_mass'] = 10.0*myones
        self.params['shell_mass'] = 500.0*np.ones(NPTS-1)
        self.params['stiffener_mass'] = 100.0*np.ones(NPTS-1)
        self.params['buoyancy_tank_mass'] = 20.0
        self.params['buoyancy_tank_cg'] = -15.0
        self.params['buoyancy_tank_location'] = 0.3
        self.params['buoyancy_tank_displacement'] = 300.0
        self.params['column_mass_factor'] = 1.1
        self.params['outfitting_mass_fraction'] = 0.05

        self.params['permanent_ballast_height'] = 1.0
        self.params['permanent_ballast_density'] = 2e3
        
        self.params['mooring_mass'] = 50.0
        self.params['mooring_vertical_load'] = 25.0
        self.params['mooring_restoring_force'] = 1e5
        self.params['mooring_cost'] = 1e4

        self.params['ballast_cost_rate'] = 10.0
        self.params['tapered_col_cost_rate'] = 100.0
        self.params['outfitting_cost_rate'] = 1.0

        self.params['stiffener_web_thickness'] = np.array([0.5, 0.5])
        self.params['stiffener_flange_thickness'] = np.array([0.3, 0.3])
        self.params['stiffener_web_height']  = np.array([1.0, 1.0])
        self.params['stiffener_flange_width'] = np.array([2.0, 2.0])
        self.params['stiffener_spacing'] = np.array([0.1, 0.1])
        
        self.geom = column.ColumnGeometry(NSEC, NPTS)
        self.set_geometry()

        self.mycolumn = column.ColumnProperties(NPTS)
        
    def set_geometry(self):
        tempUnknowns = {}
        self.geom.solve_nonlinear(self.params, tempUnknowns, None)
        for pairs in tempUnknowns.items():
            self.params[pairs[0]] = pairs[1]

    def testColumnMassCG(self):
        m_column, cg_column, I_column, ibox = self.mycolumn.compute_column_mass_cg(self.params, self.unknowns)

        bulk  = self.params['bulkhead_mass']
        stiff = self.params['stiffener_mass']
        shell = self.params['shell_mass']
        box   = self.params['buoyancy_tank_mass']
        boxcg = self.params['buoyancy_tank_cg']
        mycg  = 1.1*(np.dot(bulk, self.params['z_full']) + box*boxcg + np.dot(stiff+shell, self.params['z_section']))/m_column
        mysec = stiff+shell+bulk[:-1]
        mysec[-1] += bulk[-1]
        mysec[ibox] += box
        mysec *= 1.1

        I_expect = 1.05 * 1.1 * 65.0*np.r_[np.ones(3), np.zeros(3)]
        
        self.assertAlmostEqual(self.unknowns['column_mass'], 1.1*(bulk.sum()+stiff.sum()+shell.sum()+box) )
        self.assertAlmostEqual(self.unknowns['column_mass'], m_column )
        self.assertEqual(self.unknowns['outfitting_mass'], 0.05*m_column )
        self.assertAlmostEqual(cg_column, mycg )
        npt.assert_equal(self.mycolumn.section_mass, mysec)
        npt.assert_equal(I_expect, I_column)

    def testBallastMassCG(self):
        m_ballast, cg_ballast, I_ballast = self.mycolumn.compute_ballast_mass_cg(self.params, self.unknowns)

        area = np.pi * 9.5**2
        m_perm = area * 1.0 * 2e3
        cg_perm = self.params['z_full'][0] + 0.5

        I_perm = np.zeros(6)
        I_perm[2] = 0.5 * m_perm * 9.5**2
        I_perm[0] = m_perm * (3*9.5**2 + 1.0**2) / 12.0 + m_perm*0.5**2
        I_perm[1] = I_perm[0]
        
        # Unused!
        h_expect = 1e6 / area / 1000.0
        m_expect = m_perm + 1e6
        cg_water = self.params['z_full'][0] + 1.0 + 0.5*h_expect
        cg_expect = (m_perm*cg_perm + 1e6*cg_water) / m_expect
        
        self.assertAlmostEqual(self.unknowns['ballast_mass'], m_perm)
        
        self.assertAlmostEqual(m_ballast, m_perm)
        self.assertAlmostEqual(cg_ballast, cg_perm)
        npt.assert_almost_equal(I_perm, I_ballast)


    def testBalance(self):
        rho_w = self.params['water_density']

        self.mycolumn.balance_column(self.params, self.unknowns)
        m_column, cg_column, I_column, ibox = self.mycolumn.compute_column_mass_cg(self.params, self.unknowns)
        m_ballast, cg_ballast, I_ballast = self.mycolumn.compute_ballast_mass_cg(self.params, self.unknowns)
        m_out = 0.05 * m_column
        m_expect = m_column + m_ballast + m_out
        cg_system = ((m_column+m_out)*cg_column + m_ballast*cg_ballast) / m_expect

        self.assertAlmostEqual(m_expect, self.unknowns['total_mass'].sum())
        self.assertAlmostEqual(cg_system, self.unknowns['z_center_of_mass'])

        V_column = np.pi * 100.0 * 35.0
        V_box    = self.params['buoyancy_tank_displacement']
        box_cg   = self.params['buoyancy_tank_cg']
        V_expect = V_column + V_box
        cb_expect = (-17.5*V_column + V_box*box_cg) / V_expect
        Ixx = 0.25 * np.pi * 1e4
        Axx = np.pi * 1e2
        self.assertAlmostEqual(self.unknowns['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.unknowns['hydrostatic_force'].sum(), V_expect*rho_w*g)
        self.assertAlmostEqual(self.unknowns['z_center_of_buoyancy'], cb_expect)
        self.assertAlmostEqual(self.unknowns['Iwater'], Ixx)
        self.assertAlmostEqual(self.unknowns['Awater'], Axx)

        z_draft = 15.0 - 50.0
        I_expect = I_column + I_ballast
        I_expect[0] -= m_expect*(cg_system-z_draft)**2
        I_expect[1] = I_expect[0]
        npt.assert_almost_equal(self.unknowns['I_column'], I_expect)

        m_a = np.zeros(6)
        m_a[:2] = V_expect * rho_w
        m_a[2]  = 0.5 * (8.0/3.0) * rho_w * 10.0**3
        m_a[3:5] = np.pi * rho_w * 100.0 * ((0-cb_expect)**3.0 - (-35-cb_expect)**3.0) / 3.0
        npt.assert_almost_equal(self.unknowns['added_mass'], m_a, decimal=-4)
        
        # Test if everything under water
        dz = -1.5*self.params['z_full'][-1]
        self.params['z_section'] += dz 
        self.params['z_full'] += dz 
        self.mycolumn.balance_column(self.params, self.unknowns)
        V_column = np.pi * 100.0 * 50.0
        V_expect = V_column + V_box
        cb_expect = (V_column*(-25.0 + self.params['z_full'][-1])  + V_box*box_cg) / V_expect
        self.assertAlmostEqual(self.unknowns['displaced_volume'].sum(), V_expect)
        self.assertAlmostEqual(self.unknowns['hydrostatic_force'].sum(), V_expect*rho_w*g)
        self.assertAlmostEqual(self.unknowns['z_center_of_buoyancy'], cb_expect)

        # Test taper- check hydrostatic via Archimedes within 1%
        self.params['d_full'][5] -= 8.0
        self.mycolumn.balance_column(self.params, self.unknowns)
        self.assertAlmostEqual(self.unknowns['hydrostatic_force'].sum() / (self.unknowns['displaced_volume'].sum()*rho_w*g), 1.0, delta=1e-2)
        
    def testCheckCost(self):
        self.unknowns['ballast_mass'] = 50.0
        self.unknowns['column_mass'] = 200.0
        self.unknowns['outfitting_mass'] = 25.0
        self.mycolumn.compute_cost(self.params, self.unknowns)

        self.assertEqual(self.unknowns['ballast_cost'], 10.0 * 50.0)
        self.assertEqual(self.unknowns['column_cost'], 100.0 * 200.0)
        self.assertEqual(self.unknowns['outfitting_cost'], 1.0 * 25.0)
        self.assertEqual(self.unknowns['total_cost'], 10.0*50.0 + 100.0*200.0 + 1.0*25.0)

        
class TestBuckle(unittest.TestCase):
    def setUp(self):
        self.params = {}
        self.unknowns = {}
        self.resid = None

        # Use the API 2U Appendix B as a big unit test!
        ksi_to_si = 6894757.29317831
        lbperft3_to_si = 16.0185
        ft_to_si = 0.3048
        in_to_si = ft_to_si / 12.0
        kip_to_si = 4.4482216 * 1e3

        onepts  = np.ones((NPTS,))
        onesec  = np.ones((NPTS-1,))
        #onesec0 = np.ones((NSEC,))
        self.params['d_full'] = 600 * onepts * in_to_si
        self.params['t_full'] = 0.75 * onepts * in_to_si
        self.params['t_web'] = 5./8. * onesec * in_to_si
        self.params['h_web'] = 14.0 * onesec * in_to_si
        self.params['t_flange'] = 1.0 * onesec * in_to_si
        self.params['w_flange'] = 10.0 * onesec * in_to_si
        self.params['L_stiffener'] = 5.0 * onesec * ft_to_si
        #self.params['section_height'] = 50.0 * onesec0 * ft_to_si
        self.params['pressure'] = (64.0*lbperft3_to_si) * g * (60*ft_to_si) * onepts
        self.params['E'] = 29e3 * ksi_to_si
        self.params['nu'] = 0.3
        self.params['yield_stress'] = 50 * ksi_to_si
        self.params['wave_height'] = 0.0 # gives only static pressure
        self.params['stack_mass_in'] = 9000 * kip_to_si/g
        self.params['section_mass'] = 0.0 * np.ones((NPTS-1,))
        self.params['loading'] = 'radial'
        self.params['z_full'] = np.linspace(0, 1, NPTS)
        self.params['z_section'],_ = nodal2sectional( self.params['z_full'] )
        self.params['z_param'] = np.linspace(0, 1, NSEC+1)
        self.params['gamma_f'] = 1.0
        self.params['gamma_b'] = 1.0

        self.buckle = column.ColumnBuckling(NSEC, NPTS)


    def testAppliedAxial(self):
        t = self.params['t_full'][0]
        d = self.params['d_full'][0]
        kip_to_si = 4.4482216 * 1e3
        expect = 9000 * kip_to_si / (2*np.pi*t*(0.5*d-0.5*t))
        npt.assert_almost_equal(self.buckle.compute_applied_axial(self.params), expect, decimal=4)
        
    def testCheckStresses(self):
        self.buckle.solve_nonlinear(self.params, self.unknowns, self.resid)
        
        npt.assert_almost_equal(self.unknowns['web_compactness'], 24.1/22.4, decimal=3)
        npt.assert_almost_equal(self.unknowns['flange_compactness'], 9.03/5.0, decimal=3)
        self.assertAlmostEqual(self.unknowns['axial_local_api'][1], 1.07, 1)
        self.assertAlmostEqual(self.unknowns['axial_general_api'][1], 0.34, 1)
        self.assertAlmostEqual(self.unknowns['external_local_api'][1], 1.07, 1)
        self.assertAlmostEqual(self.unknowns['external_general_api'][1], 0.59, 1)

        
def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestSectional))
    suite.addTest(unittest.makeSuite(TestBulk))
    suite.addTest(unittest.makeSuite(TestBuoyancyTank))
    suite.addTest(unittest.makeSuite(TestStiff))
    suite.addTest(unittest.makeSuite(TestGeometry))
    suite.addTest(unittest.makeSuite(TestProperties))
    suite.addTest(unittest.makeSuite(TestBuckle))
    return suite

if __name__ == '__main__':
    unittest.TextTestRunner().run(suite())
