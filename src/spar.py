from openmdao.api import Component
import numpy as np
from scipy.optimize import fmin, brentq
import utils

from constants import gravity

def TBeamProperties(baseHeight, baseThickness, flangeWidth, flangeThickness):
    '''
    http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx
    '''
    # Area of T cross section is sum of the two rectangles
    area_base   = baseHeight*baseThickness
    area_flange = flangeWidth*flangeThickness
    area        = area_base + area_flange
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ( (baseHeight + 0.5*flangeThickness)*area_flange + 0.5*baseHeight*area_base ) / area
    # Moments of inertia: y-axis runs through base (spinning top),
    # x-axis runs parallel to flange through cg
    Iyy =  (area_base*baseThickness**2 + area_flange*flangeWidth    ) / 12.0
    Ixx = ((area_base*baseHeight**2    + area_flange*flangeThickness) / 12.0 +
           area_base*(y_cg - 0.5*baseHeight)**2 +
           area_flange*(baseHeight + 0.5*flangeThickness - y_cg)**2 )
    return area, y_cg, Ixx, Iyy


class Spar(Component):
    '''
    |              |
    |              |  
    |   |      |   |
    |----      ----|
    |   |      |   |
    |              |
    |              |
    '''
    def __init__(self):
        super(Spar,self).__init__()

        # Environment
        self.add_param('gust_factor', val=1.0, desc='gust factor')
        self.add_param('air_density', val=1.198, units='kg/m**3', desc='density of air')
        self.add_param('water_density', val=1025, units='kg/m**3', desc='density of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('significant_wave_height', val=0.0, units='m', desc='significant wave height')
        self.add_param('significant_wave_period', val=0.0, units='m', desc='significant wave period')
        self.add_param('wind_reference_speed', val=0.0, units='m/s', desc='reference wind speed')
        self.add_param('wind_reference_height', val=0.0, units='m', desc='reference height')
        self.add_param('alpha', val=0.0, desc='power law exponent')

        # Material properties
        self.add_param('material_density', val=7850., units='kg/m**3', desc='density of spar material')
        self.add_param('E', val=200.e9, units='Pa', desc='Modulus of elasticity (Youngs) of spar material')
        self.add_param('nu', val=0.3, desc='poissons ratio of spar material')
        self.add_param('yield_stress', val=345000000., units='Pa', desc='yield stress of spar material')        
        # Design variables
        self.add_param('number_of_sections', val=3, desc='number of sections along the spar allowing for different stiffeners and diameters')
        self.add_param('section_height', val=np.zeros((3,)), units='m', desc='length (height) or each section in the spar (length = nsection)')
        self.add_param('outer_diameter', val=np.zeros((4,)), units='m', desc='outer diameter at each section end point (length = nsection + 1)')
        self.add_param('wall_thickness', val=np.zeros((4,)), units='m', desc='shell wall thickness at each section end point (length = nsection + 1)')
        self.add_param('stiffener_web_height', val=np.zeros((3,)), units='m', desc='height of stiffener web (base of T) within each section (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((3,)), units='m', desc='thickness of stiffener web (base of T) within each section (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((3,)), units='m', desc='height of stiffener flange (top of T) within each section (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((3,)), units='m', desc='thickness of stiffener flange (top of T) within each section (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((3,)), units='m', desc='Axial distance from one ring stiffener to another within each section (length = nsection)')


        # Outputs
        self.add_output('flange_compactness', desc='check for flange compactness')
        self.add_output('web_compactness', desc='check for web compactness')
        self.add_output('VAL', desc='unity check for axial load - local buckling')
        self.add_output('VAG', desc='unity check for axial load - genenral instability')
        self.add_output('VEL', desc='unity check for external pressure - local buckling')
        self.add_output('VEG', desc='unity check for external pressure - general instability')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Check that axial and hoop loads don't exceed limits
        self.check_stresses(params, unknowns)

        # Check that the substructure provides necessary stability
        self.check_stability(params, unknowns)
        
        # Compute total mass of spar substructure
        self.compute_mass(params, unknowns)

        # Compute costs of spar substructure
        self.compute_cost(params, unknowns)


    def check_stresses(self, params, unknowns):
        '''
        This function computes the applied axial and hoop stresses in a cylinder and 
        '''
        # Unpack variables
        nsections    = params['number_of_sections']
        R_od         = params['outer_diameter'] * 0.5
        t_wall       = params['wall_thickness']
        t_web        = params['stiffener_web_thickness']
        t_flange     = params['stiffener_flange_thickness']
        h_section    = params['section_height']
        L_stiffener  = params['stiffener_spacing']
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        h_web        = params['stiffener_web_height']
        yield_stress = params['yield_stress']
        
        # Geometry computations
        R = R_od - 0.5*t_wall
        R_flange = R_od - t_wall - h_web - 0.5*
        area, y_cg, Ixx, Iyy = TBeamProperties(h_web, t_web, params['stiffener_flange_width'], t_flange)
        t_stiff = area / h_web # effective thickness(width) of stiffener section
        
        # Compute mass of each section
        section_mass = self.compute_spar_mass(params, unknowns)

        # APPLIED STRESSES (Section 11 of API Bulletin 2U)
        # Applied axial stresss at each section node 
        axial_load = np.ones((nsections,)) * (params['tower_mass'] + params['RNA_mass'])
        # Add in weight of sections above it
        axial_load[1:] += np.cumsum( section_mass[:-1] )
        # Divide by shell cross sectional area to get stress
        axial_stress = axial_load / (2 * np.pi * R * t_wall)
        
        # Compute hoop stress accounting for stiffener rings
        # This has to be done at midpoint between stiffeners and at stiffener location itself
        # Compute beta (just a local term used here)
        d    = E * t_wall**3 / (12.0 * (1 - nu*nu))
        beta = (0.25 * E * t_wall / R_od / d)**0.25
        # Compute psi-factor (just a local term used here)
        u   = 0.5 * beta * L_stiffener
        psi = (2*np.sin(u)*np.cosh(u) + np.cos(u)*np.sinh(u)) / (np.sinh(2*u) + np.sin(2*u)) 
        # Compute a couple of other local terms
        u   = beta * L_stiffener
        k_t = 8 * beta**3 * d * (np.cosh(u) - np.cos(u)) / (np.sinh(u) + np.sin(u))
        k_d = E * t_stiff * (R_od**2 - R_flange**2) / R_od / ((1+nu)*R_od**2 + (1-nu)*R_flange**2)
        # Compute the correction to hoop stress due to the presesnce of ring stiffeners
        stiffener_factor_KthL = 1 - psi * (pressure_sigma / pressure) * (k_d / (k_d + k_t))
        stiffener_factor_KthG = 1 -       (pressure_sigma / pressure) * (k_d / (k_d + k_t))
        hoop_stress_nostiff   = (pressure * R_od / t_wall)
        hoop_stress_between   = hoop_stress_nostiff * stiffener_factor_KthL
        hoop_stress_atring    = hoop_stress_nostiff * stiffener_factor_KthG

        # BUCKLING FAILURE STRESSES (Section 4 of API Bulletin 2U)
        # 1. Local shell mode buckling from axial loads
        # Compute a few parameters that define the curvature of the geometry
        m_x  = L_stiffener / np.sqrt(R * t_wall)
        z_x  = m_x**2 * np.sqrt(1 - nu**2)
        z_m  = 12.0 * z_x**2 / np.pi**4
        # Imperfection factor- empirical fit that converts theory to reality
        a_xL = 9.0 * (300.0 + (2*R/t_wall))**(-0.4)
        # Calculate buckling coefficient
        C_xL = np.sqrt( 1 + 150.0 * a_xL**2 * m_x**4 / (2*R/t_wall) )
        # Calculate elastic and inelastic final limits
        elastic_axial_local_FxeL   = C_xL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)
        inelastic_axial_local_FxcL = plasticityRF(elastic_axial_local_FxeL, yield_stress)

        # 2. Local shell mode buckling from external (pressure) loads
        # Imperfection factor- empirical fit that converts theory to reality
        a_thL = np.ones(m_x.shape)
        a_thL[m_x > 5.0] = 0.8
        # Find the buckling mode- closest integer that is root of solved equation
        n   = np.zeros((nsections,))
        for k in xrange(nsections):
            c = L_stiffener[k] / np.pi / R[k]
            f = lambda x:(c*x)**2*(1 + (c*x)**2)**4/(2 + 3*(c*x)**2) - z_m[k]
            n[k] = brentq(f, 0., 15., xtol=1e-2, rtol=1e-3)
        # Calculate beta (local term)
        beta  = np.round(n) * L_stiffener / np.pi / R
        # Calculate buckling coefficient
        C_thL = a_thL * ( (1+beta**2)**2/(0.5+beta**2) + 0.112*m_x**4/(1+beta**2)**2/(0.5+beta**2) )
        # Calculate elastic and inelastic final limits
        elastic_extern_local_FreL   = C_thL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)
        inelastic_extern_local_FrcL = plasticityRF(elastic_extern_local_FreL, yield_stress)

        # 3. General instability buckling from axial loads
        area_stiff_bar = area / L_stiffener / t_wall
        a_x = 0.85 / (1 + 0.0025*2*R/t_wall)
        a_xG = a_x
        a_xG[area_stiff_bar>=0.2] = 0.72
        a_xG[area_stiff_bar<0.06 and area_stiff_bar<0.2] = (3.6 - 5.0*a_x)*area_stiff_bar
        # Calculate elastic and inelastic final limits
        elastic_axial_general_FxeG   = 0.605 * a_xG * E * t_wall/R * np.sqrt(1 + area_stiff_bar)
        inelastic_axial_general_FxcG = plasticityRF(elastic_axial_general_FxeG, yield_stress)

        # 4. General instability buckling from external loads
        z_r = -(y_cg + 0.5*t_wall)
        a_thG = 0.8
        L_shell_effective = L_stiffener
        L_shell_effective[m_x > 1.56] = 1.1*np.sqrt(2.0*R*t_wall) + t_web
        Ier = Ir + area*z_r**2*L_shell_effective/(area+L_shell_effective) + L_shell_effective*t_wall**3/12.0
        lambda_G = np.pi * R / L_bulkhead
        pressure_failure_peG = E * (lambda_G**4*t_wall/R/(n*n+k*lambda_G**2-1)/(n*n + lambda_G**2)**2 +
                                    I_er*(n*n-1)/L_stiffener)
        # Calculate elastic and inelastic final limits
        elastic_axial_general_FreG   = a_thG * pressure_failure_peG * R_od * stiffener_factor_KthG
        inelastic_axial_general_FrcG = plasticityRF(elastic_axial_general_FreG, yield_stress)
        
        
    def compute_spar_mass(self, params, unknowns):
        '''
        This function computes the applied axial and hoop stresses in a cylinder and 
        '''
        # Unpack variables
        twall     = params['wall_thickness']
        
        # Geometry computations
        R = params['outer_diameter'] - 0.5*t_wall
        
        # Mass of each sections and compute volume
        section_mass = params['material_density'] * 2 * np.pi * R * t_wall * params['section_height']
        return section_mass

    
    def compute_mass(self, params, unknowns):
        pass
    def compute_cost(self, params, unknowns):
        pass
        
