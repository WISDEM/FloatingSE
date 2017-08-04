from openmdao.api import Component
import numpy as np
from scipy.optimize import fmin, brentq
import utils

from constants import gravity
from commonse.WindWaveDrag import cylinderDrag

def interp1d_local(y0, y1, X, x):
    return (y0 + (y1-y0) * (x/X))

def frustumVol(diam1, diam2, h): # calculates frustum volume - array or scalar inputs
    r1 = 0.5*diam1
    r2 = 0.5*diam2
    return (np.pi * (h / 3) * (r1**2 + r1*r2 + r2**2))

def frustumCG(diam1, diam2, h):  # calculates frustum center of gravity - array or scalar inputs
    # NOTE THIS IS FOR A SOLID FRUSTUM, NOT A SHELL
    r1 = 0.5*diam1
    r2 = 0.5*diam2
    return (0.25*h * (r1**2 + 2.*r1*r2 + 3.*r2**2) / (r1**2 + r1*r2 + r2**2))

def frustumShellCG(diam1, diam2, h):  # calculates frustum center of gravity - array or scalar inputs
    r1 = 0.5*diam1
    r2 = 0.5*diam2
    return (h/3 * (r1 + 2.*r2) / (r1 + r2))

def cylinder_drag_per_length(U, r, rho, mu):
    # dynamic pressure
    q = 0.5*rho*U**2

    # Reynolds number and drag
    Re = rho*U*d/mu
    cd, dcd_dRe = cylinderDrag(Re)
    return (q*cd*d)

def cylinder_forces_per_length(U, A, r, rho, mu, Cm):
    Fi = 0.5*rho*Cm*np.pi*r**2*A  # Morrison's equation
    Fd = cylinder_drag_per_length(U, r, rho, mu)
    Fp = Fi + Fd

def linear_waves(z, Dwater, hmax, T, rho, Ucurrent):
    '''
    This function computes the static and dynamic pressure at each section
    '''
    # Unpack variables
    amplitude = 0.5 * hmax

    # Compute wave number from Linear/Airy (Potential Flow) Wave Theory
    # (https://en.wikipedia.org/wiki/Airy_wave_theory)

    # circular frequency
    omega = 2.0*np.pi/T

    # compute wave number from dispersion relationship
    k = brentq(lambda x: omega**2 - gravity*x*np.tanh(Hwater*x), 0, 10*omega**2/gravity)

    # maximum velocity
    U  = amplitude * omega*np.cosh(k*(z  + Dwater))/np.sinh(k*Dwater) + Ucurrent
    U0 = amplitude * omega*np.cosh(k*(0. + Dwater))/np.sinh(k*Dwater) + Ucurrent

    # acceleration
    A  = U  * omega
    A0 = U0 * omega

    # Pressure is sum of static and dynamic contributions
    # Static is simple rho * g * z
    # Dynamic is from standard solution to Airy (Potential Flow) Wave theory
    z_section = 0.5*(self.z_nodes[:-1] + self.z_nodes[1:])
    static    = rho * gravity * np.abs(z)
    dynamic   = rho * gravity * amplitude * np.cosh(k*(z + Dwater)) / np.cosh(k*Dwater)
    pressure  = static + dynamic
    return U, A, pressure

def waveU(H,T,k,z,depth,theta):
    # TODO THETA TERM!
    return (np.pi*H/T) * (np.cosh(k*(z+depth)) / np.sinh(k*depth)) * np.cos(theta)

def waveUdot(H,T,k,z,depth,theta):
    # TODO THETA TERM!
    return (2*np.pi**2*H/T**2)* (np.cosh(k*(z+depth)) / np.sinh(k*depth)) * np.sin(theta)

def windPowerLaw(uref,href,alpha,H):
    return (uref*(H/href)**alpha)

def compute_bulkhead_mass(params):
    # Unpack variables
    twall        = params['wall_thickness']
    R_od         = params['outer_diameter']
    bulkheadTF   = params['outer_diameter']
    rho          = params['material_density']
    coeff        = params['bulkhead_mass_factor']

    # Compute bulkhead volume at every section node
    # Assume bulkheads are same thickness as shell wall
    V_bulk = np.pi * (R_od - twall)**2 * twall
    
    # Convert to mass with fudge factor for design features not captured in this simple approach
    m_bulk = coeff * rho * V_bulk

    # Zero out nodes where there is no bulkhead
    m_bulk[np.logical_not(bulkheadTF)] = 0.0

    # Assign each bulkhead to a section
    m_bulk_section      = m_bulk[:-1]
    m_bulk_section[-1] += m_bulk[-1]
    return m_bulk_section


def compute_shell_mass(params):
    # Unpack variables
    twall        = params['wall_thickness']
    R_od         = params['outer_diameter']
    h_section    = params['section_height']
    rho          = params['material_density']
    coeff        = params['shell_mass_factor']

    # Get average radius and wall thickness in each section
    R = 0.5*(R_od[1:] + R_od[:-1])
    t = 0.5*(twall[1:] + twall[:-1])

    # Shell volume for each section
    V_shell = 2.0 * np.pi * R * t * h_section

    # Ring mass by volume with fudge factor for design features not captured in this simple approach
    return (coeff * rho * V_shell)


def compute_stiffener_mass(params):
    # Unpack variables
    R_od         = params['outer_diameter']
    twall        = params['wall_thickness']
    t_web        = params['stiffener_web_thickness']
    t_flange     = params['stiffener_flange_thickness']
    h_web        = params['stiffener_web_height']
    h_flange     = params['stiffener_flange_width']
    L_stiffener  = params['stiffener_spacing']
    h_section    = params['section_height']
    rho          = params['material_density']
    coeff        = params['ring_mass_factor']

    # Outer and inner radius of web at section nodes
    R_wo = R_od - t_wall
    R_wi = R_wo - h_web
    # Outer and inner radius of flange at section nodes
    R_fo = R_wo
    R_fi = R_fo - t_flange

    # Material volumes at section nodes
    V_web    = np.pi*(R_wo**2 - R_wi**2) * t_web
    V_flange = np.pi*(R_fo**2 - R_fi**2) * h_flange

    # Ring mass by volume at section nodes
    # Include fudge factor for design features not captured in this simple approach
    m_ring = coeff*rho*(V_web + V_flange)
    
    # Average mass of rings per section
    m_ring_section = 0.5*(m_ring[1:] + m_ring[:-1])

    # Number of stiffener rings per section (height of section divided by spacing)
    nring_per_section = h_section / L_stiffener
    return (nring_per_section*m_ring_section)

def frustumVol_radius(r1, r2, h):
    return ( np.pi * (h/3.0) * (r1*r1 + r2*r2 + r1*r2) )

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


def plasticityRF(Felastic, yield_stress):
    # Plasticity reduction factor (Section 5 of API Bulletin 2U)
    Fratio = yield_stress / Felastic
    eta    = Fratio * (1.0 + 3.75*Fratio**2)**(-0.25)
    Finelastic = Felastic
    Finelastic[Felastic > 0.5*yield_stress] *= eta[Felastic > 0.5*yield_stress]
    return Finelastic


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

        # Variables local to the class and not OpenMDAO
        self.system_mass = None
        self.pressure    = None
        
        # Environment
        self.add_param('gust_factor', val=1.0, desc='gust factor')
        self.add_param('air_density', val=1.198, units='kg/m**3', desc='density of air')
        self.add_param('water_density', val=1025, units='kg/m**3', desc='density of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('wave_height', val=0.0, units='m', desc='wave height (crest to trough)')
        self.add_param('wave_period', val=0.0, units='m', desc='wave period')
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
        self.add_output('axial_local_unity', desc='unity check for axial load - local buckling')
        self.add_output('axial_general_unity', desc='unity check for axial load - genenral instability')
        self.add_output('external_local_unity', desc='unity check for external pressure - local buckling')
        self.add_output('external_general_unity', desc='unity check for external pressure - general instability')
        # Internal variables
        self.pressure = None

        
    def solve_nonlinear(self, params, unknowns, resids):
        # Compute total mass of spar substructure
        self.compute_mass(params, unknowns)

        # Stability in terms of can the spar's buyouncy support the whole structure
        # Also computers center of bouyancy
        self.compute_bouyancy(params, unknowns)

        # Static stability in that the CG should be below the Center of Bouyancy
        # This first computes the CG
        self.compute_static_stability(params, unknowns)
        
        # Preliminaries
        self.compute_pressure(params)
        
        # Check that axial and hoop loads don't exceed limits
        self.check_stresses(params, unknowns)

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
        h_web        = params['stiffener_web_height']
        h_flange     = params['stiffener_flange_width']
        h_section    = params['section_height']
        L_stiffener  = params['stiffener_spacing']
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        yield_stress = params['yield_stress']

        # TODO: PRESSURE

        # Apply quick "compactness" check on stiffener geometry
        # Constraint is that these must be >= 1
        unknown['flange_compactness'] = 0.375 * (t_flange / h_flange) * np.sqrt(E / yield_stress)
        unknown['web_compactness']    = 1.0   * (t_web    / h_web   ) * np.sqrt(E / yield_stress)
        
        # Geometry computations
        R        = R_od - 0.5*t_wall
        R_flange = R_od - t_wall - h_web - 0.5*
        area_stiff, y_cg, Ixx, Iyy = TBeamProperties(h_web, t_web, h_flange, t_flange)
        t_stiff  = area_stiff / h_web # effective thickness(width) of stiffener section
        
        # Compute mass of each section
        section_mass = self.compute_spar_mass(params)

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
        # Pressure from axial load
        pressure_sigma        = self.pressure - nu*axial_stress*t_wall/R_od
        # Compute the correction to hoop stress due to the presesnce of ring stiffeners
        stiffener_factor_KthL = 1 - psi * (pressure_sigma / self.pressure) * (k_d / (k_d + k_t))
        stiffener_factor_KthG = 1 -       (pressure_sigma / self.pressure) * (k_d / (k_d + k_t))
        hoop_stress_nostiff   = (self.pressure * R_od / t_wall)
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
            n[k] = brentq(lambda x:(c*x)**2*(1 + (c*x)**2)**4/(2 + 3*(c*x)**2) - z_m[k])
        # Calculate beta (local term)
        beta  = np.round(n) * L_stiffener / np.pi / R
        # Calculate buckling coefficient
        C_thL = a_thL * ( (1+beta**2)**2/(0.5+beta**2) + 0.112*m_x**4/(1+beta**2)**2/(0.5+beta**2) )
        # Calculate elastic and inelastic final limits
        elastic_extern_local_FreL   = C_thL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)
        inelastic_extern_local_FrcL = plasticityRF(elastic_extern_local_FreL, yield_stress)

        # 3. General instability buckling from axial loads
        area_stiff_bar = area_stiff / L_stiffener / t_wall
        # Compute imperfection factor
        a_x = 0.85 / (1 + 0.0025*2*R/t_wall)
        a_xG = a_x
        a_xG[area_stiff_bar>=0.2] = 0.72
        a_xG[area_stiff_bar<0.06 and area_stiff_bar<0.2] = (3.6 - 5.0*a_x)*area_stiff_bar
        # Calculate elastic and inelastic final limits
        elastic_axial_general_FxeG   = 0.605 * a_xG * E * t_wall/R * np.sqrt(1 + area_stiff_bar)
        inelastic_axial_general_FxcG = plasticityRF(elastic_axial_general_FxeG, yield_stress)

        # 4. General instability buckling from external loads
        # Distance from shell centerline to stiffener cg
        z_r = -(y_cg + 0.5*t_wall)
        # Imperfection factor
        a_thG = 0.8
        # Effective shell width
        L_shell_effective = L_stiffener
        L_shell_effective[m_x > 1.56] = 1.1*np.sqrt(2.0*R*t_wall) + t_web
        # Compute effective shell moment of inertia based on Ir - I of stiffener
        # TODO: is "Ir" Ixx or Iyy, guessing Iyy
        Ier = Iyy + area_stiff*z_r**2*L_shell_effective/(area_stiff+L_shell_effective) + L_shell_effective*t_wall**3/12.0
        # Lambda- a local constant
        lambda_G = np.pi * R å/ L_bulkhead
        # Compute pressure leading to elastic failure
        pressure_failure_peG = E * (lambda_G**4*t_wall/R/(n*n+k*lambda_G**2-1)/(n*n + lambda_G**2)**2 +
                                    I_er*(n*n-1)/L_stiffener)
        # Calculate elastic and inelastic final limits
        elastic_extern_general_FreG   = a_thG * pressure_failure_peG * R_od * stiffener_factor_KthG
        inelastic_extern_general_FrcG = plasticityRF(elastic_extern_general_FreG, yield_stress)
        
        # COMBINE AXIAL AND HOOP (EXTERNAL PRESSURE) LOADS TO FIND DESIGN LIMITS
        # (Section 6 of API Bulletin 2U)
        load_per_length_Nph = axial_stress        * t_wall
        load_per_length_Nth = hoop_stress_nostiff * t_wall
        load_ratio_k        = load_per_length_Nph / load_per_length_Nth
        def solveFthFph(Fxci, Frci, Kth):
            Kph = 1.0
            c1  = (Fxci + Frci) / yield_stress - 1.0
            c2  = load_ratio_k * Kph / Kth
            Fthci = brentq(lambda x: (c2*x/Fxci)**2 - c1*(c2*x/Fxci)*(x/Frci) + (x/Frci)**2 - 1.0)
            Fphci = c2 * Fthci
            return Fphci, Fthci
        
        inelastic_local_FphcL, inelastic_local_FthcL = solveFthFph(inelastic_axial_local_FxcL, inelastic_extern_local_FrcL, stiffener_factor_KthL)
        inelastic_general_FphcG, inelastic_general_FthcG = solveFthFph(inelastic_axial_general_FxcG, inelastic_extern_general_FrcG, stiffener_factor_KthG)
        
        # Use the inelastic limits and yield stress to compute required safety factors
        def safety_factor(Ficj):
            # Partial safety factor, psi
            psi = 1.4 - 0.4 * Ficj / yield_stress
            psi[Ficj <= 0.5*yield_stress] = 1.2
            psi[Ficj >= 0.5*yield_stress] = 1.0
            # Final safety factor is 25% higher to give a margin
            return 1.25*psi
        # Apply safety factors
        axial_limit_local_FaL     = inelastic_local_FphcL   / safety_factor(inelastic_local_FphcL  )
        extern_limit_local_FthL   = inelastic_local_FthcL   / safety_factor(inelastic_local_FthcL  )
        axial_limit_general_FaG   = inelastic_general_FphcG / safety_factor(inelastic_general_FphcG)
        extern_limit_general_FthG = inelastic_general_FthcG / safety_factor(inelastic_general_FthcG)

        # Compare limits to applied stresses and use this ratio as a design constraint
        # (Section 9 "Allowable Stresses" of API Bulletin 2U)
        # These values must be <= 1.0
        unknown['axial_local_unity']    = axial_stress / axial_limit_local_FaL
        unknown['axial_general_unity']  = axial_stress / axial_limit_general_FaG
        unknown['extern_local_unity']   = hoop_stress_between / extern_limit_local_FthL
        unknown['extern_general_unity'] = hoop_stress_between / extern_limit_general_FthG


    
        
    def compute_mass(self, params, unknowns):
        '''
        This function computes the spar mass by section from its components
        '''
        # Unpack variables
        coeff        = params['spar_mass_factor']

        # Find mass of all of the sub-components of the spar
        m_shell     = compute_shell_mass(params)
        m_stiffener = compute_stiffener_mass(params)
        m_bulkhead  = compute_bulkhead_mass(params)
        m_spar      = coeff*(m_shell + m_stiffener + m_bulkhead)
        # TODO: BALLAST
        m_ballast   = compute_ballast_mass(params)
        m_spar     += m_ballast
        # TODO: MOORING
        m_mooring   = params['mooring_mass']
        # Add up all contributions (this is still by section)
        m_system    = m_spar.sum() + m_ballast.sum() + m_mooring

        # Store in output dictionary and class variable
        unknowns['spar_mass']         = m_spar.sum()
        unknowns['ballas_mass']       = m_ballast.sum()
        unknowns['shell_mass']        = m_shell.sum()
        unknowns['stiffener_mass']    = m_stiffener.sum()
        unknowns['system_total_mass'] = m_system
        self.system_mass = m_system
        self.spar_mass   = m_spar

    
    def compute_bouyancy(self, params, unknowns):
        # Unpack variables
        R_od         = params['outer_diameter']
        twall        = params['wall_thickness']
        h_section    = params['section_height']
        rho_water    = params['water_density']
        
        # Mass of the turbine is tower plus RNA plus spar substructure
        m_turbine = (params['tower_mass'] + params['RNA_mass'])
        m_total   = m_turbine + self.system_mass.sum()

        # Compute volume of each section and mass of displaced water by section
        V_section   = frustumVol_radius(R_od[:-1], R_od[1:], h_section)
        m_displaced = rho_water * V_section

        # Compute the bouyancy force as a cumulative sum starting from the keel on up
        total_bouyancy = np.r_[0.0, np.cumsum(m_displaced)]
        # Find the section where the bouyancy force equals the weight- the exact point will be the waterline
        iSection       = np.argmax(total_bouyancy > m_total)
        # If the spar cannot hold the turbine, then iSection=0
        # We set freeboard = 0 to that the constraint freeboard>0 will be violated
        if iSection == 0: unknown['freeboard'] = 0.0

        # Have to now solve exactly where in this section is the waterline
        # Start with the remaining mass budget that must be supported
        m_remain = m_total - total_bouyancy[iSection-1]
        # Define a function that takes in the height local to this section and returns the bouyancy budget
        def solve_water_line(x):
            # Section radius as a function of height
            r = interp1d_local(R_od[iSection-1], R_od[iSection], h_section[iSection], x)
            return (rho_water*frustumVol_radius(R_od[iSection-1], r, x) - m_remain)
        # Solve this equation for the relative height in this section
        z_local = brentq(solve_water_line, 0.0, H)
        z_temp = np.interp(m_total, total_bouyancy, np.r_[0.0, np.cumsum(h_section)])
        print z_local, z_temp
        
        # Get cumulative height of spar along sections (keel is first node, bottom of first section)
        height_sum     = np.cumsum(h_section)
        # The draft is the whole sections below water and the
        # fractional piece of the water-line section just computerd
        self.draft     = height_sum[iSection-1] + z_local
        # Now that we know the draft, we can find the z-position of all the section nodes (keel is first node)
        self.z_nodes   = np.r_[0, height_sum] - self.draft
        # The spar length above the water line is the z-position of the top most node
        self.freeboard = self.z_nodes[-1]
        unknown['freeboard'] = self.freeboard

        # Now compute center of bouyancy- the center of mass of the displaced volume
        t_wall_ave = 0.5*(t_wall[1:] + t_wall[:-1])
        self.cm_section = frustumShellCM_radius(R_od[:-1], R_od[1:], t_wall_ave, h_section)
        # Convert local CM section heights to absolute z-position
        self.cm_section += self.z_nodes[:-1]
        # Set the volumes of every section under water
        r_od_waterline = interp1d_local(R_od[iSection-1], R_od[iSection], h_section[iSection], z_local)
        V_section[iSection] = frustumVol_radius(R_od[iSection-1], r_od_waterline, z_local)
        V_section[(iSection+1):] = 0.0
        # Compute Center of Bouyancy in z-coordinates (0=waterline)
        self.center_bouyancy = np.dot(V_section, cm_section) / V_section.sum()
        self.bouyancy_force  = rho_water * V_section.sum() * gravity

        # Compute the distance from the center of bouyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the moment of inertia of the water-plane cross section about the heel axis (without mass)
        # For a spar, we assume this is just the I of a ring about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html
        t_wall_waterline       = interp1d_local(t_wall[iSection-1], t_wall[iSection], h_section[iSection], z_local)
        r_waterline            = r_od_waterline - 0.5*t_wall_waterline
        Iwater                 = 0.25 * np.pi * r_waterline**4.0 
        bouyancy_metacentre_BM = Iwater / V_section.sum()
        self.metacentre = bouyancy_metacentre_BM + self.center_bouyancy
        

    def compute_static_stability(self, params, unknowns):
        # Unpack variables
        R_od         = params['outer_diameter']
        tower_mass   = params['tower_mass']
        tower_cg     = params['tower_center_of_gravity']
        rna_mass     = params['rna_mass']
        rna_cg       = params['rna_center_of_gravity'] # From base of tower

        # Put tower and rna cg in our coordinate system
        tower_cg += self.freeboard
        rna_cg   += self.freeboard
        
        # Compute CG
        # TODO: SHOULD THIS INCLUDE MOORING?
        self.system_cg = ( (np.dot(self.cm_section, self.spar_mass) + tower_mass*tower_cg + rna_mass*rna_cg) /
                           (self.spar_mass.sum() + tower_mass + rna_mass) )

        # Compute metacentric height: the distance from the CG to the metacentre
        self.metacentric_height = self.metacentre - self.system_cg
        
        # Measure static stability:
        # 1. Center of bouyancy should be above CG
        # 2. Metacentric height should be positive
        unknown['static_stability'  ] = self.system_cg < self.center_bouyancy
        unknown['metacentric_height'] = self.metacentric_height
        

    def compute_forces_moments(self, params):
        # Unpack variables
        rhoWater = params['water_density']
        rhoAir   = params['air_density']
        muWater  = params['water_viscosity']
        muAir    = params['air_viscosity']
        Dwater   = params['water_depth']
        hwave    = params['wave_height']
        Twave    = params['wave_period']
        uref     = params['wind_reference_speed']
        href     = params['wind_reference_height']
        alpha    = params['shear_coefficient']
        Cm       = params['morison_mass_coefficient']
        Ftower   = params['tower_wind_force']
        tower_cg = params['tower_center_of_gravity'] # from base of tower
        rna_mass = params['rna_mass']
        Frna     = params['rna_wind_force'] # Drag or thrust?
        rna_cg   = params['rna_center_of_gravity'] # z-direction From base of tower
        rna_cg_x = params['rna_center_of_gravity_x'] # x-direction from centerline
        
        npts = 100
        
        # Spar contribution
        zpts = np.linspace(self.z_nodes[0], self.z_nodes[-1], npts)
        rho = rhoWater * np.ones(zpts.shape)
        mu  = muWater  * np.ones(zpts.shape)
        rho[zpts>=0.0] = rhoAir
        mu[ zpts>=0.0] = muAir
        U, A, pressure = linear_waves(zpts, Dwater, hwave, Twave, rho, np.zeros(zpts.shape))
        # In air, set velocity to air speed instead
        A[zpts>=0.0] = 0.0
        U[zpts>=0.0] = windPowerLaw(uref, href, alpha, zpts[zpts>=0])
        # Radius along the spar
        r = np.interp(zpts, self.z_nodes, R_od)
        # Get forces along spar- good for water or wind with our vectorized inputs
        F = cylinder_forces_per_length(U, A, r, rho, mu, Cm)
        # Compute pitch moments from spar forces about CG
        M = np.trapz((zpts-self.system_cg)*F, zpts)

        # Tower contribution
        F += Ftower
        M += Ftower*(tower_cg + self.freeboard)

        # RNA contribution: wind force
        F += Frna
        M += Frna*(rna_cg + self.freeboard)

        # RNA contribution: moment due to offset mass
        # Note this is in the opposite moment direction as the wind forces
        # TODO: WHAT ABOUT THRUST?
        M -= rna_mass*gravity*rna_cg_x

        # Compute restoring moment under small angle assumptions
        M_restoring = self.metacentric_height * self.bouyancy_force
        
        # Comput heel angle
        unknown['heel_angle'] = np.rad2deg( M / M_restoring )

        # Now compute offsets from the applied force
        # First use added mass (the mass of the water that must be displaced in movement)
        # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/09/411/9411273.pdf
        mass_add_surge = rhoWater * np.pi * R_od.max() * self.draft
        # TODO!!
        
    def compute_cost(self, params, unknowns):
        pass
        
