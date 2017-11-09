from openmdao.api import Component
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import cumtrapz

from constants import gravity
from sparGeometry import nodal2sectional, NSECTIONS
from commonse.WindWaveDrag import cylinderDrag
import commonse.Frustum as frustum

# Integration points
NPTS = 100

def cylinder_drag_per_length(U, r, rho, mu):
    """This function gives the drag per unit height of a cylinder based on Reynolds number.
    NOTE: This is implemented in CommonSE as an OpenMDAO component, here it is just a function

    INPUTS:
    ----------
    U   : float (scalar/vector),  speed in fluid
    r   : float (scalar/vector),  radius of cyliner
    rho : float (scalar/vector),  density of fluid
    mu  : float (scalar/vector),  viscosity of fluid

    OUTPUTS:
    -------
    D   : float (scalar/vector),  drag per unit length/height
    """
    # dynamic pressure
    q = 0.5*rho*U**2

    # Reynolds number and drag.  Note cylinder drag wants vector inputs
    Re = rho*U*2*r/mu
    if type(Re) is type(0.0): Re = np.array(Re)
    cd, dcd_dRe = cylinderDrag(Re)
    return (q*cd*2*r)


def cylinder_forces_per_length(U, A, r, rho, mu, Cm):
    """This function gives the forces per unit height of a cylinder.
    This includes the inertial forces of Morison's equation and fluid drag.
    NOTE: This is implemented in CommonSE as an OpenMDAO component, here it is just a function

    INPUTS:
    ----------
    U   : float (scalar/vector),  speed in fluid
    A   : float (scalar/vector),  acceleration in fluid
    r   : float (scalar/vector),  radius of cyliner
    rho : float (scalar/vector),  density of fluid
    mu  : float (scalar/vector),  viscosity of fluid
    cm  : float (scalar/vector),  1+added mass coefficient

    OUTPUTS:
    -------
    F   : float (scalar/vector),  force per unit length/height
    """
    Fi = rho * Cm * np.pi * r**2 * A  # Morison's equation
    Fd = cylinder_drag_per_length(U, r, rho, mu)
    return (Fi + Fd)


def linear_waves(z, Dwater, hmax, T, rho, Ucurrent=0.0):
    """This function gives the speed, acceleration, and pressure from standard linear (Airy) wave theory
    https://en.wikipedia.org/wiki/Airy_wave_theory
    https://www.ieawind.org/task_23/Subtask_2S_docs/Meeting%2008_Berlin/GeirMoe_Chap6-6_LinearWaveTheory.pdf

    INPUTS:
    ----------
    z        : float (scalar/vector),  z-position 0=surface, -Dwater=bottom
    Dwater   : float (scalar/vector),  Water depth
    hmax     : float (scalar/vector),  Crest-to-trough wave height
    T        : float (scalar/vector),  Wave period
    rho      : float (scalar/vector),  water density
    Ucurrent : float (scalar/vector),  Current speed beyond wave motion

    OUTPUTS:
    -------
    U        : float (scalar/vector),  Water speed due to wave motion
    A        : float (scalar/vector),  Water acceleration due to wave motion
    p        : float (scalar/vector),  Pressure due to dynamic wave motion and static column
    """
    # Wave amplitude is half wave "height"
    amplitude = 0.5 * hmax

    # circular frequency
    omega = 2.0*np.pi/T

    # compute wave number from dispersion relationship
    k = brentq(lambda x: omega**2 - gravity*x*np.tanh(Dwater*x), 0, 100.0*omega**2/gravity)

    # maximum velocity
    U  = amplitude * omega*np.cosh(k*(z  + Dwater))/np.sinh(k*Dwater) + Ucurrent
    U0 = amplitude * omega*np.cosh(k*(0. + Dwater))/np.sinh(k*Dwater) + Ucurrent
    # TODO: THETA term *cos(theta)
    
    # acceleration
    A  = U  * omega
    A0 = U0 * omega
    # TODO: THETA term *sin(theta)

    # Pressure is just dynamic sum of static and dynamic contributions
    # Static is simple rho * g * z
    # Dynamic is from standard solution to Airy (Potential Flow) Wave theory
    static  = rho * gravity * np.abs(z)
    dynamic = rho * gravity * amplitude * np.cosh(k*(z + Dwater)) / np.cosh(k*Dwater)
    p       = static + dynamic
    return U, A, p


def wind_power_law(uref,href,alpha,H):
    """Computes wind speed profile within atmospheric boundary layer using shear exponent / power law relationship
    (https://en.wikipedia.org/wiki/Wind_profile_power_law)

    INPUTS:
    ----------
    uref  : float (scalar/vector),  reference wind speed
    href  : float (scalar/vector),  height of reference wind speed
    alpha : float (scalar/vector),  shear exponent
    H     : float (scalar/vector),  query height

    OUTPUTS:
    -------
    U     : float (scalar/vector),  wind speed at query height
    """
    return (uref * (H/href)**alpha)


def compute_bulkhead_mass(params):
    """Computes bulkhead masses at each section node

    INPUTS:
    ----------
    params : dictionary of input parameters

    OUTPUTS:
    -------
    m_bulk : float vector, mass of bulkhead at each section node
    """
    # Unpack variables
    twall        = params['wall_thickness'] # at section nodes
    R_od         = params['outer_radius'] # at section nodes
    bulkheadTF   = params['bulkhead_nodes'] # at section nodes
    rho          = params['material_density']
    coeff        = params['bulkhead_mass_factor']

    # Compute bulkhead volume at every section node
    # Assume bulkheads are same thickness as shell wall
    V_bulk = np.pi * (R_od - twall)**2 * twall
    
    # Convert to mass with fudge factor for design features not captured in this simple approach
    m_bulk = coeff * rho * V_bulk

    # Zero out nodes where there is no bulkhead
    m_bulk[np.logical_not(bulkheadTF)] = 0.0
    return m_bulk


def compute_shell_mass(params):
    """Computes spar shell masses by section.  Assumes wall thickness is small relative to spar radius.

    INPUTS:
    ----------
    params : dictionary of input parameters

    OUTPUTS:
    -------
    m_shell : float vector, mass of bulkhead at each section node
    """
    # Unpack variables
    twall        = params['wall_thickness'] # at section nodes
    R_od         = params['outer_radius'] # at section nodes
    h_section    = params['section_height']
    rho          = params['material_density']
    coeff        = params['shell_mass_factor']

    # Same with wall thicknesses
    Tbot = twall[:-1]
    Ttop = twall[1:]

    # Radius (to middle of shell) at base and tops of all frustum sections
    Rbot = R_od[:-1] - 0.5*Tbot
    Rtop = R_od[1:] - 0.5*Ttop

    # Shell volume for each section determined by allowing for linear variation in R & T in each section.
    # Integrate 2*pi*r*t*dz from 0 to H
    V_shell = (np.pi*h_section/3.0)*( Rbot*(2*Tbot + Ttop) + Rtop*(Tbot + 2*Ttop) )

    # Ring mass by volume with fudge factor for design features not captured in this simple approach
    return (coeff * rho * V_shell)


def compute_stiffener_mass(params):
    """Computes spar stiffener mass by section.  
    Stiffener being the ring of T-cross section members placed periodically along spar
    Assumes constant stiffener spacing along the spar, but allows for varying stiffener geometry
    Slicing the spar lengthwise would reveal the stiffener T-geometry as:
    |              |
    |              |  
    |   |      |   |
    |----      ----|
    |   |      |   |
    |              |
    |              |

    INPUTS:
    ----------
    params : dictionary of input parameters

    OUTPUTS:
    -------
    m_stiffener : float vector, mass of stiffeners by section
    """
    # Unpack variables
    R_od         = params['outer_radius'] # at section nodes
    twall        = params['wall_thickness'] # at section nodes
    t_web        = params['stiffener_web_thickness'] # by section
    t_flange     = params['stiffener_flange_thickness'] # by section
    h_web        = params['stiffener_web_height'] # by section
    w_flange     = params['stiffener_flange_width'] # by section
    L_stiffener  = params['stiffener_spacing']
    h_section    = params['section_height']
    rho          = params['material_density']
    coeff        = params['ring_mass_factor']

    # Get average radius and wall thickness in each section
    R = nodal2sectional(R_od)
    t = nodal2sectional(twall)

    # Outer and inner radius of web by section
    R_wo = R - t
    R_wi = R_wo - h_web
    # Outer and inner radius of flange by section
    R_fo = R_wi
    R_fi = R_fo - t_flange

    # Material volumes by section
    V_web    = np.pi*(R_wo**2 - R_wi**2) * t_web
    V_flange = np.pi*(R_fo**2 - R_fi**2) * w_flange

    # Ring mass by volume by section 
    # Include fudge factor for design features not captured in this simple approach
    m_ring = coeff*rho*(V_web + V_flange)
    
    # Number of stiffener rings per section (height of section divided by spacing)
    nring_per_section = h_section / L_stiffener
    return (nring_per_section*m_ring)



def TBeamProperties(h_web, t_web, w_flange, t_flange):
    """Computes T-cross section area, CG, and moments of inertia
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (T-base) height
    t_web    : float (scalar/vector),  web (T-base) thickness
    w_flange : float (scalar/vector),  flange (T-top) width/height
    t_flange : float (scalar/vector),  flange (T-top) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    Ixx  : float (scalar/vector),  Moment of intertia around axis parallel to flange, through y_cg
    Iyy  : float (scalar/vector),  Moment of intertia around y-axis
    """
    # Area of T cross section is sum of the two rectangles
    area_web    = h_web * t_web
    area_flange = w_flange * t_flange
    area        = area_web + area_flange
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ( (h_web + 0.5*t_flange)*area_flange + 0.5*h_web*area_web ) / area
    # Moments of inertia: y-axis runs through base (spinning top),
    # x-axis runs parallel to flange through cg
    Iyy =  (area_web*t_web**2 + area_flange*w_flange**2    ) / 12.0
    Ixx = ((area_web*h_web**2 + area_flange*t_flange**2) / 12.0 +
           area_web*(y_cg - 0.5*h_web)**2 +
           area_flange*(h_web + 0.5*t_flange - y_cg)**2 )
    return area, y_cg, Ixx, Iyy


def IBeamProperties(h_web, t_web, w_flange, t_flange, w_base, t_base):
    """Computes uneven I-cross section area, CG
    See: http://www.amesweb.info/SectionalPropertiesTabs/SectionalPropertiesTbeam.aspx

    INPUTS:
    ----------
    h_web    : float (scalar/vector),  web (I-stem) height
    t_web    : float (scalar/vector),  web (I-stem) thickness
    w_flange : float (scalar/vector),  flange (I-top) width/height
    t_flange : float (scalar/vector),  flange (I-top) thickness
    w_base   : float (scalar/vector),  base (I-bottom) width/height
    t_base   : float (scalar/vector),  base (I-bottom) thickness

    OUTPUTS:
    -------
    area : float (scalar/vector),  T-cross sectional area
    y_cg : float (scalar/vector),  Position of CG along y-axis (extending from base up through the T)
    """
    # Area of T cross section is sum of the two rectangles
    area_web    = h_web * t_web
    area_flange = w_flange * t_flange
    area_base   = w_base * t_base
    area        = area_web + area_flange + area_base
    # Y-position of the center of mass (Yna) measured from the base
    y_cg = ( (t_base + h_web + 0.5*t_flange)*area_flange + (t_base + 0.5*h_web)*area_web + 0.5*t_base*area_base ) / area
    return area, y_cg


def compute_applied_axial(params, section_mass):
    """Compute axial stress for spar from z-axis loading

    INPUTS:
    ----------
    params       : dictionary of input parameters
    section_mass : float (scalar/vector),  mass of each spar section as axial loading increases with spar depth

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  axial stress
    """
    # Unpack variables
    R_od    = nodal2sectional(params['outer_radius'])
    t_wall  = nodal2sectional(params['wall_thickness'])
    m_stack = params['stack_mass_in']
    R       = R_od - 0.5*t_wall
    
    # Add in weight of sections above it
    axial_load    = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
    # Divide by shell cross sectional area to get stress
    return (gravity * axial_load / (2.0 * np.pi * R * t_wall))


def compute_applied_hoop(pressure, R_od, t_wall):
    """Compute hoop stress WITHOUT accounting for stiffener rings

    INPUTS:
    ----------
    pressure : float (scalar/vector),  radial (hydrostatic) pressure
    R_od     : float (scalar/vector),  radius to outer wall of shell
    t_wall   : float (scalar/vector),  shell wall thickness

    OUTPUTS:
    -------
    stress   : float (scalar/vector),  hoop stress with no stiffeners
    """
    return (pressure * R_od / t_wall)

    
def compute_stiffener_factors(params, pressure, axial_stress):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params       : dictionary of input parameters
    pressure     : float (scalar/vector),  radial (hydrostatic) pressure
    axial_stress : float (scalar/vector),  axial loading (z-axis) stress

    OUTPUTS:
    -------
    stiffener_factor_KthL : float (scalar/vector),  Stress modifier from stiffeners for local buckling from axial loads
    stiffener_factor_KthG : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    """
    # Unpack variables
    R_od         = nodal2sectional(params['outer_radius'])
    t_wall       = nodal2sectional(params['wall_thickness'])
    t_web        = params['stiffener_web_thickness']
    t_flange     = params['stiffener_flange_thickness']
    h_web        = params['stiffener_web_height']
    w_flange     = params['stiffener_flange_width']
    L_stiffener  = params['stiffener_spacing']
    E            = params['E'] # Young's modulus
    nu           = params['nu'] # Poisson ratio

    # Geometry computations
    R_flange = R_od - h_web # Should have "- t_wall", but not in appendix B
    area_stiff, y_cg, Ixx, Iyy = TBeamProperties(h_web, t_web, w_flange, t_flange)
    t_stiff  = area_stiff / h_web # effective thickness(width) of stiffener section

    # Compute hoop stress modifiers accounting for stiffener rings
    # This has to be done at midpoint between stiffeners and at stiffener location itself
    # Compute beta (just a local term used here)
    D    = E * t_wall**3 / (12.0 * (1 - nu*nu))
    beta = (0.25 * E * t_wall / R_od**2 / D)**0.25
    # Compute psi-factor (just a local term used here)
    u     = 0.5 * beta * L_stiffener
    psi_k = 2.0 * (np.sin(u)*np.cosh(u) + np.cos(u)*np.sinh(u)) / (np.sinh(2*u) + np.sin(2*u)) 

    # Compute a couple of other local terms
    u   = beta * L_stiffener
    k_t = 8 * beta**3 * D * (np.cosh(u) - np.cos(u)) / (np.sinh(u) + np.sin(u))
    k_d = E * t_stiff * (R_od**2 - R_flange**2) / R_od / ((1+nu)*R_od**2 + (1-nu)*R_flange**2)

    # Pressure from axial load
    pressure_sigma        = pressure - nu*axial_stress*t_wall/R_od

    # Compute the correct ion to hoop stress due to the presesnce of ring stiffeners
    stiffener_factor_KthL = 1 - psi_k * (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    stiffener_factor_KthG = 1 -         (pressure_sigma / pressure) * (k_d / (k_d + k_t))
    return stiffener_factor_KthL, stiffener_factor_KthG


def compute_elastic_stress_limits(params, KthG, loading='hydrostatic'):
    """Compute modifiers to stress due to presence of stiffener rings.

    INPUTS:
    ----------
    params  : dictionary of input parameters
    KthG    : float (scalar/vector),  Stress modifier from stiffeners for general buckling from external pressure
    loading : string (hydrostatic/radial), Parameter that determines a coefficient- is only included for unit testing consistency with API 2U Appdx B and should not be used in practice

    OUTPUTS:
    -------
    elastic_axial_local_FxeL    : float (scalar/vector),  Elastic stress limit for local buckling from axial loads
    elastic_extern_local_FreL   : float (scalar/vector),  Elastic stress limit for local buckling from external pressure loads
    elastic_axial_general_FxeG  : float (scalar/vector),  Elastic stress limit for general instability from axial loads
    elastic_extern_general_FreG : float (scalar/vector),  Elastic stress limit for general instability from external pressure loads
    """
    # Unpack variables
    R_od         = nodal2sectional(params['outer_radius'])
    t_wall       = nodal2sectional(params['wall_thickness'])
    t_web        = params['stiffener_web_thickness']
    t_flange     = params['stiffener_flange_thickness']
    h_web        = params['stiffener_web_height']
    w_flange     = params['stiffener_flange_width']
    h_section    = params['section_height']
    L_stiffener  = params['stiffener_spacing']
    E            = params['E'] # Young's modulus
    nu           = params['nu'] # Poisson ratio

    # Geometry computations
    nsections = R_od.size
    area_stiff, y_cg, Ixx, Iyy = TBeamProperties(h_web, t_web, w_flange, t_flange)
    area_stiff_bar = area_stiff / L_stiffener / t_wall
    R  = R_od - 0.5*t_wall

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

    # 2. Local shell mode buckling from external (pressure) loads
    # Imperfection factor- empirical fit that converts theory to reality
    a_thL = np.ones(m_x.shape)
    a_thL[m_x > 5.0] = 0.8
    # Find the buckling mode- closest integer that is root of solved equation
    n   = np.zeros((nsections,))
    maxn = 50
    for k in xrange(nsections):
        c = L_stiffener[k] / np.pi / R[k]
        myfun = lambda x:((c*x)**2*(1 + (c*x)**2)**4/(2 + 3*(c*x)**2) - z_m[k])
        try:
            n[k] = brentq(myfun, 0, maxn)
        except:
            n[k] = maxn
    # Calculate beta (local term)
    beta  = np.round(n) * L_stiffener / np.pi / R
    # Calculate buckling coefficient
    C_thL = a_thL * ( (1+beta**2)**2/(0.5+beta**2) + 0.112*m_x**4/(1+beta**2)**2/(0.5+beta**2) )
    # Calculate elastic and inelastic final limits
    elastic_extern_local_FreL   = C_thL * np.pi**2 * E * (t_wall/L_stiffener)**2 / 12.0 / (1-nu**2)

    # 3. General instability buckling from axial loads
    # Compute imperfection factor
    a_x = 0.85 / (1 + 0.0025*2*R/t_wall)
    a_xG = a_x
    a_xG[area_stiff_bar>=0.2] = 0.72
    ind = np.logical_and(area_stiff_bar<0.06, area_stiff_bar<0.2)
    a_xG[ind] = (3.6 - 5.0*a_x[ind])*area_stiff_bar[ind]
    # Calculate elastic and inelastic final limits
    elastic_axial_general_FxeG   = 0.605 * a_xG * E * t_wall/R * np.sqrt(1 + area_stiff_bar)

    # 4. General instability buckling from external loads
    # Distance from shell centerline to stiffener cg
    z_r = -(y_cg + 0.5*t_wall)
    # Imperfection factor
    a_thG = 0.8
    # Effective shell width if the outer shell and the T-ring stiffener were to be combined to make an uneven I-beam
    L_shell_effective = 1.1*np.sqrt(2.0*R*t_wall) + t_web
    L_shell_effective[m_x <= 1.56] = L_stiffener[m_x <= 1.56]
    # Get properties of this effective uneven I-beam
    _, yna_eff = IBeamProperties(h_web, t_web, w_flange, t_flange, L_shell_effective, t_wall)
    Rc = R_od - yna_eff
    # Compute effective shell moment of inertia based on Ir - I of stiffener
    Ier = Ixx + area_stiff*z_r**2*L_shell_effective*t_wall/(area_stiff+L_shell_effective*t_wall) + L_shell_effective*t_wall**3/12.0
    # Lambda- a local constant
    lambda_G = np.pi * R / h_section
    # Coefficient factor listed as 'k' in peG equation
    coeff = 0.5 if loading in ['hydro','h','hydrostatic','static'] else 0.0    
    # Compute pressure leading to elastic failure
    n = np.zeros(R_od.shape)
    pressure_failure_peG = np.zeros(R_od.shape)
    for k in xrange(nsections):
        peG = lambda x: ( E*lambda_G[k]**4*t_wall[k]/R[k]/(x**2+0.0*lambda_G[k]**2-1)/(x**2 + lambda_G[k]**2)**2 +
                          E*Ier[k]*(x**2-1)/L_stiffener[k]/Rc[k]**2/R_od[k] )
        minout = minimize_scalar(peG, bounds=(2.0, 15.0), method='bounded')
        n[k] = minout.x
        pressure_failure_peG[k] = peG(n[k])
    # Calculate elastic and inelastic final limits
    elastic_extern_general_FreG   = a_thG * pressure_failure_peG * R_od * KthG / t_wall

    return elastic_axial_local_FxeL, elastic_extern_local_FreL, elastic_axial_general_FxeG, elastic_extern_general_FreG


def plasticityRF(Felastic, yield_stress):
    """Computes plasticity reduction factor for elastic stresses near the yield stress to obtain an inelastic stress
    This is defined in Section 5 of API Bulletin 2U

    INPUTS:
    ----------
    Felastic     : float (scalar/vector),  elastic stress
    yield_stress : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    Finelastic   : float (scalar/vector),  modified (in)elastic stress
    """
    Fratio = np.array(yield_stress / Felastic)
    eta    = Fratio * (1.0 + 3.75*Fratio**2)**(-0.25)
    Finelastic = np.array(Felastic)
    Finelastic[Felastic > 0.5*yield_stress] *= eta[Felastic > 0.5*yield_stress]
    return Finelastic


def safety_factor(Ficj, yield_stress):
    """Use the inelastic limits and yield stress to compute required safety factors
    This is defined in Section 9 of API Bulletin 2U

    INPUTS:
    ----------
    Ficj          : float (scalar/vector),  inelastic stress
    yield_stress  : float (scalar/vector),  yield stress

    OUTPUTS:
    -------
    safety_factor : float (scalar/vector),  margin applied to inelastic stress limits
    """
    # Partial safety factor, psi
    psi = np.array(1.4 - 0.4 * Ficj / yield_stress)
    psi[Ficj <= 0.5*yield_stress] = 1.2
    psi[Ficj >= yield_stress] = 1.0
    # Final safety factor is 25% higher to give a margin
    return 1.25*psi


class Cylinder(Component):
    """
    OpenMDAO Component class for cylinder substructure elements in floating offshore wind turbines.
    """

    def __init__(self):
        super(Cylinder,self).__init__()

        # Variables local to the class and not OpenMDAO
        self.section_mass   = None # Weight of spar by section
        self.z_permanent_ballast = None
        
        # Environment
        self.add_param('air_density', val=1.198, units='kg/m**3', desc='density of air')
        self.add_param('air_viscosity', val=1.81e-5, units='kg/s/m', desc='viscosity of air')
        self.add_param('water_density', val=1025.0, units='kg/m**3', desc='density of water')
        self.add_param('water_viscosity', val=8.90e-4, units='kg/s/m', desc='viscosity of water')
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('wave_height', val=0.0, units='m', desc='wave height (crest to trough)')
        self.add_param('wave_period', val=0.0, units='m', desc='wave period')
        self.add_param('wind_reference_speed', val=0.0, units='m/s', desc='reference wind speed')
        self.add_param('wind_reference_height', val=0.0, units='m', desc='reference height')
        self.add_param('alpha', val=0.0, desc='power law exponent')
        self.add_param('morison_mass_coefficient', val=2.0, desc='One plus the added mass coefficient')
        self.add_param('stack_mass_in', val=0.0, units='kg', desc='Weight above the cylinder column')
        
        # Material properties
        self.add_param('material_density', val=7850., units='kg/m**3', desc='density of spar material')
        self.add_param('E', val=200.e9, units='Pa', desc='Modulus of elasticity (Youngs) of spar material')
        self.add_param('nu', val=0.3, desc='poissons ratio of spar material')
        self.add_param('yield_stress', val=345000000., units='Pa', desc='yield stress of spar material')
        self.add_param('permanent_ballast_density', val=4492.0, units='kg/m**3', desc='density of permanent ballast')

        # Inputs from SparGeometry
        self.add_param('z_nodes', val=np.zeros((NSECTIONS+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_section', val=np.zeros((NSECTIONS,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Design variables
        self.add_param('section_height', val=np.zeros((NSECTIONS,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('outer_radius', val=np.zeros((NSECTIONS+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('wall_thickness', val=np.zeros((NSECTIONS+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('stiffener_web_height', val=np.zeros((NSECTIONS,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((NSECTIONS,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((NSECTIONS,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((NSECTIONS,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((NSECTIONS,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')
        self.add_param('bulkhead_nodes', val=[True]*(NSECTIONS+1), desc='Nodal locations where there is a bulkhead bottom to top (length = nsection + 1)', pass_by_obj=True)
        self.add_param('permanent_ballast_height', val=0.0, units='m', desc='height of permanent ballast')
        
        # Mass correction factors from simple rules here to real life
        self.add_param('bulkhead_mass_factor', val=1.0, desc='Bulkhead mass correction factor')
        self.add_param('ring_mass_factor', val=1.0, desc='Stiffener ring mass correction factor')
        self.add_param('spar_mass_factor', val=1.0, desc='Overall spar mass correction factor')
        self.add_param('shell_mass_factor', val=1.0, desc='Spar shell mass correction factor')
        self.add_param('outfitting_mass_fraction', val=0.0, desc='Mass fraction added for outfitting')
        
        # Cost rates
        self.add_param('ballast_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of ballast')
        self.add_param('tapered_col_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of tapered columns (frustums)')
        self.add_param('outfitting_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass for outfitting spar')

        # Outputs
        self.add_output('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_output('ballast_mass', val=0.0, units='kg', desc='mass of permanent ballast')
        self.add_output('variable_ballast_interp_mass', val=np.zeros((NPTS,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_output('variable_ballast_interp_zpts', val=np.zeros((NPTS,)), units='m', desc='z-points of potential ballast mass')

        self.add_output('z_center_of_gravity', val=0.0, units='m', desc='z-position CofG of cylinder')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position CofB of cylinder')
        self.add_output('Awater', val=0.0, units='m**2', desc='Area of waterplace cross section')
        self.add_output('Iwater', val=0.0, units='m**4', desc='Second moment of area of waterplace cross section')
        self.add_output('displaced_volume', val=0.0, units='m**3', desc='Volume of water displaced by cylinder')
 
        self.add_output('spar_cost', val=0.0, units='USD', desc='cost of spar structure')
        self.add_output('spar_mass', val=0.0, units='kg', desc='mass of spar structure')
        self.add_output('shell_mass', val=0.0, units='kg', desc='mass of spar shell')
        self.add_output('stiffener_mass', val=0.0, units='kg', desc='mass of spar stiffeners')
        self.add_output('bulkhead_mass', val=0.0, units='kg', desc='mass of spar bulkheads')

        self.add_output('surge_force_vector', val=np.zeros((NPTS,)), units='N', desc='Force in surge direction')
        self.add_output('surge_force_points', val=np.zeros((NPTS,)), units='m', desc='z-points for surge force vector')
        
        self.add_output('outfitting_cost', val=0.0, units='USD', desc='cost of outfitting the spar')
        self.add_output('outfitting_mass', val=0.0, units='kg', desc='cost of outfitting the spar')

        self.add_output('total_mass', val=0.0, units='kg', desc='total mass of cylinder')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of cylinder')
        
        # Output constraints
        self.add_output('flange_spacing_ratio', val=np.zeros((NSECTIONS,)), desc='ratio between flange and stiffener spacing')
        self.add_output('web_radius_ratio', val=np.zeros((NSECTIONS,)), desc='ratio between web height and radius')
        self.add_output('flange_compactness', val=np.zeros((NSECTIONS,)), desc='check for flange compactness')
        self.add_output('web_compactness', val=np.zeros((NSECTIONS,)), desc='check for web compactness')
        self.add_output('axial_local_unity', val=np.zeros((NSECTIONS,)), desc='unity check for axial load - local buckling')
        self.add_output('axial_general_unity', val=np.zeros((NSECTIONS,)), desc='unity check for axial load - genenral instability')
        self.add_output('external_local_unity', val=np.zeros((NSECTIONS,)), desc='unity check for external pressure - local buckling')
        self.add_output('external_general_unity', val=np.zeros((NSECTIONS,)), desc='unity check for external pressure - general instability')

        
        
    def solve_nonlinear(self, params, unknowns, resids):
        """Main entry point for OpenMDAO and the 'main' function for executing spar substructure sizing analysis
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        resids   : OpenMDAO residuals dictionary
        
        OUTPUTS  : (none)
        """
        # Balance the design by adding ballast to achieve desired draft and freeboard heights
        # This requires a full mass tally as well.
        # Compute the CG, CB, and metacentric heights- use these for a static stability check
        self.balance_cylinder(params, unknowns)

        # Calculate hydrodynamic forces on cylinder
        self.compute_surge_pitch(params, unknowns)
        
        # Check that axial and hoop loads don't exceed limits
        self.check_stresses(params, unknowns)

        # Compute costs of spar substructure
        self.compute_cost(params, unknowns)


    def compute_spar_mass_cg(self, params, unknowns):
        """Computes spar mass from components: Shell, Stiffener rings, Bulkheads
        Also computes center of mass of the shell by weighted sum of the components' position
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS:
        ----------
        section_mass class variable set
        m_spar   : spar mass
        z_cg     : center of mass along z-axis for the spar
        spar_mass       in 'unknowns' dictionary set
        shell_mass      in 'unknowns' dictionary set
        stiffener_mass  in 'unknowns' dictionary set
        bulkhead_mass   in 'unknowns' dictionary set
        outfitting_mass in 'unknowns' dictionary set
        """
        # Unpack variables
        coeff        = params['spar_mass_factor']
        z_nodes      = params['z_nodes']
        z_section    = params['z_section']
        
        m_spar = 0.0
        z_cg = 0.0
        
        # Find mass of all of the sub-components of the spar
        # Masses assumed to be focused at section centroids
        m_shell     = compute_shell_mass(params)
        m_stiffener = compute_stiffener_mass(params)
        m_spar     += (m_shell + m_stiffener).sum()
        z_cg       += np.dot(m_shell+m_stiffener, z_section)

        # Masses assumed to be centered at nodes
        m_bulkhead  = compute_bulkhead_mass(params)
        m_spar     += m_bulkhead.sum()
        z_cg       += np.dot(m_bulkhead, z_nodes)

        # Account for components not explicitly calculated here
        m_spar     *= coeff

        # Compute CG position of the spar
        z_cg       *= coeff / m_spar

        # Apportion every mass to a section for buckling stress computation later
        self.section_mass = coeff*(m_shell + m_stiffener + m_bulkhead[:-1])
        self.section_mass[-1] += coeff*m_bulkhead[-1]
        
        # Store outputs addressed so far
        unknowns['spar_mass']         = m_spar
        unknowns['shell_mass']        = m_shell.sum()
        unknowns['stiffener_mass']    = m_stiffener.sum()
        unknowns['bulkhead_mass']     = m_bulkhead.sum()
        unknowns['outfitting_mass']   = params['outfitting_mass_fraction'] * m_spar

        # Return total spar mass and position of spar cg
        return m_spar, z_cg


    def compute_ballast_mass_cg(self, params, unknowns):
        """Computes permanent ballast mass and center of mass
        Assumes permanent ballast is located at bottom of spar (at the keel)
        From the user/optimizer input of ballast height, computes the mass based on varying radius of the spar
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS:
        ----------
        m_ballast     : permanent ballast mass
        z_cg          : center of mass along z-axis for the ballast
        z_ballast_var : z-position of where variable ballast starts
        ballast_mass in 'unknowns' dictionary set

        """
        # Unpack variables
        R_od        = params['outer_radius']
        t_wall      = params['wall_thickness']
        h_ballast   = params['permanent_ballast_height']
        rho_ballast = params['permanent_ballast_density']
        rho_water   = params['water_density']
        z_nodes     = params['z_nodes']

        # Geometry of the spar in our coordinate system (z=0 at waterline)
        z_draft     = z_nodes[0]

        # Fixed and total ballast mass and cg
        # Assume they are bottled in cylinders a the keel of the spar- first the permanent then the fixed
        zpts      = np.linspace(z_draft, z_draft+h_ballast, NPTS)
        R_id      = np.interp(zpts, z_nodes, R_od-t_wall)
        V_perm    = np.trapz(np.pi*R_id**2, zpts)
        m_perm    = rho_ballast * V_perm
        z_cg_perm = rho_ballast * np.trapz(zpts*np.pi*R_id**2, zpts) / m_perm

        # Water ballast will start at top of fixed ballast
        z_water_start = z_draft + h_ballast

        # Find height of water ballast numerically by finding the height that integrates to the mass we want
        zpts    = np.linspace(z_water_start, z_nodes[-1], NPTS)
        R_id    = np.interp(zpts, z_nodes, R_od-t_wall)
        m_water = rho_water * cumtrapz(np.pi*R_id**2, zpts)
        unknowns['variable_ballast_interp_mass'] = np.r_[0.0, m_water] #cumtrapz has length-1
        unknowns['variable_ballast_interp_zpts'] = zpts
        
        # Save permanent ballast mass and variable height
        unknowns['ballast_mass'] = m_perm

        return m_perm, z_cg_perm

        
    def balance_cylinder(self, params, unknowns):
        """Balances the weight of the spar with buoyancy force by setting variable (water) ballast
        Once this is determined, can set the system center of gravity and determine static stability margins
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : (none)
        ----------
        system_cg class variable set
        variable_ballast_height in 'unknowns' dictionary set
        variable_ballast_mass   in 'unknowns' dictionary set
        total_mass              in 'unknowns' dictionary set
        static_stability        in 'unknowns' dictionary set
        metacentric_height      in 'unknowns' dictionary set
        """
        # Unpack variables
        R_od             = params['outer_radius']
        t_wall           = params['wall_thickness']
        z_nodes          = params['z_nodes']
        
        # Add in contributions from the spar and permanent ballast assumed to start at draft point
        m_spar   , cg_spar     = self.compute_spar_mass_cg(params, unknowns)
        m_ballast, cg_ballast  = self.compute_ballast_mass_cg(params, unknowns)
        m_outfit               = unknowns['outfitting_mass']
        m_total                = m_spar + m_ballast + m_outfit
        unknowns['total_mass'] = m_total
        unknowns['z_center_of_gravity'] = ( (m_spar+m_outfit)*cg_spar + m_ballast*cg_ballast ) / m_total
        
        # Compute volume of each section and mass of displaced water by section
        # Find the radius at the waterline so that we can compute the submerged volume as a sum of frustum sections
        r_waterline = np.interp(0.0, z_nodes, R_od)
        t_waterline = np.interp(0.0, z_nodes, t_wall)
        z_under     = np.r_[z_nodes[z_nodes < 0.0], 0.0]
        r_under     = np.r_[R_od[z_nodes < 0.0], r_waterline]
        V_under     = frustum.frustumVol_radius(r_under[:-1], r_under[1:], np.diff(z_under))
        unknowns['displaced_volume'] = V_under.sum()

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under  = frustum.frustumCG_radius(r_under[:-1], r_under[1:], np.diff(z_under))
        z_cg_under += z_under[:-1]
        # Now take weighted average of these CG points with volume
        unknowns['z_center_of_buoyancy'] = np.dot(V_under, z_cg_under) / V_under.sum()

        # 2nd moment of area for circular cross section
        # Note: Assuming Iwater here depends on "water displacement" cross-section
        # and not actual moment of inertia type of cross section (thin hoop)
        unknowns['Iwater'] = 0.25 * np.pi * r_waterline**4.0
        unknowns['Awater'] = np.pi * r_waterline**2.0

        
    def compute_surge_pitch(self, params, unknowns):
        """Balances the weight of the spar with buoyancy force by setting variable (water) ballast
        Once this is determined, can set the system center of gravity and determine static stability margins
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : (none)
        ----------
        heel_angle   in 'unknowns' dictionary set
        offset_force_ratio in 'unknowns' dictionary set
        """
        # Unpack variables
        R_od      = params['outer_radius']
        rhoWater  = params['water_density']
        rhoAir    = params['air_density']
        muWater   = params['water_viscosity']
        muAir     = params['air_viscosity']
        Dwater    = params['water_depth']
        hwave     = params['wave_height']
        Twave     = params['wave_period']
        uref      = params['wind_reference_speed']
        href      = params['wind_reference_height']
        alpha     = params['alpha']
        Cm        = params['morison_mass_coefficient']
        z_nodes   = params['z_nodes']
        
        # Spar contribution
        zpts = np.linspace(z_nodes[0], z_nodes[-1], NPTS)
        r    = np.interp(zpts, z_nodes, R_od)
        rho  = rhoWater * np.ones(zpts.shape)
        mu   = muWater  * np.ones(zpts.shape)
        rho[zpts>=0.0] = rhoAir
        mu[ zpts>=0.0] = muAir
        uvel, accel, _ = linear_waves(zpts, Dwater, hwave, Twave, rho, np.zeros(zpts.shape))
        
        # In air, set velocity to air speed instead
        accel[zpts>=0.0] = 0.0
        uvel[ zpts>=0.0] = wind_power_law(uref, href, alpha, zpts[zpts>=0])
        
        # Get forces along spar- good for water or wind with our vectorized inputs
        # By setting acceleration to zero above waterline, hydrodynamic forces will be zero and only drag will remain
        F = cylinder_forces_per_length(uvel, accel, r, rho, mu, Cm)
        unknowns['surge_force_vector'] = F
        unknowns['surge_force_points'] = zpts
         

    def check_stresses(self, params, unknowns, loading='hydro'):
        '''
        This function computes the applied axial and hoop stresses in a cylinder and compares that to 
        limits established by the API standard.  Some physcial geometry checks are also performed.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        loading  : Main loading source (default 'hydro')
        
        OUTPUTS  : (none)
        ----------
        flange_spacing_ratio   in 'unknowns' dictionary set
        web_radius_ratio       in 'unknowns' dictionary set
        flange_compactness     in 'unknowns' dictionary set
        web_compactness        in 'unknowns' dictionary set
        axial_local_unity      in 'unknowns' dictionary set
        axial_general_unity    in 'unknowns' dictionary set
        extern_local_unity     in 'unknowns' dictionary set
        extern_general_unity   in 'unknowns' dictionary set
        '''
        # Unpack variables
        R_od         = nodal2sectional(params['outer_radius'])
        t_wall       = nodal2sectional(params['wall_thickness'])
        t_web        = params['stiffener_web_thickness']
        t_flange     = params['stiffener_flange_thickness']
        h_web        = params['stiffener_web_height']
        w_flange     = params['stiffener_flange_width']
        L_stiffener  = params['stiffener_spacing']
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        yield_stress = params['yield_stress']
        z_nodes      = params['z_nodes']
        z_section    = params['z_section']

        # Create some constraints for reasonable stiffener designs for an optimizer
        unknowns['flange_spacing_ratio'] = w_flange / L_stiffener
        unknowns['web_radius_ratio']     = h_web    / R_od
        
        # Apply quick "compactness" check on stiffener geometry
        # Constraint is that these must be >= 1
        unknowns['flange_compactness'] = 0.375 * (t_flange / (0.5*w_flange)) * np.sqrt(E / yield_stress)
        unknowns['web_compactness']    = 1.0   * (t_web    / h_web         ) * np.sqrt(E / yield_stress)

        # APPLIED STRESSES (Section 11 of API Bulletin 2U)
        _, _, pressure      = linear_waves(z_section, params['water_depth'], params['wave_height'], params['wave_period'], params['water_density'])
        axial_stress        = compute_applied_axial(params, self.section_mass)
        stiffener_factor_KthL, stiffener_factor_KthG = compute_stiffener_factors(params, pressure, axial_stress)
        hoop_stress_nostiff = compute_applied_hoop(pressure, R_od, t_wall)
        hoop_stress_between = hoop_stress_nostiff * stiffener_factor_KthL
        hoop_stress_atring  = hoop_stress_nostiff * stiffener_factor_KthG
        
        # BUCKLING FAILURE STRESSES (Section 4 of API Bulletin 2U)
        elastic_axial_local_FxeL, elastic_extern_local_FreL, elastic_axial_general_FxeG, elastic_extern_general_FreG = compute_elastic_stress_limits(params, stiffener_factor_KthG, loading=loading)
        inelastic_axial_local_FxcL    = plasticityRF(elastic_axial_local_FxeL   , yield_stress)
        inelastic_axial_general_FxcG  = plasticityRF(elastic_axial_general_FxeG , yield_stress)
        inelastic_extern_local_FrcL   = plasticityRF(elastic_extern_local_FreL  , yield_stress)
        inelastic_extern_general_FrcG = plasticityRF(elastic_extern_general_FreG, yield_stress)
        
        # COMBINE AXIAL AND HOOP (EXTERNAL PRESSURE) LOADS TO FIND DESIGN LIMITS
        # (Section 6 of API Bulletin 2U)
        load_per_length_Nph = axial_stress        * t_wall
        load_per_length_Nth = hoop_stress_nostiff * t_wall
        load_ratio_k        = load_per_length_Nph / load_per_length_Nth
        def solveFthFph(Fxci, Frci, Kth):
            Fphci = np.zeros(Fxci.shape)
            Fthci = np.zeros(Fxci.shape)
            Kph   = 1.0
            c1    = (Fxci + Frci) / yield_stress - 1.0
            c2    = load_ratio_k * Kph / Kth
            for k in xrange(Fxci.size):
                try:
                    Fthci[k] = brentq(lambda x: (c2[k]*x/Fxci[k])**2 - c1[k]*(c2[k]*x/Fxci[k])*(x/Frci[k]) + (x/Frci[k])**2 - 1.0, 0, Fxci[k]+Frci[k], maxiter=20)
                except:
                    Fthci[k] = Fxci[k] + Frci[k]
                Fphci[k] = c2[k] * Fthci[k]
            return Fphci, Fthci
        
        inelastic_local_FphcL, inelastic_local_FthcL = solveFthFph(inelastic_axial_local_FxcL, inelastic_extern_local_FrcL, stiffener_factor_KthL)
        inelastic_general_FphcG, inelastic_general_FthcG = solveFthFph(inelastic_axial_general_FxcG, inelastic_extern_general_FrcG, stiffener_factor_KthG)

        # Use the inelastic limits and yield stress to compute required safety factors
        # and adjust the limits accordingly
        axial_limit_local_FaL     = inelastic_local_FphcL   / safety_factor(inelastic_local_FphcL  , yield_stress)
        extern_limit_local_FthL   = inelastic_local_FthcL   / safety_factor(inelastic_local_FthcL  , yield_stress)
        axial_limit_general_FaG   = inelastic_general_FphcG / safety_factor(inelastic_general_FphcG, yield_stress)
        extern_limit_general_FthG = inelastic_general_FthcG / safety_factor(inelastic_general_FthcG, yield_stress)

        # Compare limits to applied stresses and use this ratio as a design constraint
        # (Section 9 "Allowable Stresses" of API Bulletin 2U)
        # These values must be <= 1.0
        unknowns['axial_local_unity']      = axial_stress / axial_limit_local_FaL
        unknowns['axial_general_unity']    = axial_stress / axial_limit_general_FaG
        unknowns['external_local_unity']   = hoop_stress_between / extern_limit_local_FthL
        unknowns['external_general_unity'] = hoop_stress_between / extern_limit_general_FthG

        
    def compute_cost(self, params, unknowns):
        unknowns['ballast_cost']    = params['ballast_cost_rate'] * unknowns['ballast_mass']
        unknowns['spar_cost']       = params['tapered_col_cost_rate'] * unknowns['spar_mass']
        unknowns['outfitting_cost'] = params['outfitting_cost_rate'] * unknowns['outfitting_mass']
        self.cost                   = unknowns['ballast_cost'] + unknowns['spar_cost'] + unknowns['outfitting_cost']
        unknowns['total_cost']      = self.cost
