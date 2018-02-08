from openmdao.api import Component
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import cumtrapz

from commonse import gravity, eps
from floatingInstance import nodal2sectional
from commonse.WindWaveDrag import cylinderDrag
import commonse.frustum as frustum
from commonse.UtilizationSupplement import shellBuckling_withStiffeners

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
    R_od         = 0.5*params['outer_diameter'] # at section nodes
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
    R_od         = 0.5*params['outer_diameter'] # at section nodes
    h_section    = params['section_height']
    rho          = params['material_density']
    coeff        = params['shell_mass_factor']

    # Same with wall thicknesses
    Tbot = twall[:-1]
    Ttop = twall[1:]

    # Radius (to middle of shell) at base and tops of all frustum sections
    Rbot = R_od[:-1]
    Rtop = R_od[1:]

    # Shell volume for each section determined by allowing for linear variation in R & T in each section.
    V_shell = frustum.frustumShellVolume(Rbot, Rtop, Tbot, Ttop, h_section)

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
    R_od         = 0.5*params['outer_diameter'] # at section nodes
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
    R_od    = 0.5*nodal2sectional(params['outer_diameter'])
    t_wall  = nodal2sectional(params['wall_thickness'])
    m_stack = params['stack_mass_in']
    R       = R_od - 0.5*t_wall
    
    # Add in weight of sections above it
    axial_load    = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
    # Divide by shell cross sectional area to get stress
    return (gravity * axial_load / (2.0 * np.pi * R * t_wall))


class CylinderGeometry(Component):
    """
    OpenMDAO Component class for vertical cylinders in substructure for floating offshore wind turbines.
    """

    def __init__(self, nSection):
        super(CylinderGeometry,self).__init__()

        # Design variables
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('freeboard', val=0.0, units='m', desc='Length of spar above water line')
        self.add_param('fairlead', val=0.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('section_height', val=np.zeros((nSection,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer diameter at each section node bottom to top (length = nsection + 1)')
        self.add_param('wall_thickness', val=np.zeros((nSection+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('fairlead_offset_from_shell', val=0.5, units='m',desc='fairlead offset from shell')

        # Outputs
        self.add_output('draft', val=0.0, units='m', desc='Spar draft (length of body under water)')
        self.add_output('z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('z_section', val=np.zeros((nSection,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')
        self.add_output('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')

        # Output constraints
        self.add_output('draft_depth_ratio', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('fairlead_draft_ratio', val=0.0, desc='Ratio of fairlead to draft')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

    def solve_nonlinear(self, params, unknowns, resids):
        """Sets nodal points and sectional centers of mass in z-coordinate system with z=0 at the waterline.
        Nodal points are the beginning and end points of each section.
        Nodes and sections start at bottom and move upwards.
        
        INPUTS:
        ----------
        params   : dictionary of input parameters
        unknowns : dictionary of output parameters
        
        OUTPUTS  : none (all unknown dictionary values set)
        """
        # Unpack variables
        D_water   = params['water_depth']
        R_od      = 0.5*params['outer_diameter']
        t_wall    = params['wall_thickness']
        h_section = params['section_height']
        freeboard = params['freeboard']
        fairlead  = params['fairlead'] # depth of mooring attachment point
        fair_off  = params['fairlead_offset_from_shell']

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of spar and move up
        z_nodes             = np.flipud( freeboard - np.r_[0.0, np.cumsum(np.flipud(h_section))] )
        unknowns['draft']   = np.abs(z_nodes[0])
        unknowns['z_nodes'] = z_nodes

        # Determine radius at mooring connection point (fairlead)
        unknowns['fairlead_radius'] = fair_off + np.interp(-fairlead, z_nodes, R_od)
        
        # With waterline at z=0, set the z-position of section centroids
        cm_section = frustum.frustumShellCG(R_od[:-1], R_od[1:], t_wall[:-1], t_wall[1:], h_section)
        unknowns['z_section'] = z_nodes[:-1] + cm_section

        # Create constraint output that draft is less than water depth and fairlead is less than draft
        unknowns['draft_depth_ratio'] = unknowns['draft'] / D_water
        unknowns['fairlead_draft_ratio'] = 0.0 if z_nodes[0] == 0.0 else fairlead / unknowns['draft'] 





class Cylinder(Component):
    """
    OpenMDAO Component class for cylinder substructure elements in floating offshore wind turbines.
    """

    def __init__(self, nSection, nIntPts):
        super(Cylinder,self).__init__()

        # Variables local to the class and not OpenMDAO
        self.section_mass   = None # Weight of spar by section
        self.z_permanent_ballast = None
        self.npts = nIntPts
        
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
        self.add_param('stack_mass_in', val=eps, units='kg', desc='Weight above the cylinder column')
        
        # Material properties
        self.add_param('material_density', val=7850., units='kg/m**3', desc='density of material')
        self.add_param('E', val=200e9, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('nu', val=0.3, desc='poissons ratio of spar material')
        self.add_param('yield_stress', val=345e6, units='Pa', desc='yield stress of material')
        self.add_param('permanent_ballast_density', val=4492.0, units='kg/m**3', desc='density of permanent ballast')

        # Inputs from SparGeometry
        self.add_param('z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_section', val=np.zeros((nSection,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Design variables
        self.add_param('section_height', val=np.zeros((nSection,)), units='m', desc='length (height) or each section in the spar bottom to top (length = nsection)')
        self.add_param('outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer diameter at each section node bottom to top (length = nsection + 1)')
        self.add_param('wall_thickness', val=np.zeros((nSection+1,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('stiffener_web_height', val=np.zeros((nSection,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((nSection,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((nSection,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')
        self.add_param('bulkhead_nodes', val=[True]*(nSection+1), desc='Nodal locations where there is a bulkhead bottom to top (length = nsection + 1)', pass_by_obj=True)
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
        self.add_output('variable_ballast_interp_mass', val=np.zeros((self.npts,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_output('variable_ballast_interp_zpts', val=np.zeros((self.npts,)), units='m', desc='z-points of potential ballast mass')

        self.add_output('z_center_of_gravity', val=0.0, units='m', desc='z-position CofG of cylinder')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position CofB of cylinder')
        self.add_output('Awater', val=0.0, units='m**2', desc='Area of waterplace cross section')
        self.add_output('Iwater', val=0.0, units='m**4', desc='Second moment of area of waterplace cross section')
        self.add_output('displaced_volume', val=np.zeros((nSection,)), units='m**3', desc='Volume of water displaced by cylinder by section')
 
        self.add_output('spar_cost', val=0.0, units='USD', desc='cost of spar structure')
        self.add_output('spar_mass', val=0.0, units='kg', desc='mass of spar structure')
        self.add_output('shell_mass', val=0.0, units='kg', desc='mass of spar shell')
        self.add_output('stiffener_mass', val=0.0, units='kg', desc='mass of spar stiffeners')
        self.add_output('bulkhead_mass', val=0.0, units='kg', desc='mass of spar bulkheads')

        self.add_output('surge_force_vector', val=np.zeros((self.npts,)), units='N', desc='Force in surge direction')
        self.add_output('surge_force_points', val=np.zeros((self.npts,)), units='m', desc='z-points for surge force vector')
        
        self.add_output('outfitting_cost', val=0.0, units='USD', desc='cost of outfitting the spar')
        self.add_output('outfitting_mass', val=0.0, units='kg', desc='cost of outfitting the spar')

        self.add_output('total_mass', val=np.zeros((nSection,)), units='kg', desc='total mass of cylinder by section')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of cylinder')
        
        # Output constraints
        self.add_output('flange_spacing_ratio', val=np.zeros((nSection,)), desc='ratio between flange and stiffener spacing')
        self.add_output('web_radius_ratio', val=np.zeros((nSection,)), desc='ratio between web height and radius')
        self.add_output('flange_compactness', val=np.zeros((nSection,)), desc='check for flange compactness')
        self.add_output('web_compactness', val=np.zeros((nSection,)), desc='check for web compactness')
        self.add_output('axial_local_unity', val=np.zeros((nSection,)), desc='unity check for axial load - local buckling')
        self.add_output('axial_general_unity', val=np.zeros((nSection,)), desc='unity check for axial load - genenral instability')
        self.add_output('external_local_unity', val=np.zeros((nSection,)), desc='unity check for external pressure - local buckling')
        self.add_output('external_general_unity', val=np.zeros((nSection,)), desc='unity check for external pressure - general instability')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
        
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
        R_od        = 0.5*params['outer_diameter']
        t_wall      = params['wall_thickness']
        h_ballast   = params['permanent_ballast_height']
        rho_ballast = params['permanent_ballast_density']
        rho_water   = params['water_density']
        z_nodes     = params['z_nodes']

        # Geometry of the spar in our coordinate system (z=0 at waterline)
        z_draft     = z_nodes[0]

        # Fixed and total ballast mass and cg
        # Assume they are bottled in cylinders a the keel of the spar- first the permanent then the fixed
        zpts      = np.linspace(z_draft, z_draft+h_ballast, self.npts)
        R_id      = np.interp(zpts, z_nodes, R_od-t_wall)
        V_perm    = np.pi * np.trapz(R_id**2, zpts)
        m_perm    = rho_ballast * V_perm
        z_cg_perm = rho_ballast * np.pi * np.trapz(zpts*R_id**2, zpts) / m_perm
        for k in xrange(z_nodes.size-1):
            ind = np.logical_and(zpts>=z_nodes[k], zpts<=z_nodes[k+1]) 
            self.section_mass[k] += rho_ballast * np.pi * np.trapz(R_id[ind]**2, zpts[ind])
        
        # Water ballast will start at top of fixed ballast
        z_water_start = z_draft + h_ballast

        # Find height of water ballast numerically by finding the height that integrates to the mass we want
        zpts    = np.linspace(z_water_start, z_nodes[-1], self.npts)
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
        R_od              = 0.5*params['outer_diameter']
        t_wall            = params['wall_thickness']
        z_nodes           = params['z_nodes']
        self.section_mass = np.zeros((z_nodes.size-1,))
        
        # Add in contributions from the spar and permanent ballast assumed to start at draft point
        m_spar   , cg_spar     = self.compute_spar_mass_cg(params, unknowns)
        m_ballast, cg_ballast  = self.compute_ballast_mass_cg(params, unknowns)
        m_outfit               = unknowns['outfitting_mass']
        m_total                = m_spar + m_ballast + m_outfit
        self.section_mass     += m_outfit / self.section_mass.size
        unknowns['total_mass'] = self.section_mass
        unknowns['z_center_of_gravity'] = ( (m_spar+m_outfit)*cg_spar + m_ballast*cg_ballast ) / m_total
        
        # Compute volume of each section and mass of displaced water by section
        # Find the radius at the waterline so that we can compute the submerged volume as a sum of frustum sections
        if z_nodes[-1] > 0.0:
            r_waterline = np.interp(0.0, z_nodes, R_od)
            z_under     = np.r_[z_nodes[z_nodes < 0.0], 0.0]
            r_under     = np.r_[R_od[z_nodes < 0.0], r_waterline]
        else:
            r_waterline = R_od[-1]
            r_under     = R_od
            z_under     = z_nodes
            
        V_under     = frustum.frustumVol(r_under[:-1], r_under[1:], np.diff(z_under))
        # 0-pad so that it has the length of sections
        add0        = np.maximum(0, self.section_mass.size-V_under.size)
        V_under     = np.r_[V_under, np.zeros((add0,))]
        unknowns['displaced_volume'] = V_under

        # Compute Center of Buoyancy in z-coordinates (0=waterline)
        # First get z-coordinates of CG of all frustums
        z_cg_under  = frustum.frustumCG(r_under[:-1], r_under[1:], np.diff(z_under))
        z_cg_under += z_under[:-1]
        z_cg_under  = np.r_[z_cg_under, np.zeros((add0,))]
        # Now take weighted average of these CG points with volume
        V_under += eps
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
        R_od      = 0.5*params['outer_diameter']
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
        zpts = np.linspace(z_nodes[0], z_nodes[-1], self.npts)
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
        R_od         = 0.5*nodal2sectional(params['outer_diameter'])
        h_section    = params['section_height']
        t_wall       = nodal2sectional(params['wall_thickness'])
        t_web        = params['stiffener_web_thickness']
        t_flange     = params['stiffener_flange_thickness']
        h_web        = params['stiffener_web_height']
        w_flange     = params['stiffener_flange_width']
        L_stiffener  = params['stiffener_spacing']
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        yield_stress = params['yield_stress']
        z_section    = params['z_section']

        # Create some constraints for reasonable stiffener designs for an optimizer
        unknowns['flange_spacing_ratio'] = w_flange / L_stiffener
        unknowns['web_radius_ratio']     = h_web    / R_od

        _, _, pressure      = linear_waves(z_section, params['water_depth'], params['wave_height'], params['wave_period'], params['water_density'])
        axial_stress        = compute_applied_axial(params, self.section_mass)

        (flange_compactness, web_compactness, axial_local_unity, axial_general_unity,
         external_local_unity, external_general_unity) = shellBuckling_withStiffeners(pressure, axial_stress, R_od, t_wall, h_section,
                                                                                      h_web, t_web, w_flange, t_flange,
                                                                                      L_stiffener, E, nu, yield_stress, loading)
        unknowns['flange_compactness']     = flange_compactness
        unknowns['web_compactness']        = web_compactness
        unknowns['axial_local_unity']      = axial_local_unity
        unknowns['axial_general_unity']    = axial_general_unity
        unknowns['external_local_unity']   = external_local_unity
        unknowns['external_general_unity'] = external_general_unity

        
    def compute_cost(self, params, unknowns):
        unknowns['ballast_cost']    = params['ballast_cost_rate'] * unknowns['ballast_mass']
        unknowns['spar_cost']       = params['tapered_col_cost_rate'] * unknowns['spar_mass']
        unknowns['outfitting_cost'] = params['outfitting_cost_rate'] * unknowns['outfitting_mass']
        self.cost                   = unknowns['ballast_cost'] + unknowns['spar_cost'] + unknowns['outfitting_cost']
        unknowns['total_cost']      = self.cost
