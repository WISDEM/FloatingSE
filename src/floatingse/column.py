from openmdao.api import Component, Group
import numpy as np
from scipy.integrate import cumtrapz

from commonse.utilities import nodal2sectional
import commonse.frustum as frustum
from commonse.UtilizationSupplement import shellBuckling_withStiffeners, GeometricConstraints
from commonse import gravity, eps, AeroHydroLoads, CylinderWindDrag, CylinderWaveDrag
from commonse.vertical_cylinder import CylinderDiscretization, CylinderMass
from commonse.environment import PowerWind, LinearWaves


class BulkheadMass(Component):
    """Computes bulkhead masses at each section node"""
    def __init__(self, nSection, nFull):
        super(BulkheadMass,self).__init__()
        self.bulk_full = np.zeros( nFull, dtype=np.int_)

        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')
        self.add_param('bulkhead_nodes', val=[False]*(nSection+1), desc='Nodal locations where there is a bulkhead bottom to top (length = nsection + 1)', pass_by_obj=True)
        self.add_param('bulkhead_mass_factor', val=0.0, desc='Bulkhead mass correction factor')

        self.add_output('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of spar bulkheads')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        z_full     = params['z_full'] # at section nodes
        z_param    = params['z_param']
        R_od       = 0.5*params['d_full'] # at section nodes
        twall      = params['t_full'] # at section nodes
        bulkheadTF = params['bulkhead_nodes'] # at section nodes
        
        # Map bulkhead locations to finer computation grid
        Zf,Zp = np.meshgrid(z_full, z_param)
        idx = np.argmin( np.abs(Zf-Zp), axis=1 )
        bulk_param = np.zeros( z_param.shape, dtype=np.int_)
        bulk_param[ np.where(bulkheadTF) ] = 1
        self.bulk_full = np.zeros( z_full.shape, dtype=np.int_)
        self.bulk_full[idx] = bulk_param
        
        # Compute bulkhead volume at every section node
        # Assume bulkheads are same thickness as shell wall
        V_bulk = np.pi * (R_od - twall)**2 * twall
        
        # Convert to mass with fudge factor for design features not captured in this simple approach
        m_bulk = self.bulk_full * params['bulkhead_mass_factor'] * params['rho'] * V_bulk
        
        # Zero out nodes where there is no bulkhead
        #m_bulk[np.logical_not(self.bulk_full)] = 0.0

        unknowns['bulkhead_mass'] = m_bulk

    def list_deriv_vars(self):
        inputs = ('d_full','t_full')
        outputs = ('bulkhead_mass',)
        return inputs, outputs
    
    def linearize(self, params, unknowns, resids):
        R_od   = 0.5*params['d_full'] # at section nodes
        twall  = params['t_full'] # at section nodes
        dVdR   = 2.0 * np.pi * (R_od - twall) * twall
        dVdt   = np.pi * (R_od - twall) * ( R_od - 3.0*twall )
        coeff  = params['bulkhead_mass_factor'] * params['rho']
        myones = np.ones(dVdR.shape) * self.bulk_full

        J = {}
        J['bulkhead_mass','d_full'] = coeff * np.diag(dVdR*myones) * 0.5 # 0.5 for d->r
        J['bulkhead_mass','t_full'] = coeff * np.diag(dVdt*myones)
        return J

    

class StiffenerMass(Component):
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
    """
    def __init__(self, nSection, nFull):
        super(StiffenerMass,self).__init__()

        self.add_param('d_full', val=np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', val=np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('rho', val=0.0, units='kg/m**3', desc='material density')

        self.add_param('stiffener_web_height', val=np.zeros((nSection,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((nSection,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((nSection,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')

        self.add_param('ring_mass_factor', val=0.0, desc='Stiffener ring mass correction factor')
        
        self.add_output('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of spar stiffeners')
        self.add_output('flange_spacing_ratio', val=np.zeros((nFull-1,)), desc='ratio between flange and stiffener spacing')
        self.add_output('stiffener_radius_ratio', val=np.zeros((nFull-1,)), desc='ratio between stiffener height and radius')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        R_od,_       = nodal2sectional(params['d_full']) # at section nodes
        R_od        *= 0.5
        t_wall,_     = nodal2sectional( params['t_full'] ) # at section nodes
        z_full       = params['z_full'] # at section nodes
        z_param,_    = nodal2sectional( params['z_param'] )
        z_section,_  = nodal2sectional( params['z_full'] )
        h_section    = np.diff(z_full)
        t_web        = np.interp(z_section, z_param, params['stiffener_web_thickness'])
        t_flange     = np.interp(z_section, z_param, params['stiffener_flange_thickness'])
        h_web        = np.interp(z_section, z_param, params['stiffener_web_height'])
        w_flange     = np.interp(z_section, z_param, params['stiffener_flange_width'])
        L_stiffener  = np.interp(z_section, z_param, params['stiffener_spacing'])
        
        # Outer and inner radius of web by section
        R_wo = R_od - t_wall
        R_wi = R_wo - h_web
        # Outer and inner radius of flange by section
        R_fo = R_wi
        R_fi = R_fo - t_flange

        # Material volumes by section
        V_web    = np.pi*(R_wo**2 - R_wi**2) * t_web
        V_flange = np.pi*(R_fo**2 - R_fi**2) * w_flange

        # Ring mass by volume by section 
        # Include fudge factor for design features not captured in this simple approach
        m_ring = params['ring_mass_factor']*params['rho']*(V_web + V_flange)

        # Number of stiffener rings per section (height of section divided by spacing)
        nring_per_section = h_section / L_stiffener
        unknowns['stiffener_mass'] =  nring_per_section * m_ring
        
        # Create some constraints for reasonable stiffener designs for an optimizer
        unknowns['flange_spacing_ratio']   = w_flange / (0.5*L_stiffener)
        unknowns['stiffener_radius_ratio'] = (h_web + t_flange + t_wall) / R_od

                

        
class ColumnGeometry(Component):
    """
    OpenMDAO Component class for vertical columns in substructure for floating offshore wind turbines.
    """

    def __init__(self, nSection, nFull):
        super(ColumnGeometry,self).__init__()

        # Design variables
        self.add_param('water_depth', val=0.0, units='m', desc='water depth')
        self.add_param('freeboard', val=0.0, units='m', desc='Length of spar above water line')
        self.add_param('fairlead', val=0.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('z_full_in', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_param_in', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('section_center_of_mass', val=np.zeros(nFull-1), units='m', desc='z position of center of mass of each can in the cylinder')

        # Outputs
        self.add_output('z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_output('draft', val=0.0, units='m', desc='Spar draft (length of body under water)')
        self.add_output('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Output constraints
        self.add_output('draft_depth_ratio', val=0.0, desc='Ratio of draft to water depth')
        self.add_output('fairlead_draft_ratio', val=0.0, desc='Ratio of fairlead to draft')

        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'

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
        freeboard = params['freeboard']
        fairlead  = params['fairlead'] # depth of mooring attachment point

        # With waterline at z=0, set the z-position of section nodes
        # Note sections and nodes start at bottom of spar and move up
        draft  = params['z_param_in'][-1] - freeboard
        z_full = params['z_full_in'] - draft 
        unknowns['draft']     = draft
        unknowns['z_full']    = z_full
        unknowns['z_param']   = params['z_param_in'] - draft 
        unknowns['z_section'] = params['section_center_of_mass'] - draft 

        # Create constraint output that draft is less than water depth and fairlead is less than draft
        unknowns['draft_depth_ratio'] = draft / params['water_depth']
        unknowns['fairlead_draft_ratio'] = 0.0 if z_full[0] == 0.0 else fairlead / draft



class ColumnProperties(Component):
    """
    OpenMDAO Component class for column substructure elements in floating offshore wind turbines.
    """

    def __init__(self, nFull):
        super(ColumnProperties,self).__init__()

        # Variables local to the class and not OpenMDAO
        self.section_mass = np.zeros((nFull-1,)) # Weight of spar by section
        
        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')
        
        # Material properties
        self.add_param('permanent_ballast_density', val=0.0, units='kg/m**3', desc='density of permanent ballast')

        # Inputs from Geometry
        self.add_param('z_full', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')

        # Design variables
        self.add_param('d_full', val=np.zeros((nFull,)), units='m', desc='outer diameter at each section node bottom to top (length = nsection + 1)')
        self.add_param('t_full', val=np.zeros((nFull,)), units='m', desc='shell wall thickness at each section node bottom to top (length = nsection + 1)')
        self.add_param('permanent_ballast_height', val=0.0, units='m', desc='height of permanent ballast')
        
        # Mass correction factors from simple rules here to real life
        self.add_param('shell_mass', val=np.zeros(nFull-1), units='kg', desc='mass of spar shell')
        self.add_param('stiffener_mass', val=np.zeros(nFull-1), units='kg', desc='mass of spar stiffeners')
        self.add_param('bulkhead_mass', val=np.zeros(nFull), units='kg', desc='mass of spar bulkheads')
        self.add_param('column_mass_factor', val=0.0, desc='Overall spar mass correction factor')
        self.add_param('outfitting_mass_fraction', val=0.0, desc='Mass fraction added for outfitting')
        
        # Cost rates
        self.add_param('ballast_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of ballast')
        self.add_param('tapered_col_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass of tapered columns (frustums)')
        self.add_param('outfitting_cost_rate', val=0.0, units='USD/kg', desc='Cost per unit mass for outfitting spar')

        # Outputs
        self.add_output('ballast_cost', val=0.0, units='USD', desc='cost of permanent ballast')
        self.add_output('ballast_mass', val=0.0, units='kg', desc='mass of permanent ballast')
        self.add_output('variable_ballast_interp_mass', val=np.zeros((nFull,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_output('variable_ballast_interp_zpts', val=np.zeros((nFull,)), units='m', desc='z-points of potential ballast mass')

        self.add_output('z_center_of_mass', val=0.0, units='m', desc='z-position CofG of column')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='z-position CofB of column')
        self.add_output('Awater', val=0.0, units='m**2', desc='Area of waterplace cross section')
        self.add_output('Iwater', val=0.0, units='m**4', desc='Second moment of area of waterplace cross section')
        self.add_output('displaced_volume', val=np.zeros((nFull-1,)), units='m**3', desc='Volume of water displaced by column by section')
 
        self.add_output('spar_cost', val=0.0, units='USD', desc='cost of spar structure')
        self.add_output('spar_mass', val=0.0, units='kg', desc='mass of spar structure')
        
        self.add_output('outfitting_cost', val=0.0, units='USD', desc='cost of outfitting the spar')
        self.add_output('outfitting_mass', val=0.0, units='kg', desc='cost of outfitting the spar')

        self.add_output('total_mass', val=np.zeros((nFull-1,)), units='kg', desc='total mass of column by section')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of column')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
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
        self.balance_column(params, unknowns)

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
        coeff        = params['column_mass_factor']
        z_nodes      = params['z_full']
        z_section    = params['z_section']
        m_shell      = params['shell_mass']
        m_stiffener  = params['stiffener_mass']
        m_bulkhead   = params['bulkhead_mass']
        
        m_spar = 0.0
        z_cg = 0.0
        
        # Find mass of all of the sub-components of the spar
        # Masses assumed to be focused at section centroids
        m_spar     += (m_shell + m_stiffener).sum()
        z_cg       += np.dot(m_shell+m_stiffener, z_section)

        # Masses assumed to be centered at nodes
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
        unknowns['spar_mass']       = m_spar
        unknowns['outfitting_mass'] = params['outfitting_mass_fraction'] * m_spar

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
        R_od        = 0.5*params['d_full']
        t_wall      = params['t_full']
        h_ballast   = params['permanent_ballast_height']
        rho_ballast = params['permanent_ballast_density']
        rho_water   = params['water_density']
        z_nodes     = params['z_full']

        npts = R_od.size
        
        # Geometry of the spar in our coordinate system (z=0 at waterline)
        z_draft     = z_nodes[0]

        # Fixed and total ballast mass and cg
        # Assume they are bottled in columns a the keel of the spar- first the permanent then the fixed
        zpts      = np.linspace(z_draft, z_draft+h_ballast, npts)
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
        # This step is completed in spar.py or semi.py because we must account for other substructure elements too
        zpts    = np.linspace(z_water_start, z_nodes[-1], npts)
        R_id    = np.interp(zpts, z_nodes, R_od-t_wall)
        m_water = rho_water * cumtrapz(np.pi*R_id**2, zpts)
        unknowns['variable_ballast_interp_mass'] = np.r_[0.0, m_water] #cumtrapz has length-1
        unknowns['variable_ballast_interp_zpts'] = zpts
        
        # Save permanent ballast mass and variable height
        unknowns['ballast_mass'] = m_perm

        return m_perm, z_cg_perm

        
    def balance_column(self, params, unknowns):
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
        R_od              = 0.5*params['d_full']
        t_wall            = params['t_full']
        z_nodes           = params['z_full']
        self.section_mass = np.zeros((z_nodes.size-1,))
        
        # Add in contributions from the spar and permanent ballast assumed to start at draft point
        m_spar   , cg_spar     = self.compute_spar_mass_cg(params, unknowns)
        m_ballast, cg_ballast  = self.compute_ballast_mass_cg(params, unknowns)
        m_outfit               = unknowns['outfitting_mass']
        m_total                = m_spar + m_ballast + m_outfit
        self.section_mass     += m_outfit / self.section_mass.size
        unknowns['total_mass'] = self.section_mass
        unknowns['z_center_of_mass'] = ( (m_spar+m_outfit)*cg_spar + m_ballast*cg_ballast ) / m_total

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

        
    def compute_cost(self, params, unknowns):
        unknowns['ballast_cost']    = params['ballast_cost_rate'] * unknowns['ballast_mass']
        unknowns['spar_cost']       = params['tapered_col_cost_rate'] * unknowns['spar_mass']
        unknowns['outfitting_cost'] = params['outfitting_cost_rate'] * unknowns['outfitting_mass']
        unknowns['total_cost']      = unknowns['ballast_cost'] + unknowns['spar_cost'] + unknowns['outfitting_cost']


        
class ColumnBuckling(Component):
    '''
    This function computes the applied axial and hoop stresses in a column and compares that to 
    limits established by the API standard.  Some physcial geometry checks are also performed.
    '''
    def __init__(self, nSection, nFull):
        super(ColumnBuckling,self).__init__()

        # From other modules
        self.add_param('stack_mass_in', val=eps, units='kg', desc='Weight above the cylinder column')
        self.add_param('section_mass', val=np.zeros((nFull-1,)), units='kg', desc='total mass of column by section')
        self.add_param('pressure', np.zeros(nFull), units='N/m**2', desc='Dynamic (and static)? pressure')
        
        self.add_param('d_full', np.zeros(nFull), units='m', desc='cylinder diameter at corresponding locations')
        self.add_param('t_full', np.zeros(nFull), units='m', desc='shell thickness at corresponding locations')
        self.add_param('z_full', val=np.zeros(nFull), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('z_section', val=np.zeros((nFull-1,)), units='m', desc='z-coordinates of section centers of mass (length = nsection)')
        self.add_param('z_param', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')

        self.add_param('stiffener_web_height', val=np.zeros((nSection,)), units='m', desc='height of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_web_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener web (base of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_width', val=np.zeros((nSection,)), units='m', desc='height of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_flange_thickness', val=np.zeros((nSection,)), units='m', desc='thickness of stiffener flange (top of T) within each section bottom to top (length = nsection)')
        self.add_param('stiffener_spacing', val=np.zeros((nSection,)), units='m', desc='Axial distance from one ring stiffener to another within each section bottom to top (length = nsection)')

        self.add_param('E', val=0.0, units='Pa', desc='Modulus of elasticity (Youngs) of material')
        self.add_param('nu', val=0.0, desc='poissons ratio of spar material')
        self.add_param('yield_stress', val=0.0, units='Pa', desc='yield stress of material')

        self.add_param('loading', val='hydro', desc='Loading type in API checks [hydro/radial]', pass_by_obj=True)
        
        # Output constraints
        self.add_output('flange_compactness', val=np.zeros((nFull-1,)), desc='check for flange compactness')
        self.add_output('web_compactness', val=np.zeros((nFull-1,)), desc='check for web compactness')
        self.add_output('axial_local_unity', val=np.zeros((nFull-1,)), desc='unity check for axial load - local buckling')
        self.add_output('axial_general_unity', val=np.zeros((nFull-1,)), desc='unity check for axial load - genenral instability')
        self.add_output('external_local_unity', val=np.zeros((nFull-1,)), desc='unity check for external pressure - local buckling')
        self.add_output('external_general_unity', val=np.zeros((nFull-1,)), desc='unity check for external pressure - general instability')
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5

        
    def compute_applied_axial(self, params):
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
        R_od,_         = nodal2sectional(params['d_full'])
        R_od          *= 0.5
        t_wall,_       = nodal2sectional(params['t_full'])
        section_mass   = params['section_mass']
        m_stack        = params['stack_mass_in']
        
        # Middle radius
        R = R_od - 0.5*t_wall
        # Add in weight of sections above it
        axial_load = m_stack + np.r_[0.0, np.cumsum(section_mass[:-1])]
        # Divide by shell cross sectional area to get stress
        return (gravity * axial_load / (2.0 * np.pi * R * t_wall))

    
    def solve_nonlinear(self, params, unknowns, resids):
        # Unpack variables
        R_od,_       = nodal2sectional( params['d_full'] )
        R_od        *= 0.5
        h_section    = np.diff( params['z_full'] )
        t_wall,_     = nodal2sectional( params['t_full'] )
        z_param,_    = nodal2sectional( params['z_param'] )
        z_section    = params['z_section']
        t_web        = np.interp(z_section, z_param, params['stiffener_web_thickness'])
        t_flange     = np.interp(z_section, z_param, params['stiffener_flange_thickness'])
        h_web        = np.interp(z_section, z_param, params['stiffener_web_height'])
        w_flange     = np.interp(z_section, z_param, params['stiffener_flange_width'])
        L_stiffener  = np.interp(z_section, z_param, params['stiffener_spacing'])
        E            = params['E'] # Young's modulus
        nu           = params['nu'] # Poisson ratio
        sigma_y      = params['yield_stress']
        loading      = params['loading']
        pressure,_   = nodal2sectional( params['pressure'] )

        # Compute applied axial stress simply, like API guidelines (as opposed to running frame3dd)
        sigma_ax = self.compute_applied_axial(params)
        (flange_compactness, web_compactness, axial_local_unity, axial_general_unity,
         external_local_unity, external_general_unity) = shellBuckling_withStiffeners(pressure, sigma_ax, R_od, t_wall, h_section,
                                                                                      h_web, t_web, w_flange, t_flange,
                                                                                      L_stiffener, E, nu, sigma_y, loading)
        unknowns['flange_compactness']     = flange_compactness
        unknowns['web_compactness']        = web_compactness
        unknowns['axial_local_unity']      = axial_local_unity
        unknowns['axial_general_unity']    = axial_general_unity
        unknowns['external_local_unity']   = external_local_unity
        unknowns['external_general_unity'] = external_general_unity


class Column(Group):
    def __init__(self, nSection, nFull):
        super(Column,self).__init__()
        
        self.add('cyl_geom', CylinderDiscretization(nSection+1, nFull), promotes=['section_height','diameter','wall_thickness',
                                                                                  'd_full','t_full'])
        
        self.add('cyl_mass', CylinderMass(nFull), promotes=['d_full','t_full','material_density'])

        self.add('col_geom', ColumnGeometry(nSection, nFull), promotes=['water_depth','freeboard','fairlead','z_full','z_param','z_section',
                                                                        'draft','draft_depth_ratio','fairlead_draft_ratio'])

        self.add('gc', GeometricConstraints(nSection+1, diamFlag=True), promotes=['min_taper','min_d_to_t','manufacturability','weldability'])

        self.add('bulk', BulkheadMass(nSection, nFull), promotes=['z_full','z_param','d_full','t_full','rho',
                                                                  'bulkhead_mass_factor','bulkhead_nodes','bulkhead_mass'])

        self.add('stiff', StiffenerMass(nSection,nFull), promotes=['d_full','t_full','z_full','z_param','rho','ring_mass_factor',
                                                                   'stiffener_mass','stiffener_web_height',
                                                                   'stiffener_web_thickness','stiffener_flange_width',
                                                                   'stiffener_flange_thickness','stiffener_spacing',
                                                                   'flange_spacing_ratio','stiffener_radius_ratio'])

        self.add('col', ColumnProperties(nFull), promotes=['water_density','d_full','t_full','z_full','z_section',
                                                           'permanent_ballast_density','permanent_ballast_height',
                                                           'bulkhead_mass','stiffener_mass','column_mass_factor','outfitting_mass_fraction',
                                                           'ballast_cost_rate','tapered_col_cost_rate','outfitting_cost_rate',
                                                           'variable_ballast_interp_mass','variable_ballast_interp_zpts',
                                                           'z_center_of_mass','z_center_of_buoyancy','Awater','Iwater',
                                                           'displaced_volume','total_mass','total_cost'])

        self.add('wind', PowerWind(nFull), promotes=['Uref','zref','shearExp','z0'])
        self.add('wave', LinearWaves(nFull), promotes=['Uc','hmax','T'])
        self.add('windLoads', CylinderWindDrag(nFull), promotes=['cd_usr','beta'])
        self.add('waveLoads', CylinderWaveDrag(nFull), promotes=['cm','cd_usr'])
        self.add('distLoads', AeroHydroLoads(nFull), promotes=['Px','Py','Pz','qdyn','yaw'])

        self.add('buck', ColumnBuckling(nSection, nFull), promotes=['d_full','t_full','z_full','z_section','z_param','E','nu','yield_stress',
                                                                    'loading','stack_mass_in','stiffener_web_height',
                                                                    'stiffener_web_thickness','stiffener_flange_width',
                                                                    'stiffener_flange_thickness','stiffener_spacing',
                                                                    'flange_compactness','web_compactness',
                                                                    'axial_local_unity','axial_general_unity',
                                                                    'external_local_unity','external_general_unity'])
        
        self.connect('diameter', 'gc.d')
        self.connect('wall_thickness', 'gc.t')
        self.connect('cyl_geom.z_param', 'col_geom.z_param_in')
        self.connect('cyl_geom.z_full', ['cyl_mass.z_full','col_geom.z_full_in'])
        
        self.connect('cyl_mass.section_center_of_mass', 'col_geom.section_center_of_mass')
        
        self.connect('cyl_mass.mass', 'col.shell_mass')
        self.connect('material_density','rho')
        
        self.connect('total_mass', 'buck.section_mass')

        self.connect('water_depth','wave.z_floor')
        self.connect('z_full', ['wind.z', 'wave.z', 'windLoads.z','waveLoads.z','distLoads.z'])
        self.connect('d_full', ['windLoads.d','waveLoads.d'])
        self.connect('beta','waveLoads.beta')
        self.connect('z0', 'wave.z_surface')

        self.connect('wind.U', 'windLoads.U')

        self.connect('water_density',['wave.rho','waveLoads.rho'])
        self.connect('wave.U', 'waveLoads.U')
        self.connect('wave.A', 'waveLoads.A')
        self.connect('wave.p', 'waveLoads.p')
        
        # connections to distLoads1
        self.connect('windLoads.windLoads:Px', 'distLoads.windLoads:Px')
        self.connect('windLoads.windLoads:Py', 'distLoads.windLoads:Py')
        self.connect('windLoads.windLoads:Pz', 'distLoads.windLoads:Pz')
        self.connect('windLoads.windLoads:qdyn', 'distLoads.windLoads:qdyn')
        self.connect('windLoads.windLoads:beta', 'distLoads.windLoads:beta')
        self.connect('windLoads.windLoads:z', 'distLoads.windLoads:z')
        self.connect('windLoads.windLoads:d', 'distLoads.windLoads:d')
        
        self.connect('waveLoads.waveLoads:Px', 'distLoads.waveLoads:Px')
        self.connect('waveLoads.waveLoads:Py', 'distLoads.waveLoads:Py')
        self.connect('waveLoads.waveLoads:Pz', 'distLoads.waveLoads:Pz')
        self.connect('waveLoads.waveLoads:pt', 'distLoads.waveLoads:qdyn')
        self.connect('waveLoads.waveLoads:beta', 'distLoads.waveLoads:beta')
        self.connect('waveLoads.waveLoads:z', 'distLoads.waveLoads:z')
        self.connect('waveLoads.waveLoads:d', 'distLoads.waveLoads:d')

        self.connect('qdyn', 'buck.pressure')
