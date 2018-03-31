from openmdao.api import Component
import numpy as np

from commonse import gravity, eps, DirectionVector


class SubstructureBase(Component):
    def __init__(self, nFull):
        super(SubstructureBase,self).__init__()
        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')

        # From other components
        self.add_param('max_heel', val=0.0, units='deg',desc='Maximum angle of heel allowable')
        self.add_param('mooring_mass', val=0.0, units='kg', desc='Mass of mooring lines')
        self.add_param('mooring_effective_mass', val=0.0, units='kg', desc='Mass of mooring lines that weigh on structure, ignoring mass of mooring lines on sea floor')
        self.add_param('mooring_surge_restoring_force', val=0.0, units='N', desc='Restoring force from mooring system after surge motion')
        self.add_param('mooring_pitch_restoring_force', val=np.zeros((10,3)), units='N', desc='Restoring force from mooring system after pitch motion')
        self.add_param('mooring_cost', val=0.0, units='USD', desc='Cost of mooring system')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')

        self.add_param('base_column_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('base_column_cost', val=0.0, units='USD', desc='Cost of spar structure')
        self.add_param('base_freeboard', val=0.0, units='m', desc='Length of spar above water line')

        self.add_param('water_ballast_mass_vector', val=np.zeros((nFull,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_param('water_ballast_zpts_vector', val=np.zeros((nFull,)), units='m', desc='z-points of potential ballast mass')

        self.add_param('structural_mass', val=0.0, units='kg', desc='Mass of whole turbine except for mooring lines')
        self.add_param('structure_center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of whole turbine')
        self.add_param('z_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of gravity (x,y = 0,0)')
        self.add_param('total_displacement', val=0.0, units='m**3', desc='Total volume of water displaced by floating turbine (except for mooring lines)')
        self.add_param('total_force', val=np.zeros(3), units='N', desc='Net forces on turbine')
        self.add_param('total_moment', val=np.zeros(3), units='N*m', desc='Moments on whole turbine')

        
        # Outputs
        self.add_output('total_mass', val=0.0, units='kg', desc='total mass of spar and moorings')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of spar and moorings')
        self.add_output('metacentric_height', val=0.0, units='m', desc='measure of static overturning stability')
        self.add_output('buoyancy_to_gravity', val=0.0, desc='static stability margin based on position of centers of gravity and buoyancy')
        self.add_output('offset_force_ratio', val=0.0, desc='total surge force divided by restoring force')
        self.add_output('heel_moment_ratio', val=0.0, desc='total pitch moment divided by restoring moment')

        self.add_output('center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity (x,y = 0,0)')

        self.add_output('variable_ballast_mass', val=0.0, units='kg', desc='Amount of variable water ballast')
        self.add_output('variable_ballast_height_ratio', val=0.0, units='m', desc='height of water ballast to balance spar')


        
class SubstructureGeometry(Component):
    """
    OpenMDAO Component class for Semi substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self, nFull):
        super(SubstructureGeometry,self).__init__()

        # Design variables
        self.add_param('base_outer_diameter', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('auxiliary_outer_diameter', val=np.zeros((nFull,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('auxiliary_z_nodes', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('base_z_nodes', val=np.zeros((nFull,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_offset_from_shell', val=0.0, units='m',desc='fairlead offset from shell')
        self.add_param('radius_to_auxiliary_column', val=0.0, units='m',desc='Distance from base column centerpoint to ballast column centerpoint')
        self.add_param('number_of_auxiliary_columns', val=0, desc='Number of ballast columns evenly spaced around base column', pass_by_obj=True)
        self.add_param('tower_base', val=0.0, units='m', desc='tower base diameter')
        
        # Output constraints
        self.add_output('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')
        self.add_output('base_auxiliary_spacing', val=0.0, desc='Radius of base and ballast columns relative to spacing')
        self.add_output('transition_buffer', val=0.0, units='m', desc='Buffer between substructure base and tower base')

        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
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
        R_od_base       = 0.5*params['base_outer_diameter']
        R_od_ballast    = 0.5*params['auxiliary_outer_diameter']
        R_semi          = params['radius_to_auxiliary_column']
        R_tower         = 0.5*params['tower_base']
        z_nodes_ballast = params['auxiliary_z_nodes']
        z_nodes_base    = params['base_z_nodes']
        fairlead        = params['fairlead'] # depth of mooring attachment point
        fair_off        = params['fairlead_offset_from_shell']

        # Set spacing constraint
        unknowns['base_auxiliary_spacing'] = (R_od_base.max() + R_od_ballast.max()) / R_semi

        # Determine radius at mooring connection point (fairlead)
        if params['number_of_auxiliary_columns'] > 0:
            unknowns['fairlead_radius'] = R_semi + fair_off + np.interp(-fairlead, z_nodes_ballast, R_od_ballast)
        else:
            unknowns['fairlead_radius'] = fair_off + np.interp(-fairlead, z_nodes_base, R_od_base)

        # Constrain spar top to be at least greater than tower base
        unknowns['transition_buffer'] = R_od_base[-1] - R_tower



class SemiStable(SubstructureBase):
    """
    OpenMDAO Component class for Semisubmersible substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self, nFull):
        super(SemiStable,self).__init__(nFull)

        self.add_param('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        
        self.add_param('auxiliary_column_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('auxiliary_column_Awaterplane', val=0.0, units='m**2', desc='Area of waterplane cross-section')
        self.add_param('auxiliary_column_cost', val=0.0, units='USD', desc='Cost of spar structure')
        
        self.add_param('number_of_auxiliary_columns', val=0, desc='Number of ballast columns evenly spaced around base column', pass_by_obj=True)
        self.add_param('radius_to_auxiliary_column', val=0.0, units='m',desc='Distance from base column centerpoint to ballast column centerpoint')

        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):
        # TODO: Get centerlines right- in sparGeometry?
        # Determine ballast and cg of system
        self.balance_semi(params, unknowns)
        
        # Determine stability, metacentric height from waterplane profile, displaced volume
        self.compute_stability(params, unknowns)
        
        # Sum all costs
        self.compute_costs(params, unknowns)

        
    def balance_semi(self, params, unknowns):
        # Unpack variables
        m_struct     = params['structural_mass']
        m_mooringE   = params['mooring_effective_mass']
        m_mooring    = params['mooring_mass']
        
        V_system     = params['total_displacement']

        cg_struct    = params['structure_center_of_mass']
        
        m_water_data = params['water_ballast_mass_vector']
        z_water_data = params['water_ballast_zpts_vector']
        rhoWater     = params['water_density']
        npts         = z_water_data.size
        
        # SEMI TODO: Make water_ballast in base only?  columns too?  How to apportion?

        # Make sure total mass of system with variable water ballast balances against displaced volume
        # Water ballast should be buried in m_column
        m_water  = V_system*rhoWater - (m_struct + m_mooringE)
        m_system = m_struct + m_water

        # Output substructure total turbine mass
        unknowns['total_mass'] = m_struct + m_mooring

        # Find height given interpolant functions from columns
        if m_water_data[-1] < m_water:
            # Don't have enough space, so max out variable balast here and constraints will catch this
            z_end = z_water_data[-1]
            coeff = m_water / m_water_data[-1]
        elif m_water < 0.0:
            z_end = z_water_data[0]
            coeff = 0.0
        else:
            z_end = np.interp(m_water, m_water_data, z_water_data)
            coeff = 1.0
        h_water = z_end - z_water_data[0]
        unknowns['variable_ballast_mass']   = m_water
        unknowns['variable_ballast_height_ratio'] = coeff * h_water / (z_water_data[-1] - z_water_data[0])
        
        # Find cg of whole system
        # First find cg of water variable ballast by taking derivative of cumulative integral
        # 2nd-order accurate derivative of data
        dmdz    = np.gradient(m_water_data, z_water_data)
        # Put derivative on new data points for integration
        zpts    = np.linspace(z_water_data[0], z_end, npts)
        dmdz    = np.interp(zpts, z_water_data, dmdz)
        z_water = np.trapz(zpts * dmdz, zpts) / m_water
        unknowns['center_of_mass'] = (m_struct*cg_struct + m_water*np.r_[0.0, 0.0, z_water]) / m_system

        
    def compute_stability(self, params, unknowns):
        # Unpack variables
        ncolumn         = params['number_of_auxiliary_columns']
        z_cb            = params['z_center_of_buoyancy']
        z_cg            = unknowns['center_of_mass'][-1]
        V_system        = params['total_displacement']
        
        Iwater_base     = params['base_column_Iwaterplane']
        Iwater_column   = params['auxiliary_column_Iwaterplane']
        Awater_column   = params['auxiliary_column_Awaterplane']

        F_surge         = params['total_force'][0]
        M_pitch         = params['total_moment'][1]
        F_restore       = params['mooring_surge_restoring_force']
        rhoWater        = params['water_density']
        R_semi          = params['radius_to_auxiliary_column']

        F_restore_pitch = params['mooring_pitch_restoring_force']
        z_fairlead      = params['fairlead']*(-1)
        R_fairlead      = params['fairlead_radius']
        max_heel        = params['max_heel']
        
        # Compute the distance from the center of buoyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the area moment of inertia (meters^4) of the water-plane cross section about the heel axis
        # For a spar, we assume this is just the I of a circle about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html

        # Water plane area of all components with parallel axis theorem
        Iwater_system = Iwater_base
        radii = R_semi * np.cos( np.linspace(0, 2*np.pi, ncolumn+1) )
        for k in xrange(ncolumn):
            Iwater_system += Iwater_column + Awater_column*radii[k]**2
        
        # Measure static stability:
        # 1. Center of buoyancy should be above CG (difference should be positive)
        # 2. Metacentric height should be positive
        buoyancy2metacentre_BM         = Iwater_system / V_system
        unknowns['buoyancy_to_gravity'] = z_cg - z_cb
        unknowns['metacentric_height' ] = buoyancy2metacentre_BM - unknowns['buoyancy_to_gravity']
        
        F_buoy     = V_system * rhoWater * gravity
        M_restore  = unknowns['metacentric_height'] * np.sin(np.deg2rad(max_heel)) * F_buoy 

        # Convert mooring restoring force after pitch to a restoring moment
        nlines = np.count_nonzero(F_restore_pitch[:,2])
        F_restore_pitch = F_restore_pitch[:nlines,:]
        moorx  = R_fairlead * np.cos( np.linspace(0, 2*np.pi, nlines+1)[:-1] )
        moory  = R_fairlead * np.sin( np.linspace(0, 2*np.pi, nlines+1)[:-1] )
        r_moor = np.c_[moorx, moory, (z_fairlead - z_cg)*np.ones(moorx.shape)]
        Msum   = 0.0
        for k in xrange(nlines):
            dvF   = DirectionVector.fromArray(F_restore_pitch[k,:])
            dvR   = DirectionVector.fromArray(r_moor[k,:]).yawToHub(max_heel)
            M     = dvR.cross(dvF)
            Msum += M.y

        M_restore += Msum
        
        # Comput heel angle
        unknowns['heel_moment_ratio'] =  np.abs( M_pitch / M_restore )

        # Now compute offsets from the applied force
        # First use added mass (the mass of the water that must be displaced in movement)
        # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/09/411/9411273.pdf
        #mass_add_surge = rhoWater * np.pi * R_od.max() * draft
        #T_surge        = 2*np.pi*np.sqrt( (unknowns['total_mass']+mass_add_surge) / kstiff_horiz_mooring)

        # Compare restoring force from mooring to force of worst case spar displacement
        unknowns['offset_force_ratio'] = np.abs(F_surge / F_restore)

        
    def compute_costs(self, params, unknowns):
        # Unpack variables
        ncolumn    = params['number_of_auxiliary_columns']
        c_mooring  = params['mooring_cost']
        c_aux      = params['auxiliary_column_cost']
        c_base     = params['base_column_cost']
        c_pontoon  = params['pontoon_cost']

        unknowns['total_cost'] = c_mooring + ncolumn*c_aux + c_base + c_pontoon
        

