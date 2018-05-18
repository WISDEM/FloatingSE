from openmdao.api import Component
import numpy as np
from scipy.integrate import cumtrapz

from commonse import gravity, eps, DirectionVector
from commonse.utilities import assembleI, unassembleI
from map_mooring import NLINES_MAX
        
class SubstructureGeometry(Component):
    """
    OpenMDAO Component class for substructure geometry for floating offshore wind turbines.
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
        self.add_param('number_of_auxiliary_columns', val=0, desc='Number of ballast columns evenly spaced around base column')
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
        if int(params['number_of_auxiliary_columns']) > 0:
            unknowns['fairlead_radius'] = R_semi + fair_off + np.interp(-fairlead, z_nodes_ballast, R_od_ballast)
        else:
            unknowns['fairlead_radius'] = fair_off + np.interp(-fairlead, z_nodes_base, R_od_base)

        # Constrain spar top to be at least greater than tower base
        unknowns['transition_buffer'] = R_od_base[-1] - R_tower




class Substructure(Component):
    def __init__(self, nFull):
        super(Substructure,self).__init__()
        # Environment
        self.add_param('water_density', val=0.0, units='kg/m**3', desc='density of water')
        self.add_param('wave_period', 0.0, units='s', desc='period of maximum wave height')

        # From other components
        self.add_param('max_heel', val=0.0, units='deg',desc='Maximum angle of heel allowable')
        self.add_param('mooring_mass', val=0.0, units='kg', desc='Mass of mooring lines')
        self.add_param('mooring_neutral_load', val=np.zeros((NLINES_MAX,3)), units='N', desc='z-force of mooring lines on structure')
        self.add_param('mooring_surge_restoring_force', val=0.0, units='N', desc='Restoring force from mooring system after surge motion')
        self.add_param('mooring_pitch_restoring_force', val=np.zeros((NLINES_MAX,3)), units='N', desc='Restoring force from mooring system after pitch motion')
        self.add_param('mooring_cost', val=0.0, units='USD', desc='Cost of mooring system')
        self.add_param('mooring_stiffness', val=np.zeros((6,6)), units='N/m', desc='Linearized stiffness matrix of mooring system at neutral (no offset) conditions.')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')
        
        self.add_param('number_of_auxiliary_columns', val=0, desc='Number of ballast columns evenly spaced around base column')
        self.add_param('radius_to_auxiliary_column', val=0.0, units='m',desc='Distance from base column centerpoint to ballast column centerpoint')

        self.add_param('base_column_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('base_column_Awaterplane', val=0.0, units='m**2', desc='Area of waterplane cross-section')
        self.add_param('base_column_cost', val=0.0, units='USD', desc='Cost of spar structure')
        self.add_param('base_column_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of base column by section')
        self.add_param('base_freeboard', val=0.0, units='m', desc='Length of spar above water line')
        self.add_param('base_column_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('base_column_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('base_column_moments_of_inertia', val=np.zeros(6), units='kg*m**2', desc='mass moment of inertia of column about base [xx yy zz xy xz yz]')
        self.add_param('base_column_added_mass', val=np.zeros(6), units='kg', desc='Diagonal of added mass matrix- masses are first 3 entries, moments are last 3')

        self.add_param('auxiliary_column_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('auxiliary_column_Awaterplane', val=0.0, units='m**2', desc='Area of waterplane cross-section')
        self.add_param('auxiliary_column_cost', val=0.0, units='USD', desc='Cost of spar structure')
        self.add_param('auxiliary_column_mass', val=np.zeros((nFull-1,)), units='kg', desc='mass of ballast column by section')
        self.add_param('auxiliary_column_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of column buoyancy force')
        self.add_param('auxiliary_column_center_of_mass', val=0.0, units='m', desc='z-position of center of column mass')
        self.add_param('auxiliary_column_moments_of_inertia', val=np.zeros(6), units='kg*m**2', desc='mass moment of inertia of column about base [xx yy zz xy xz yz]')
        self.add_param('auxiliary_column_added_mass', val=np.zeros(6), units='kg', desc='Diagonal of added mass matrix- masses are first 3 entries, moments are last 3')
        
        self.add_param('water_ballast_zpts_vector', val=np.zeros((nFull,)), units='m', desc='z-points of potential ballast mass')
        self.add_param('water_ballast_radius_vector', val=np.zeros((nFull,)), units='m', desc='Inner radius of potential ballast mass')

        self.add_param('structural_mass', val=0.0, units='kg', desc='Mass of whole turbine except for mooring lines')
        self.add_param('structure_center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity of whole turbine')
        self.add_param('structural_frequencies', val=np.zeros(6), units='Hz', desc='')
        self.add_param('z_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of gravity (x,y = 0,0)')
        self.add_param('total_displacement', val=0.0, units='m**3', desc='Total volume of water displaced by floating turbine (except for mooring lines)')
        self.add_param('total_force', val=np.zeros(3), units='N', desc='Net forces on turbine')
        self.add_param('total_moment', val=np.zeros(3), units='N*m', desc='Moments on whole turbine')

        self.add_param('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')

        
        # Outputs
        self.add_output('total_mass', val=0.0, units='kg', desc='total mass of spar and moorings')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of spar and moorings')
        self.add_output('metacentric_height', val=0.0, units='m', desc='measure of static overturning stability')
        self.add_output('buoyancy_to_gravity', val=0.0, desc='static stability margin based on position of centers of gravity and buoyancy')
        self.add_output('offset_force_ratio', val=0.0, desc='total surge force divided by restoring force')
        self.add_output('heel_moment_ratio', val=0.0, desc='total pitch moment divided by restoring moment')

        self.add_output('center_of_mass', val=np.zeros(3), units='m', desc='xyz-position of center of gravity (x,y = 0,0)')

        self.add_output('variable_ballast_mass', val=0.0, units='kg', desc='Amount of variable water ballast')
        self.add_output('variable_ballast_center_of_mass', val=0.0, units='m', desc='Center of mass for variable ballast')
        self.add_output('variable_ballast_moments_of_inertia', val=np.zeros(6), units='kg*m**2', desc='mass moment of inertia of variable ballast [xx yy zz xy xz yz]')
        self.add_output('variable_ballast_height_ratio', val=0.0, units='m', desc='height of water ballast to balance spar')

        self.add_output('mass_matrix', val=np.zeros(6), units='kg', desc='Summary mass matrix of structure (minus pontoons)')
        self.add_output('added_mass_matrix', val=np.zeros(6), units='kg', desc='Summary hydrodynamic added mass matrix of structure (minus pontoons)')
        self.add_output('hydrostatic_stiffness', val=np.zeros(6), units='N/m', desc='Summary hydrostatic stiffness of structure')
        self.add_output('rigid_body_periods', val=np.zeros(6), units='s', desc='Natural periods of oscillation in 6 DOF')
        self.add_output('period_margin', val=np.zeros(6), desc='Margin between natural periods and wave periods')
        self.add_output('modal_margin', val=np.zeros(6), desc='Margin between structural modes and wave periods')
        
        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
        self.deriv_options['check_form'] = 'central'
        self.deriv_options['step_calc'] = 'relative'
        self.deriv_options['step_size'] = 1e-5
        
    def solve_nonlinear(self, params, unknowns, resids):
        # TODO: Get centerlines right- in sparGeometry?
        # Determine ballast and cg of system
        self.balance(params, unknowns)
        
        # Determine stability, metacentric height from waterplane profile, displaced volume
        self.compute_stability(params, unknowns)

        # Compute natural periods of osciallation
        self.compute_rigid_body_periods(params, unknowns)
        
        # Check margins of natural and eigenfrequencies against waves
        self.check_frequency_margins(params, unknowns)
        
        # Sum all costs
        self.compute_costs(params, unknowns)

        
    def balance(self, params, unknowns):
        # Unpack variables
        m_struct     = params['structural_mass']
        Fz_mooring   = np.sum( params['mooring_neutral_load'][:,-1] )
        m_mooring    = params['mooring_mass']
        
        V_system     = params['total_displacement']

        cg_struct    = params['structure_center_of_mass']
        
        z_water_data = params['water_ballast_zpts_vector']
        r_water_data = params['water_ballast_radius_vector']
        rhoWater     = params['water_density']
        
        # SEMI TODO: Make water_ballast in base only?  columns too?  How to apportion?

        # Make sure total mass of system with variable water ballast balances against displaced volume
        # Water ballast should be buried in m_column
        m_water  = V_system*rhoWater - (m_struct + Fz_mooring/gravity)
        m_system = m_struct + m_water

        # Output substructure total turbine mass
        unknowns['total_mass'] = m_struct + m_mooring

        # Find height given interpolant functions from columns
        m_water_data = rhoWater * np.pi * cumtrapz(r_water_data**2, z_water_data)
        m_water_data = np.r_[0.0, m_water_data] #cumtrapz has length-1
        
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
        # First find cg of water variable ballast by finding midpoint of mass sum
        z_cg  = np.interp(0.5*coeff*m_water, m_water_data, z_water_data)
        unknowns['center_of_mass'] = (m_struct*cg_struct + m_water*np.r_[0.0, 0.0, z_cg]) / m_system
        unknowns['variable_ballast_center_of_mass'] = z_cg

        # Integrate for moment of inertia of variable ballast
        npts  = 1e2
        z_int = np.linspace(z_water_data[0], z_end, npts)
        r_int = np.interp(z_int, z_water_data, r_water_data)
        Izz   = 0.5 * rhoWater * np.pi * np.trapz(r_int**4, z_int)
        Ixx   = rhoWater * np.pi * np.trapz(0.25*r_int**4 + r_int**2*(z_int-z_cg)**2, z_int)
        unknowns['variable_ballast_moments_of_inertia'] = np.array([Ixx, Ixx, Izz, 0.0, 0.0, 0.0])
            
        
    def compute_stability(self, params, unknowns):
        # Unpack variables
        ncolumn         = int(params['number_of_auxiliary_columns'])
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


    def compute_rigid_body_periods(self, params, unknowns):
        # Unpack variables
        ncolumn         = int(params['number_of_auxiliary_columns'])
        R_semi          = params['radius_to_auxiliary_column']
        
        m_base          = np.sum(params['base_column_mass'])
        m_column        = np.sum(params['auxiliary_column_mass'])
        m_struct        = params['structural_mass']
        m_water         = np.maximum(0.0, unknowns['variable_ballast_mass'])
        m_a_base        = params['base_column_added_mass']
        m_a_column      = params['auxiliary_column_added_mass']
        
        rhoWater        = params['water_density']
        V_system        = params['total_displacement']
        h_metacenter    = unknowns['metacentric_height']

        Awater_base     = params['base_column_Awaterplane']
        Awater_column   = params['auxiliary_column_Awaterplane']
        I_base          = params['base_column_moments_of_inertia']
        I_column        = params['auxiliary_column_moments_of_inertia']
        I_water         = unknowns['variable_ballast_moments_of_inertia']

        z_cg_base       = params['base_column_center_of_mass']
        z_cb_base       = params['base_column_center_of_buoyancy']
        z_cg_column     = params['auxiliary_column_center_of_mass']
        z_cb_column     = params['auxiliary_column_center_of_buoyancy']
        z_cg_water      = unknowns['variable_ballast_center_of_mass']
        r_cg            = unknowns['center_of_mass']
        
        K_moor          = np.diag( params['mooring_stiffness'] )

        
        # Number of degrees of freedom
        nDOF = 6
        
        # Compute elements on mass matrix diagonal
        M_mat = np.zeros((nDOF,))
        # Surge, sway, heave just use normal inertia
        M_mat[:3] = m_struct + m_water
        # Add in moments of inertia of primary column
        I_total = assembleI( np.zeros(6) )
        I_base  = assembleI( I_base )
        R       = np.array([0.0, 0.0, z_cg_base]) - r_cg
        I_total += I_base + m_base*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        # Add up moments of intertia of other columns
        radii_x   = R_semi * np.cos( np.linspace(0, 2*np.pi, ncolumn+1) )
        radii_y   = R_semi * np.sin( np.linspace(0, 2*np.pi, ncolumn+1) )
        I_column  = assembleI( I_column )
        for k in xrange(ncolumn):
            R        = np.array([radii_x[k], radii_y[k], z_cg_column]) - r_cg
            I_total += I_column + m_column*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        # Add in variable ballast
        R         = np.array([0.0, 0.0, z_cg_water]) - r_cg
        I_total  += assembleI(I_water) + m_water*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        # Stuff moments of inertia into mass matrix
        M_mat[3:] = unassembleI( I_total )[:3]
        unknowns['mass_matrix'] = M_mat
        
        # Add up all added mass entries in a similar way
        A_mat = np.zeros((nDOF,))
        # Surge, sway, heave just use normal inertia
        A_mat[:3] = m_a_base[:3] + ncolumn*m_a_column[:3]
        # Add up moments of inertia, move added mass moments from CofB to CofG
        I_base    = assembleI( np.r_[m_a_base[3:]  , np.zeros(3)] )
        R         = np.array([0.0, 0.0, z_cb_base]) - r_cg
        I_total   = I_base + m_a_base[0]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        # Add up added moments of intertia of all columns for other entries
        I_column  = assembleI( np.r_[m_a_column[3:], np.zeros(3)] )
        for k in xrange(ncolumn):
            R        = np.array([radii_x[k], radii_y[k], z_cb_column]) - r_cg
            I_total += I_column + m_a_column[0]*(np.dot(R, R)*np.eye(3) - np.outer(R, R))
        A_mat[3:] = unassembleI( I_total )[:3]
        unknowns['added_mass_matrix'] = A_mat
        
        # Hydrostatic stiffness has contributions in heave (K33) and roll/pitch (K44/55)
        # See DNV-RP-H103: Modeling and Analyis of Marine Operations
        K_hydro = np.zeros((nDOF,))
        K_hydro[2]   = rhoWater * gravity * (Awater_base + ncolumn*Awater_column)
        K_hydro[3:5] = rhoWater * gravity * V_system * h_metacenter
        unknowns['hydrostatic_stiffness'] = K_hydro

        # Now compute all six natural periods at once
        epsilon = 1e-6 # Avoids numerical issues
        K_total = np.maximum(K_hydro + K_moor, 0.0)
        unknowns['rigid_body_periods'] = 2*np.pi * np.sqrt( (M_mat + A_mat) / (K_total + epsilon) )

        
    def check_frequency_margins(self, params, unknowns):
        # Unpack variables
        T_sys    = unknowns['rigid_body_periods']
        T_wave   = params['wave_period']
        f_struct = params['structural_frequencies']

        # Compute margins between wave forcing and natural periods
        unknowns['period_margin'] = np.abs(T_sys - T_wave) / T_wave

        # Compute margins bewteen wave forcing and structural frequencies
        T_struct = 1.0 / f_struct
        unknowns['modal_margin'] = np.abs(T_struct - T_wave) / T_wave
        
        
    def compute_costs(self, params, unknowns):
        # Unpack variables
        ncolumn    = int(params['number_of_auxiliary_columns'])
        c_mooring  = params['mooring_cost']
        c_aux      = params['auxiliary_column_cost']
        c_base     = params['base_column_cost']
        c_pontoon  = params['pontoon_cost']

        unknowns['total_cost'] = c_mooring + ncolumn*c_aux + c_base + c_pontoon
        

