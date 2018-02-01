from openmdao.api import Component
import numpy as np

from commonse import gravity, eps



class SemiGeometry(Component):
    """
    OpenMDAO Component class for Semi substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self, nSection):
        super(SemiGeometry,self).__init__()

        # Design variables
        self.add_param('base_outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_outer_diameter', val=np.zeros((nSection+1,)), units='m', desc='outer radius at each section node bottom to top (length = nsection + 1)')
        self.add_param('ballast_z_nodes', val=np.zeros((nSection+1,)), units='m', desc='z-coordinates of section nodes (length = nsection+1)')
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('fairlead_offset_from_shell', val=0.5, units='m',desc='fairlead offset from shell')
        self.add_param('radius_to_ballast_cylinder', val=10.0, units='m',desc='Distance from base cylinder centerpoint to ballast cylinder centerpoint')

        # Output constraints
        self.add_output('fairlead_radius', val=0.0, units='m', desc='Outer spar radius at fairlead depth (point of mooring attachment)')
        self.add_output('base_ballast_spacing', val=0.0, desc='Radius of base and ballast cylinders relative to spacing')

        
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
        R_od_base       = 0.5*params['base_outer_diameter']
        R_od_ballast    = 0.5*params['ballast_outer_diameter']
        R_semi          = params['radius_to_ballast_cylinder']
        z_nodes_ballast = params['ballast_z_nodes']
        fairlead        = params['fairlead'] # depth of mooring attachment point
        fair_off        = params['fairlead_offset_from_shell']

        # Set spacing constraint
        unknowns['base_ballast_spacing'] = (R_od_base.max() + R_od_ballast.max()) / R_semi

        # Determine radius at mooring connection point (fairlead)
        unknowns['fairlead_radius'] = R_semi + fair_off + np.interp(-fairlead, z_nodes_ballast, R_od_ballast)




class Semi(Component):
    """
    OpenMDAO Component class for Semisubmersible substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self, nSection, nIntPts):
        super(Semi,self).__init__()

        # Environment
        self.add_param('water_density', val=1025.0, units='kg/m**3', desc='density of water')

        # From other components
        self.add_param('turbine_mass', val=eps, units='kg', desc='mass of tower and rna')
        self.add_param('turbine_center_of_gravity', val=np.zeros((3,)), units='m', desc='xyz-position of center of turbine mass')
        self.add_param('turbine_surge_force', val=0.0, units='N', desc='Force in surge direction on turbine')
        self.add_param('turbine_pitch_moment', val=0.0, units='N*m', desc='Pitching moment (Myy) about turbine base')
        
        self.add_param('mooring_mass', val=0.0, units='kg', desc='Mass of mooring lines')
        self.add_param('mooring_effective_mass', val=0.0, units='kg', desc='Mass of mooring lines that weigh on structure, ignoring mass of mooring lines on sea floor')
        self.add_param('mooring_surge_restoring_force', val=0.0, units='N', desc='Restoring force in surge direction from mooring system')
        self.add_param('mooring_cost', val=0.0, units='USD', desc='Cost of mooring system')

        self.add_param('pontoon_mass', val=0.0, units='kg', desc='Mass of pontoon elements and connecting truss')
        self.add_param('pontoon_cost', val=0.0, units='USD', desc='Cost of pontoon elements and connecting truss')
        self.add_param('pontoon_buoyancy', val=0.0, units='N', desc='Buoyancy force of submerged pontoon elements')
        self.add_param('pontoon_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of pontoon buoyancy force')
        self.add_param('pontoon_center_of_gravity', val=0.0, units='m', desc='z-position of center of pontoon mass')
        
        self.add_param('base_cylinder_mass', val=np.zeros((nSection,)), units='kg', desc='mass of cylinder by section')
        self.add_param('base_cylinder_displaced_volume', val=np.zeros((nSection,)), units='m**3', desc='cylinder volume of water displaced by section')
        self.add_param('base_cylinder_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of cylinder buoyancy force')
        self.add_param('base_cylinder_center_of_gravity', val=0.0, units='m', desc='z-position of center of cylinder mass')
        self.add_param('base_cylinder_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('base_cylinder_surge_force', val=np.zeros((nIntPts,)), units='N', desc='Force vector in surge direction on cylinder')
        self.add_param('base_cylinder_force_points', val=np.zeros((nIntPts,)), units='m', desc='zpts for force vector')
        self.add_param('base_cylinder_cost', val=0.0, units='USD', desc='Cost of spar structure')
        self.add_param('base_freeboard', val=0.0, units='m', desc='Length of spar above water line')
        
        self.add_param('ballast_cylinder_mass', val=np.zeros((nSection,)), units='kg', desc='mass of cylinder by section')
        self.add_param('ballast_cylinder_displaced_volume', val=np.zeros((nSection,)), units='m**3', desc='cylinder volume of water displaced by section')
        self.add_param('ballast_cylinder_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of cylinder buoyancy force')
        self.add_param('ballast_cylinder_center_of_gravity', val=0.0, units='m', desc='z-position of center of cylinder mass')
        self.add_param('ballast_cylinder_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('ballast_cylinder_Awaterplane', val=0.0, units='m**2', desc='Area of waterplane cross-section')
        self.add_param('ballast_cylinder_surge_force', val=np.zeros((nIntPts,)), units='N', desc='Force vector in surge direction on cylinder')
        self.add_param('ballast_cylinder_force_points', val=np.zeros((nIntPts,)), units='m', desc='zpts for force vector')
        self.add_param('ballast_cylinder_cost', val=0.0, units='USD', desc='Cost of spar structure')
        
        self.add_param('number_of_ballast_cylinders', val=3, desc='Number of ballast cylinders evenly spaced around base cylinder', pass_by_obj=True)
        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('water_ballast_mass_vector', val=np.zeros((nIntPts,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_param('water_ballast_zpts_vector', val=np.zeros((nIntPts,)), units='m', desc='z-points of potential ballast mass')
        self.add_param('radius_to_ballast_cylinder', val=10.0, units='m',desc='Distance from base cylinder centerpoint to ballast cylinder centerpoint')

        # Outputs
        self.add_output('total_displacement', val=0.0, units='m**3', desc='total volume of water displaced by semi structure')
        self.add_output('total_mass', val=0.0, units='kg', desc='total mass of spar and moorings')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of spar and moorings')
        self.add_output('metacentric_height', val=0.0, units='m', desc='measure of static overturning stability')
        self.add_output('static_stability', val=0.0, desc='static stability margin based on position of centers of gravity and buoyancy')
        self.add_output('offset_force_ratio', val=0.0, units='m', desc='maximum surge offset')
        self.add_output('heel_angle', val=0.0, units='deg', desc='static angle of heel for turbine and spar substructure')
        self.add_output('variable_ballast_mass', val=0.0, units='kg', desc='Amount of variable water ballast')
        self.add_output('variable_ballast_height', val=0.0, units='m', desc='height of water ballast to balance spar')
        self.add_output('z_center_of_gravity', val=0.0, units='m', desc='Z-position of center of gravity (x,y = 0,0)')
        self.add_output('z_center_of_buoyancy', val=0.0, units='m', desc='Z-position of center of gravity (x,y = 0,0)')

        
        # Derivatives
        self.deriv_options['type'] = 'fd'
        self.deriv_options['form'] = 'central'
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
        ncylinder    = params['number_of_ballast_cylinders']
        m_turb       = params['turbine_mass']
        m_mooringE   = params['mooring_effective_mass']
        m_mooring    = params['mooring_mass']
        m_base       = params['base_cylinder_mass']
        m_cylinder   = params['ballast_cylinder_mass']
        m_pontoon    = params['pontoon_mass']
        
        V_base       = params['base_cylinder_displaced_volume']
        V_cylinder   = params['ballast_cylinder_displaced_volume']
        F_pontoon    = params['pontoon_buoyancy']
        
        z_base       = params['base_cylinder_center_of_gravity']
        z_cylinder   = params['ballast_cylinder_center_of_gravity']
        z_pontoon    = params['pontoon_center_of_gravity']
        z_turb       = params['turbine_center_of_gravity'][-1]
        z_fairlead   = params['fairlead']*(-1)

        z_cb_base     = params['base_cylinder_center_of_buoyancy']
        z_cb_cylinder = params['ballast_cylinder_center_of_buoyancy']
        z_cb_pontoon  = params['pontoon_center_of_buoyancy']
        
        m_water_data = params['water_ballast_mass_vector']
        z_water_data = params['water_ballast_zpts_vector']
        rhoWater     = params['water_density']
        npts         = z_water_data.size
        
        # SEMI TODO: Make water_ballast in base only?  cylinders too?  How to apportion?

        # Make sure total mass of system with variable water ballast balances against displaced volume
        # Water ballast should be buried in m_cylinder
        m_system  = m_base.sum() + ncylinder*m_cylinder.sum() + m_mooringE + m_turb + m_pontoon
        V_pontoon = F_pontoon/rhoWater/gravity
        V_system  = V_base.sum() + ncylinder*V_cylinder.sum() + V_pontoon
        m_water   = V_system*rhoWater - m_system

        # Output substructure total mass, different than system mass
        unknowns['total_mass'] = m_system - m_turb - m_mooringE + m_mooring
        unknowns['total_displacement'] = V_system
        m_system += m_water

        # Find height given interpolant functions from cylinders
        if m_water_data[-1] < m_water:
            # Don't have enough space, so max out variable balast here and constraints will catch this
            z_end = z_water_data[-1]
        elif m_water < 0.0:
            z_end = z_water_data[0]
        else:
            z_end = np.interp(m_water, m_water_data, z_water_data)
        h_water = z_end - z_water_data[0]
        unknowns['variable_ballast_mass']   = m_water
        unknowns['variable_ballast_height'] = h_water

        # Find cb (center of buoyancy) for whole system
        z_cb = (V_base.sum()*z_cb_base + ncylinder*V_cylinder.sum()*z_cb_cylinder + V_pontoon*z_cb_pontoon) / V_system
        unknowns['z_center_of_buoyancy'] = z_cb
        
        # Find cg of whole system
        # First find cg of water variable ballast by taking derivative of cumulative integral
        # 2nd-order accurate derivative of data
        dmdz    = np.gradient(m_water_data, z_water_data)
        # Put derivative on new data points for integration
        zpts    = np.linspace(z_water_data[0], z_end, npts)
        dmdz    = np.interp(zpts, z_water_data, dmdz)
        z_water = np.trapz(zpts * dmdz, zpts) / m_water
        z_cg = (m_turb*z_turb + ncylinder*m_cylinder.sum()*z_cylinder + m_base.sum()*z_base +
                m_mooringE*z_fairlead + m_water*z_water + m_pontoon*z_pontoon) / m_system
        unknowns['z_center_of_gravity'] = z_cg 

        
    def compute_stability(self, params, unknowns):
        # Unpack variables
        ncylinder         = params['number_of_ballast_cylinders']
        z_cb              = unknowns['z_center_of_buoyancy']
        z_cg              = unknowns['z_center_of_gravity']
        freeboard         = params['base_freeboard']
        
        Iwater_base       = params['base_cylinder_Iwaterplane']
        Iwater_cylinder   = params['ballast_cylinder_Iwaterplane']
        Awater_cylinder   = params['ballast_cylinder_Awaterplane']
        m_cylinder        = params['ballast_cylinder_mass']

        V_cylinder        = params['base_cylinder_displaced_volume']
        V_system          = unknowns['total_displacement']
        
        F_base_vector     = params['base_cylinder_surge_force']
        zpts_base         = params['base_cylinder_force_points']
        
        F_cylinder_vector = params['ballast_cylinder_surge_force']
        zpts_cylinder     = params['ballast_cylinder_force_points']
        
        F_turb            = params['turbine_surge_force']
        M_turb            = params['turbine_pitch_moment']
        
        F_restore         = params['mooring_surge_restoring_force']
        rhoWater          = params['water_density']
        R_semi            = params['radius_to_ballast_cylinder']
        
        # Compute the distance from the center of buoyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the area moment of inertia (meters^4) of the water-plane cross section about the heel axis
        # For a spar, we assume this is just the I of a circle about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html

        # Water plane area of all components with parallel axis theorem
        Iwater_system = Iwater_base
        radii = R_semi * np.cos( np.linspace(0, 2*np.pi, ncylinder+1) )
        for k in xrange(ncylinder):
            Iwater_system += Iwater_cylinder + Awater_cylinder*radii[k]**2
        
        # Measure static stability:
        # 1. Center of buoyancy should be above CG (difference should be positive)
        # 2. Metacentric height should be positive
        buoyancy2metacentre_BM         = Iwater_system / V_system
        unknowns['static_stability'  ] = z_cb - z_cg
        unknowns['metacentric_height'] = buoyancy2metacentre_BM + unknowns['static_stability']
        
        # Compute restoring moment under small angle assumptions
        # Metacentric height computed during spar balancing calculation
        F_base = np.trapz(F_base_vector, zpts_base)
        M_base = np.trapz((zpts_base-z_cg)*F_base_vector, zpts_base)

        F_cylinder = np.trapz(F_cylinder_vector, zpts_cylinder)
        M_cylinder = np.trapz((zpts_cylinder-z_cg)*F_cylinder_vector, zpts_cylinder)

        M_buoy_weight = 0.0
        F_buoy_cylinder = V_cylinder.sum() * rhoWater * gravity
        W_cylinder      = m_cylinder.sum() * gravity
        for k in xrange(ncylinder):
            M_buoy_weight += radii[k]*(F_buoy_cylinder - W_cylinder)
        
        # Correct moment from base of turbine to cg of spar M' = F(x+dx) = m+Fdx
        M_turb    += F_turb * (freeboard-z_cg) 

        F_surge    = F_base + ncylinder*F_cylinder + F_turb
        M_pitch    = M_base + ncylinder*M_cylinder + M_buoy_weight + M_turb

        F_buoy     = V_system * rhoWater * gravity
        M_restore  = unknowns['metacentric_height'] * F_buoy
            
        # Comput heel angle
        unknowns['heel_angle'] = np.abs( np.rad2deg( M_pitch / M_restore ) )

        # Now compute offsets from the applied force
        # First use added mass (the mass of the water that must be displaced in movement)
        # http://www.iaea.org/inis/collection/NCLCollectionStore/_Public/09/411/9411273.pdf
        #mass_add_surge = rhoWater * np.pi * R_od.max() * draft
        #T_surge        = 2*np.pi*np.sqrt( (unknowns['total_mass']+mass_add_surge) / kstiff_horiz_mooring)

        # Compare restoring force from mooring to force of worst case spar displacement
        unknowns['offset_force_ratio'] = np.abs(F_surge / F_restore)

        
    def compute_costs(self, params, unknowns):
        # Unpack variables
        ncylinder  = params['number_of_ballast_cylinders']
        c_mooring  = params['mooring_cost']
        c_cylinder = params['ballast_cylinder_cost']
        c_base     = params['base_cylinder_cost']
        c_pontoon  = params['pontoon_cost']

        unknowns['total_cost'] = c_mooring + ncylinder*c_cylinder + c_base + c_pontoon
        
