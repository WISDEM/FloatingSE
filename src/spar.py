from openmdao.api import Component
import numpy as np

from floatingInstance import NSECTIONS
from constants import gravity

NPTS = 100

class Spar(Component):
    """
    OpenMDAO Component class for Spar substructure for floating offshore wind turbines.
    Should be tightly coupled with MAP Mooring class for full system representation.
    """

    def __init__(self):
        super(Spar,self).__init__()

        # Environment
        self.add_param('water_density', val=1025.0, units='kg/m**3', desc='density of water')

        # From other components
        self.add_param('turbine_mass', val=0.0, units='kg', desc='mass of tower and rna')
        self.add_param('turbine_center_of_gravity', val=0.0, units='m', desc='z-position of center of turbine mass')
        self.add_param('turbine_surge_force', val=np.zeros((2,)), units='N', desc='Force in surge direction on turbine')
        self.add_param('turbine_force_points', val=np.zeros((2,)), units='m', desc='zpts for force vector')
        self.add_param('turbine_pitch_moment', val=0.0, units='N*m', desc='Pitching moment from turbine that does not depend on sytem z-center of gravity')

        self.add_param('mooring_mass', val=0.0, units='kg', desc='Mass of mooring lines')
        self.add_param('mooring_effective_mass', val=0.0, units='kg', desc='Mass of mooring lines that weigh on structure, ignoring mass of mooring lines on sea floor')
        self.add_param('mooring_surge_restoring_force', val=0.0, units='N', desc='Restoring force in surge direction from mooring system')
        self.add_param('mooring_cost', val=0.0, units='USD', desc='Cost of mooring system')

        self.add_param('base_cylinder_mass', val=np.zeros((NSECTIONS,)), units='kg', desc='mass of cylinder')
        self.add_param('base_cylinder_displaced_volume', val=np.zeros((NSECTIONS,)), units='m**3', desc='cylinder volume of water displaced')
        self.add_param('base_cylinder_center_of_buoyancy', val=0.0, units='m', desc='z-position of center of cylinder buoyancy force')
        self.add_param('base_cylinder_center_of_gravity', val=0.0, units='m', desc='z-position of center of cylinder mass')
        self.add_param('base_cylinder_Iwaterplane', val=0.0, units='m**4', desc='Second moment of area of waterplane cross-section')
        self.add_param('base_cylinder_surge_force', val=np.zeros((NPTS,)), units='N', desc='Force vector in surge direction on cylinder')
        self.add_param('base_cylinder_force_points', val=np.zeros((NPTS,)), units='m', desc='zpts for force vector')
        self.add_param('base_cylinder_cost', val=0.0, units='USD', desc='Cost of spar structure')

        self.add_param('fairlead', val=1.0, units='m', desc='Depth below water for mooring line attachment')
        self.add_param('water_ballast_mass_vector', val=np.zeros((NPTS,)), units='kg', desc='mass vector of potential ballast mass')
        self.add_param('water_ballast_zpts_vector', val=np.zeros((NPTS,)), units='m', desc='z-points of potential ballast mass')

        # Outputs
        self.add_output('total_mass', val=0.0, units='kg', desc='total mass of spar and moorings')
        self.add_output('total_cost', val=0.0, units='USD', desc='total cost of spar and moorings')
        self.add_output('metacentric_height', val=0.0, units='m', desc='measure of static overturning stability')
        self.add_output('static_stability', val=0.0, desc='static stability margin based on position of centers of gravity and buoyancy')
        self.add_output('offset_force_ratio', val=0.0, units='m', desc='maximum surge offset')
        self.add_output('heel_angle', val=0.0, units='deg', desc='static angle of heel for turbine and spar substructure')
        self.add_output('variable_ballast_mass', val=0.0, units='kg', desc='Amount of variable water ballast')
        self.add_output('variable_ballast_height', val=0.0, units='m', desc='height of water ballast to balance spar')
        self.add_output('z_center_of_gravity', val=0.0, units='m', desc='Z-position of center of gravity (x,y = 0,0)')

        
    def solve_nonlinear(self, params, unknowns, resids):
        # TODO: Get centerlines right- in sparGeometry?
        # Determine ballast and cg of spar
        self.balance_spar(params, unknowns)
        
        # Determine stability, metacentric height from waterplane profile, displaced volume
        self.compute_stability(params, unknowns)
        
        # Sum all costs
        self.compute_costs(params, unknowns)

        
    def balance_spar(self, params, unknowns):
        # TODO SEMI: set n_cylinder and ballast_cylinder_mass/vol add n*param for total contributions
        # Unpack variables
        m_turb       = params['turbine_mass']
        m_mooringE   = params['mooring_effective_mass']
        m_mooring    = params['mooring_mass']
        m_cylinder   = params['base_cylinder_mass']
        m_water_data = params['water_ballast_mass_vector']
        V_cylinder   = params['base_cylinder_displaced_volume']
        z_cylinder   = params['base_cylinder_center_of_gravity']
        z_turb       = params['turbine_center_of_gravity']
        z_fairlead   = params['fairlead']*(-1)
        z_water_data = params['water_ballast_zpts_vector']
        rhoWater     = params['water_density']

        # SEMI TODO: Make water_ballast per cylinder

        # Make sure total mass of system with variable water ballast balances against displaced volume
        # Water ballast should be buried in m_cylinder
        m_system  = m_cylinder.sum() + m_mooringE + m_turb
        V_system  = V_cylinder.sum()
        m_water   = V_system*rhoWater - m_system

        # Output substructure total mass, different than system mass
        unknowns['total_mass'] = m_system - m_turb - m_mooringE + m_mooring
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
        # 2nd-order accurate derivative of data
        dmdz = np.gradient(m_water_data, z_water_data)
        # Put derivative on new data points for integration
        zpts = np.linspace(z_water_data[0], z_end, NPTS)
        dmdz = np.interp(zpts, z_water_data, dmdz)
        
        # Find cg of water
        z_water = np.trapz(zpts * dmdz, zpts) / m_water

        # Find cg of whole system
        z_cg = (m_turb*z_turb + m_cylinder.sum()*z_cylinder + m_mooringE*z_fairlead + m_water*z_water) / m_system
        unknowns['variable_ballast_mass']   = m_water
        unknowns['variable_ballast_height'] = h_water
        unknowns['z_center_of_gravity']     = z_cg
        #print unknowns['total_mass'], z_cg, V_system*rhoWater*gravity

        
    def compute_stability(self, params, unknowns):
        # Unpack variables
        z_cg              = unknowns['z_center_of_gravity']
        V_cylinder        = params['base_cylinder_displaced_volume']
        z_cb              = params['base_cylinder_center_of_buoyancy']
        Iwater_system     = params['base_cylinder_Iwaterplane']
        F_cylinder_vector = params['base_cylinder_surge_force']
        cylinder_zpts     = params['base_cylinder_force_points']
        F_turb_vector     = params['turbine_surge_force']
        turb_zpts         = params['turbine_force_points']
        M_turb            = params['turbine_pitch_moment']
        F_restore         = params['mooring_surge_restoring_force']
        rhoWater          = params['water_density']
        
        # Compute the distance from the center of buoyancy to the metacentre (BM is naval architecture)
        # BM = Iw / V where V is the displacement volume (just computed)
        # Iw is the area moment of inertia (meters^4) of the water-plane cross section about the heel axis
        # For a spar, we assume this is just the I of a circle about x or y
        # See https://en.wikipedia.org/wiki/Metacentric_height
        # https://en.wikipedia.org/wiki/List_of_second_moments_of_area
        # and http://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node30.html

        # Water plane area of all components
        #Iwater_system = 0.0
        #Iwater_system += Iwater_cylinder #+ Awater_cylinder*r_cylinder**2
        
        # Measure static stability:
        # 1. Center of buoyancy should be above CG (difference should be positive)
        # 2. Metacentric height should be positive
        V_system = V_cylinder.sum()
        buoyancy2metacentre_BM         = Iwater_system / V_system
        unknowns['static_stability'  ] = z_cb - z_cg
        unknowns['metacentric_height'] = buoyancy2metacentre_BM + unknowns['static_stability']
        
        # Compute restoring moment under small angle assumptions
        # Metacentric height computed during spar balancing calculation
        F_cylinder = np.trapz(F_cylinder_vector, cylinder_zpts)
        M_cylinder = np.trapz((cylinder_zpts-z_cg)*F_cylinder_vector, cylinder_zpts)
        F_turb     = F_turb_vector.sum()
        for f,z in zip(F_turb_vector, turb_zpts):
            M_turb += f*(z-z_cg) 
            
        F_surge    = F_cylinder + F_turb
        F_buoy     = V_system * rhoWater * gravity
        M_pitch    = M_cylinder + M_turb
        M_restore  = unknowns['metacentric_height'] * F_buoy
        # TODO SEMI: r x F to cb and cg of all components
            
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
        # TODO SEMI: set n_cylinder and ballast_cylinder_cost add n*param for total contributions
        # Unpack variables
        c_mooring  = params['mooring_cost']
        c_cylinder = params['base_cylinder_cost']

        unknowns['total_cost'] = c_mooring + c_cylinder
        
