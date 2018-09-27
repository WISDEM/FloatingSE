from openmdao.api import Group, IndepVarComp, Problem, Component
from .column import Column, ColumnGeometry
from .substructure import Substructure, SubstructureGeometry
from .loading import Loading
from .map_mooring import MapMooring
from towerse.tower import TowerLeanSE
import numpy as np

    
class FloatingSE(Group):

    def __init__(self, nSection):
        super(FloatingSE, self).__init__()

        self.nFull = 3*nSection+1

        self.add('tow', TowerLeanSE(nSection+1,self.nFull), promotes=['material_density','tower_section_height',
                                                                      'tower_outer_diameter','tower_wall_thickness','tower_outfitting_factor',
                                                                      'tower_buckling_length','max_taper','min_d_to_t','rna_mass','rna_cg','rna_I',
                                                                      'tower_mass','tower_I_base','hub_height'])
        
        # Next do base and ballast columns
        # Ballast columns are replicated from same design in the components
        self.add('base', Column(nSection, self.nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                                 'Uref','zref','shearExp','beta','yaw','Uc','Hs','T','cd_usr','cm','loading',
                                                                 'max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height','fairlead',
                                                                 'permanent_ballast_density','bulkhead_mass_factor','ballast_heave_box_mass_factor',
                                                                 'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                 'tapered_col_cost_rate','outfitting_cost_rate'])
        self.add('aux', Column(nSection, self.nFull), promotes=['water_depth','water_density','material_density','E','nu','yield_stress','z0',
                                                                'Uref','zref','shearExp','beta','yaw','Uc','Hs','T','cd_usr','cm','loading',
                                                                'max_taper','min_d_to_t','gamma_f','gamma_b','foundation_height','fairlead',
                                                                'permanent_ballast_density','bulkhead_mass_factor','ballast_heave_box_mass_factor',
                                                                'ring_mass_factor','column_mass_factor','outfitting_mass_fraction','ballast_cost_rate',
                                                                'tapered_col_cost_rate','outfitting_cost_rate'])

        # Run Semi Geometry for interfaces
        self.add('sg', SubstructureGeometry(self.nFull), promotes=['*'])

        # Next run MapMooring
        self.add('mm', MapMooring(), promotes=['*'])
        
        # Add in the connecting truss
        self.add('load', Loading(nSection, self.nFull), promotes=['*'])#water_density','material_density','E','G','yield_stress',

        # Run main Semi analysis
        self.add('subs', Substructure(self.nFull), promotes=['*'])

        # Define all input variables from all models
        
        # SemiGeometry
        self.add('radius_to_auxiliary_column', IndepVarComp('radius_to_auxiliary_column', 0.0), promotes=['*'])
        self.add('number_of_auxiliary_columns',  IndepVarComp('number_of_auxiliary_columns', 0), promotes=['*'])
        
        self.add('fairlead',                   IndepVarComp('fairlead', 0.0), promotes=['*'])
        self.add('fairlead_offset_from_shell', IndepVarComp('fairlead_offset_from_shell', 0.0), promotes=['*'])
        self.add('z_offset',                   IndepVarComp('z_offset', 0.0), promotes=['*'])


        # Mooring
        self.add('mooring_line_length',        IndepVarComp('mooring_line_length', 0.0), promotes=['*'])
        self.add('anchor_radius',              IndepVarComp('anchor_radius', 0.0), promotes=['*'])
        self.add('mooring_diameter',           IndepVarComp('mooring_diameter', 0.0), promotes=['*'])
        self.add('number_of_mooring_connections', IndepVarComp('number_of_mooring_connections', 0), promotes=['*'])
        self.add('mooring_lines_per_connection', IndepVarComp('mooring_lines_per_connection', 0), promotes=['*'])
        self.add('mooring_type',               IndepVarComp('mooring_type', 'chain', pass_by_obj=True), promotes=['*'])
        self.add('anchor_type',                IndepVarComp('anchor_type', 'SUCTIONPILE', pass_by_obj=True), promotes=['*'])
        self.add('max_offset',         IndepVarComp('max_offset', 0.0), promotes=['*'])
        self.add('operational_heel',   IndepVarComp('operational_heel', 0.0), promotes=['*'])
        self.add('mooring_cost_rate',          IndepVarComp('mooring_cost_rate', 0.0), promotes=['*'])
        self.add('max_survival_heel',          IndepVarComp('max_survival_heel', 0.0), promotes=['*'])

        # Column
        self.add('permanent_ballast_density',  IndepVarComp('permanent_ballast_density', 0.0), promotes=['*'])
        
        self.add('base_freeboard',             IndepVarComp('base_freeboard', 0.0), promotes=['*'])
        self.add('base_section_height',        IndepVarComp('base_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('base_outer_diameter',        IndepVarComp('base_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('base_wall_thickness',        IndepVarComp('base_wall_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('base_stiffener_web_height',       IndepVarComp('base_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_web_thickness',    IndepVarComp('base_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_flange_width',     IndepVarComp('base_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_flange_thickness', IndepVarComp('base_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('base_stiffener_spacing',          IndepVarComp('base_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('base_bulkhead_thickness',             IndepVarComp('base_bulkhead_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('base_permanent_ballast_height',   IndepVarComp('base_permanent_ballast_height', 0.0), promotes=['*'])
        self.add('base_ballast_heave_box_diameter',   IndepVarComp('base_ballast_heave_box_diameter', 0.0), promotes=['*'])
        self.add('base_ballast_heave_box_height',   IndepVarComp('base_ballast_heave_box_height', 0.0), promotes=['*'])
        self.add('base_ballast_heave_box_location',   IndepVarComp('base_ballast_heave_box_location', 0.0), promotes=['*'])

        self.add('auxiliary_freeboard',          IndepVarComp('auxiliary_freeboard', 0.0), promotes=['*'])
        self.add('auxiliary_section_height',     IndepVarComp('auxiliary_section_height', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_outer_diameter',     IndepVarComp('auxiliary_outer_diameter', np.zeros((nSection+1,))), promotes=['*'])
        self.add('auxiliary_wall_thickness',     IndepVarComp('auxiliary_wall_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('auxiliary_stiffener_web_height',       IndepVarComp('auxiliary_stiffener_web_height', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_stiffener_web_thickness',    IndepVarComp('auxiliary_stiffener_web_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_stiffener_flange_width',     IndepVarComp('auxiliary_stiffener_flange_width', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_stiffener_flange_thickness', IndepVarComp('auxiliary_stiffener_flange_thickness', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_stiffener_spacing',          IndepVarComp('auxiliary_stiffener_spacing', np.zeros((nSection,))), promotes=['*'])
        self.add('auxiliary_bulkhead_thickness',             IndepVarComp('auxiliary_bulkhead_thickness', np.zeros((nSection+1,))), promotes=['*'])
        self.add('auxiliary_permanent_ballast_height',   IndepVarComp('auxiliary_permanent_ballast_height', 0.0), promotes=['*'])
        self.add('auxiliary_ballast_heave_box_diameter',   IndepVarComp('auxiliary_ballast_heave_box_diameter', 0.0), promotes=['*'])
        self.add('auxiliary_ballast_heave_box_height',   IndepVarComp('auxiliary_ballast_heave_box_height', 0.0), promotes=['*'])
        self.add('auxiliary_ballast_heave_box_location',   IndepVarComp('auxiliary_ballast_heave_box_location', 0.0), promotes=['*'])

        self.add('bulkhead_mass_factor',       IndepVarComp('bulkhead_mass_factor', 0.0), promotes=['*'])
        self.add('ring_mass_factor',           IndepVarComp('ring_mass_factor', 0.0), promotes=['*'])
        self.add('shell_mass_factor',          IndepVarComp('shell_mass_factor', 0.0), promotes=['*'])
        self.add('column_mass_factor',           IndepVarComp('column_mass_factor', 0.0), promotes=['*'])
        self.add('outfitting_mass_fraction',   IndepVarComp('outfitting_mass_fraction', 0.0), promotes=['*'])
        self.add('ballast_cost_rate',          IndepVarComp('ballast_cost_rate', 0.0), promotes=['*'])
        self.add('tapered_col_cost_rate',      IndepVarComp('tapered_col_cost_rate', 0.0), promotes=['*'])
        self.add('outfitting_cost_rate',       IndepVarComp('outfitting_cost_rate', 0.0), promotes=['*'])
        self.add('loading',                    IndepVarComp('loading', val='hydrostatic', pass_by_obj=True), promotes=['*'])
        
        self.add('max_taper_ratio',            IndepVarComp('max_taper_ratio', 0.0), promotes=['*'])
        self.add('min_diameter_thickness_ratio', IndepVarComp('min_diameter_thickness_ratio', 0.0), promotes=['*'])

        # Pontoons
        #self.add('G',                          IndepVarComp('G', 0.0), promotes=['*'])

        # Other Constraints
        self.add('wave_period_range_low',   IndepVarComp('wave_period_range_low', 2.0), promotes=['*'])
        self.add('wave_period_range_high',  IndepVarComp('wave_period_range_high', 20.0), promotes=['*'])

        # Connect all input variables from all models
        self.connect('radius_to_auxiliary_column', ['radius_to_auxiliary_column', 'radius_to_auxiliary_column'])

        self.connect('base_freeboard', ['tow.foundation_height', 'base.freeboard', 'base_freeboard'])
        self.connect('base_section_height', 'base.section_height')
        self.connect('base_outer_diameter', 'base.diameter')
        self.connect('base_wall_thickness', 'base.wall_thickness')
        self.connect('z_offset', 'foundation_height')

        self.connect('tow.d_full', ['windLoads.d','tower_d_full']) # includes tower_d_full
        self.connect('tow.t_full', 'tower_t_full')
        self.connect('tow.z_full', ['loadingWind.z','tower_z_full']) # includes tower_z_full
        self.connect('tow.cm.mass','tower_mass_section')
        self.connect('tower_buckling_length','tower_buckling_length')
        self.connect('tow.turbine_mass','base.stack_mass_in')
        self.connect('tow.tower_center_of_mass','tower_center_of_mass')
        
        self.connect('auxiliary_freeboard', ['aux.freeboard','auxiliary_freeboard'])
        self.connect('auxiliary_section_height', 'aux.section_height')
        self.connect('auxiliary_outer_diameter', 'aux.diameter')
        self.connect('auxiliary_wall_thickness', 'aux.wall_thickness')

        self.connect('max_taper_ratio', 'max_taper')
        self.connect('min_diameter_thickness_ratio', 'min_d_to_t')
        
        # To do: connect these to independent variables
        self.connect('base.windLoads.rho',['aux.windLoads.rho','windLoads.rho'])
        self.connect('base.windLoads.mu',['aux.windLoads.mu','windLoads.mu'])
        self.connect('base.waveLoads.mu','aux.waveLoads.mu')

        
        self.connect('base_stiffener_web_height', 'base.stiffener_web_height')
        self.connect('base_stiffener_web_thickness', 'base.stiffener_web_thickness')
        self.connect('base_stiffener_flange_width', 'base.stiffener_flange_width')
        self.connect('base_stiffener_flange_thickness', 'base.stiffener_flange_thickness')
        self.connect('base_stiffener_spacing', 'base.stiffener_spacing')
        self.connect('base_bulkhead_thickness', 'base.bulkhead_thickness')
        self.connect('base_permanent_ballast_height', 'base.permanent_ballast_height')
        self.connect('base.L_stiffener','base_buckling_length')
        self.connect('base_ballast_heave_box_diameter', 'base.ballast_heave_box_diameter')
        self.connect('base_ballast_heave_box_height', 'base.ballast_heave_box_height')
        self.connect('base_ballast_heave_box_location', 'base.ballast_heave_box_location')

        self.connect('auxiliary_stiffener_web_height', 'aux.stiffener_web_height')
        self.connect('auxiliary_stiffener_web_thickness', 'aux.stiffener_web_thickness')
        self.connect('auxiliary_stiffener_flange_width', 'aux.stiffener_flange_width')
        self.connect('auxiliary_stiffener_flange_thickness', 'aux.stiffener_flange_thickness')
        self.connect('auxiliary_stiffener_spacing', 'aux.stiffener_spacing')
        self.connect('auxiliary_bulkhead_thickness', 'aux.bulkhead_thickness')
        self.connect('auxiliary_permanent_ballast_height', 'aux.permanent_ballast_height')
        self.connect('aux.L_stiffener','auxiliary_buckling_length')
        self.connect('auxiliary_ballast_heave_box_diameter', 'aux.ballast_heave_box_diameter')
        self.connect('auxiliary_ballast_heave_box_height', 'aux.ballast_heave_box_height')
        self.connect('auxiliary_ballast_heave_box_location', 'aux.ballast_heave_box_location')
        
        self.connect('bulkhead_mass_factor', 'ballast_heave_box_mass_factor')
        self.connect('shell_mass_factor', ['base.cyl_mass.outfitting_factor', 'aux.cyl_mass.outfitting_factor'])

        self.connect('base.z_full', ['base_z_nodes', 'base_z_full'])
        self.connect('base.d_full', 'base_d_full')
        self.connect('base.t_full', 'base_t_full')

        self.connect('aux.z_full', ['auxiliary_z_nodes', 'auxiliary_z_full'])
        self.connect('aux.d_full', 'auxiliary_d_full')
        self.connect('aux.t_full', 'auxiliary_t_full')

        self.connect('max_offset_restoring_force', 'mooring_surge_restoring_force')
        self.connect('operational_heel_restoring_force', 'mooring_pitch_restoring_force')
        
        self.connect('base.z_center_of_mass', 'base_center_of_mass')
        self.connect('base.z_center_of_buoyancy', 'base_center_of_buoyancy')
        self.connect('base.I_column', 'base_moments_of_inertia')
        self.connect('base.Iwater', 'base_Iwaterplane')
        self.connect('base.Awater', 'base_Awaterplane')
        self.connect('base.displaced_volume', 'base_displaced_volume')
        self.connect('base.hydrostatic_force', 'base_hydrostatic_force')
        self.connect('base.added_mass', 'base_added_mass')
        self.connect('base.total_mass', 'base_mass')
        self.connect('base.total_cost', 'base_cost')
        self.connect('base.variable_ballast_interp_zpts', 'water_ballast_zpts_vector')
        self.connect('base.variable_ballast_interp_radius', 'water_ballast_radius_vector')
        self.connect('base.Px', 'base_Px')
        self.connect('base.Py', 'base_Py')
        self.connect('base.Pz', 'base_Pz')
        self.connect('base.qdyn', 'base_qdyn')

        self.connect('aux.z_center_of_mass', 'auxiliary_center_of_mass')
        self.connect('aux.z_center_of_buoyancy', 'auxiliary_center_of_buoyancy')
        self.connect('aux.I_column', 'auxiliary_moments_of_inertia')
        self.connect('aux.Iwater', 'auxiliary_Iwaterplane')
        self.connect('aux.Awater', 'auxiliary_Awaterplane')
        self.connect('aux.displaced_volume', 'auxiliary_displaced_volume')
        self.connect('aux.hydrostatic_force', 'auxiliary_hydrostatic_force')
        self.connect('aux.added_mass', 'auxiliary_added_mass')
        self.connect('aux.total_mass', 'auxiliary_mass')
        self.connect('aux.total_cost', 'auxiliary_cost')
        self.connect('aux.Px', 'auxiliary_Px')
        self.connect('aux.Py', 'auxiliary_Py')
        self.connect('aux.Pz', 'auxiliary_Pz')
        self.connect('aux.qdyn', 'auxiliary_qdyn')
        self.connect('aux.draft', 'auxiliary_draft')

         # Use complex number finite differences
        typeStr = 'fd'
        formStr = 'central'
        stepVal = 1e-5
        stepStr = 'relative'
        
        self.deriv_options['type'] = typeStr
        self.deriv_options['form'] = formStr
        self.deriv_options['check_form'] = formStr
        self.deriv_options['step_size'] = stepVal
        self.deriv_options['step_calc'] = stepStr



def sparExample():
    # Number of sections to be used in the design
    nsection = 5

    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem(root=FloatingSE(nsection))
    prob.setup()

    # Remove all auxiliary columns
    prob['number_of_auxiliary_columns'] = 0
    prob['cross_attachment_pontoons_int']   = 0
    prob['lower_attachment_pontoons_int']   = 0
    prob['upper_attachment_pontoons_int']   = 0
    prob['lower_ring_pontoons_int']         = 0
    prob['upper_ring_pontoons_int']         = 0
    prob['outer_cross_pontoons_int']        = 0

    # Set environment to that used in OC3 testing campaign
    prob['water_depth'] = 320.0  # Distance to sea floor [m]
    prob['Hs']        = 10.8   # Significant wave height [m]
    prob['T']           = 9.8    # Wave period [s]
    prob['Uref']        = 11.0   # Wind reference speed [m/s]
    prob['zref']        = 119.0  # Wind reference height [m]
    prob['shearExp']    = 0.11   # Shear exponent in wind power law
    prob['cm']          = 2.0    # Added mass coefficient
    prob['Uc']          = 0.0    # Mean current speed
    prob['z0']          = 0.0    # Water line
    prob['yaw']         = 0.0    # Turbine yaw angle
    prob['beta']        = 0.0    # Wind beta angle
    prob['cd_usr']      = np.inf # Compute drag coefficient

    # Wind and water properties
    prob['base.windLoads.rho'] = 1.226   # Density of air [kg/m^3]
    prob['base.windLoads.mu']  = 1.78e-5 # Viscosity of air [kg/m/s]
    prob['water_density']      = 1025.0  # Density of water [kg/m^3]
    prob['base.waveLoads.mu']  = 1.08e-3 # Viscosity of water [kg/m/s]
    
    # Material properties
    prob['material_density'] = 7850.0          # Steel [kg/m^3]
    prob['E']                = 200e9           # Young's modulus [N/m^2]
    prob['G']                = 79.3e9          # Shear modulus [N/m^2]
    prob['yield_stress']     = 3.45e8          # Elastic yield stress [N/m^2]
    prob['nu']               = 0.26            # Poisson's ratio
    prob['permanent_ballast_density'] = 4492.0 # [kg/m^3]

    # Mass and cost scaling factors
    prob['bulkhead_mass_factor']     = 1.0     # Scaling for unaccounted bulkhead mass
    prob['ring_mass_factor']         = 1.0     # Scaling for unaccounted stiffener mass
    prob['shell_mass_factor']        = 1.0     # Scaling for unaccounted shell mass
    prob['column_mass_factor']       = 1.05    # Scaling for unaccounted column mass
    prob['outfitting_mass_fraction'] = 0.06    # Fraction of additional outfitting mass for each column
    prob['ballast_cost_rate']        = 100.0   # Cost factor for ballast mass [$/kg]
    prob['tapered_col_cost_rate']    = 4720.0  # Cost factor for column mass [$/kg]
    prob['outfitting_cost_rate']     = 6980.0  # Cost factor for outfitting mass [$/kg]
    prob['mooring_cost_rate']        = 1.1     # Cost factor for mooring mass [$/kg]
    
    # Safety factors
    prob['gamma_f'] = 1.35 # Safety factor on loads
    prob['gamma_b'] = 1.1  # Safety factor on buckling
    prob['gamma_m'] = 1.1  # Safety factor on materials
    prob['gamma_n'] = 1.0  # Safety factor on consequence of failure
    prob['gamma_fatigue'] = 1.755 # Not used

    # Column geometry
    prob['base_permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['base_freeboard']                = 10.0 # Height extension above waterline [m]
    prob['base_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])  # Length of each section [m]
    prob['base_outer_diameter'] = np.array([9.4, 9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['base_wall_thickness'] = 0.05 * np.ones(nsection+1)               # Shell thickness at each section node (linear lofting between) [m]
    prob['base_bulkhead_thickness'] = 0.05*np.array([1,1,0,0,0,0]) # Locations/thickness of internal bulkheads at section interfaces [m]
    
    # Column ring stiffener parameters
    prob['base_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]
    
    # Mooring parameters
    prob['number_of_mooring_connections'] = 3             # Evenly spaced around structure
    prob['mooring_lines_per_connection'] = 1             # Evenly spaced around structure
    prob['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
    prob['anchor_type']                = 'suctionpile' # Options are SUCTIONPILE or DRAGEMBEDMENT
    prob['mooring_diameter']           = 0.09          # Diameter of mooring line/chain [m]
    prob['fairlead']                   = 70.0          # Distance below waterline for attachment [m]
    prob['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
    prob['mooring_line_length']        = 902.2         # Unstretched mooring line length
    prob['anchor_radius']              = 853.87        # Distance from centerline to sea floor landing [m]
    prob['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
    prob['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]

    # Porperties of turbine tower
    prob['hub_height']              = 77.6                              # Length from tower base to top (not including freeboard) [m]
    prob['tower_section_height']    = 77.6/nsection * np.ones(nsection) # Length of each tower section [m]
    prob['tower_outer_diameter']    = np.linspace(6.5, 3.87, nsection+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_wall_thickness']    = np.linspace(0.027, 0.019, nsection+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_buckling_length']   = 30.0                              # Tower buckling reinforcement spacing [m]
    prob['tower_outfitting_factor'] = 1.07                              # Scaling for unaccounted tower mass in outfitting

    # Properties of rotor-nacelle-assembly (RNA)
    prob['rna_mass']   = 350e3 # Mass [kg]
    prob['rna_I']      = 1e5*np.array([1149.307, 220.354, 187.597, 0, 5.037, 0]) # Moment of intertia (xx,yy,zz,xy,xz,yz) [kg/m^2]
    prob['rna_cg']     = np.array([-1.132, 0, 0.509])                       # Offset of RNA center of mass from tower top (x,y,z) [m]
    # Max thrust
    prob['rna_force']  = np.array([1284744.196, 0, -112400.5527])           # Net force acting on RNA (x,y,z) [N]
    prob['rna_moment'] = np.array([3963732.762, 896380.8464, -346781.682]) # Net moment acting on RNA (x,y,z) [N*m]
    # Max wind speed
    #prob['rna_force']  = np.array([188038.8045, 0,  -16451.2637]) # Net force acting on RNA (x,y,z) [N]
    #prob['rna_moment'] = np.array([0.0, 131196.8431,  0.0]) # Net moment acting on RNA (x,y,z) [N*m]
    
    # Mooring constraints
    prob['mooring_max_offset'] = 0.1*prob['water_depth'] # Max surge/sway offset [m]      
    prob['mooring_operational_heel']   = 10.0 # Max heel (pitching) angle [deg]

    # Design constraints
    prob['max_taper_ratio'] = 0.2                # For manufacturability of rolling steel
    prob['min_diameter_thickness_ratio'] = 120.0 # For weld-ability

    # API 2U flag
    prob['loading'] = 'hydrostatic'

    # Other variables to avoid divide by zeros, even though it won't matter
    prob['radius_to_auxiliary_column'] = 15.0
    prob['auxiliary_section_height'] = 1.0 * np.ones(nsection)
    prob['auxiliary_outer_diameter'] = 5.0 * np.ones(nsection+1)
    prob['auxiliary_wall_thickness'] = 0.1 * np.ones(nsection+1)
    prob['auxiliary_permanent_ballast_height'] = 0.1
    prob['auxiliary_stiffener_web_height'] = 0.1 * np.ones(nsection)
    prob['auxiliary_stiffener_web_thickness'] =  0.1 * np.ones(nsection)
    prob['auxiliary_stiffener_flange_width'] =  0.1 * np.ones(nsection)
    prob['auxiliary_stiffener_flange_thickness'] =  0.1 * np.ones(nsection)
    prob['auxiliary_stiffener_spacing'] =  0.1 * np.ones(nsection)
    prob['pontoon_outer_diameter'] = 1.0
    prob['pontoon_wall_thickness'] = 0.1
    
    prob.run()

    '''
    f = open('deriv_spar.dat','w')
    out = prob.check_total_derivatives(f)
    #out = prob.check_partial_derivatives(f, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        for k in out[comp].keys():
            if ( (out[comp][k]['rel error'][0] > tol) and (out[comp][k]['abs error'][0] > tol) ):
                print k
    '''





def semiExample():
    # Number of sections to be used in the design
    nsection = 5

    # Initialize OpenMDAO problem and FloatingSE Group
    prob = Problem(root=FloatingSE(nsection))
    prob.setup()

    # Add in auxiliary columns and truss elements
    prob['number_of_auxiliary_columns'] = 3
    prob['cross_attachment_pontoons_int']   = 1 # Lower-Upper base-to-auxiliary connecting cross braces
    prob['lower_attachment_pontoons_int']   = 1 # Lower base-to-auxiliary connecting pontoons
    prob['upper_attachment_pontoons_int']   = 1 # Upper base-to-auxiliary connecting pontoons
    prob['lower_ring_pontoons_int']         = 1 # Lower ring of pontoons connecting auxiliary columns
    prob['upper_ring_pontoons_int']         = 1 # Upper ring of pontoons connecting auxiliary columns
    prob['outer_cross_pontoons_int']        = 1 # Auxiliary ring connecting V-cross braces

    # Set environment to that used in OC4 testing campaign
    prob['water_depth'] = 200.0  # Distance to sea floor [m]
    prob['Hs']        = 10.8   # Significant wave height [m]
    prob['T']           = 9.8    # Wave period [s]
    prob['Uref']        = 11.0   # Wind reference speed [m/s]
    prob['zref']        = 119.0  # Wind reference height [m]
    prob['shearExp']    = 0.11   # Shear exponent in wind power law
    prob['cm']          = 2.0    # Added mass coefficient
    prob['Uc']          = 0.0    # Mean current speed
    prob['z0']          = 0.0    # Water line
    prob['yaw']         = 0.0    # Turbine yaw angle
    prob['beta']        = 0.0    # Wind beta angle
    prob['cd_usr']      = np.inf # Compute drag coefficient

    # Wind and water properties
    prob['base.windLoads.rho'] = 1.226   # Density of air [kg/m^3]
    prob['base.windLoads.mu']  = 1.78e-5 # Viscosity of air [kg/m/s]
    prob['water_density']      = 1025.0  # Density of water [kg/m^3]
    prob['base.waveLoads.mu']  = 1.08e-3 # Viscosity of water [kg/m/s]
    
    # Material properties
    prob['material_density'] = 7850.0          # Steel [kg/m^3]
    prob['E']                = 200e9           # Young's modulus [N/m^2]
    prob['G']                = 79.3e9          # Shear modulus [N/m^2]
    prob['yield_stress']     = 3.45e8          # Elastic yield stress [N/m^2]
    prob['nu']               = 0.26            # Poisson's ratio
    prob['permanent_ballast_density'] = 4492.0 # [kg/m^3]

    # Mass and cost scaling factors
    prob['bulkhead_mass_factor']     = 1.0     # Scaling for unaccounted bulkhead mass
    prob['ring_mass_factor']         = 1.0     # Scaling for unaccounted stiffener mass
    prob['shell_mass_factor']        = 1.0     # Scaling for unaccounted shell mass
    prob['column_mass_factor']       = 1.05    # Scaling for unaccounted column mass
    prob['outfitting_mass_fraction'] = 0.06    # Fraction of additional outfitting mass for each column
    prob['ballast_cost_rate']        = 100.0   # Cost factor for ballast mass [$/kg]
    prob['tapered_col_cost_rate']    = 4720.0  # Cost factor for column mass [$/kg]
    prob['outfitting_cost_rate']     = 6980.0  # Cost factor for outfitting mass [$/kg]
    prob['mooring_cost_rate']        = 1.1     # Cost factor for mooring mass [$/kg]
    prob['pontoon_cost_rate']        = 6.250   # Cost factor for pontoons [$/kg]
    
    # Safety factors
    prob['gamma_f'] = 1.35 # Safety factor on loads
    prob['gamma_b'] = 1.1  # Safety factor on buckling
    prob['gamma_m'] = 1.1  # Safety factor on materials
    prob['gamma_n'] = 1.0  # Safety factor on consequence of failure
    prob['gamma_fatigue'] = 1.755 # Not used

    # Column geometry
    prob['base_permanent_ballast_height'] = 10.0 # Height above keel for permanent ballast [m]
    prob['base_freeboard']                = 10.0 # Height extension above waterline [m]
    prob['base_section_height'] = np.array([36.0, 36.0, 36.0, 8.0, 14.0])  # Length of each section [m]
    prob['base_outer_diameter'] = np.array([9.4, 9.4, 9.4, 9.4, 6.5, 6.5]) # Diameter at each section node (linear lofting between) [m]
    prob['base_wall_thickness'] = 0.05 * np.ones(nsection+1)               # Shell thickness at each section node (linear lofting between) [m]
    prob['base_bulkhead_thickness'] = 0.05*np.array([1,1,0,0,0,0]) # Locations/thickness of internal bulkheads at section interfaces [m]

    # Auxiliary column geometry
    prob['radius_to_auxiliary_column']         = 33.333 * np.cos(np.pi/6) # Centerline of base column to centerline of auxiliary column [m]
    prob['auxiliary_permanent_ballast_height'] = 0.1                      # Height above keel for permanent ballast [m]
    prob['auxiliary_freeboard']                = 12.0                     # Height extension above waterline [m]
    prob['auxiliary_section_height']           = np.array([6.0, 0.1, 7.9, 8.0, 10]) # Length of each section [m]
    prob['auxiliary_outer_diameter']           = np.array([24, 24, 12, 12, 12, 12]) # Diameter at each section node (linear lofting between) [m]
    prob['auxiliary_wall_thickness']           = 0.06 * np.ones(nsection+1)         # Shell thickness at each section node (linear lofting between) [m]

    # Column ring stiffener parameters
    prob['base_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_flange_width']     = 0.10 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['base_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

    # Auxiliary column ring stiffener parameters
    prob['auxiliary_stiffener_web_height']       = 0.10 * np.ones(nsection) # (by section) [m]
    prob['auxiliary_stiffener_web_thickness']    = 0.04 * np.ones(nsection) # (by section) [m]
    prob['auxiliary_stiffener_flange_width']     = 0.01 * np.ones(nsection) # (by section) [m]
    prob['auxiliary_stiffener_flange_thickness'] = 0.02 * np.ones(nsection) # (by section) [m]
    prob['auxiliary_stiffener_spacing']          = 0.40 * np.ones(nsection) # (by section) [m]

    # Pontoon parameters
    prob['pontoon_outer_diameter']    = 3.2    # Diameter of all pontoon/truss elements [m]
    prob['pontoon_wall_thickness']    = 0.0175 # Thickness of all pontoon/truss elements [m]
    prob['base_pontoon_attach_lower'] = -20.0  # Lower z-coordinate on base where truss attaches [m]
    prob['base_pontoon_attach_upper'] = 10.0   # Upper z-coordinate on base where truss attaches [m]
    
    # Mooring parameters
    prob['number_of_mooring_connections'] = 3             # Evenly spaced around structure
    prob['mooring_lines_per_connection'] = 1             # Evenly spaced around structure
    prob['mooring_type']               = 'chain'       # Options are chain, nylon, polyester, fiber, or iwrc
    prob['anchor_type']                = 'suctionpile' # Options are SUCTIONPILE or DRAGEMBEDMENT
    prob['mooring_diameter']           = 0.0766        # Diameter of mooring line/chain [m]
    prob['fairlead']                   = 14.0          # Distance below waterline for attachment [m]
    prob['fairlead_offset_from_shell'] = 0.5           # Offset from shell surface for mooring attachment [m]
    prob['mooring_line_length']        = 835.5+300         # Unstretched mooring line length
    prob['anchor_radius']              = 837.6+300.0         # Distance from centerline to sea floor landing [m]
    prob['fairlead_support_outer_diameter'] = 3.2    # Diameter of all fairlead support elements [m]
    prob['fairlead_support_wall_thickness'] = 0.0175 # Thickness of all fairlead support elements [m]

    # Porperties of turbine tower
    prob['hub_height']              = 77.6                              # Length from tower base to top (not including freeboard) [m]
    prob['tower_section_height']    = 77.6/nsection * np.ones(nsection) # Length of each tower section [m]
    prob['tower_outer_diameter']    = np.linspace(6.5, 3.87, nsection+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_wall_thickness']    = np.linspace(0.027, 0.019, nsection+1) # Diameter at each tower section node (linear lofting between) [m]
    prob['tower_buckling_length']   = 30.0                              # Tower buckling reinforcement spacing [m]
    prob['tower_outfitting_factor'] = 1.07                              # Scaling for unaccounted tower mass in outfitting

    # Properties of rotor-nacelle-assembly (RNA)
    prob['rna_mass']   = 350e3 # Mass [kg]
    prob['rna_I']      = 1e5*np.array([1149.307, 220.354, 187.597, 0, 5.037, 0]) # Moment of intertia (xx,yy,zz,xy,xz,yz) [kg/m^2]
    prob['rna_cg']     = np.array([-1.132, 0, 0.509])                       # Offset of RNA center of mass from tower top (x,y,z) [m]
    # Max thrust
    prob['rna_force']  = np.array([1284744.196, 0, -112400.5527])           # Net force acting on RNA (x,y,z) [N]
    prob['rna_moment'] = np.array([3963732.762, 896380.8464, -346781.682]) # Net moment acting on RNA (x,y,z) [N*m]
    # Max wind speed
    #prob['rna_force']  = np.array([188038.8045, 0,  -16451.2637]) # Net force acting on RNA (x,y,z) [N]
    #prob['rna_moment'] = np.array([0.0, 131196.8431,  0.0]) # Net moment acting on RNA (x,y,z) [N*m]
    
    # Mooring constraints
    prob['mooring_max_offset'] = 0.1*prob['water_depth'] # Max surge/sway offset [m]      
    prob['mooring_operational_heel']   = 10.0 # Max heel (pitching) angle [deg]

    # Design constraints
    prob['max_taper_ratio'] = 0.2                # For manufacturability of rolling steel
    prob['min_diameter_thickness_ratio'] = 120.0 # For weld-ability
    prob['connection_ratio_max']      = 0.25 # For welding pontoons to columns

    # API 2U flag
    prob['loading'] = 'hydrostatic'
    
    prob.run()
    
    '''
    f = open('deriv_semi.dat','w')
    out = prob.check_total_derivatives(f)
    #out = prob.check_partial_derivatives(f, compact_print=True)
    f.close()
    tol = 1e-4
    for comp in out.keys():
        for k in out[comp].keys():
            if ( (out[comp][k]['rel error'][0] > tol) and (out[comp][k]['abs error'][0] > tol) ):
                print k
    '''

if __name__ == "__main__":
    from openmdao.api import Problem
    import sys

    if len(sys.argv) > 1 and sys.argv[1].lower() in ['spar','column','col','oc3']:
        sparExample()
    else:
        semiExample()
        
