from openmdao.api import Group, Problem
from spar import Spar
from sparGeometry import SparGeometry
from mapMooring import MapMooring

class SparAssembly(Group):

    def __init__(self):
        super(SparAssembly, self).__init__()

        # Define environment type variables here?  Link them to all?
        
        # Run Spar Geometry component first
        self.add('sg', SparGeometry())

        # Next run MapMooring
        self.add('mm', MapMooring())

        # Run main Spar analysis
        self.add('spar', Spar())

        # Connect SparGeometry and MapMooring variables that are the same
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')

        # Connect SparGeometry and MapMooring to the Spar
        self.connect('sg.water_depth', ['mm.water_depth', 'spar.water_depth'])
        self.connect('sg.fairlead', ['mm.fairlead','spar.fairlead'])
        self.connect('sg.freeboard', 'spar.freeboard')
        self.connect('sg.section_height', 'spar.section_height')
        self.connect('sg.outer_radius', 'spar.outer_radius')
        self.connect('sg.wall_thickness', 'spar.wall_thickness')
        self.connect('sg.draft', 'spar.draft')
        self.connect('sg.z_nodes', 'spar.z_nodes')
        self.connect('sg.z_section', 'spar.z_section')

        self.connect('mm.water_density', 'spar.water_density')
        self.connect('mm.total_mass', 'spar.mooring_mass')
        self.connect('mm.total_cost', 'spar.mooring_cost')
        self.connect('mm.vertical_load', 'spar.mooring_vertical_load')
        self.connect('mm.max_offset_restoring_force', 'spar.mooring_restoring_force')

        
        
def optimize_spar(params):

    # Setup the problem
    prob = Problem(SparAssembly)
    
    # Vectorize inputs as starting point
    nsections = int(params['number_of_section'])
    nodeOnes  = np.ones((nsections+1,))
    secOnes   = np.ones((nsections,))

    # INPUT PARAMETER SETTING
    # Parameters heading into Spar Geometry first
    prob['sg.water_depth'] = params['water_depth']
    prob['sg.freeboard'] = params['freeboard']
    prob['sg.fairlead'] = params['fairlead']
    prob['sg.section_height'] = (params['spar_length']/nsections) * secOnes
    prob['sg.outer_radius'] = params['outer_radius'] * nodeOnes
    prob['sg.wall_thickness'] = params['wall_thickness'] * nodeOnes
    prob['sg.fairlead_offset_from_shell'] = params['fairlead_offset_from_shell']

    # Parameters heading into MAP Mooring second
    prob['mm.scope_ratio'] = params['scope_ratio']
    prob['mm.anchor_radius'] = params['anchor_radius']
    prob['mm.mooring_diameter'] = params['mooring_diameter']
    prob['mm.number_of_mooring_lines'] = params['number_of_mooring_lines']
    prob['mm.mooring_type'] = params['mooring_type']
    prob['mm.anchor_type'] = params['anchor_type']
    prob['mm.max_offset'] = 0.1*params['water_depth'] # Assumption!
    prob['mm.mooring_cost_rate'] = params['mooring_cost_rate']
    prob['mm.anchor_cost_rate'] = params['anchor_cost_rate']
    prob['mm.misc_cost_factor'] = params['misc_cost_factor']

    #Parameters heading into Spar
    prob['spar.air_density'] = params['air_density']
    prob['spar.air_viscosity'] = params['air_viscosity']
    prob['spar.water_density'] = params['water_density']
    prob['spar.water_viscosity'] = params['water_viscosity']
    prob['spar.wave_height'] = params['wave_height']
    prob['spar.wave_period'] = params['wave_period']
    prob['spar.wind_reference_speed'] = params['wind_reference_speed']
    prob['spar.wind_reference_height'] = params['wind_reference_height']
    prob['spar.alpha'] = params['alpha']
    prob['spar.morison_mass_coefficient'] = params['morison_mass_coefficient']
    prob['spar.material_density'] = params['material_density']
    prob['spar.E'] = params['E']
    prob['spar.nu'] = params['nu']
    prob['spar.yield_stress'] = params['yield_stress']
    prob['spar.permanent_ballast_density'] = params['permanent_ballast_density']
    
    prob['spar.stiffener_web_height'] = params['stiffener_web_height'] * secOnes
    prob['spar.stiffener_web_thickness'] = params['stiffener_web_thickness'] * secOnes
    prob['spar.stiffener_flange_width'] = params['stiffener_flange_width'] * secOnes
    prob['spar.stiffener_flange_thickness'] = params['stiffener_flange_thickness'] * secOnes
    prob['spar.stiffener_spacing'] = params['stiffener_spacing'] * secOnes
    prob['spar.bulkhead_nodes'] = [True] * nsections
    prob['spar.permanent_ballast_height'] = params['permanent_ballast_height']
    
    prob['spar.bulkhead_mass_factor'] = params['bulkhead_mass_factor']
    prob['spar.ring_mass_factor'] = params['ring_mass_factor']
    prob['spar.spar_mass_factor'] = params['spar_mass_factor']
    prob['spar.shell_mass_factor'] = params['shell_mass_factor']
    prob['spar.outfitting_mass_factor'] = params['outfitting_mass_factor']
    prob['spar.ballast_cost_rate'] = params['ballast_cost_rate']
    prob['spar.tapered_col_cost_rate'] = params['tapered_col_cost_rate']
    prob['spar.outfitting_cost_rate'] = params['outfitting_cost_rate']
    
    prob['spar.rna_mass'] = params['rna_mass']
    prob['spar.rna_center_of_gravity'] = params['rna_center_of_gravity']
    prob['spar.rna_center_of_gravity_x'] = params['rna_center_of_gravity_x']
    prob['spar.rna_wind_force'] = params['rna_wind_force']
    prob['spar.tower_mass'] = params['tower_mass']
    prob['spar.tower_center_of_gravity'] = params['tower_center_of_gravity']
    prob['spar.tower_wind_force'] = params['tower_wind_force']

    
    # Establish the optimization driver, then set design variables and constraints
    prob.driver = ScipyOptimizer()

    
    # DESIGN VARIABLES
    prob.driver.add_desvar('sg.freeboard',lower=0.0)
    prob.driver.add_desvar('sg.fairlead',lower=0.0)
    prob.driver.add_desvar('sg.fairlead_offset_from_shell',lower=0.0)
    prob.driver.add_desvar('sg.section_height',lower=lower=1e-2)
    prob.driver.add_desvar('sg.outer_radius',lower=lower=1.0)
    prob.driver.add_desvar('sg.wall_thickness',lower=1e-2)

    prob.driver.add_desvar('mm.scope_ratio', lower=1.0) #>1 means longer than water depth
    prob.driver.add_desvar('mm.anchor_radius', lower=0.0)
    prob.driver.add_desvar('mm.mooring_diameter', lower=1e-2)
    prob.driver.add_desvar('mm.number_of_mooring_lines', lower=1)
    # TODO: Integer design variables
    #prob.driver.add_desvar('mm.mooring_type')
    #prob.driver.add_desvar('mm.anchor_type')

    prob.driver.add_desvar('spar.stiffener_web_height', lower=1e-3)
    prob.driver.add_desvar('spar.stiffener_web_thickness', lower=1e-3)
    prob.driver.add_desvar('spar.stiffener_flange_width', lower=1e-3)
    prob.driver.add_desvar('spar.stiffener_flange_thickness', lower=1e-3)
    prob.driver.add_desvar('spar.stiffener_spacing', lower=1e-2)
    prob.driver.add_desvar('spar.permanent_ballast_height', lower=0.0)
    # TODO: Boolean design variables
    #prob.driver.add_desvar('spar.bulkhead_nodes')

    
    # CONSTRAINTS
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=1.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

    # API Bulletin 2U constraints
    prob.driver.add_constraint('flange_compactness', lower=1.0)
    prob.driver.add_constraint('web_compactness', lower=1.0)
    prob.driver.add_constraint('axial_local_unity', upper=1.0)
    prob.driver.add_constraint('axial_general_unity', upper=1.0)
    prob.driver.add_constraint('external_local_unity', upper=1.0)
    prob.driver.add_constraint('external_general_unity', upper=1.0)

    # Metacentric height should be positive for static stability
    prob.driver.add_constraint('metacentric_height', lower=0.0)

    # Center of bouyancy should be above CG (difference should be positive)
    prob.driver.add_constraint('static_stability', lower=0.0)

    # Achieving non-zero variable ballast height means the spar can be balanced
    prob.driver.add_constraint('variable_ballast_height', lower=0.0, upper=params['spar_length'])

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    prob.driver.add_constraint('offset_force_ratio', lower=0.0, upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    prob.driver.add_constraint('heel_angle', upper=10.0)

    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('total_cost', val=0.0, units='USD', desc='cost of spar structure')
        
    
    prob.setup()
    prob.run()
