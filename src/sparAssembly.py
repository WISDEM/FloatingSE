from openmdao.api import Group, Problem, ScipyOptimizer
from spar import Spar
from sparGeometry import SparGeometry
from mapMooring import MapMooring
import numpy as np

class SparAssembly(Group):

    def __init__(self):
        super(SparAssembly, self).__init__()

        # Define environment type variables here?  Link them to all?
        
        # Run Spar Geometry component first
        self.add('sg', SparGeometry())

        # Next run MapMooring
        self.add('mm', MapMooring())

        # Run main Spar analysis
        self.add('sp', Spar())

        # Connect SparGeometry and MapMooring variables that are the same
        self.connect('sg.fairlead_radius', 'mm.fairlead_radius')

        # Connect SparGeometry and MapMooring to the Spar
        self.connect('sg.water_depth', ['mm.water_depth', 'sp.water_depth'])
        self.connect('sg.fairlead', ['mm.fairlead','sp.fairlead'])
        self.connect('sg.freeboard', 'sp.freeboard')
        self.connect('sg.section_height', 'sp.section_height')
        self.connect('sg.outer_radius', 'sp.outer_radius')
        self.connect('sg.wall_thickness', 'sp.wall_thickness')
        self.connect('sg.draft', 'sp.draft')
        self.connect('sg.z_nodes', 'sp.z_nodes')
        self.connect('sg.z_section', 'sp.z_section')

        self.connect('mm.water_density', 'sp.water_density')
        self.connect('mm.total_mass', 'sp.mooring_mass')
        self.connect('mm.total_cost', 'sp.mooring_cost')
        self.connect('mm.vertical_load', 'sp.mooring_vertical_load')
        self.connect('mm.max_offset_restoring_force', 'sp.mooring_restoring_force')

        
        
def optimize_spar(params):

    # Setup the problem
    prob = Problem()
    prob.root = SparAssembly()
    prob.setup()
    
    # Vectorize inputs as starting point
    nsections = int(params['number_of_sections'])
    nodeOnes  = np.ones((nsections+1,))
    secOnes   = np.ones((nsections,))

    # INPUT PARAMETER SETTING
    # Parameters heading into Spar Geometry first
    prob['sg.water_depth']                = params['water_depth']
    prob['sg.freeboard']                  = params['freeboard']
    prob['sg.fairlead']                   = params['fairlead']
    prob['sg.section_height']             = (params['spar_length']/nsections) * secOnes
    prob['sg.outer_radius']               = params['outer_radius'] * nodeOnes
    prob['sg.wall_thickness']             = params['wall_thickness'] * nodeOnes
    prob['sg.fairlead_offset_from_shell'] = params['fairlead_offset_from_shell']

    # Parameters heading into MAP Mooring second
    prob['mm.scope_ratio']             = params['scope_ratio']
    prob['mm.anchor_radius']           = params['anchor_radius']
    prob['mm.mooring_diameter']        = params['mooring_diameter']
    prob['mm.number_of_mooring_lines'] = params['number_of_mooring_lines']
    prob['mm.mooring_type']            = params['mooring_type']
    prob['mm.anchor_type']             = params['anchor_type']
    prob['mm.max_offset']              = 0.1*params['water_depth'] # Assumption!
    prob['mm.mooring_cost_rate']       = params['mooring_cost_rate']

    #Parameters heading into Spar
    prob['sp.air_density']               = params['air_density']
    prob['sp.air_viscosity']             = params['air_viscosity']
    prob['sp.water_density']             = params['water_density']
    prob['sp.water_viscosity']           = params['water_viscosity']
    prob['sp.wave_height']               = params['wave_height']
    prob['sp.wave_period']               = params['wave_period']
    prob['sp.wind_reference_speed']      = params['wind_reference_speed']
    prob['sp.wind_reference_height']     = params['wind_reference_height']
    prob['sp.alpha']                     = params['alpha']
    prob['sp.morison_mass_coefficient']  = params['morison_mass_coefficient']
    prob['sp.material_density']          = params['material_density']
    prob['sp.E']                         = params['E']
    prob['sp.nu']                        = params['nu']
    prob['sp.yield_stress']              = params['yield_stress']
    prob['sp.permanent_ballast_density'] = params['permanent_ballast_density']
    
    prob['sp.stiffener_web_height']       = params['stiffener_web_height'] * secOnes
    prob['sp.stiffener_web_thickness']    = params['stiffener_web_thickness'] * secOnes
    prob['sp.stiffener_flange_width']     = params['stiffener_flange_width'] * secOnes
    prob['sp.stiffener_flange_thickness'] = params['stiffener_flange_thickness'] * secOnes
    prob['sp.stiffener_spacing']          = params['stiffener_spacing'] * secOnes
    prob['sp.bulkhead_nodes']             = [True] * nsections
    prob['sp.permanent_ballast_height']   = params['permanent_ballast_height']
    
    prob['sp.bulkhead_mass_factor']     = params['bulkhead_mass_factor']
    prob['sp.ring_mass_factor']         = params['ring_mass_factor']
    prob['sp.spar_mass_factor']         = params['spar_mass_factor']
    prob['sp.shell_mass_factor']        = params['shell_mass_factor']
    prob['sp.outfitting_mass_fraction'] = params['outfitting_mass_fraction']
    prob['sp.ballast_cost_rate']        = params['ballast_cost_rate']
    prob['sp.tapered_col_cost_rate']    = params['tapered_col_cost_rate']
    prob['sp.outfitting_cost_rate']     = params['outfitting_cost_rate']
    
    prob['sp.rna_mass']                = params['rna_mass']
    prob['sp.rna_center_of_gravity']   = params['rna_center_of_gravity']
    prob['sp.rna_center_of_gravity_x'] = params['rna_center_of_gravity_x']
    prob['sp.rna_wind_force']          = params['rna_wind_force']
    prob['sp.tower_mass']              = params['tower_mass']
    prob['sp.tower_center_of_gravity'] = params['tower_center_of_gravity']
    prob['sp.tower_wind_force']        = params['tower_wind_force']

    
    # Establish the optimization driver, then set design variables and constraints
    prob.driver = ScipyOptimizer() #COBYLAdriver()

    
    # DESIGN VARIABLES
    prob.driver.add_desvar('sg.freeboard',lower=0.0)
    prob.driver.add_desvar('sg.fairlead',lower=0.0)
    prob.driver.add_desvar('sg.fairlead_offset_from_shell',lower=0.0)
    prob.driver.add_desvar('sg.section_height',lower=1e-2)
    prob.driver.add_desvar('sg.outer_radius',lower=1.0)
    prob.driver.add_desvar('sg.wall_thickness',lower=1e-2)

    prob.driver.add_desvar('mm.scope_ratio', lower=1.0) #>1 means longer than water depth
    prob.driver.add_desvar('mm.anchor_radius', lower=0.0)
    prob.driver.add_desvar('mm.mooring_diameter', lower=1e-2)
    prob.driver.add_desvar('mm.number_of_mooring_lines', lower=1)
    # TODO: Integer design variables
    #prob.driver.add_desvar('mm.mooring_type')
    #prob.driver.add_desvar('mm.anchor_type')

    prob.driver.add_desvar('sp.stiffener_web_height', lower=1e-3)
    prob.driver.add_desvar('sp.stiffener_web_thickness', lower=1e-3)
    prob.driver.add_desvar('sp.stiffener_flange_width', lower=1e-3)
    prob.driver.add_desvar('sp.stiffener_flange_thickness', lower=1e-3)
    prob.driver.add_desvar('sp.stiffener_spacing', lower=1e-2)
    prob.driver.add_desvar('sp.permanent_ballast_height', lower=0.0)
    # TODO: Boolean design variables
    #prob.driver.add_desvar('sp.bulkhead_nodes')

    
    # CONSTRAINTS
    # Ensure that draft is greater than 0 (spar length>0) and that less than water depth
    prob.driver.add_constraint('sg.draft_depth_ratio',lower=0.0, upper=1.0)

    # Ensure max mooring line tension is less than X% of MBL: 60% for intact mooring, 80% for damanged
    prob.driver.add_constraint('mm.safety_factor',lower=0.0, upper=0.8)

    # API Bulletin 2U constraints
    prob.driver.add_constraint('sp.flange_compactness', lower=1.0)
    prob.driver.add_constraint('sp.web_compactness', lower=1.0)
    prob.driver.add_constraint('sp.axial_local_unity', upper=1.0)
    prob.driver.add_constraint('sp.axial_general_unity', upper=1.0)
    prob.driver.add_constraint('sp.external_local_unity', upper=1.0)
    prob.driver.add_constraint('sp.external_general_unity', upper=1.0)

    # Metacentric height should be positive for static stability
    prob.driver.add_constraint('sp.metacentric_height', lower=0.0)

    # Center of bouyancy should be above CG (difference should be positive)
    prob.driver.add_constraint('sp.static_stability', lower=0.0)

    # Achieving non-zero variable ballast height means the spar can be balanced
    prob.driver.add_constraint('sp.variable_ballast_height', lower=0.0, upper=params['spar_length'])

    # Surge restoring force should be greater than wave-wind forces (ratio < 1)
    prob.driver.add_constraint('sp.offset_force_ratio', lower=0.0, upper=1.0)

    # Heel angle should be less than 6deg for ordinary operation, less than 10 for extreme conditions
    prob.driver.add_constraint('sp.heel_angle', upper=10.0)

    
    # OBJECTIVE FUNCTION: Minimize total cost!
    prob.driver.add_objective('sp.total_cost')
        
    # Execute the optimization
    prob.setup()
    prob.run()

    return prob



if __name__ == '__main__':
    params = {}
    params['water_depth'] = 218.0
    params['freeboard'] = 15.0
    params['fairlead'] = 10.0
    params['spar_length'] = 75.0
    params['number_of_sections'] = 5
    params['outer_radius'] = 3.0
    params['wall_thickness'] = 0.05
    params['fairlead_offset_from_shell'] = 0.5
    params['scope_ratio'] = 1.5
    params['anchor_radius'] = 50.0
    params['mooring_diameter'] = 0.1
    params['number_of_mooring_lines'] = 3
    params['mooring_type'] = 'chain'
    params['anchor_type'] = 'pile'
    params['mooring_cost_rate'] = 1.1
    params['air_density'] = 1.198
    params['air_viscosity'] = 1.81e-5
    params['water_density'] = 1025.0
    params['water_viscosity'] = 8.9e-4
    params['wave_height'] = 10.8
    params['wave_period'] = 9.8
    params['wind_reference_speed'] = 11.0
    params['wind_reference_height'] = 97.0
    params['alpha'] = 0.11
    params['morison_mass_coefficient'] = 2.0
    params['material_density'] = 7850.0
    params['E'] = 200e9
    params['nu'] = 0.3
    params['yield_stress'] = 3.45e8
    params['permanent_ballast_density'] = 4492.0
    params['stiffener_web_height']= 0.1
    params['stiffener_web_thickness'] = 0.01
    params['stiffener_flange_width'] = 0.1
    params['stiffener_flange_thickness'] = 0.01
    params['stiffener_spacing'] = 0.2
    params['permanent_ballast_height'] = 3.0
    params['bulkhead_mass_factor'] = 1.0
    params['ring_mass_factor'] = 1.0
    params['shell_mass_factor'] = 1.0
    params['spar_mass_factor'] = 1.05
    params['outfitting_mass_fraction'] = 0.06
    params['ballast_cost_rate'] = 100.0
    params['tapered_col_cost_rate'] = 4720.0
    params['outfitting_cost_rate'] = 6980.0
    params['rna_mass']= 3.655e5
    params['rna_center_of_gravity'] = 3.5 + 80.0
    params['rna_center_of_gravity_x'] = 5.75
    params['tower_mass'] = 3.66952e5
    params['tower_center_of_gravity'] = 35.0
    params['rna_wind_force'] = 1e4 # TODO: Better guess
    params['tower_wind_force'] = 1e4 # TODO: Better guess

    prob = optimize_spar(params)
    
